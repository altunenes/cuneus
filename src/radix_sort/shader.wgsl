// GPU Radix Sort Shader
// Ported from wgpu_sort (BSD 2-Clause License)
// Copyright (c) 2024, Simon Niedermayr, Josef Stumpfegger

/*
BSD 2-Clause License
Copyright (c) 2024, Simon Niedermayr, Josef Stumpfegger 
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

struct GeneralInfo {
    num_keys: u32,
    padded_size: u32,
    even_pass: u32,
    odd_pass: u32,
    sort_failed: atomic<u32>,
};

@group(0) @binding(0) var<storage, read_write> infos: GeneralInfo;
@group(0) @binding(1) var<storage, read_write> histograms: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> keys: array<u32>;
@group(0) @binding(3) var<storage, read_write> keys_b: array<u32>;
@group(0) @binding(4) var<storage, read_write> payload_a: array<u32>;
@group(0) @binding(5) var<storage, read_write> payload_b: array<u32>;

var<workgroup> smem: array<atomic<u32>, rs_radix_size>;
var<private> kv: array<u32, rs_histogram_block_rows>;

fn zero_smem(lid: u32) {
    if lid < rs_radix_size {
        atomicStore(&smem[lid], 0u);
    }
}

@compute @workgroup_size({histogram_wg_size})
fn zero_histograms(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    if gid.x == 0u {
        infos.even_pass = 0u;
        infos.odd_pass = 1u;
        atomicStore(&infos.sort_failed, 0u);  // Reset global failure flag
    }

    let scatter_wg_size = histogram_wg_size;
    let scatter_block_kvs = scatter_wg_size * rs_scatter_block_rows;
    let scatter_blocks_ru = (infos.num_keys + scatter_block_kvs - 1u) / scatter_block_kvs;
    let histo_size = rs_radix_size;

    var n = (rs_keyval_size + scatter_blocks_ru - 1u) * histo_size;
    let b = n;
    if infos.num_keys < infos.padded_size {
        n += infos.padded_size - infos.num_keys;
    }

    let line_size = nwg.x * {histogram_wg_size}u;
    for (var cur_index = gid.x; cur_index < n; cur_index += line_size) {
        if cur_index < rs_keyval_size * histo_size {
            atomicStore(&histograms[cur_index], 0u);
        } else if cur_index < b {
            atomicStore(&histograms[cur_index], 0u);
        } else {
            keys[infos.num_keys + cur_index - b] = 0xFFFFFFFFu;
        }
    }
}

fn histogram_pass(pass_: u32, lid: u32) {
    zero_smem(lid);
    workgroupBarrier();

    for (var j = 0u; j < rs_histogram_block_rows; j++) {
        let u_val = kv[j];
        let digit = extractBits(u_val, pass_ * rs_radix_log2, rs_radix_log2);
        atomicAdd(&smem[digit], 1u);
    }

    workgroupBarrier();
    let histogram_offset = rs_radix_size * pass_ + lid;
    if lid < rs_radix_size && atomicLoad(&smem[lid]) >= 0u {
        atomicAdd(&histograms[histogram_offset], atomicLoad(&smem[lid]));
    }
}

fn fill_kv(wid: u32, lid: u32) {
    let rs_block_keyvals = rs_histogram_block_rows * histogram_wg_size;
    let kv_in_offset = wid * rs_block_keyvals + lid;
    for (var i = 0u; i < rs_histogram_block_rows; i++) {
        let pos = kv_in_offset + i * histogram_wg_size;
        kv[i] = keys[pos];
    }
}

fn fill_kv_keys_b(wid: u32, lid: u32) {
    let rs_block_keyvals = rs_histogram_block_rows * histogram_wg_size;
    let kv_in_offset = wid * rs_block_keyvals + lid;
    for (var i = 0u; i < rs_histogram_block_rows; i++) {
        let pos = kv_in_offset + i * histogram_wg_size;
        kv[i] = keys_b[pos];
    }
}

@compute @workgroup_size({histogram_wg_size})
fn calculate_histogram(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    fill_kv(wid.x, lid.x);
    histogram_pass(3u, lid.x);
    histogram_pass(2u, lid.x);
    histogram_pass(1u, lid.x);
    histogram_pass(0u, lid.x);
}

fn prefix_reduce_smem(lid: u32) {
    var offset = 1u;
    for (var d = rs_radix_size >> 1u; d > 0u; d = d >> 1u) {
        workgroupBarrier();
        if lid < d {
            let ai = offset * (2u * lid + 1u) - 1u;
            let bi = offset * (2u * lid + 2u) - 1u;
            atomicAdd(&smem[bi], atomicLoad(&smem[ai]));
        }
        offset = offset << 1u;
    }

    if lid == 0u {
        atomicStore(&smem[rs_radix_size - 1u], 0u);
    }

    for (var d = 1u; d < rs_radix_size; d = d << 1u) {
        offset = offset >> 1u;
        workgroupBarrier();
        if lid < d {
            let ai = offset * (2u * lid + 1u) - 1u;
            let bi = offset * (2u * lid + 2u) - 1u;
            let t = atomicLoad(&smem[ai]);
            atomicStore(&smem[ai], atomicLoad(&smem[bi]));
            atomicAdd(&smem[bi], t);
        }
    }
}

@compute @workgroup_size({prefix_wg_size})
fn prefix_histogram(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let histogram_base = (rs_keyval_size - 1u - wid.x) * rs_radix_size;
    let histogram_offset = histogram_base + lid.x;

    atomicStore(&smem[lid.x], atomicLoad(&histograms[histogram_offset]));
    atomicStore(&smem[lid.x + {prefix_wg_size}u], atomicLoad(&histograms[histogram_offset + {prefix_wg_size}u]));

    prefix_reduce_smem(lid.x);
    workgroupBarrier();

    atomicStore(&histograms[histogram_offset], atomicLoad(&smem[lid.x]));
    atomicStore(&histograms[histogram_offset + {prefix_wg_size}u], atomicLoad(&smem[lid.x + {prefix_wg_size}u]));
}

// Scatter variables
var<workgroup> scatter_smem: array<u32, rs_mem_dwords>;
var<workgroup> scatter_failed: atomic<u32>;
var<private> kr: array<u32, rs_scatter_block_rows>;
var<private> pv: array<u32, rs_scatter_block_rows>;

const rs_partition_mask_status: u32 = 0xC0000000u;
const rs_partition_mask_count: u32 = 0x3FFFFFFFu;

fn partitions_base_offset() -> u32 { return rs_keyval_size * rs_radix_size; }
fn smem_prefix_offset() -> u32 { return rs_radix_size + rs_radix_size; }

fn histogram_load(digit: u32) -> u32 {
    return atomicLoad(&smem[digit]);
}

fn histogram_store(digit: u32, count: u32) {
    atomicStore(&smem[digit], count);
}

fn fill_kv_even(wid: u32, lid: u32) {
    let subgroup_id = lid / histogram_sg_size;
    let subgroup_invoc_id = lid - subgroup_id * histogram_sg_size;
    let subgroup_keyvals = rs_scatter_block_rows * histogram_sg_size;
    let rs_block_keyvals = rs_histogram_block_rows * histogram_wg_size;
    let kv_in_offset = wid * rs_block_keyvals + subgroup_id * subgroup_keyvals + subgroup_invoc_id;

    for (var i = 0u; i < rs_histogram_block_rows; i++) {
        let pos = kv_in_offset + i * histogram_sg_size;
        kv[i] = keys[pos];
    }
    for (var i = 0u; i < rs_histogram_block_rows; i++) {
        let pos = kv_in_offset + i * histogram_sg_size;
        pv[i] = payload_a[pos];
    }
}

fn fill_kv_odd(wid: u32, lid: u32) {
    let subgroup_id = lid / histogram_sg_size;
    let subgroup_invoc_id = lid - subgroup_id * histogram_sg_size;
    let subgroup_keyvals = rs_scatter_block_rows * histogram_sg_size;
    let rs_block_keyvals = rs_histogram_block_rows * histogram_wg_size;
    let kv_in_offset = wid * rs_block_keyvals + subgroup_id * subgroup_keyvals + subgroup_invoc_id;

    for (var i = 0u; i < rs_histogram_block_rows; i++) {
        let pos = kv_in_offset + i * histogram_sg_size;
        kv[i] = keys_b[pos];
    }
    for (var i = 0u; i < rs_histogram_block_rows; i++) {
        let pos = kv_in_offset + i * histogram_sg_size;
        pv[i] = payload_b[pos];
    }
}

fn scatter(pass_: u32, lid: vec3<u32>, gid: vec3<u32>, wid: vec3<u32>, nwg: vec3<u32>,
           partition_status_invalid: u32, partition_status_reduction: u32, partition_status_prefix: u32) -> bool {
    // Initialize failure flag
    if lid.x == 0u {
        atomicStore(&scatter_failed, 0u);
    }
    workgroupBarrier();

    let partition_mask_invalid = partition_status_invalid << 30u;
    let partition_mask_reduction = partition_status_reduction << 30u;
    let partition_mask_prefix = partition_status_prefix << 30u;

    let subgroup_id = lid.x / histogram_sg_size;
    let subgroup_offset = subgroup_id * histogram_sg_size;
    let subgroup_tid = lid.x - subgroup_offset;
    let subgroup_count = {scatter_wg_size}u / histogram_sg_size;

    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        let u_val = kv[i];
        let digit = extractBits(u_val, pass_ * rs_radix_log2, rs_radix_log2);
        atomicStore(&smem[lid.x], digit);
        var count = 0u;
        var rank = 0u;

        for (var j = 0u; j < histogram_sg_size; j++) {
            if atomicLoad(&smem[subgroup_offset + j]) == digit {
                count += 1u;
                if j <= subgroup_tid {
                    rank += 1u;
                }
            }
        }
        kr[i] = (count << 16u) | rank;
    }

    zero_smem(lid.x);
    workgroupBarrier();

    for (var i = 0u; i < subgroup_count; i++) {
        if subgroup_id == i {
            for (var j = 0u; j < rs_scatter_block_rows; j++) {
                let v = kv[j];
                let digit = extractBits(v, pass_ * rs_radix_log2, rs_radix_log2);
                let prev = histogram_load(digit);
                let rank = kr[j] & 0xFFFFu;
                let count = kr[j] >> 16u;
                kr[j] = prev + rank;
                if rank == count {
                    histogram_store(digit, prev + count);
                }
            }
        }
        workgroupBarrier();
    }

    let partition_offset = lid.x + partitions_base_offset();
    let partition_base = wid.x * rs_radix_size;

    if wid.x == 0u {
        let hist_offset = pass_ * rs_radix_size + lid.x;
        if lid.x < rs_radix_size {
            let exc = atomicLoad(&histograms[hist_offset]);
            let red = histogram_load(lid.x);
            scatter_smem[lid.x] = exc;
            let inc = exc + red;
            atomicStore(&histograms[partition_offset], inc | partition_mask_prefix);
        }
    } else {
        if lid.x < rs_radix_size && wid.x < nwg.x - 1u {
            let red = histogram_load(lid.x);
            atomicStore(&histograms[partition_offset + partition_base], red | partition_mask_reduction);
        }

        if lid.x < rs_radix_size {
            var partition_base_prev = partition_base - rs_radix_size;
            var exc = 0u;
            var spin_count = 0u;
            let max_spins = 1000000u;

            loop {
                spin_count += 1u;
                // Check if another workgroup already failed - exit early
                if spin_count % 1000u == 0u && atomicLoad(&infos.sort_failed) != 0u {
                    atomicStore(&scatter_failed, 1u);
                    break;
                }
                if spin_count > max_spins {
                    // Deadlock prevention: signal failure globally and abort
                    atomicStore(&scatter_failed, 1u);
                    atomicStore(&infos.sort_failed, 1u);  // Global flag
                    break;
                }
                let prev = atomicLoad(&histograms[partition_base_prev + partition_offset]);
                if (prev & rs_partition_mask_status) == partition_mask_invalid {
                    continue;
                }
                exc += prev & rs_partition_mask_count;
                if (prev & rs_partition_mask_status) != partition_mask_prefix {
                    partition_base_prev -= rs_radix_size;
                    continue;
                }
                scatter_smem[lid.x] = exc;
                if wid.x < nwg.x - 1u {
                    atomicAdd(&histograms[partition_offset + partition_base], exc | (1u << 30u));
                }
                break;
            }
        }
    }

    prefix_reduce_smem(lid.x);
    workgroupBarrier();

    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        let v = kv[i];
        let digit = extractBits(v, pass_ * rs_radix_log2, rs_radix_log2);
        let exc = histogram_load(digit);
        let idx = exc + kr[i];
        kr[i] |= (idx << 16u);
    }
    workgroupBarrier();

    let smem_reorder_offset = rs_radix_size;
    let smem_base = smem_reorder_offset + lid.x;

    for (var j = 0u; j < rs_scatter_block_rows; j++) {
        let smem_idx = smem_reorder_offset + (kr[j] >> 16u) - 1u;
        scatter_smem[smem_idx] = kv[j];
    }
    workgroupBarrier();

    for (var j = 0u; j < rs_scatter_block_rows; j++) {
        kv[j] = scatter_smem[smem_base + j * {scatter_wg_size}u];
    }
    workgroupBarrier();

    for (var j = 0u; j < rs_scatter_block_rows; j++) {
        let smem_idx = smem_reorder_offset + (kr[j] >> 16u) - 1u;
        scatter_smem[smem_idx] = pv[j];
    }
    workgroupBarrier();

    for (var j = 0u; j < rs_scatter_block_rows; j++) {
        pv[j] = scatter_smem[smem_base + j * {scatter_wg_size}u];
    }
    workgroupBarrier();

    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        let smem_idx = smem_reorder_offset + (kr[i] >> 16u) - 1u;
        scatter_smem[smem_idx] = kr[i];
    }
    workgroupBarrier();

    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        kr[i] = scatter_smem[smem_base + i * {scatter_wg_size}u] & 0xFFFFu;
    }

    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        let v = kv[i];
        let digit = extractBits(v, pass_ * rs_radix_log2, rs_radix_log2);
        let exc = scatter_smem[digit];
        kr[i] += exc - 1u;
    }

    // Return true if scatter succeeded, false if deadlock was detected
    workgroupBarrier();
    return atomicLoad(&scatter_failed) == 0u;
}

@compute @workgroup_size({scatter_wg_size})
fn scatter_even(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
                @builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    if gid.x == 0u {
        infos.odd_pass = (infos.odd_pass + 1u) % 2u;
    }
    let cur_pass = infos.even_pass * 2u;

    fill_kv_even(wid.x, lid.x);
    let success = scatter(cur_pass, lid, gid, wid, nwg, 0u, 1u, 2u);

    // Only write if scatter succeeded - otherwise keep previous order
    if success {
        for (var i = 0u; i < rs_scatter_block_rows; i++) {
            keys_b[kr[i]] = kv[i];
        }
        for (var i = 0u; i < rs_scatter_block_rows; i++) {
            payload_b[kr[i]] = pv[i];
        }
    }
}

@compute @workgroup_size({scatter_wg_size})
fn scatter_odd(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
               @builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    if gid.x == 0u {
        infos.even_pass = (infos.even_pass + 1u) % 2u;
    }
    let cur_pass = infos.odd_pass * 2u + 1u;

    fill_kv_odd(wid.x, lid.x);
    let success = scatter(cur_pass, lid, gid, wid, nwg, 2u, 3u, 0u);

    // Only write if scatter succeeded - otherwise keep previous order
    if success {
        for (var i = 0u; i < rs_scatter_block_rows; i++) {
            keys[kr[i]] = kv[i];
        }
        for (var i = 0u; i < rs_scatter_block_rows; i++) {
            payload_a[kr[i]] = pv[i];
        }
    }
}

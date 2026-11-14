// 2D FFT workflow with butterworth filter, Enes Altun 2025.
//  Some FFT operations adapted from:
//  fadaaszhi, 2025: https://compute.toys/view/1187: Fast Fourier Transform
//  FabriceNeyret2, 2023 https://www.shadertoy.com/view/DtGfWV : Fourier Workflow 3 / phases info
//  FabriceNeyret2, 2017  https://www.shadertoy.com/view/XtScWt: Fourier Workflow 2 / phases info
//  SPIR-V-based Stockham + mixed-radix kernels   https://github.com/DTolm/VkFFT
//  Moreland, K., & Angel, E. (2003). The FFT on a GPU. SIGGRAPH/EUROGRAPHICS Conference On Graphics Hardware.
//
// V1 model: zero-mean Gabor simple cells (Jones & Palmer 1987; Daugman 1985), complex cells as energy
//  (Adelson & Bergen 1985), Naka-Rushton contrast response and divisive normalisation (Heeger 1992),
//  on cone-opponent DKL axes (Derrington, Krauskopf & Lennie 1984), cones after Vienot, Brettel & Mollon 1999.
//
// Frequencies are in cycles per image, measured along the image's long side. The image is
// letterboxed into the N x N square with its mean colour, so its aspect is kept.

const RADIX = 4;

struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

struct FFTParams {
    // 0 none, 1 low-pass, 2 high-pass, 3 band-pass, 4 orientation, 5 Gabor (V1)
    filter_type: i32,
    // low/high-pass cutoff, cycles per image
    cutoff: f32,
    // Butterworth order
    order: f32,
    // band-pass / Gabor centre (cycles per image) and full width at half height (octaves)
    band_center: f32,
    band_octaves: f32,
    // kept orientation of the image structure (degrees from horizontal) and its full width at half height
    orientation: f32,
    orient_width: f32,
    // 0 filtered, 1 original, 2 spectrum
    view: i32,
    resolution: u32,
    is_bw: i32,
    // Tukey window on the image edges, and the fraction of the image it tapers
    window: i32,
    keep_mean: i32,
    // decades of amplitude shown in the spectrum view
    spec_decades: f32,
    show_radial: i32,
    taper: f32,
    // phase scramble amount 0..1 and its seed
    phase_amount: f32,
    phase_seed: u32,
    // reshape the amplitude spectrum to 1/f^slope_target
    slope_on: i32,
    slope_target: f32,
    // 1 on frames where the FFT is recomputed (a change, or video); otherwise the last result is shown
    run: u32,
    // 0 even cell, 1 odd cell, 2 complex cell
    gabor_part: i32,
    // preferred SF (c/deg), image width (deg), SF bandwidth (oct), orientation and its bandwidth (deg, FWHH)
    v1_cpd: f32,
    v1_deg: f32,
    v1_bw: f32,
    v1_ori: f32,
    v1_ow: f32,
    // 0 luminance, 1 L-M, 2 S-(L+M), 3 all
    v1_chan: i32,
    // 8-orientation population
    v1_pop: i32,
    // Naka-Rushton semi-saturation contrast
    v1_c50: f32,
    v1_norm: i32,
    v1_gain: f32,
    v1_rf: i32,
    // animation: sweep period (s), orientation spin (deg/s), SF and cutoff sweeps (oct), phase drift (deg/s),
    // scramble and slope swings; 0 = still
    an_per: f32,
    an_rot: f32,
    an_sf: f32,
    an_cut: f32,
    an_ph: f32,
    an_scr: f32,
    an_slp: f32,
    show_key: i32,
};
// Group 1: Primary Pass I/O & Parameters
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: FFTParams;
@group(1) @binding(2) var input_texture: texture_2d<f32>;
@group(1) @binding(3) var input_sampler: sampler;

// font atlas (16 x 16 ASCII grid), for the slopes printed in the radial plot
struct FontUniforms { atlas_size: vec2f, char_size: vec2f, screen_size: vec2f, grid_size: vec2f };
@group(2) @binding(0) var<uniform> font: FontUniforms;
@group(2) @binding(1) var font_atlas: texture_2d<f32>;

// Storage buffer for FFT data
@group(3) @binding(0) var<storage, read_write> image_data: array<vec2f>;
// measurements: radial amplitude of the input and the result, the 1/f fits and the image mean
@group(3) @binding(1) var<storage, read_write> stats: array<f32>;
const S_IN = 0u;
const S_OUT = 1024u;
// alpha_in, intercept_in, alpha_out, intercept_out (log10 amplitude)
const S_FIT = 2048u;
const S_MEAN = 2052u;

const PI = 3.1415927;

fn lg10(x: f32) -> f32 { return log2(x) * 0.30103; }
const LOG2_N_MAX = 11;
const N_MAX = 2048;
const N_CHANNELS = 3u;
const LUMA = vec3f(0.2126, 0.7152, 0.0722);
// linear rgb -> L, M, S normalised to white (Vienot et al. 1999), and back
const LMS = mat3x3f(vec3f(0.272939, 0.100227, 0.017817), vec3f(0.664187, 0.787611, 0.109619), vec3f(0.062874, 0.112162, 0.872564));
const RGB = mat3x3f(vec3f(5.303306, -0.671462, -0.023933), vec3f(-4.499556, 1.862489, -0.142106), vec3f(0.19625, -0.191027, 1.166039));
const NORI = 8u;
var<workgroup> X: array<vec2f, 2048>;
var<workgroup> R: array<vec4f, 256>;

fn mul(x: vec2f, y: vec2f) -> vec2f {
    return vec2(x.x * y.x - x.y * y.y, x.x * y.y + x.y * y.x);
}

fn cis(x: f32) -> vec2f {
    return vec2(cos(x), sin(x));
}

fn index(channel: u32, y: u32, x: u32) -> u32 {
    let N = params.resolution;
    return channel * N * N + y * N + x;
}

fn reverse_bits(x: u32, bits: u32) -> u32 {
    var ret = 0u;
    var val = x;

    for(var i = 0u; i < bits; i++) {
        ret = (ret << 1u) | (val & 1u);
        val = val >> 1u;
    }

    return ret;
}

fn reverse_digits_base_4(x: u32, n: u32) -> u32 {
    var v = x;
    var y = 0u;

    for (var i = 0u; i < n; i++) {
        y = (y << 2u) | (v & 3u);
        v >>= 2u;
    }

    return y;
}

fn swing() -> f32 { return sin(2.0 * PI * time_data.time / max(params.an_per, 0.5)); }
fn a_ori(o: f32) -> f32 { return o + params.an_rot * time_data.time; }
fn a_cut(f: f32) -> f32 { return f * exp2(params.an_cut * swing()); }
fn a_phase() -> f32 { return radians(select(0.0, 90.0, params.gabor_part == 1) + params.an_ph * time_data.time); }

// V1 needs extra planes: 3..5 single cell (lum, L-M, S), 3..10 population; fits at N <= 1024
fn v1_on() -> bool { return params.filter_type == 5 && params.resolution <= 1024u; }
fn inv_planes() -> vec2u {
    if (v1_on()) { return vec2u(3u, select(3u, NORI, params.v1_pop != 0)); }
    return vec2u(0u, N_CHANNELS);
}

struct Gab { su: f32, sv: f32, f0: f32, k: f32 };
// frequency sigmas from the half-amplitude bandwidths
fn gab_shape() -> Gab {
    let f0 = max(params.v1_cpd * params.v1_deg * exp2(params.an_sf * swing()), 0.5);
    let b = exp2(max(params.v1_bw, 0.1));
    let su = f0 * (b - 1.0) / ((b + 1.0) * 1.17741);
    let sv = f0 * tan(radians(clamp(params.v1_ow, 1.0, 179.0)) * 0.5) / 1.17741;
    return Gab(su, sv, f0, exp(-0.5 * f0 * f0 / (su * su)));
}
// one-sided transfer at k (c/img, y up) for stripes at `ori` degrees; the Morlet term removes the DC leak
fn gab(k: vec2f, g: Gab, ori: f32) -> f32 {
    let phi = radians(ori + 90.0);
    let u = dot(k, vec2f(cos(phi), sin(phi)));
    let v = dot(k, vec2f(-sin(phi), cos(phi)));
    let a = exp(-0.5 * (u - g.f0) * (u - g.f0) / (g.su * g.su)) - g.k * exp(-0.5 * u * u / (g.su * g.su));
    return a * exp(-0.5 * v * v / (g.sv * g.sv));
}

// where the image sits in the N x N square: xy offset, zw size, in cells
fn content_rect() -> vec4f {
    let n = f32(params.resolution);
    let isz = vec2f(textureDimensions(input_texture));
    let sz = isz * (n / max(isz.x, isz.y));
    return vec4f((n - sz) * 0.5, sz);
}

// mean colour of the input, from a 64 x 64 grid of samples
@compute @workgroup_size(256, 1, 1)
fn image_mean(@builtin(local_invocation_index) li: u32) {
    if (params.run == 0u) { return; }
    var acc = vec3f(0.0);
    for (var i = 0u; i < 16u; i++) {
        let k = li * 16u + i;
        let uv = (vec2f(f32(k % 64u), f32(k / 64u)) + 0.5) / 64.0;
        acc += textureSampleLevel(input_texture, input_sampler, uv, 0.0).rgb;
    }
    R[li] = vec4f(acc / 16.0, 0.0);
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (li < s) { R[li] += R[li + s]; }
        workgroupBarrier();
    }
    if (li == 0u) {
        let m = R[0].rgb / 256.0;
        stats[S_MEAN] = m.r;
        stats[S_MEAN + 1u] = m.g;
        stats[S_MEAN + 2u] = m.b;
    }
}

@compute @workgroup_size(16, 16, 1)
fn initialize_data(@builtin(global_invocation_id) id: vec3u) {
    if (params.run == 0u) { return; }
    let N = params.resolution;

    if (any(id.xy >= vec2(N))) {
        return;
    }

    // letterbox: outside the image the square holds the mean colour
    let mean = vec3f(stats[S_MEAN], stats[S_MEAN + 1u], stats[S_MEAN + 2u]);
    let rect = content_rect();
    let q = (vec2f(id.xy) + 0.5 - rect.xy) / rect.zw;
    var color = mean;
    if (all(q >= vec2f(0.0)) && all(q < vec2f(1.0))) {
        // average the cell's whole footprint, so a large image is reduced without aliasing
        let foot = vec2f(textureDimensions(input_texture)) / rect.zw;
        let taps = vec2u(clamp(ceil(foot), vec2f(1.0), vec2f(8.0)));
        var acc = vec3f(0.0);
        for (var ty = 0u; ty < taps.y; ty++) {
            for (var tx = 0u; tx < taps.x; tx++) {
                let o = (vec2f(f32(tx), f32(ty)) + 0.5) / vec2f(taps) - 0.5;
                acc += textureSampleLevel(input_texture, input_sampler, (vec2f(id.xy) + 0.5 + o - rect.xy) / rect.zw, 0.0).rgb;
            }
        }
        color = acc / f32(taps.x * taps.y);
        // Tukey window: the edges fade into the mean, so the FFT sees no seam
        if (params.window != 0) {
            let e = clamp(min(q, 1.0 - q) / max(params.taper, 0.0001), vec2f(0.0), vec2f(1.0));
            let w = 0.5 - 0.5 * cos(PI * e);
            color = mean + (color - mean) * w.x * w.y;
        }
    }

    for (var i = 0u; i < N_CHANNELS; i++) {
        image_data[index(i, id.y, id.x)] = vec2(color[i], 0.0);
    }
}

// FFT on rows
@compute @workgroup_size(64, 1, 1)
fn fft_horizontal(@builtin(workgroup_id) workgroup_id: vec3u, @builtin(local_invocation_index) local_index: u32) {
    if (params.run == 0u || workgroup_id.x >= params.resolution) { return; }
    let LOG2_N = firstLeadingBit(params.resolution);
    let LOG4_N = LOG2_N / 2u;
    let N = params.resolution;
    
    let row = workgroup_id.x;
    if (row >= N) { return; }
    
    for (var ch = 0u; ch < N_CHANNELS; ch++) {
        // Load data with bit-reversal permutation
        for (var i = 0u; i < N / 64u; i++) {
            let j = local_index + i * 64u;
            
            var k: u32;
            if (RADIX == 2) {
                k = reverse_bits(j, LOG2_N);
            } else {
                k = reverse_digits_base_4(j >> (LOG2_N & 1u), LOG4_N);
                k |= (j & (LOG2_N & 1u)) << (LOG2_N - 1u);
            }
            
            X[k] = image_data[index(ch, row, j)];
        }
        
        workgroupBarrier();
        
        // Radix-4 FFT passes
        for (var p = 0u; RADIX == 4 && p < LOG4_N; p++) {
            let s = 1u << (2u * p);
            
            for (var i = 0u; i < N / 64u / 4u; i++) {
                let j = local_index + i * 64u;
                let k = j & (s - 1u);
                let t = -2.0 * PI / f32(s * 4u) * f32(k);
                let k0 = ((j >> (2u * p)) << (2u * p + 2u)) + k;
                let k1 = k0 + 1u * s;
                let k2 = k0 + 2u * s;
                let k3 = k0 + 3u * s;
                
                let x0 = X[k0];
                let x1 = mul(cis(t), X[k1]);
                let x2 = mul(cis(t * 2.0), X[k2]);
                let x3 = mul(cis(t * 3.0), X[k3]);
                
                X[k0] = x0 + x1 + x2 + x3;
                X[k1] = x0 - mul(vec2(0.0, 1.0), x1) - x2 + mul(vec2(0.0, 1.0), x3);
                X[k2] = x0 - x1 + x2 - x3;
                X[k3] = x0 + mul(vec2(0.0, 1.0), x1) - x2 - mul(vec2(0.0, 1.0), x3);
            }
            
            workgroupBarrier();
        }
        
        for (var p = select(0u, 2u * LOG4_N, RADIX == 4); p < LOG2_N; p++) {
            let s = 1u << p;
            
            for (var i = 0u; i < N / 64u / 2u; i++) {
                let j = local_index + i * 64u;
                let k = j & (s - 1u);
                let k0 = ((j >> p) << (p + 1u)) + k;
                let k1 = k0 + s;
                
                let x0 = X[k0];
                let x1 = mul(cis(-2.0 * PI / f32(s * 2u) * f32(k)), X[k1]);
                
                X[k0] = x0 + x1;
                X[k1] = x0 - x1;
            }
            
            workgroupBarrier();
        }
        
        // Store results back to storage
        for (var i = 0u; i < N / 64u; i++) {
            let j = local_index + i * 64u;
            image_data[index(ch, row, j)] = X[j];
        }
    }
}

// FFT on columns
@compute @workgroup_size(64, 1, 1)
fn fft_vertical(@builtin(workgroup_id) workgroup_id: vec3u, @builtin(local_invocation_index) local_index: u32) {
    if (params.run == 0u || workgroup_id.x >= params.resolution) { return; }
    let LOG2_N = firstLeadingBit(params.resolution);
    let LOG4_N = LOG2_N / 2u;
    let N = params.resolution;
    
    let col = workgroup_id.x;
    if (col >= N) { return; }
    
    for (var ch = 0u; ch < N_CHANNELS; ch++) {
        // Load data with bit-reversal permutation
        for (var i = 0u; i < N / 64u; i++) {
            let j = local_index + i * 64u;
            
            var k: u32;
            if (RADIX == 2) {
                k = reverse_bits(j, LOG2_N);
            } else {
                k = reverse_digits_base_4(j >> (LOG2_N & 1u), LOG4_N);
                k |= (j & (LOG2_N & 1u)) << (LOG2_N - 1u);
            }
            
            X[k] = image_data[index(ch, j, col)];
        }
        
        workgroupBarrier();
        
        // Radix-4 FFT passes
        for (var p = 0u; RADIX == 4 && p < LOG4_N; p++) {
            let s = 1u << (2u * p);
            
            for (var i = 0u; i < N / 64u / 4u; i++) {
                let j = local_index + i * 64u;
                let k = j & (s - 1u);
                let t = -2.0 * PI / f32(s * 4u) * f32(k);
                let k0 = ((j >> (2u * p)) << (2u * p + 2u)) + k;
                let k1 = k0 + 1u * s;
                let k2 = k0 + 2u * s;
                let k3 = k0 + 3u * s;
                
                let x0 = X[k0];
                let x1 = mul(cis(t), X[k1]);
                let x2 = mul(cis(t * 2.0), X[k2]);
                let x3 = mul(cis(t * 3.0), X[k3]);
                
                X[k0] = x0 + x1 + x2 + x3;
                X[k1] = x0 - mul(vec2(0.0, 1.0), x1) - x2 + mul(vec2(0.0, 1.0), x3);
                X[k2] = x0 - x1 + x2 - x3;
                X[k3] = x0 + mul(vec2(0.0, 1.0), x1) - x2 - mul(vec2(0.0, 1.0), x3);
            }
            
            workgroupBarrier();
        }
        
        for (var p = select(0u, 2u * LOG4_N, RADIX == 4); p < LOG2_N; p++) {
            let s = 1u << p;
            
            for (var i = 0u; i < N / 64u / 2u; i++) {
                let j = local_index + i * 64u;
                let k = j & (s - 1u);
                let k0 = ((j >> p) << (p + 1u)) + k;
                let k1 = k0 + s;
                
                let x0 = X[k0];
                let x1 = mul(cis(-2.0 * PI / f32(s * 2u) * f32(k)), X[k1]);
                
                X[k0] = x0 + x1;
                X[k1] = x0 - x1;
            }
            
            workgroupBarrier();
        }
        
        // Store results back to storage
        for (var i = 0u; i < N / 64u; i++) {
            let j = local_index + i * 64u;
            image_data[index(ch, j, col)] = X[j];
        }
    }
}

// rotational average of the luminance amplitude at radius `r` (cycles per image), normalised so a
// sinusoid of contrast c reads c/2 at any resolution. Half the ring is enough: the spectrum is symmetric
fn radial(r: u32, li: u32, off: u32) {
    let N = params.resolution;
    let rf = f32(r);
    let M = max(1u, u32(ceil(PI * rf)));
    var acc = 0.0;
    for (var i = li; i < M; i += 64u) {
        let th = (f32(i) + 0.5) / f32(M) * PI;
        let k = vec2i(round(rf * vec2f(cos(th), sin(th))));
        let x = u32((k.x + i32(N)) % i32(N));
        let y = u32((k.y + i32(N)) % i32(N));
        var z = vec2f(0.0);
        for (var c = 0u; c < N_CHANNELS; c++) { z += LUMA[c] * image_data[index(c, y, x)]; }
        acc += length(z);
    }
    R[li] = vec4f(acc, 0.0, 0.0, 0.0);
    workgroupBarrier();
    for (var s = 32u; s > 0u; s >>= 1u) {
        if (li < s) { R[li] += R[li + s]; }
        workgroupBarrier();
    }
    if (li == 0u) { stats[off + r] = R[0].x / f32(M) / f32(N * N); }
}

@compute @workgroup_size(64, 1, 1)
fn radial_in(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    if (params.run == 0u || wid.x >= params.resolution / 2u) { return; }
    radial(wid.x, li, S_IN);
}

@compute @workgroup_size(64, 1, 1)
fn radial_out(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    if (params.run == 0u || v1_on() || wid.x >= params.resolution / 2u) { return; }
    radial(wid.x, li, S_OUT);
}

// the fit range skips the lowest bins and the top octave, where aliasing and the window dominate
fn fit_range() -> vec2u { return vec2u(2u, params.resolution / 4u); }

// least-squares line through log10 amplitude vs log10 frequency: amplitude ~ 10^b / f^alpha
fn fit(li: u32, off: u32, dst: u32) {
    let fr = fit_range();
    var acc = vec4f(0.0);
    for (var r = fr.x + li; r <= fr.y; r += 64u) {
        let x = lg10(f32(r));
        let y = lg10(max(stats[off + r], 1e-12));
        acc += vec4f(x, y, x * x, x * y);
    }
    R[li] = acc;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s >>= 1u) {
        if (li < s) { R[li] += R[li + s]; }
        workgroupBarrier();
    }
    if (li == 0u) {
        let n = f32(fr.y - fr.x + 1u);
        let t = R[0];
        let slope = (n * t.w - t.x * t.y) / max(n * t.z - t.x * t.x, 1e-9);
        stats[dst] = -slope;
        stats[dst + 1u] = (t.y - slope * t.x) / n;
    }
}

@compute @workgroup_size(64, 1, 1)
fn fit_in(@builtin(local_invocation_index) li: u32) {
    if (params.run == 0u) { return; }
    fit(li, S_IN, S_FIT);
}

@compute @workgroup_size(64, 1, 1)
fn fit_out(@builtin(local_invocation_index) li: u32) {
    if (params.run == 0u || v1_on()) { return; }
    fit(li, S_OUT, S_FIT + 2u);
}

fn hash_u(x: u32) -> u32 {
    var v = x;
    v ^= v >> 16u; v *= 0x7feb352du;
    v ^= v >> 15u; v *= 0x846ca68bu;
    v ^= v >> 16u;
    return v;
}

// random phase in -pi..pi for one frequency, from the seed
fn rnd_phase(fx: i32, fy: i32) -> f32 {
    let h = hash_u(u32(fx + 4096) * 73856093u ^ u32(fy + 4096) * 19349663u ^ hash_u(params.phase_seed));
    return (f32(h >> 8u) / 16777216.0 * 2.0 - 1.0) * PI;
}

// phase noise with n(-k) = -n(k), so the scrambled image stays real;
// bins that are their own mirror (DC and the Nyquist corners) keep their phase
fn phase_noise(fx: i32, fy: i32, half: i32) -> f32 {
    let self_x = fx == 0 || fx == -half;
    let self_y = fy == 0 || fy == -half;
    if (self_x && self_y) { return 0.0; }
    if (fy > 0 || (self_y && fx > 0)) { return rnd_phase(fx, fy); }
    let mx = select(-fx, fx, fx == -half);
    let my = select(-fy, fy, fy == -half);
    return -rnd_phase(mx, my);
}

// Frequency domain operations: spectral slope, phase scramble, then the filter
@compute @workgroup_size(16, 16, 1)
fn modify_frequencies(@builtin(global_invocation_id) id: vec3u) {
    if (params.run == 0u) { return; }
    let N = params.resolution;

    if (any(id.xy >= vec2(N))) {
        return;
    }

    // signed frequency of this bin, cycles per image (0,0 is DC)
    let half = i32(N / 2u);
    let fx = i32((id.x + N / 2u) % N) - half;
    let fy = i32((id.y + N / 2u) % N) - half;
    let f = length(vec2f(f32(fx), f32(fy)));

    // amplitude gain from the slope change, pivoting at the middle of the fit range
    var gain = 1.0;
    if (params.slope_on != 0 && f > 0.0) {
        let fr = fit_range();
        let pivot = sqrt(f32(fr.x * fr.y));
        gain = pow(f / pivot, stats[S_FIT] - params.slope_target - params.an_slp * swing());
    }
    let dphi = phase_noise(fx, fy, half) * clamp(params.phase_amount + params.an_scr * (0.5 + 0.5 * swing()), 0.0, 1.0);

    var scale = 1.0;
    switch params.filter_type {
        // Butterworth low-pass
        case 1: {
            scale = 1.0 / (1.0 + pow(f / max(a_cut(params.cutoff), 0.01), 2.0 * params.order));
        }
        // Butterworth high-pass
        case 2: {
            scale = select(0.0, 1.0 / (1.0 + pow(a_cut(params.cutoff) / max(f, 0.0001), 2.0 * params.order)), f > 0.0);
        }
        // log-Gaussian band-pass, width as full width at half height in octaves
        case 3: {
            let sigma = max(params.band_octaves, 0.05) / 2.3548;
            let d = log2(max(f, 0.0001) / max(a_cut(params.band_center), 0.01));
            scale = select(0.0, exp(-d * d / (2.0 * sigma * sigma)), f > 0.0);
        }
        // orientation: stripes at angle a have their frequency vector at a + 90 degrees
        case 4: {
            let a = degrees(atan2(-f32(fy), f32(fx))) - 90.0;
            var d = a - a_ori(params.orientation);
            d -= 180.0 * round(d / 180.0);
            let w = max(params.orient_width, 1.0);
            scale = select(1.0, exp(-4.0 * 0.6931 * d * d / (w * w)), f > 0.0);
        }
        default: {}
    }
    if (params.keep_mean != 0 && f == 0.0) { scale = 1.0; }

    let rot = cis(dphi);
    for (var i = 0u; i < N_CHANNELS; i++) {
        let j = index(i, id.y, id.x);
        image_data[j] = mul(image_data[j], rot) * gain * scale;
    }

    // V1: cone contrasts on DKL axes, then the cells; planes 0..2 keep the input spectrum
    if (v1_on()) {
        let c0 = max(LMS * vec3f(stats[S_MEAN], stats[S_MEAN + 1u], stats[S_MEAN + 2u]), vec3f(1e-4));
        var cs: array<vec2f, 3>;
        for (var r = 0u; r < 3u; r++) {
            var z = vec2f(0.0);
            for (var i = 0u; i < 3u; i++) { z += LMS[i][r] * image_data[index(i, id.y, id.x)]; }
            cs[r] = z / c0[r];
        }
        let lm = (cs[0] + cs[1]) * 0.5;
        var op = array<vec2f, 3>(lm, cs[0] - cs[1], cs[2] - lm);
        let k = vec2f(f32(fx), -f32(fy));
        let g = gab_shape();
        if (params.v1_pop != 0) {
            for (var o = 0u; o < NORI; o++) { image_data[index(3u + o, id.y, id.x)] = op[0] * gab(k, g, f32(o) * 180.0 / f32(NORI)); }
        } else {
            let h = gab(k, g, a_ori(params.v1_ori));
            for (var i = 0u; i < 3u; i++) { image_data[index(3u + i, id.y, id.x)] = op[i] * h; }
        }
    }
}

// inverse FFT on rows
@compute @workgroup_size(64, 1, 1)
fn ifft_horizontal(@builtin(workgroup_id) workgroup_id: vec3u, @builtin(local_invocation_index) local_index: u32) {
    if (params.run == 0u || params.view != 0 || workgroup_id.x >= params.resolution) { return; }
    let LOG2_N = firstLeadingBit(params.resolution);
    let LOG4_N = LOG2_N / 2u;
    let N = params.resolution;
    
    let row = workgroup_id.x;
    if (row >= N) { return; }
    
    let pl = inv_planes();
    for (var ch = pl.x; ch < pl.x + pl.y; ch++) {
        // Load data with bit-reversal permutation
        for (var i = 0u; i < N / 64u; i++) {
            let j = local_index + i * 64u;
            
            var k: u32;
            if (RADIX == 2) {
                k = reverse_bits(j, LOG2_N);
            } else {
                k = reverse_digits_base_4(j >> (LOG2_N & 1u), LOG4_N);
                k |= (j & (LOG2_N & 1u)) << (LOG2_N - 1u);
            }
            
            X[k] = image_data[index(ch, row, j)];
        }
        
        workgroupBarrier();
        
        for (var p = 0u; RADIX == 4 && p < LOG4_N; p++) {
            let s = 1u << (2u * p);
            
            for (var i = 0u; i < N / 64u / 4u; i++) {
                let j = local_index + i * 64u;
                let k = j & (s - 1u);
                let t = 2.0 * PI / f32(s * 4u) * f32(k);
                let k0 = ((j >> (2u * p)) << (2u * p + 2u)) + k;
                let k1 = k0 + 1u * s;
                let k2 = k0 + 2u * s;
                let k3 = k0 + 3u * s;
                
                let x0 = X[k0];
                let x1 = mul(cis(t), X[k1]);
                let x2 = mul(cis(t * 2.0), X[k2]);
                let x3 = mul(cis(t * 3.0), X[k3]);
                
                X[k0] = x0 + x1 + x2 + x3;
                X[k1] = x0 + mul(vec2(0.0, 1.0), x1) - x2 - mul(vec2(0.0, 1.0), x3);
                X[k2] = x0 - x1 + x2 - x3;
                X[k3] = x0 - mul(vec2(0.0, 1.0), x1) - x2 + mul(vec2(0.0, 1.0), x3);
            }
            
            workgroupBarrier();
        }
        
        for (var p = select(0u, 2u * LOG4_N, RADIX == 4); p < LOG2_N; p++) {
            let s = 1u << p;
            
            for (var i = 0u; i < N / 64u / 2u; i++) {
                let j = local_index + i * 64u;
                let k = j & (s - 1u);
                let k0 = ((j >> p) << (p + 1u)) + k;
                let k1 = k0 + s;
                
                let x0 = X[k0];
                let x1 = mul(cis(2.0 * PI / f32(s * 2u) * f32(k)), X[k1]);
                
                X[k0] = x0 + x1;
                X[k1] = x0 - x1;
            }
            
            workgroupBarrier();
        }
        
        for (var i = 0u; i < N / 64u; i++) {
            let j = local_index + i * 64u;
            image_data[index(ch, row, j)] = X[j] / f32(N);
        }
    }
}

// now on columns inverse... 
@compute @workgroup_size(64, 1, 1)
fn ifft_vertical(@builtin(workgroup_id) workgroup_id: vec3u, @builtin(local_invocation_index) local_index: u32) {
    if (params.run == 0u || params.view != 0 || workgroup_id.x >= params.resolution) { return; }
    let LOG2_N = firstLeadingBit(params.resolution);
    let LOG4_N = LOG2_N / 2u;
    let N = params.resolution;
    
    let col = workgroup_id.x;
    if (col >= N) { return; }
    
    let pl = inv_planes();
    for (var ch = pl.x; ch < pl.x + pl.y; ch++) {
        for (var i = 0u; i < N / 64u; i++) {
            let j = local_index + i * 64u;
            
            var k: u32;
            if (RADIX == 2) {
                k = reverse_bits(j, LOG2_N);
            } else {
                k = reverse_digits_base_4(j >> (LOG2_N & 1u), LOG4_N);
                k |= (j & (LOG2_N & 1u)) << (LOG2_N - 1u);
            }
            
            X[k] = image_data[index(ch, j, col)];
        }
        
        workgroupBarrier();
        
        for (var p = 0u; RADIX == 4 && p < LOG4_N; p++) {
            let s = 1u << (2u * p);
            
            for (var i = 0u; i < N / 64u / 4u; i++) {
                let j = local_index + i * 64u;
                let k = j & (s - 1u);
                let t = 2.0 * PI / f32(s * 4u) * f32(k);
                let k0 = ((j >> (2u * p)) << (2u * p + 2u)) + k;
                let k1 = k0 + 1u * s;
                let k2 = k0 + 2u * s;
                let k3 = k0 + 3u * s;
                
                let x0 = X[k0];
                let x1 = mul(cis(t), X[k1]);
                let x2 = mul(cis(t * 2.0), X[k2]);
                let x3 = mul(cis(t * 3.0), X[k3]);
                
                X[k0] = x0 + x1 + x2 + x3;
                X[k1] = x0 + mul(vec2(0.0, 1.0), x1) - x2 - mul(vec2(0.0, 1.0), x3);
                X[k2] = x0 - x1 + x2 - x3;
                X[k3] = x0 - mul(vec2(0.0, 1.0), x1) - x2 + mul(vec2(0.0, 1.0), x3);
            }
            
            workgroupBarrier();
        }
        
        for (var p = select(0u, 2u * LOG4_N, RADIX == 4); p < LOG2_N; p++) {
            let s = 1u << p;
            
            for (var i = 0u; i < N / 64u / 2u; i++) {
                let j = local_index + i * 64u;
                let k = j & (s - 1u);
                let k0 = ((j >> p) << (p + 1u)) + k;
                let k1 = k0 + s;
                
                let x0 = X[k0];
                let x1 = mul(cis(2.0 * PI / f32(s * 2u) * f32(k)), X[k1]);
                
                X[k0] = x0 + x1;
                X[k1] = x0 - x1;
            }
            
            workgroupBarrier();
        }
        
        for (var i = 0u; i < N / 64u; i++) {
            let j = local_index + i * 64u;
            image_data[index(ch, j, col)] = X[j] / f32(N);
        }
    }
}

// log10 amplitude of a stored radial profile at frequency f, linearly between bins
fn radial_log(off: u32, f: f32) -> f32 {
    let top = params.resolution / 2u - 1u;
    let r0 = min(u32(f), top);
    let a = mix(stats[off + r0], stats[off + min(r0 + 1u, top)], fract(f));
    return lg10(max(a, 1e-12));
}

// one ASCII glyph of height sz with its top-left at pos
fn glyph(p: vec2f, pos: vec2f, code: u32, sz: f32) -> f32 {
    let r = (p - pos) / sz;
    if (any(r < vec2f(0.0)) || any(r >= vec2f(1.0))) { return 0.0; }
    let uv = (vec2f(f32(code % 16u), f32(code / 16u)) + 0.05 + r * 0.9) / 16.0;
    return smoothstep(0.1, 0.9, textureLoad(font_atlas, vec2i(uv * font.atlas_size), 0).r * 0.8);
}

// "alpha in  1.92" / "alpha out 1.00": the fitted slope, rounded to two decimals
fn slope_label(p: vec2f, pos: vec2f, sz: f32, out: bool, x: f32) -> f32 {
    var c = array<u32, 16>(97u, 108u, 112u, 104u, 97u, 32u, 105u, 110u, 32u, 32u, 0u, 0u, 0u, 0u, 0u, 0u);
    if (out) { c[6] = 111u; c[7] = 117u; c[8] = 116u; }
    var n = 10u;
    let v = u32(round(min(abs(x), 99.99) * 100.0));
    if (x < 0.0 && v > 0u) { c[n] = 45u; n++; }
    if (v >= 1000u) { c[n] = 48u + v / 1000u; n++; }
    c[n] = 48u + (v / 100u) % 10u;
    c[n + 1u] = 46u;
    c[n + 2u] = 48u + (v / 10u) % 10u;
    c[n + 3u] = 48u + v % 10u;
    n += 4u;
    var a = 0.0;
    for (var i = 0u; i < n; i++) { a = max(a, glyph(p, pos + vec2f(f32(i) * sz * 0.5, 0.0), c[i], sz)); }
    return a;
}

// log-log plot of the radial amplitude: input white, result orange, their 1/f fits dashed.
// x spans 1 .. N/2 cycles per image, y spans 1e-6 .. 1
fn radial_plot(p: vec2f, D: vec2f) -> vec4f {
    let size = min(D.x, D.y) * 0.36;
    let lo = D - vec2f(size) - 16.0;
    let q = (p - lo) / size;
    if (any(q < vec2f(0.0)) || any(q > vec2f(1.0))) { return vec4f(0.0); }
    let qy = 1.0 - q.y;
    let lmax = lg10(f32(params.resolution / 2u));
    let lf = q.x * lmax;
    let f = pow(10.0, lf);
    let step_f = pow(10.0, lf + lmax / size);

    var col = vec3f(0.02, 0.025, 0.04);
    // decade grid
    let gx = abs(fract(lf + 0.5) - 0.5) * size / lmax;
    let gy = abs(fract(qy * 6.0 + 0.5) - 0.5) * size / 6.0;
    col += vec3f(0.08) * (smoothstep(1.0, 0.0, gx) + smoothstep(1.0, 0.0, gy));

    let fr = fit_range();
    let in_fit = f >= f32(fr.x) && f <= f32(fr.y);
    let offs = array<u32, 2>(S_IN, S_OUT);
    let tints = array<vec3f, 2>(vec3f(0.9, 0.9, 0.95), vec3f(1.0, 0.55, 0.15));
    for (var k = 0; k < 2; k++) {
        let y0 = (radial_log(offs[k], f) + 6.0) / 6.0;
        let dy = (radial_log(offs[k], step_f) + 6.0) / 6.0 - y0;
        let d = abs(qy - y0) * size / sqrt(1.0 + dy * dy * size * size);
        col = mix(col, tints[k], smoothstep(1.8, 0.6, d));
        // fitted line, dashed, over the fit range
        let yf = (stats[S_FIT + u32(k) * 2u + 1u] - stats[S_FIT + u32(k) * 2u] * lf + 6.0) / 6.0;
        let dash = step(0.5, fract(q.x * 30.0));
        col = mix(col, tints[k] * 0.7, smoothstep(1.4, 0.4, abs(qy - yf) * size) * dash * select(0.35, 1.0, in_fit));
    }
    let e = min(min(q.x, 1.0 - q.x), min(q.y, 1.0 - q.y)) * size;
    col += vec3f(0.25) * smoothstep(1.2, 0.0, e);
    // the fitted slopes, bottom left where the curves never reach, each in its curve's colour
    let ts = max(size * 0.06, 11.0);
    for (var k = 0u; k < 2u; k++) {
        let pos = lo + vec2f(8.0, size - 8.0 - f32(2u - k) * ts * 1.15);
        col = mix(col, tints[k], slope_label(p, pos, ts, k == 1u, stats[S_FIT + k * 2u]));
    }
    return vec4f(col, 0.92);
}

// display: responses are shown as the cone contrast that drives them, around a mid grey (linear output)
fn s2l(x: vec3f) -> vec3f { return select(pow((x + 0.055) / 1.055, vec3f(2.4)), x / 12.92, x <= vec3f(0.04045)); }
fn show_cc(dl: vec3f) -> vec3f { return max(RGB * (0.214 * (1.0 + dl)), vec3f(0.0)); }
fn hsv(h: f32, s: f32, v: f32) -> vec3f { return v * mix(vec3f(1.0), clamp(abs(fract(h + vec3f(0.0, 2.0, 1.0) / 3.0) * 6.0 - 3.0) - 1.0, vec3f(0.0), vec3f(1.0)), s); }
// Naka-Rushton, exponent 2
fn nr(e2: f32, pool: f32) -> f32 { let c = params.v1_c50 * params.v1_c50; return e2 / (c + pool); }

fn v1_view(c: vec2u) -> vec3f {
    let gn = 2.0 * params.v1_gain;
    if (params.v1_pop != 0) {
        var r: array<f32, 8>;
        var tot = 0.0;
        for (var o = 0u; o < NORI; o++) { let e = length(image_data[index(3u + o, c.y, c.x)]) * gn; r[o] = e * e; tot += r[o]; }
        var v = vec2f(0.0);
        var sum = 0.0;
        var mx = 0.0;
        for (var o = 0u; o < NORI; o++) {
            let q = nr(r[o], select(r[o], tot / f32(NORI), params.v1_norm != 0));
            let a = 2.0 * PI * f32(o) / f32(NORI);
            v += q * vec2f(cos(a), sin(a));
            sum += q;
            mx = max(mx, q);
        }
        // hue = preferred orientation, saturation = selectivity, value = strongest cell
        let h = fract(atan2(v.y, v.x) / (2.0 * PI));
        return s2l(hsv(h, clamp(length(v) / max(sum, 1e-6), 0.0, 1.0), clamp(mx, 0.0, 1.0)));
    }
    var z: array<vec2f, 3>;
    for (var i = 0u; i < 3u; i++) { z[i] = image_data[index(3u + i, c.y, c.x)] * gn; }
    let ch = params.v1_chan;
    if (params.gabor_part == 2) {
        var e = vec3f(dot(z[0], z[0]), dot(z[1], z[1]), dot(z[2], z[2]));
        e = vec3f(nr(e.x, e.x), nr(e.y, e.y), nr(e.z, e.z));
        switch ch {
            case 0: { return s2l(vec3f(e.x)); }
            case 1: { return s2l(e.y * vec3f(1.0, 0.35, 0.45)); }
            case 2: { return s2l(e.z * vec3f(0.55, 0.45, 1.0)); }
            default: { return s2l(min(vec3f(e.x) * 0.7 + e.y * vec3f(0.6, 0.1, 0.2) + e.z * vec3f(0.2, 0.15, 0.7), vec3f(1.0))); }
        }
    }
    // phase 0 even, 90 odd, in between a mix of the pair
    let cp = cis(a_phase());
    let r = vec3f(dot(z[0], cp), dot(z[1], cp), dot(z[2], cp));
    switch ch {
        case 0: { return show_cc(vec3f(r.x)); }
        case 1: { return show_cc(vec3f(r.y * 0.5, -r.y * 0.5, 0.0)); }
        case 2: { return show_cc(vec3f(0.0, 0.0, r.z)); }
        default: { return show_cc(vec3f(r.x + r.y * 0.5, r.x - r.y * 0.5, r.z + r.x)); }
    }
}

// half-height contours of the cells over the spectrum view (both lobes of the real cell)
fn v1_spec(p: vec2f, D: vec2f, s: f32, region: vec4f) -> vec4f {
    let n = f32(params.resolution);
    let k = ((p - (D - region.zw * s) * 0.5) / s - n * 0.5) * vec2f(1.0, -1.0);
    let g = gab_shape();
    var a = 0.0;
    var col = vec3f(0.0);
    let no = select(1u, NORI, params.v1_pop != 0);
    for (var o = 0u; o < no; o++) {
        let ori = select(a_ori(params.v1_ori), f32(o) * 180.0 / f32(NORI), params.v1_pop != 0);
        let h = max(gab(k, g, ori), gab(-k, g, ori));
        let w = 1.0 - smoothstep(0.0, 0.05, abs(h - 0.5));
        if (w > a) { a = w; col = select(vec3f(0.3, 0.9, 1.0), s2l(hsv(f32(o) / f32(NORI), 0.8, 1.0)), params.v1_pop != 0); }
    }
    return vec4f(col, a * 0.9);
}

// the cell's receptive field at true scale, even and odd, top left
fn rf_inset(p: vec2f, D: vec2f, s: f32) -> vec4f {
    let n = f32(params.resolution);
    let g = gab_shape();
    let sx = n / (2.0 * PI * g.su);
    let sy = n / (2.0 * PI * g.sv);
    let hb = clamp(3.0 * max(sx, sy) * s, 24.0, D.y * 0.18);
    for (var i = 0; i < 2; i++) {
        let ctr = vec2f(16.0 + hb + f32(i) * (2.0 * hb + 10.0), 16.0 + hb);
        let q = p - ctr;
        if (all(abs(q) <= vec2f(hb))) {
            let e = min(hb - abs(q.x), hb - abs(q.y));
            if (e < 1.5) { return vec4f(vec3f(0.6), 1.0); }
            let d = q / s * vec2f(1.0, -1.0);
            let phi = radians(a_ori(params.v1_ori) + 90.0);
            let u = dot(d, vec2f(cos(phi), sin(phi)));
            let v = dot(d, vec2f(-sin(phi), cos(phi)));
            let env = exp(-0.5 * (u * u / (sx * sx) + v * v / (sy * sy)));
            let w = 2.0 * PI * g.f0 * u / n;
            let ph = a_phase() + select(0.0, PI * 0.5, i == 1);
            let val = env * (cos(ph) * (cos(w) - g.k) + sin(ph) * sin(w));
            return vec4f(s2l(vec3f(0.5 + 0.5 * val)), 1.0);
        }
    }
    return vec4f(0.0);
}

// colour key: each population colour drawn as a bar at its orientation, bottom left
fn ori_legend(p: vec2f, D: vec2f) -> vec4f {
    let c = clamp(min(D.x, D.y) * 0.035, 16.0, 34.0);
    let q = p - vec2f(16.0, D.y - 16.0 - c);
    if (q.x < 0.0 || q.y < 0.0 || q.x > c * f32(NORI) || q.y > c) { return vec4f(0.0); }
    let i = min(u32(q.x / c), NORI - 1u);
    let a = radians(f32(i) * 180.0 / f32(NORI));
    let l = (q - vec2f((f32(i) + 0.5) * c, 0.5 * c)) * vec2f(1.0, -1.0);
    let along = dot(l, vec2f(cos(a), sin(a)));
    let across = abs(dot(l, vec2f(-sin(a), cos(a))));
    let bar = (1.0 - smoothstep(1.2, 2.2, across)) * (1.0 - smoothstep(c * 0.36, c * 0.42, abs(along)));
    return vec4f(mix(vec3f(0.02), s2l(hsv(f32(i) / f32(NORI), 0.85, 1.0)), bar), 0.85);
}

//render
@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let dimensions = vec2u(textureDimensions(output));

    if (any(id.xy >= dimensions)) {
        return;
    }

    let N = params.resolution;
    let nf = f32(N);
    let D = vec2f(dimensions);

    // the image rect for picture views, the whole square for the spectrum, fitted to the window
    let rect = content_rect();
    var region = rect;
    if (params.view == 2) { region = vec4f(0.0, 0.0, nf, nf); }
    let s = min(D.x / region.z, D.y / region.w);
    let cell = (vec2f(id.xy) + 0.5 - (D - region.zw * s) * 0.5) / s;

    var color = vec3f(0.03);
    if (all(cell >= vec2f(0.0)) && all(cell < region.zw)) {
        let c = vec2u(clamp(region.xy + cell, vec2f(0.0), vec2f(nf - 1.0)));
        if (params.view == 1) {
            color = textureSampleLevel(input_texture, input_sampler, cell / region.zw, 0.0).rgb;
        } else if (params.view == 2) {
            // log amplitude, normalised like the radial plot, `spec_decades` decades below 1
            let x = (c.x + N / 2u) % N;
            let y = (c.y + N / 2u) % N;
            for (var i = 0u; i < N_CHANNELS; i++) {
                let a = length(image_data[index(i, y, x)]) / f32(N * N);
                color[i] = 1.0 + lg10(max(a, 1e-12)) / max(params.spec_decades, 0.5);
            }
        } else if (v1_on()) {
            color = v1_view(c);
        } else {
            for (var i = 0u; i < N_CHANNELS; i++) {
                color[i] = image_data[index(i, c.y, c.x)].x;
            }
        }
    }

    color = clamp(color, vec3(0.0), vec3(1.0));
    if (params.is_bw != 0) {
        color = vec3(dot(color, LUMA));
    }

    if (v1_on()) {
        if (params.view == 2) {
            let o = v1_spec(vec2f(id.xy) + 0.5, D, s, region);
            color = mix(color, o.rgb, o.a);
        }
        if (params.v1_rf != 0 && params.v1_pop == 0) {
            let o = rf_inset(vec2f(id.xy) + 0.5, D, s);
            color = mix(color, o.rgb, o.a);
        }
        if (params.v1_pop != 0 && params.view == 0 && params.show_key != 0) {
            let o = ori_legend(vec2f(id.xy) + 0.5, D);
            color = mix(color, o.rgb, o.a);
        }
    } else if (params.show_radial != 0) {
        let plot = radial_plot(vec2f(id.xy) + 0.5, D);
        color = mix(color, plot.rgb, plot.a);
    }
    textureStore(output, id.xy, vec4(color, 1.0));
}
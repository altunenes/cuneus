// 2D FFT workflow with butterworth filter, Enes Altun 2025.
//  Some FFT operations adapted from:
//  fadaaszhi, 2025: https://compute.toys/view/1187: Fast Fourier Transform
//  FabriceNeyret2, 2023 https://www.shadertoy.com/view/DtGfWV : Fourier Workflow 3 / phases info
//  FabriceNeyret2, 2017  https://www.shadertoy.com/view/XtScWt: Fourier Workflow 2 / phases info
//  SPIR-V-based Stockham + mixed-radix kernels   https://github.com/DTolm/VkFFT
//  Moreland, K., & Angel, E. (2003). The FFT on a GPU. SIGGRAPH/EUROGRAPHICS Conference On Graphics Hardware.
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
    // 0 none, 1 low-pass, 2 high-pass, 3 band-pass, 4 orientation
    filter_type: i32,
    // low/high-pass cutoff, cycles per image
    cutoff: f32,
    // Butterworth order
    order: f32,
    // band-pass centre (cycles per image) and full width at half height (octaves)
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
    _padding2: u32,
};
// Group 1: Primary Pass I/O & Parameters
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: FFTParams;
@group(1) @binding(2) var input_texture: texture_2d<f32>;
@group(1) @binding(3) var input_sampler: sampler;

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
fn radial_in(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) { radial(wid.x, li, S_IN); }

@compute @workgroup_size(64, 1, 1)
fn radial_out(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) { radial(wid.x, li, S_OUT); }

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
fn fit_in(@builtin(local_invocation_index) li: u32) { fit(li, S_IN, S_FIT); }

@compute @workgroup_size(64, 1, 1)
fn fit_out(@builtin(local_invocation_index) li: u32) { fit(li, S_OUT, S_FIT + 2u); }

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
        gain = pow(f / pivot, stats[S_FIT] - params.slope_target);
    }
    let dphi = phase_noise(fx, fy, half) * params.phase_amount;

    var scale = 1.0;
    switch params.filter_type {
        // Butterworth low-pass
        case 1: {
            scale = 1.0 / (1.0 + pow(f / max(params.cutoff, 0.01), 2.0 * params.order));
        }
        // Butterworth high-pass
        case 2: {
            scale = select(0.0, 1.0 / (1.0 + pow(params.cutoff / max(f, 0.0001), 2.0 * params.order)), f > 0.0);
        }
        // log-Gaussian band-pass, width as full width at half height in octaves
        case 3: {
            let sigma = max(params.band_octaves, 0.05) / 2.3548;
            let d = log2(max(f, 0.0001) / max(params.band_center, 0.01));
            scale = select(0.0, exp(-d * d / (2.0 * sigma * sigma)), f > 0.0);
        }
        // orientation: stripes at angle a have their frequency vector at a + 90 degrees
        case 4: {
            let a = degrees(atan2(-f32(fy), f32(fx))) - 90.0;
            var d = a - params.orientation;
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
}

// inverse FFT on rows
@compute @workgroup_size(64, 1, 1)
fn ifft_horizontal(@builtin(workgroup_id) workgroup_id: vec3u, @builtin(local_invocation_index) local_index: u32) {
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
    let LOG2_N = firstLeadingBit(params.resolution);
    let LOG4_N = LOG2_N / 2u;
    let N = params.resolution;
    
    let col = workgroup_id.x;
    if (col >= N) { return; }
    
    for (var ch = 0u; ch < N_CHANNELS; ch++) {
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
    return vec4f(col, 0.92);
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

    if (params.show_radial != 0) {
        let plot = radial_plot(vec2f(id.xy) + 0.5, D);
        color = mix(color, plot.rgb, plot.a);
    }
    textureStore(output, id.xy, vec4(color, 1.0));
}

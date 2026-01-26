// 2D FFT workflow with butterworth filter, Enes Altun 2025.
//  Some FFT operations adapted from: 
//  fadaaszhi, 2025: https://compute.toys/view/1187: Fast Fourier Transform
//  FabriceNeyret2, 2023 https://www.shadertoy.com/view/DtGfWV : Fourier Workflow 3 / phases info 
//  FabriceNeyret2, 2017  https://www.shadertoy.com/view/XtScWt: Fourier Workflow 2 / phases info 
//  SPIR-V-based Stockham + mixed-radix kernels   https://github.com/DTolm/VkFFT
//  Moreland, K., & Angel, E. (2003). The FFT on a GPU. SIGGRAPH/EUROGRAPHICS Conference On Graphics Hardware.


const RADIX = 4;

struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

struct FFTParams {
    filter_type: i32,     
    filter_strength: f32, 
    filter_direction: f32,
    filter_radius: f32,   
    show_freqs: i32,      
    resolution: u32,      
    is_bw: i32,
    _padding2: u32,
};
// Group 1: Primary Pass I/O & Parameters  
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: FFTParams;
@group(1) @binding(2) var input_texture: texture_2d<f32>;
@group(1) @binding(3) var input_sampler: sampler;

// Storage buffer for FFT data
@group(3) @binding(0) var<storage, read_write> image_data: array<vec2f>;

const PI = 3.1415927;
const LOG2_N_MAX = 11;
const N_MAX = 2048;
const N_CHANNELS = 3u;
var<workgroup> X: array<vec2f, 2048>;

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

fn magnitude(z: vec2f) -> f32 {
    return sqrt(z.x * z.x + z.y * z.y);
}

fn phase(z: vec2f) -> f32 {
    return atan2(z.y, z.x);
}

fn fftshift(i: u32, N: u32) -> u32 {
    return (i + N / 2u) % N;
}

@compute @workgroup_size(16, 16, 1)
fn initialize_data(@builtin(global_invocation_id) id: vec3u) {
    let N = params.resolution;
    
    if (any(id.xy >= vec2(N))) {
        return;
    }
    
    let uv = (vec2f(id.xy) + 0.5) / f32(N);
    
    // Load color from input texture (uploaded by user) or black screen if no texture
    let input_size = vec2u(textureDimensions(input_texture));
    let sample_coord = vec2u(uv * vec2f(input_size));
    let clamped_coord = clamp(sample_coord, vec2u(0), input_size - vec2u(1));
    var color = textureLoad(input_texture, clamped_coord, 0).rgb;
    
    for (var i = 0u; i < N_CHANNELS; i++) {
        image_data[index(i, id.y, id.x)] = vec2(color[i], 0.0);
    }
}

// FFT on rows
@compute @workgroup_size(256, 1, 1)
fn fft_horizontal(@builtin(workgroup_id) workgroup_id: vec3u, @builtin(local_invocation_index) local_index: u32) {
    let LOG2_N = firstLeadingBit(params.resolution);
    let LOG4_N = LOG2_N / 2u;
    let N = params.resolution;
    
    let row = workgroup_id.x;
    if (row >= N) { return; }
    
    for (var ch = 0u; ch < N_CHANNELS; ch++) {
        // Load data with bit-reversal permutation
        for (var i = 0u; i < N / 256u; i++) {
            let j = local_index + i * 256u;
            
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
            
            for (var i = 0u; i < N / 256u / 4u; i++) {
                let j = local_index + i * 256u;
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
            
            for (var i = 0u; i < N / 256u / 2u; i++) {
                let j = local_index + i * 256u;
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
        for (var i = 0u; i < N / 256u; i++) {
            let j = local_index + i * 256u;
            image_data[index(ch, row, j)] = X[j];
        }
    }
}

// FFT on columns
@compute @workgroup_size(256, 1, 1)
fn fft_vertical(@builtin(workgroup_id) workgroup_id: vec3u, @builtin(local_invocation_index) local_index: u32) {
    let LOG2_N = firstLeadingBit(params.resolution);
    let LOG4_N = LOG2_N / 2u;
    let N = params.resolution;
    
    let col = workgroup_id.x;
    if (col >= N) { return; }
    
    for (var ch = 0u; ch < N_CHANNELS; ch++) {
        // Load data with bit-reversal permutation
        for (var i = 0u; i < N / 256u; i++) {
            let j = local_index + i * 256u;
            
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
            
            for (var i = 0u; i < N / 256u / 4u; i++) {
                let j = local_index + i * 256u;
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
            
            for (var i = 0u; i < N / 256u / 2u; i++) {
                let j = local_index + i * 256u;
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
        for (var i = 0u; i < N / 256u; i++) {
            let j = local_index + i * 256u;
            image_data[index(ch, j, col)] = X[j];
        }
    }
}

fn butterworth(f: f32, cutoff: f32, order: f32, highpass: bool) -> f32 {
    let ratio = f / cutoff;
    var result: f32;
    
    if (highpass) {
        result = 1.0 / (1.0 + pow(cutoff / max(f, 0.001), 2.0 * order));
    } else {
        result = 1.0 / (1.0 + pow(ratio, 2.0 * order));
    }
    
    return result;
}

// Frequency domain operations
@compute @workgroup_size(16, 16, 1)
fn modify_frequencies(@builtin(global_invocation_id) id: vec3u) {
    let N = params.resolution;
    
    if (any(id.xy >= vec2(N))) {
        return;
    }
    
    // Calculate shifted coordinates for centered frequency representation
    let shifted_x = (id.x + N / 2u) % N;
    let shifted_y = (id.y + N / 2u) % N;
    
    // Calculate frequency coordinates (0,0 is DC, center of the image)
    let freq_x = f32(shifted_x) - f32(N / 2u);
    let freq_y = f32(shifted_y) - f32(N / 2u);
    
    // Normalized frequency (distance from DC). range: [0, 1]
    let freq_coords = vec2f(freq_x, freq_y);
    let f = length(freq_coords) / f32(N / 2u);
    
    var scale = 1.0;
    let t = 1.0 - params.filter_strength;
    let order = 7.0; //I generally use 7.0 :-P
    let safe_t = max(t, 0.01);
    switch params.filter_type {
        // LSF
        case 0: {
            // Butterworth low-pass
            let cutoff = 0.3 * safe_t;
            scale = butterworth(f, cutoff, order, false);
            break;
        }
        // HSF
        case 1: {
            // Butterworth high-pass
            let cutoff = 0.05 + 0.3 * (1.0 - safe_t);
            scale = butterworth(f, cutoff, order, true);
            break;
        }
        // Band-pass filter
        case 2: {
            // Center frequency (radius) and bandwidth
            let center = params.filter_radius / 6.28;
            let bandwidth = 0.05 + 0.2 * safe_t;
            // Gaussian band-pass
            scale = exp(-pow((f - center) / bandwidth, 2.0));
            break;
        }
        // Directional
        case 3: {
            let angle = atan2(freq_coords.y, freq_coords.x);
            let direction = params.filter_direction;
             let angular_width = 0.1 + 1.0 * safe_t;
            scale = exp(-pow(sin(angle - direction) / angular_width, 2.0));
            break;
        }
        default: {
            scale = 1.0;
        }
    }
    
    // Apply the filter to each channel
    for (var i = 0u; i < N_CHANNELS; i++) {
        //(not shifted) position
        image_data[index(i, id.y, id.x)] *= scale;
    }
}

// inverse FFT on rows
@compute @workgroup_size(256, 1, 1)
fn ifft_horizontal(@builtin(workgroup_id) workgroup_id: vec3u, @builtin(local_invocation_index) local_index: u32) {
    let LOG2_N = firstLeadingBit(params.resolution);
    let LOG4_N = LOG2_N / 2u;
    let N = params.resolution;
    
    let row = workgroup_id.x;
    if (row >= N) { return; }
    
    for (var ch = 0u; ch < N_CHANNELS; ch++) {
        // Load data with bit-reversal permutation
        for (var i = 0u; i < N / 256u; i++) {
            let j = local_index + i * 256u;
            
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
            
            for (var i = 0u; i < N / 256u / 4u; i++) {
                let j = local_index + i * 256u;
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
            
            for (var i = 0u; i < N / 256u / 2u; i++) {
                let j = local_index + i * 256u;
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
        
        for (var i = 0u; i < N / 256u; i++) {
            let j = local_index + i * 256u;
            image_data[index(ch, row, j)] = X[j] / f32(N);
        }
    }
}

// now on columns inverse... 
@compute @workgroup_size(256, 1, 1)
fn ifft_vertical(@builtin(workgroup_id) workgroup_id: vec3u, @builtin(local_invocation_index) local_index: u32) {
    let LOG2_N = firstLeadingBit(params.resolution);
    let LOG4_N = LOG2_N / 2u;
    let N = params.resolution;
    
    let col = workgroup_id.x;
    if (col >= N) { return; }
    
    for (var ch = 0u; ch < N_CHANNELS; ch++) {
        for (var i = 0u; i < N / 256u; i++) {
            let j = local_index + i * 256u;
            
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
            
            for (var i = 0u; i < N / 256u / 4u; i++) {
                let j = local_index + i * 256u;
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
            
            for (var i = 0u; i < N / 256u / 2u; i++) {
                let j = local_index + i * 256u;
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
        
        for (var i = 0u; i < N / 256u; i++) {
            let j = local_index + i * 256u;
            image_data[index(ch, j, col)] = X[j] / f32(N);
        }
    }
}

//render
@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let dimensions = vec2u(textureDimensions(output));
    
    if (any(id.xy >= dimensions)) {
        return;
    }
    
    let N = params.resolution;
    
    // Calculate position in FFT image (centered)
    var p = vec2i(id.xy) - vec2i(dimensions) / 2 + vec2i(N / 2u);
    
    // Check if position is within the FFT image bounds
    if (any((p < vec2i(0)) | (p >= vec2i(N)))) {
        textureStore(output, id.xy, vec4(0.0, 0.0, 0.0, 1.0));
        return;
    }
    
    var color = vec3(0.0);
    
    if (params.show_freqs == 1) {
        // Frequency domain for better vis also log scaling for better dynamic rang
        
        let shift_x = (u32(p.x) + N / 2u) % N;
        let shift_y = (u32(p.y) + N / 2u) % N;
        
        for (var i = 0u; i < N_CHANNELS; i++) {
            let data = image_data[index(i, shift_y, shift_x)];
            

            let pixel_count = f32(N * N);
            let normalized_amp = length(data) / pixel_count;
            

            let exposure = 5000.0;
            color[i] = log(1.0 + normalized_amp * exposure) / log(exposure + 1.0);
        }
    } else {
        // Spatial domain (Filtered Image)
        for (var i = 0u; i < N_CHANNELS; i++) {
            let data = image_data[index(i, u32(p.y), u32(p.x))];
            color[i] = data.x;
        }
    }
    
    color = clamp(color, vec3(0.0), vec3(1.0));
    
    if (N_CHANNELS == 1u) {
        color = vec3(color.r);
    }
    
    if (params.is_bw != 0) {
        let luminance = dot(color, vec3(0.299, 0.587, 0.114));
        color = vec3(luminance);
    }
    textureStore(output, id.xy, vec4(color, 1.0));
}
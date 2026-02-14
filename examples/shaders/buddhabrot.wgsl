// Experimental Buddhabrot Compute Shader, Enes Altun, 2025
// A special rendering of the Mandelbrot set tracking escape trajectories
struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

struct BuddhabrotParams {
    max_iterations: u32,
    escape_radius: f32,
    zoom: f32,
    offset_x: f32,
    offset_y: f32,
    rotation: f32,
    exposure: f32,
    sample_density: f32,
    motion_speed: f32,
    dithering: f32,
    wavelength_min: f32,
    wavelength_max: f32,
    gamma: f32,
    saturation: f32,
    color_shift: f32,
    intensity_scale: f32,
    white_balance_r: f32,
    white_balance_g: f32,
    white_balance_b: f32,
    min_trajectory_len: u32,
}
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: BuddhabrotParams;

@group(2) @binding(0) var<storage, read_write> atomic_buffer: array<atomic<u32>>;

alias v4 = vec4<f32>;
alias v3 = vec3<f32>;
alias v2 = vec2<f32>;
alias m2 = mat2x2<f32>;
alias m3 = mat3x3<f32>;
const pi = 3.14159265359;
const tau = 6.28318530718;

// CIE 1931 2-degree color matching functions (390nm - 830nm, 10nm steps)
const spectrum = array<v3, 45>(
    v3(0.002362, 0.000253, 0.010482), v3(0.019110, 0.002004, 0.086011),
    v3(0.084736, 0.008756, 0.389366), v3(0.204492, 0.021391, 0.972542),
    v3(0.314679, 0.038676, 1.553480), v3(0.383734, 0.062077, 1.967280),
    v3(0.370702, 0.089456, 1.994800), v3(0.302273, 0.128201, 1.745370),
    v3(0.195618, 0.185190, 1.317560), v3(0.080507, 0.253589, 0.772125),
    v3(0.016172, 0.339133, 0.415254), v3(0.003816, 0.460777, 0.218502),
    v3(0.037465, 0.606741, 0.112044), v3(0.117749, 0.761757, 0.060709),
    v3(0.236491, 0.875211, 0.030451), v3(0.376772, 0.961988, 0.013676),
    v3(0.529826, 0.991761, 0.003988), v3(0.705224, 0.997340, 0.000000),
    v3(0.878655, 0.955552, 0.000000), v3(1.014160, 0.868934, 0.000000),
    v3(1.118520, 0.777405, 0.000000), v3(1.123990, 0.658341, 0.000000),
    v3(1.030480, 0.527963, 0.000000), v3(0.856297, 0.398057, 0.000000),
    v3(0.647467, 0.283493, 0.000000), v3(0.431567, 0.179828, 0.000000),
    v3(0.268329, 0.107633, 0.000000), v3(0.152568, 0.060281, 0.000000),
    v3(0.081261, 0.031800, 0.000000), v3(0.040851, 0.015905, 0.000000),
    v3(0.019941, 0.007749, 0.000000), v3(0.009577, 0.003718, 0.000000),
    v3(0.004553, 0.001768, 0.000000), v3(0.002175, 0.000846, 0.000000),
    v3(0.001045, 0.000407, 0.000000), v3(0.000508, 0.000199, 0.000000),
    v3(0.000251, 0.000098, 0.000000), v3(0.000126, 0.000050, 0.000000),
    v3(0.000065, 0.000025, 0.000000), v3(0.000033, 0.000013, 0.000000),
    v3(0.000018, 0.000007, 0.000000), v3(0.000009, 0.000004, 0.000000),
    v3(0.000005, 0.000002, 0.000000), v3(0.000003, 0.000001, 0.000000),
    v3(0.000002, 0.000001, 0.000000)
);

const xyz_to_rgb = m3(
     3.2404542, -0.9692660,  0.0556434,
    -1.5371385,  1.8760108, -0.2040259,
    -0.4985314,  0.0415560,  1.0572252
);

fn wl_to_xyz(wl: f32) -> v3 {
    let x = (wl - 390.0) * 0.1;
    let index = u32(clamp(x, 0.0, 43.0));
    return mix(spectrum[index], spectrum[index + 1u], fract(x));
}

// Convert a single wavelength to linear sRGB
fn wl_to_rgb(wl: f32) -> v3 {
    return max(v3(0.0), xyz_to_rgb * wl_to_xyz(wl));
}

var<private> R: v2;
var<private> seed: u32;

fn hash_u(_a: u32) -> u32 {
    var a = _a;
    a ^= a >> 16;
    a *= 0x7feb352du;
    a ^= a >> 15;
    a *= 0x846ca68bu;
    a ^= a >> 16;
    return a;
}

fn hash_f() -> f32 {
    var s = hash_u(seed);
    seed = s;
    return (f32(s) / f32(0xffffffffu));
}

fn rot(a: f32) -> m2 {
    return m2(cos(a), -sin(a), sin(a), cos(a));
}

fn cmul(a: v2, b: v2) -> v2 {
    return v2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn complex_to_screen(p: v2) -> v2 {
    var uv = (p - v2(params.offset_x, params.offset_y)) * params.zoom;
    uv = rot(-params.rotation) * uv;
    uv.x /= R.x / R.y;
    return uv * 0.5 + 0.5;
}

fn aces_tonemap(color: v3) -> v3 {
    const m1 = m3(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777
    );
    const m2 = m3(
        1.60475, -0.10208, -0.00327,
        -0.53108,  1.10813, -0.07276,
        -0.07367, -0.00605,  1.07602
    );
    var v = m1 * color;
    var a = v * (v + 0.0245786) - 0.000090537;
    var b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return m2 * (a / b);
}

fn escape_count(c: v2, max_iters: u32) -> u32 {
    var z = v2(0.0, 0.0);
    for (var n: u32 = 0; n < max_iters; n++) {
        z = cmul(z, z) + c;
        if (dot(z, z) > params.escape_radius) {
            return n;
        }
    }
    return 0u;
}

@compute @workgroup_size(64, 1, 1)
fn Splat(@builtin(global_invocation_id) id: vec3<u32>) {
    let Ru = vec2<u32>(textureDimensions(output));
    R = v2(Ru);
    seed = id.x + hash_u(time_data.frame);

    let samples_per_thread = 8u + u32(params.sample_density * 12.0);
    let pixel_count = Ru.x * Ru.y;

    // Iteration range boundaries for 3-channel split
    let range = params.max_iterations - params.min_trajectory_len;
    let third = range / 3u;
    let boundary_low = params.min_trajectory_len + third;
    let boundary_high = params.min_trajectory_len + 2u * third;

    for (var s: u32 = 0u; s < samples_per_thread; s++) {
        var c: v2;
        let sample_strategy = (s + time_data.frame) % 3u;

        if (sample_strategy == 0u) {
            let angle = hash_f() * tau;
            let radius = 0.1 + hash_f() * 0.35;
            c = v2(cos(angle) * radius - 0.25, sin(angle) * radius);
        } else if (sample_strategy == 1u) {
            c = v2(hash_f() * 3.0 - 2.0, hash_f() * 2.5 - 1.25);
        } else {
            let angle = hash_f() * tau;
            let base_radius = 0.75 + hash_f() * 0.15;
            let distortion = 0.15 * (1.0 + cos(angle * 3.0));
            c = v2(cos(angle) * (base_radius + distortion) - 0.5, sin(angle) * base_radius);
        }

        let n_escape = escape_count(c, params.max_iterations);
        if (n_escape < params.min_trajectory_len) {
            continue;
        }

        // ch 0 = short escapes
        // ch 1 = mid escapes
        // ch 2 = long escapes
        var channel: u32;
        if (n_escape < boundary_low) {
            channel = 0u;
        } else if (n_escape < boundary_high) {
            channel = 1u;
        } else {
            channel = 2u;
        }

        let buffer_offset = channel * pixel_count;

        var z = v2(0.0, 0.0);
        let w_angle = time_data.time * .2;
        let wind = v2(cos(w_angle), sin(w_angle)) * 0.0001; 
        for (var n: u32 = 0u; n < n_escape; n++) {
            z = cmul(z, z) + c;
            
            z = z + wind; 

            if (n < 5u) { continue; }
            if (abs(z.x) > 3.0 || abs(z.y) > 3.0) { continue; }

            let uv = complex_to_screen(z);

            if (uv.x >= 0.0 && uv.x < 1.0 && uv.y >= 0.0 && uv.y < 1.0) {
                let pixel_x = u32(uv.x * f32(Ru.x));
                let pixel_y = u32(uv.y * f32(Ru.y));
                let pixel_idx = pixel_x + Ru.x * pixel_y;

                if (pixel_idx < pixel_count) {
                    atomicAdd(&atomic_buffer[pixel_idx + buffer_offset], 1u);
                }
            }
        }
    }
}

@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let res = vec2<u32>(textureDimensions(output));
    if (id.x >= res.x || id.y >= res.y) { return; }
    let idx = id.x + id.y * res.x;
    let layer_offset = res.x * res.y;

    // Read raw counts per channel
    let count_short = f32(atomicLoad(&atomic_buffer[idx]));
    let count_mid   = f32(atomicLoad(&atomic_buffer[idx + layer_offset]));
    let count_long  = f32(atomicLoad(&atomic_buffer[idx + 2u * layer_offset]));

    // Normalize by frame count
    let frame_norm = 1.0 / f32(max(time_data.frame, 1u));

    // Derive 3 wavelengths from the user range
    // color_shift controls the midpoint position (0 = centered, <1 = toward min, >1 = toward max)
    let wl_short = params.wavelength_min;
    let wl_mid   = mix(params.wavelength_min, params.wavelength_max, clamp(params.color_shift, 0.0, 2.0) * 0.5);
    let wl_long  = params.wavelength_max;

    // Convert each wavelength to linear RGB via CIE XYZ
    let rgb_short = wl_to_rgb(wl_short);
    let rgb_mid   = wl_to_rgb(wl_mid);
    let rgb_long  = wl_to_rgb(wl_long);

    // Combine: each channel's count weighted by its spectral color
    var col = (count_short * rgb_short + count_mid * rgb_mid + count_long * rgb_long)
              * frame_norm * params.intensity_scale;

    // White balance
    col *= v3(params.white_balance_r, params.white_balance_g, params.white_balance_b);

    // Normalization + exposure
    let s_size = f32(res.x * res.y);
    col = col * s_size * 2e-9 / 40.0;
    col = col * pow(2.0, params.exposure);
    col = max(v3(0.0), col);

    // Saturation
    let lum = dot(col, v3(0.2126, 0.7152, 0.0722));
    col = mix(v3(lum), col, params.saturation);
    col = max(v3(0.0), col);

    // Dithering
    if (params.dithering > 0.0) {
        seed = idx + hash_u(time_data.frame);
        let noise = (hash_f() * 2.0 - 1.0) * params.dithering * 0.01;
        col += v3(noise);
    }

    col = aces_tonemap(col);
    col = pow(max(v3(0.0), col), v3(1.0 / params.gamma));

    textureStore(output, vec2<i32>(id.xy), v4(col, 1.0));

    // Clear buffer if animating
    if (params.motion_speed > 0.0) {
        atomicStore(&atomic_buffer[idx], 0u);
        atomicStore(&atomic_buffer[idx + layer_offset], 0u);
        atomicStore(&atomic_buffer[idx + 2u * layer_offset], 0u);
    }
}

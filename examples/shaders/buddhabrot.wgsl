// Spectral Buddhabrot in 4D (orbit points as (z, c)), Metropolis-Hastings sampled, Enes Altun, 2025
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
    // 0 uniform, 1 Metropolis
    sampling: u32,
    // 4D angles (degrees): Re z-Re c, Im z-Im c, Re z-Im c, Im z-Re c; spin of the first two (degrees/s)
    rot_a: f32,
    rot_b: f32,
    spin_a: f32,
    spin_b: f32,
    persist: f32,
    z0x: f32,
    z0y: f32,
    mut_size: f32,
    rot_c: f32,
    rot_d: f32,
    _p0: f32,
    _p1: f32,
}
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: BuddhabrotParams;

// Metropolis chain per thread: c, score
@group(2) @binding(0) var<storage, read_write> chains: array<vec4<f32>>;
@group(2) @binding(1) var<storage, read_write> atomic_buffer: array<atomic<u32>>;

alias v4 = vec4<f32>;
alias v3 = vec3<f32>;
alias v2 = vec2<f32>;
alias m2 = mat2x2<f32>;
alias m3 = mat3x3<f32>;
const pi = 3.14159265359;
const tau = 6.28318530718;
const K_MH: f32 = 16.0;

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
fn gauss2() -> v2 {
    let r = sqrt(-2.0 * log(max(hash_f(), 1e-7)));
    let a = hash_f() * tau;
    return r * v2(cos(a), sin(a));
}

fn rot(a: f32) -> m2 {
    return m2(cos(a), -sin(a), sin(a), cos(a));
}

fn cmul(a: v2, b: v2) -> v2 {
    return v2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// the first two rows of the 4D rotation, set once per thread
var<private> px4: v4;
var<private> py4: v4;
fn project4(z: v2, c: v2) -> v2 { let p = v4(z, c); return v2(dot(px4, p), dot(py4, p)); }
fn plane(i: u32, j: u32, a: f32) -> mat4x4<f32> {
    var m = mat4x4<f32>(v4(1.0, 0.0, 0.0, 0.0), v4(0.0, 1.0, 0.0, 0.0), v4(0.0, 0.0, 1.0, 0.0), v4(0.0, 0.0, 0.0, 1.0));
    m[i][i] = cos(a); m[j][j] = cos(a); m[j][i] = sin(a); m[i][j] = -sin(a);
    return m;
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

// escape length (0 = none), points on screen
fn score(c: v2) -> v2 {
    var z = v2(params.z0x, params.z0y);
    var on = 0.0;
    for (var n: u32 = 0u; n < params.max_iterations; n++) {
        z = cmul(z, z) + c;
        if (dot(z, z) > params.escape_radius) {
            if (n < params.min_trajectory_len) { return v2(0.0); }
            return v2(f32(n), on);
        }
        let uv = complex_to_screen(project4(z, c));
        if (n >= 5u && all(uv >= v2(0.0)) && all(uv < v2(1.0))) { on += 1.0; }
    }
    return v2(0.0);
}

fn band_of(n: u32) -> u32 {
    let third = (params.max_iterations - params.min_trajectory_len) / 3u;
    return select(select(2u, 1u, n < params.min_trajectory_len + 2u * third), 0u, n < params.min_trajectory_len + third);
}
fn splat(c: v2, n_escape: u32, w: f32) {
    let Ru = vec2<u32>(R);
    let pixel_count = Ru.x * Ru.y;
    let band = band_of(n_escape);
    let off = band * pixel_count;
    var z = v2(params.z0x, params.z0y);
    for (var n: u32 = 0u; n < n_escape; n++) {
        z = cmul(z, z) + c;
        if (n < 5u) { continue; }
        let uv = complex_to_screen(project4(z, c));
        if (all(uv >= v2(0.0)) && all(uv < v2(1.0))) {
            let k = u32(w + hash_f());
            if (k > 0u) { atomicAdd(&atomic_buffer[u32(uv.x * R.x) + Ru.x * u32(uv.y * R.y) + off], k); }
        }
    }
}

fn random_c() -> v2 { return v2(hash_f() * 3.0 - 2.0, hash_f() * 3.0 - 1.5); }

@compute @workgroup_size(64, 1, 1)
fn Splat(@builtin(global_invocation_id) id: vec3<u32>) {
    R = v2(textureDimensions(output));
    seed = hash_u(id.x * 747796405u + hash_u(time_data.frame + 1u));
    let r = plane(0u, 3u, radians(params.rot_d)) * plane(1u, 2u, radians(params.rot_c))
          * plane(1u, 3u, radians(params.rot_b + time_data.time * params.spin_b)) * plane(0u, 2u, radians(params.rot_a + time_data.time * params.spin_a));
    px4 = v4(r[0].x, r[1].x, r[2].x, r[3].x);
    py4 = v4(r[0].y, r[1].y, r[2].y, r[3].y);
    let steps = 4u + u32(params.sample_density * 10.0);

    if (params.sampling == 0u) {
        for (var s = 0u; s < steps * 2u; s++) {
            let c = random_c();
            let sc = score(c);
            if (sc.y > 0.0) { splat(c, u32(sc.x), 1.0); }
        }
        return;
    }

    // Metropolis on the score; splatting at 1 / score keeps the image unbiased
    var ch = chains[id.x];
    var c = ch.xy;
    var cur = score(c);
    for (var tries = 0; tries < 8 && cur.y <= 0.0; tries++) {
        c = random_c();
        cur = score(c);
    }
    let step = params.mut_size * 0.1 / max(params.zoom, 1e-4);
    for (var s = 0u; s < steps; s++) {
        var cn = c + gauss2() * step * pow(10.0, -2.0 * hash_f());
        if (hash_f() < 0.2) { cn = random_c(); }
        let sn = score(cn);
        if (sn.y > 0.0 && (cur.y <= 0.0 || hash_f() < sn.y / cur.y)) {
            c = cn;
            cur = sn;
        }
        if (cur.y > 0.0) { splat(c, u32(cur.x), K_MH / cur.y); }
    }
    chains[id.x] = v4(c, cur.y, 0.0);
}

@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let res = vec2<u32>(textureDimensions(output));
    if (id.x >= res.x || id.y >= res.y) { return; }
    R = v2(res);
    let idx = id.x + id.y * res.x;
    let layer_offset = res.x * res.y;

    let count_short = f32(atomicLoad(&atomic_buffer[idx]));
    let count_mid   = f32(atomicLoad(&atomic_buffer[idx + layer_offset]));
    let count_long  = f32(atomicLoad(&atomic_buffer[idx + 2u * layer_offset]));

    // frames held in the buffer (a geometric sum while fading)
    let spinning = params.spin_a != 0.0 || params.spin_b != 0.0;
    let f = f32(max(time_data.frame, 1u));
    let p = clamp(params.persist, 0.0, 0.99);
    let frames = select(f, (1.0 - pow(p, f)) / (1.0 - p), spinning);

    let wl_short = params.wavelength_min;
    let wl_mid   = mix(params.wavelength_min, params.wavelength_max, clamp(params.color_shift, 0.0, 2.0) * 0.5);
    let wl_long  = params.wavelength_max;
    var col = (count_short * wl_to_rgb(wl_short) + count_mid * wl_to_rgb(wl_mid) + count_long * wl_to_rgb(wl_long))
              * params.intensity_scale / frames;
    col *= v3(params.white_balance_r, params.white_balance_g, params.white_balance_b);
    col = max(v3(0.0), col * f32(res.x * res.y) * 2e-9 / 40.0 * pow(2.0, params.exposure));

    let lum = dot(col, v3(0.2126, 0.7152, 0.0722));
    col = max(v3(0.0), mix(v3(lum), col, params.saturation));

    if (params.dithering > 0.0) {
        seed = idx + hash_u(time_data.frame);
        col += v3((hash_f() * 2.0 - 1.0) * params.dithering * 0.01);
    }

    col = aces_tonemap(col);
    col = pow(max(v3(0.0), col), v3(1.0 / params.gamma));
    textureStore(output, vec2<i32>(id.xy), v4(col, 1.0));

    if (spinning) {
        for (var k = 0u; k < 3u; k++) {
            let i = idx + k * layer_offset;
            atomicStore(&atomic_buffer[i], u32(f32(atomicLoad(&atomic_buffer[i])) * p));
        }
    }
}

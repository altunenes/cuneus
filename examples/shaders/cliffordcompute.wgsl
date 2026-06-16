// Enes Altun, 2025 cc 3.0 
// Spectral CIE-XYZ accumulation; the look was inspired by sintel's "spectral clusters"
// (compute.toys/view/1517). 
struct TimeUniform { time: f32, delta: f32, frame: u32, _padding: u32 };
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

struct CliffordParams {
    a: f32, b: f32, c: f32, d: f32,
    motion_speed: f32,
    rotation_x: f32, rotation_y: f32,
    brightness: f32,
    scale: f32,
    dof_amount: f32, dof_focal_dist: f32,
    dispersion: f32, warp: f32, _pad3: f32,
    wl_center: f32, wl_spread: f32,
    symmetry: f32,
    _pad0: f32, _pad1: f32, _pad2: f32,
}
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: CliffordParams;
@group(2) @binding(0) var<storage, read_write> atomic_buffer: array<atomic<u32>>;

alias v4 = vec4<f32>;
alias v3 = vec3<f32>;
alias v2 = vec2<f32>;
alias m2 = mat2x2<f32>;
alias m3 = mat3x3<f32>;
const pi = 3.14159265359;
const tau = 6.28318530718;

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

var<private> R: v2;
var<private> seed: u32;

fn rot(a: f32) -> m2 { return m2(cos(a), -sin(a), sin(a), cos(a)); }
fn rotX(a: f32) -> m3 { let r = rot(a); return m3(1., 0., 0., 0., r[0][0], r[0][1], 0., r[1][0], r[1][1]); }
fn rotY(a: f32) -> m3 { let r = rot(a); return m3(r[0][0], 0., r[0][1], 0., 1., 0., r[1][0], 0., r[1][1]); }
fn rotZ(a: f32) -> m3 { let r = rot(a); return m3(r[0][0], r[0][1], 0., r[1][0], r[1][1], 0., 0., 0., 1.); }

fn hash_u(_a: u32) -> u32 { var a = _a; a ^= a >> 16u; a *= 0x7feb352du; a ^= a >> 15u; a *= 0x846ca68bu; a ^= a >> 16u; return a; }
fn hash_f() -> f32 { var s = hash_u(seed); seed = s; return (f32(s) / f32(0xffffffffu)); }
fn hash_v3() -> v3 { return v3(hash_f(), hash_f(), hash_f()); }

fn sample_disk() -> v2 {
    let r = sqrt(hash_f());
    let theta = tau * hash_f();
    return r * v2(cos(theta), sin(theta));
}

// 3D Pickover
fn attractor(p: v3, a: f32, b: f32, c: f32, d: f32) -> v3 {
    return v3(
        sin(a * p.y) - p.z * cos(b * p.x),
        p.z * sin(c * p.x) - cos(d * p.y),
        sin(p.x)
    );
}

fn projParticle(_p: v3) -> v4 {
    var p = _p * (params.scale * 1.5);
    let spin = time_data.time * 0.04;
    p = rotY(params.rotation_x * tau + spin) * p;
    p = rotX(params.rotation_y * pi - 0.3) * p;
    p = rotZ(sin(time_data.time * 0.1) * 0.2) * p;
    let z = p.z + 3.0;
    let zs = max(0.1, z);
    let proj = p.xy / zs;
    return v4(proj.x / (R.x / R.y), proj.y, zs, 1.0);
}

fn lens(_p: v2, pz: f32, focal: f32) -> v2 {
    let defocus = abs(pz - focal);
    let radial = dot(_p, _p) * 0.4;
    let coc = min((defocus + radial) * params.dof_amount * 0.025, 0.06);
    return _p + sample_disk() * coc;
}

fn aces_tonemap(color: v3) -> v3 {
    const m1 = m3(0.59719, 0.07600, 0.02840, 0.35458, 0.90834, 0.13383, 0.04823, 0.01566, 0.83777);
    const m2 = m3(1.60475, -0.10208, -0.00327, -0.53108, 1.10813, -0.07276, -0.07367, -0.00605, 1.07602);
    var v = m1 * color;
    var a = v * (v + 0.0245786) - 0.000090537;
    var b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return m2 * (a / b);
}
@compute @workgroup_size(256, 1, 1)
fn Splat(@builtin(global_invocation_id) id: vec3<u32>) {
    let Ru = vec2<u32>(textureDimensions(output));
    R = v2(Ru);
    let WH = Ru.x * Ru.y;

    seed = hash_u(id.x + hash_u(Ru.x * id.y * 200u) * 20u + hash_u(id.z) * 250u);
    seed = hash_u(seed + time_data.frame);

    let iters = 45;
    let t = time_data.time * 0.5;
    let spd = params.motion_speed * 0.3;

    // one wavelength per particle
    let wl = clamp(params.wl_center + (hash_f() - 0.5) * params.wl_spread, 390.0, 700.0);
    let wl_x = (wl - 550.0) / 150.0;
    let col_xyz = wl_to_xyz(wl);

    // wavelength couples into the attractor coefficients -> structural dispersion
    let cs = params.symmetry * 0.5 * wl_x;
    let a = params.a + 0.15 * sin(t * 0.31 * spd) + cs;
    let b = params.b + 0.15 * sin(t * 0.37 * spd) - cs;
    let c = params.c + 0.15 * sin(t * 0.43 * spd);
    let d = params.d + 0.15 * sin(t * 0.47 * spd);

    let focal = params.dof_focal_dist * 2.0 + 2.0;

    var p = (hash_v3() - 0.5) * 2.0;
    for (var i = 0; i < 18; i++) { p = attractor(p, a, b, c, d); }

    for (var i = 0; i < iters; i++) {
        p = attractor(p, a, b, c, d);
        if (i < 5) { continue; }

        let proj = projParticle(p);
        let z = proj.z;
        var uv = lens(proj.xy, z, focal) * 0.5 + 0.5;

        // cauchy dispersion
        let fp = uv * 4.0 + z * 0.4;
        let tw = time_data.time * 0.2;
        let grad = v2(
            cos(fp.x * 2.5 + tw) * sin(fp.y * 3.1) + cos(fp.x * 5.0 - tw),
            sin(fp.x * 2.1) * cos(fp.y * 2.8 - tw) + sin(fp.y * 4.0 + tw)
        );
        let wl_um = wl / 1000.0;
        let ior = 1.0 + params.dispersion / (wl_um * wl_um);
        uv += grad * (ior - 1.0) * params.warp * 0.01;

        let l = min(2.0, 0.15 / (1.0 + z * 0.1));
        let cc = vec2<u32>(uv * R);
        let idx = cc.x + Ru.x * cc.y;

        if (z > 0.01 && uv.x > 0.001 && uv.x < 0.999 && uv.y > 0.001 && uv.y < 0.999 && idx < WH) {
            let w = col_xyz * l * 256.0 * params.brightness;
            atomicAdd(&atomic_buffer[idx], u32(w.x));
            atomicAdd(&atomic_buffer[idx + WH], u32(w.y));
            atomicAdd(&atomic_buffer[idx + 2u * WH], u32(w.z));
        }
    }
}

@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let res = vec2<u32>(textureDimensions(output));
    if (id.x >= res.x || id.y >= res.y) { return; }
    let hist_id = id.x + u32(res.x) * id.y;
    let WH = res.x * res.y;

    var col_xyz = v3(
        f32(atomicLoad(&atomic_buffer[hist_id])),
        f32(atomicLoad(&atomic_buffer[hist_id + WH])),
        f32(atomicLoad(&atomic_buffer[hist_id + 2u * WH]))
    );
    col_xyz *= v3(0.95, 1.0, 1.08);
    var col = xyz_to_rgb * col_xyz;
    col = col * f32(WH) * 2e-9 / 64.0;
    col = max(v3(0.0), col);
    col = aces_tonemap(col);

    let uv = v2(f32(id.x), f32(id.y)) / v2(res);
    col *= 1.0 - 0.28 * dot(uv - 0.5, uv - 0.5) * 2.0;
    col += v3(0.001, 0.001, 0.003);

    textureStore(output, vec2<i32>(id.xy), v4(col, 1.0));

    atomicStore(&atomic_buffer[hist_id], 0u);
    atomicStore(&atomic_buffer[hist_id + WH], 0u);
    atomicStore(&atomic_buffer[hist_id + 2u * WH], 0u);
}

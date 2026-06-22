// Enes Altun, 2026;
// This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 Unported License.
//
// Schwarzschild (non rot) black hole. This started after staring at the gorgeous black holes
// on Shadertoy: baopinsui's Kerr one (couldn't look too long, my gpu melted XD) and sonicether's
// procedural Gargantua. There are many black hole code examples available online. Instead of simply using the same techniques to add yet another one, 
// I tried to create one by approaching it from different angles to create cuneus's signature at least.
// I'm not a physicist so they looked like alien relativity formulas at
// first, but adapting them into code turned out surprisingly easy once I found the right pages
// (below). But this "easy" part probably actually raise while trying to understand Baopinsui's Kerr code. hardest/time consuming part was the disk for me.. I refused to drop in a big rgba noise texture for the volume
// look, so squeezing that depth out of pure procedural noise took most of the effort tbh.
//
//   light bending, photon ring, RK4   https://en.wikipedia.org/wiki/Schwarzschild_geodesics
//   schwarzschild radius (my unit)     https://en.wikipedia.org/wiki/Schwarzschild_radius
//   disk glow = blackbody              https://en.wikipedia.org/wiki/Planck%27s_law
//   colour shift from orbital motion   https://en.wikipedia.org/wiki/Relativistic_Doppler_effect
//   colour shift from gravity          https://en.wikipedia.org/wiki/Gravitational_redshift
//   brightening on the approaching side https://en.wikipedia.org/wiki/Relativistic_beaming
//
// TAA resolve, adapted from gelami/mrange (CC0: https://www.shadertoy.com/view/fXSGR1).

// Inspirations: (even though I didn't use any of their code, they were very helpful to understand the physics and the techniques
//   baopinsui's Kerr black hole (https://www.shadertoy.com/view/wXdfzj) note that this is a really heavy shader and always crashes my browser :-( really helpful to understand how to turn the actual physics into code when you black hole
//   sonicether's Gargantua (https://www.shadertoy.com/view/lstSRS) gorgeous procedural black hole which is inspiring for how should you use cinematic blooms when you code a black hole :-P

struct TimeUniform { time: f32, delta: f32, frame: u32, _padding: u32 };
@group(0) @binding(0) var<uniform> u_t: TimeUniform;

struct BlackHoleParams {
    disk_inner: f32, disk_outer: f32, disk_thickness: f32, disk_brightness: f32,
    disk_density: f32, noise_scale: f32, swirl_speed: f32, temperature: f32,
    doppler: f32, redshift: f32, beaming: f32, ring_glow: f32,
    cam_x: f32, cam_y: f32, cam_z: f32, cam_pitch: f32,
    cam_yaw: f32, cam_roll: f32, fov: f32, taa_weight: f32,
    exposure: f32, bloom: f32, star_density: f32, gamma: f32,
    spectral_shift: f32, saturation: f32, reddening: f32, sharpen: f32,
    vividness: f32, opacity: f32, highlight: f32, spectral: f32,
}
@group(1) @binding(0) var out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> p: BlackHoleParams;

@group(3) @binding(0) var tex0: texture_2d<f32>;
@group(3) @binding(1) var sam0: sampler;
@group(3) @binding(2) var tex1: texture_2d<f32>;
@group(3) @binding(3) var sam1: sampler;

alias v2 = vec2<f32>; alias v3 = vec3<f32>; alias v4 = vec4<f32>;
alias m3 = mat3x3<f32>;

const PI: f32 = 3.14159265359;
const RS: f32 = 1.0;          // my length unit: 1 = one schwarzschild radius (rs=2GM/c²)
const HORIZON: f32 = 1.0;     // horizon sits at rs
const PHOTON_R: f32 = 1.5;    // photon sphere at 1.5 rs
const ESCAPE_R: f32 = 60.0;   // ray's escaped once it passes this
const MAX_STEPS: i32 = 320;

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

fn rotX(a: f32) -> m3 { let s = sin(a); let c = cos(a); return m3(1.,0.,0., 0.,c,-s, 0.,s,c); }
fn rotY(a: f32) -> m3 { let s = sin(a); let c = cos(a); return m3(c,0.,s, 0.,1.,0., -s,0.,c); }
fn rotZ(a: f32) -> m3 { let s = sin(a); let c = cos(a); return m3(c,-s,0., s,c,0., 0.,0.,1.); }

// Dave Hoskins (https://www.shadertoy.com/view/4djSRW)
fn hash13(q: v3) -> f32 {
    var p3 = fract(q * v3(0.1031, 0.1107, 0.0973));
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}

fn vnoise(x: v3) -> f32 {
    let i = floor(x);
    let f = fract(x);
    let u = f * f * (3.0 - 2.0 * f);
    let n000 = hash13(i + v3(0.,0.,0.));
    let n100 = hash13(i + v3(1.,0.,0.));
    let n010 = hash13(i + v3(0.,1.,0.));
    let n110 = hash13(i + v3(1.,1.,0.));
    let n001 = hash13(i + v3(0.,0.,1.));
    let n101 = hash13(i + v3(1.,0.,1.));
    let n011 = hash13(i + v3(0.,1.,1.));
    let n111 = hash13(i + v3(1.,1.,1.));
    let nx00 = mix(n000, n100, u.x);
    let nx10 = mix(n010, n110, u.x);
    let nx01 = mix(n001, n101, u.x);
    let nx11 = mix(n011, n111, u.x);
    return mix(mix(nx00, nx10, u.y), mix(nx01, nx11, u.y), u.z);
}

fn fbm(x: v3) -> f32 {
    var v = 0.0;
    var amp = 0.5;
    var pos = x;
    for (var i = 0; i < 4; i++) {
        v += amp * vnoise(pos);
        pos = pos * 2.02 + v3(11.3, 7.7, 3.1);
        amp *= 0.5;
    }
    return v;
}

// multifractal noise -> wispy filaments
fn cloud_noise(p_in: v3, octaves: i32, contrast: f32) -> f32 {
    var acc = 1.0;
    var freq = 1.0;
    for (var i = 0; i < octaves; i++) {
        acc *= 1.0 + 0.1 * (vnoise(p_in * freq) * 2.0 - 1.0);
        freq *= 3.0;
    }
    return log(1.0 + pow(max(0.0, acc), contrast));
}

fn curl2(p: v3) -> v2 {
    let e = 0.5;
    let nx1 = vnoise(p + v3(e, 0.0, 0.0));
    let nx0 = vnoise(p - v3(e, 0.0, 0.0));
    let ny1 = vnoise(p + v3(0.0, e, 0.0));
    let ny0 = vnoise(p - v3(0.0, e, 0.0));
    return v2(ny1 - ny0, -(nx1 - nx0)) / (2.0 * e);
}

fn wl_to_xyz(wl: f32) -> v3 {
    let x = (wl - 390.0) * 0.1;
    let index = u32(clamp(x, 0.0, 43.0));
    return mix(spectrum[index], spectrum[index + 1u], fract(x));
}

fn planck(wl: f32, T: f32) -> f32 {
    let x = 1.43877e7 / (wl * max(T, 1.0)); // hc/k, nm·K
    let l = wl * 1e-3;
    return 1.0 / (l * l * l * l * l * (exp(x) - 1.0));
}

// blackbody hue at T, normalized to unit luminance
fn blackbody_rgb(T: f32, wl_shift: f32) -> v3 {
    var xyz = v3(0.0);
    for (var i = 0; i < 16; i++) {
        let idx = i * 2;
        let wl = 390.0 + f32(idx) * 10.0;
        // wl_shift slides the hue
        xyz += wl_to_xyz(wl + wl_shift) * planck(wl, T);
    }
    var rgb = max(xyz_to_rgb * xyz, v3(0.0));
    let Y = dot(rgb, v3(0.2126, 0.7152, 0.0722));
    let bb = rgb / max(Y, 1e-4);

    // vividness deepens its own chroma
    let bb_lum = dot(bb, v3(0.2126, 0.7152, 0.0722));
    return max(v3(0.0), v3(bb_lum) + (bb - v3(bb_lum)) * (1.0 + p.vividness * 3.0));
}

// read the blackbody colour straight from the LUT (tex0 row 0)
const BB_T_MIN: f32 = 1000.0;
const BB_T_MAX: f32 = 30000.0;
fn blackbody_lut(T: f32) -> v3 {
    let w = textureDimensions(tex0).x;
    let t01 = clamp((T - BB_T_MIN) / (BB_T_MAX - BB_T_MIN), 0.0, 1.0);
    let x = u32(t01 * f32(w - 1u));
    return textureLoad(tex0, vec2<u32>(x, 0u), 0).rgb;
}

// YCoCg: inspiration: (gelami/mrange)
fn RGBtoYCoCg(c: v3) -> v3 {
    return m3(0.25, 0.5, -0.25, 0.5, 0.0, 0.5, 0.25, -0.5, -0.25) * c;
}
fn YCoCgToRGB(c: v3) -> v3 {
    return m3(1.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 1.0, -1.0) * c;
}

fn starfield(dir: v3) -> v3 {
    var col = v3(0.0);
    var d = normalize(dir);
    var scale = 200.0;
    for (var k = 0; k < 3; k++) {
        let pos = d * scale;
        let cell = floor(pos);
        let h = hash13(cell);
        if (h > 0.986) {
            let local = fract(pos) - 0.5;
            let dist = length(local);
            let bright = pow(smoothstep(0.16, 0.0, dist), 2.0); // tight so it survives bloom
            let tone = hash13(cell + 5.3);
            let tint = mix(v3(1.0, 0.82, 0.62), v3(0.72, 0.84, 1.0), tone);
            col += bright * tint * (0.3 + 0.8 * fract(h * 113.0));
        }
        d = d.yzx * 1.7 + 4.1;
        scale *= 2.1;
    }
    return col * p.star_density * 0.7;
}

fn galaxy(dir: v3) -> v3 {
    let d = normalize(dir);
    let bn = normalize(v3(0.62, 0.70, 0.36));
    let warp = fbm(d * 1.3 + 4.0) - 0.5;
    let h = dot(d, bn) + 0.25 * warp;
    let band = exp(-h * h * 9.0);
    if (band < 0.002) { return v3(0.0); }

    let neb = fbm(d * 2.5 + 21.0);
    let wisp = fbm(d * 6.0 + warp * 2.0 + 7.0);
    let fine = fbm(d * 13.0 + 40.0);
    let dust = smoothstep(0.4, 0.85, fbm(d * 4.0 + 50.0));

    let hue = fbm(d * 1.8 + 60.0);
    var col = mix(v3(0.010, 0.022, 0.065), v3(0.03, 0.09, 0.14), neb);
    col = mix(col, v3(0.10, 0.06, 0.09), smoothstep(0.55, 0.95, hue) * 0.5);
    col += v3(0.06, 0.09, 0.18) * pow(wisp, 3.0);
    col += v3(0.10, 0.12, 0.16) * pow(fine, 5.0);

    col *= band * (0.25 + 0.9 * neb) * (1.0 - 0.6 * dust);
    return col * p.star_density * 0.5;
}

// colour/brightness shift of the orbiting gas: orbital Doppler × gravitational redshift (see top)
fn disk_gfactor(pos: v3, photon_dir: v3) -> f32 {
    let rho = max(length(pos.xz), HORIZON + 1e-3);
     // orbital speedy
    let beta = clamp(sqrt(0.5 / rho), 0.0, 0.95);
    let gamma = inverseSqrt(max(1e-4, 1.0 - beta * beta));
    let tangent = normalize(v3(-pos.z, 0.0, pos.x)); // prograde
    let cos_a = dot(tangent, -photon_dir);
    let doppler = 1.0 / max(1e-3, gamma * (1.0 - beta * cos_a));
    let grav = sqrt(max(1e-3, 1.0 - RS / rho));
    return doppler * grav;
}

// "cheap" gas density (one tap toward the core, for the lighting gradient: those are mostly artistic choisses and I probably update it in the future..)
fn disk_density_cheap(pos: v3) -> f32 {
    let rho = length(pos.xz);
    if (rho < p.disk_inner || rho > p.disk_outer) { return 0.0; }
    let edge = (rho - p.disk_inner) / max(1e-3, p.disk_outer - p.disk_inner);
    let half_h = p.disk_thickness * (0.35 + 1.25 * (1.0 - edge));
    let yr = pos.y / max(1e-3, half_h);
    let vfall = exp(-yr * yr * 2.4);
    if (vfall < 0.01) { return 0.0; }
    let ang = atan2(pos.z, pos.x);
    let spiral = ang + p.swirl_speed * u_t.time - log(max(rho, 1e-3)) * 2.4;
    let q = v3(rho * 0.55, spiral * 0.65, pos.y * 0.9) * p.noise_scale;
    let radial = pow(clamp(p.disk_inner / max(rho, 1e-3), 0.0, 1.0), 1.6);
    return smoothstep(0.33, 1.05, cloud_noise(q, 4, 46.0)) * radial * vfall;
}

// disk gas
fn disk_sample(pos: v3, photon_dir: v3) -> v4 {
    let rho = length(pos.xz);
    if (rho < p.disk_inner || rho > p.disk_outer) { return v4(0.0); }

    // v falloff
    let edge = (rho - p.disk_inner) / max(1e-3, p.disk_outer - p.disk_inner);
    let half_h = p.disk_thickness * (0.35 + 1.25 * (1.0 - edge));
    let yr = pos.y / max(1e-3, half_h);
    let vfall = exp(-yr * yr * 2.4);
    if (vfall < 0.01) { return v4(0.0); }

    let radial = pow(clamp(p.disk_inner / max(rho, 1e-3), 0.0, 1.0), 1.6);
    if (radial * vfall * 1.4 < 0.004) { return v4(0.0); }

    // log spiral coords
    let ang = atan2(pos.z, pos.x);
    let spiral = ang + p.swirl_speed * u_t.time - log(max(rho, 1e-3)) * 2.4;
    // anisotropic
    var q = v3(rho * 0.9, spiral * 0.22, pos.y * 1.1) * p.noise_scale;

    let eddy = curl2(q * 2.2 + 9.0) * 0.3;
    q += v3(eddy.x, eddy.y, 0.0);

    let cloud = cloud_noise(q, 6, 46.0);
    let fine = cloud_noise(q * 2.6 + 21.0, 4, 30.0);
    let n = cloud * (0.5 + 0.85 * fine);
    var dens = smoothstep(0.33, 1.05, n);
    let macro_n = cloud_noise(q * 0.22 + 50.0, 3, 16.0);
    dens *= mix(0.2, 1.4, smoothstep(0.15, 1.0, macro_n));
    let dust = smoothstep(0.35, 0.92, cloud_noise(q * 0.45 + 30.0, 3, 22.0));
    dens *= (1.0 - 0.72 * dust);
    dens *= radial * vfall;
    if (dens < 0.004) { return v4(0.0); }

    // selfshadow
    let shadow_q = q + v3(0.0, 0.0, half_h * 0.9 * p.noise_scale + 0.4);
    let occ = smoothstep(0.3, 1.2, cloud_noise(shadow_q, 3, 38.0));
    let shade = 1.0 - 0.45 * occ;

    // temperature falls with radius, rises in dense filaments...
    let T_radial = (2200.0 + 4200.0 * p.temperature)
                 * pow(clamp(p.disk_inner / max(rho, 1e-3), 0.0, 1.0), 0.85);
    let T_local = T_radial * mix(0.78, 1.3, smoothstep(0.3, 1.2, cloud));
    let g_raw = disk_gfactor(pos, photon_dir);
    let g = 1.0 + (g_raw - 1.0) * p.doppler;        // doppler/redshift
    let T_obs = clamp(T_local * pow(max(g, 1e-3), p.redshift), 1200.0, 26000.0);
    var emis = blackbody_lut(T_obs);
    if (p.spectral > 0.001) {
        let wl_base = mix(660.0, 430.0, smoothstep(2000.0, 13000.0, T_local));
        let wl_obs = clamp(wl_base / max(g, 0.2), 400.0, 690.0);
        var spec = max(v3(0.0), xyz_to_rgb * wl_to_xyz(wl_obs));
        let sl = dot(spec, v3(0.2126, 0.7152, 0.0722));
        spec = spec / max(sl, 1e-4); 
        let sl2 = dot(spec, v3(0.2126, 0.7152, 0.0722));
        spec = max(v3(0.0), v3(sl2) + (spec - v3(sl2)) * (1.0 + p.vividness * 3.0));
        emis = mix(emis, spec, p.spectral);
    }

    let lum = dot(emis, v3(0.2126, 0.7152, 0.0722));
    emis = max(v3(0.0), mix(v3(lum), emis, p.saturation));

    let core_d = (rho - p.disk_inner) / max(1e-3, p.disk_inner * 0.5);
    emis += v3(1.0, 0.34, 0.10) * exp(-core_d * core_d) * 1.5;

    let lit_d = (rho - p.disk_inner) / max(1e-3, p.disk_inner * 0.01);
    emis += v3(1.0, 0.74, 0.48) * exp(-lit_d * lit_d) * p.ring_glow * 0.7;

    let face = abs(photon_dir.y);
    let view = mix(0.4, 1.15, smoothstep(0.0, 0.6, face));

    let beam_raw = pow(max(g, 1e-3), p.beaming);
    let beam = beam_raw / (1.0 + 0.12 * max(0.0, beam_raw - 1.0));
    emis *= p.disk_brightness * (0.18 + 4.0 * radial) * beam * shade * view;

    //dir light
    let light_dir = normalize(-pos + v3(1e-4, 1e-4, 1e-4));
    let grad = disk_density_cheap(pos + light_dir * 0.5) - dens;
    let lvl = max(emis.r, max(emis.g, emis.b));
    emis += v3(1.0, 0.45, 0.16) * max(0.0, grad) * lvl * 1.3;
    emis += v3(0.22, 0.5, 1.0) * max(0.0, -grad) * lvl * 3.6;

    let el = max(emis.r, max(emis.g, max(emis.b, 1e-4)));
    let comp = el * (1.0 + el * 0.012) / (1.0 + el * max(0.005, p.highlight));
    emis *= comp / el;

    let alpha = mix(clamp(dens * p.disk_density, 0.0, 1.0), 1.0, p.opacity);
    return v4(emis * dens, alpha);
}

struct Ray { pos: v3, vel: v3 }

// photon accel a = -1.5 h² r / r⁵ (rantonels, rs=1)
fn accel(pos: v3, h2: f32) -> v3 {
    let r2 = dot(pos, pos);
    let r = sqrt(r2);
    let r5 = max(r2 * r2 * r, 1e-6);
    return -1.5 * h2 * pos / r5;
}

// rk4 geodesic step
fn rk4(rin: Ray, h2: f32, dt: f32) -> Ray {
    let k1v = rin.vel;
    let k1a = accel(rin.pos, h2);
    let k2v = rin.vel + 0.5 * dt * k1a;
    let k2a = accel(rin.pos + 0.5 * dt * k1v, h2);
    let k3v = rin.vel + 0.5 * dt * k2a;
    let k3a = accel(rin.pos + 0.5 * dt * k2v, h2);
    let k4v = rin.vel + dt * k3a;
    let k4a = accel(rin.pos + dt * k3v, h2);
    var r: Ray;
    r.pos = rin.pos + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
    r.vel = rin.vel + (dt / 6.0) * (k1a + 2.0 * k2a + 2.0 * k3a + k4a);
    return r;
}

// blackbody colour LUT
@compute @workgroup_size(16, 16, 1)
fn bb_lut(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = textureDimensions(out);
    if (id.x >= dim.x || id.y != 0u) { return; }
    let t01 = f32(id.x) / f32(dim.x - 1u);
    let T = mix(BB_T_MIN, BB_T_MAX, t01);
    textureStore(out, vec2<u32>(id.x, 0u), v4(blackbody_rgb(T, p.spectral_shift), 1.0));
}

// geodesic raytrace (raw frame; TAA happens in resolve)
@compute @workgroup_size(16, 16, 1)
fn scene(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = textureDimensions(out);
    if (id.x >= dim.x || id.y >= dim.y) { return; }
    let R = v2(f32(dim.x), f32(dim.y));

    // R2 jitter (Roberts)
    let jamt = select(1.0, 0.0, p.taa_weight <= 0.001);
    let jitter = (fract(f32(u_t.frame) * v2(0.7548776662, 0.5698402910)) - 0.5) * jamt;
    let fuv = (v2(id.xy) + 0.5 + jitter) / R;

    let cam_rot = rotY(p.cam_yaw) * rotX(p.cam_pitch) * rotZ(p.cam_roll);
    let ro = v3(p.cam_x, p.cam_y, p.cam_z);
    let f = tan(p.fov * PI / 180.0 * 0.5);
    let ndc = (2.0 * fuv - 1.0) * v2(R.x / R.y, -1.0);
    let rd = normalize(cam_rot * normalize(v3(ndc * f, 1.0)));
    // conserved |r×v|² for photon orbits, used in the accel formula
    let h2 = dot(cross(ro, rd), cross(ro, rd));

    var ray = Ray(ro, rd);
    var col = v4(0.0);
    var last = ro;
    var photon_dir = rd;
    var turn = 0.0;
    var captured = false;

    for (var i = 0; i < MAX_STEPS; i++) {
        if (col.a > 0.99) { break; }
        let r = length(ray.pos);
        if (r < HORIZON * 1.001) { captured = true; break; }
        if (r > ESCAPE_R && dot(ray.pos, ray.vel) > 0.0) { break; }

        let dt = clamp(0.10 * (r - HORIZON * 0.9), 0.015, 1.2);
        let next = rk4(ray, h2, dt);

        let seg = next.pos - ray.pos;
        let seglen = length(seg);
        if (seglen > 1e-6) {
            let nd = seg / seglen;
            turn += acos(clamp(dot(photon_dir, nd), -1.0, 1.0));
            photon_dir = nd;
        }

        // march the disk slab; skip rays nowhere near it
        let rho_a = length(ray.pos.xz);
        let rho_b = length(next.pos.xz);
        let in_annulus = max(rho_a, rho_b) > p.disk_inner - 1.0
                      && min(rho_a, rho_b) < p.disk_outer + 1.0;
        let crossing = (ray.pos.y * next.pos.y < 0.0);
        let near_plane = abs(ray.pos.y) < p.disk_thickness * 2.0 + seglen;
        if (in_annulus && (crossing || near_plane)) {
            let dither = hash13(v3(f32(id.x), f32(id.y), f32(u_t.frame)));
            let subs = clamp(i32(seglen / 0.06) + 1, 1, 24);
            let inv = 1.0 / f32(subs);
            for (var s = 0; s < subs; s++) {
                if (col.a > 0.99) { break; }
                let t = (f32(s) + dither) * inv;
                let sp = mix(ray.pos, next.pos, t);
                let smp = disk_sample(sp, photon_dir);
                if (smp.a > 0.0) {
                    let a = clamp(smp.a * seglen * inv * 1.6, 0.0, 1.0);
                    // wavelength extinction: blue dies faster through gas -> deep gas reddens
                    let t = clamp(1.0 - col.a, 0.001, 1.0);
                    let trans = v3(t, pow(t, 1.4 + p.reddening * 1.6), pow(t, 1.8 + p.reddening * 4.0));
                    col = v4(col.rgb + smp.rgb * seglen * inv * 1.3 * trans,
                             col.a + (1.0 - col.a) * a);
                }
            }
        }

        ray = next;
        last = ray.pos;
    }

    // photon ring glow
    if (!captured) {
        let glow = smoothstep(1.6, 7.0, turn) * p.ring_glow;
        if (glow > 0.001) {
            let glow_col = mix(v3(1.0, 0.32, 0.12), v3(1.0, 0.62, 0.38), smoothstep(5.0, 9.0, turn));
            let ang = atan2(photon_dir.z, photon_dir.x);
            let tex = 0.55 + 0.55 * cloud_noise(v3(ang * 2.4 + p.swirl_speed * u_t.time, turn * 0.5, 5.0), 3, 18.0);
            col = v4(col.rgb + glow * glow_col * tex * (1.0 - col.a) * 0.9, col.a);
        }
    }
    if (!captured && col.a < 0.99) {
        let bg = starfield(photon_dir) + galaxy(photon_dir);
        col = v4(col.rgb + bg * (1.0 - col.a), col.a);
    }

    textureStore(out, id.xy, v4(col.rgb, 1.0));
}

// TAA resolve (gelami/mrange)
@compute @workgroup_size(16, 16, 1)
fn resolve(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = textureDimensions(out);
    if (id.x >= dim.x || id.y >= dim.y) { return; }
    let R = v2(f32(dim.x), f32(dim.y));
    let uv = (v2(id.xy) + 0.5) / R;

    // current pixel + its 3x3 neighbours
    let cur = RGBtoYCoCg(textureSampleLevel(tex0, sam0, uv, 0.0).rgb);
    var mn = cur;
    var mx = cur;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let o = v2(f32(x), f32(y)) / R;
            let n = RGBtoYCoCg(textureSampleLevel(tex0, sam0, uv + o, 0.0).rgb);
            mn = min(mn, n);
            mx = max(mx, n);
        }
    }

    var hist = RGBtoYCoCg(textureSampleLevel(tex1, sam1, uv, 0.0).rgb);
    hist = clamp(hist, mn, mx);

    let blend = select(p.taa_weight, 0.0, u_t.frame < 4u);
    let outc = mix(cur, hist, blend);
    textureStore(out, id.xy, v4(max(v3(0.0), YCoCgToRGB(outc)), 1.0));
}

@compute @workgroup_size(16, 16, 1)
fn bright(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = textureDimensions(out);
    if (id.x >= dim.x || id.y >= dim.y) { return; }
    let uv = (v2(id.xy) + 0.5) / v2(f32(dim.x), f32(dim.y));
    let c = textureSampleLevel(tex0, sam0, uv, 0.0).rgb;
    let l = dot(c, v3(0.2126, 0.7152, 0.0722));
    let k = smoothstep(0.8, 2.2, l); // so only bright gas blooms.
    textureStore(out, id.xy, v4(c * k, 1.0));
}

// separable gaussian blur (dir = axis * spacing)
fn blur5(uv: v2, dir: v2, R: v2) -> v3 {
    let o = dir / R;
    var s = textureSampleLevel(tex0, sam0, uv, 0.0).rgb * 0.382928;
    s += textureSampleLevel(tex0, sam0, uv + o, 0.0).rgb * 0.241732;
    s += textureSampleLevel(tex0, sam0, uv - o, 0.0).rgb * 0.241732;
    s += textureSampleLevel(tex0, sam0, uv + 2.0 * o, 0.0).rgb * 0.060598;
    s += textureSampleLevel(tex0, sam0, uv - 2.0 * o, 0.0).rgb * 0.060598;
    return s;
}

// bloom pyramid: 3 gaussian levels, each accumulating the previous -> big soft halo
@compute @workgroup_size(16, 16, 1)
fn blur1_h(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = textureDimensions(out); if (id.x >= dim.x || id.y >= dim.y) { return; }
    let R = v2(f32(dim.x), f32(dim.y)); let uv = (v2(id.xy) + 0.5) / R;
    textureStore(out, id.xy, v4(blur5(uv, v2(2.0, 0.0), R), 1.0));
}
@compute @workgroup_size(16, 16, 1)
fn blur1_v(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = textureDimensions(out); if (id.x >= dim.x || id.y >= dim.y) { return; }
    let R = v2(f32(dim.x), f32(dim.y)); let uv = (v2(id.xy) + 0.5) / R;
    textureStore(out, id.xy, v4(blur5(uv, v2(0.0, 2.0), R), 1.0));
}
@compute @workgroup_size(16, 16, 1)
fn blur2_h(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = textureDimensions(out); if (id.x >= dim.x || id.y >= dim.y) { return; }
    let R = v2(f32(dim.x), f32(dim.y)); let uv = (v2(id.xy) + 0.5) / R;
    textureStore(out, id.xy, v4(blur5(uv, v2(5.0, 0.0), R), 1.0));
}
@compute @workgroup_size(16, 16, 1)
fn blur2_v(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = textureDimensions(out); if (id.x >= dim.x || id.y >= dim.y) { return; }
    let R = v2(f32(dim.x), f32(dim.y)); let uv = (v2(id.xy) + 0.5) / R;
    let prev = textureSampleLevel(tex1, sam1, uv, 0.0).rgb; // + level 1
    textureStore(out, id.xy, v4(blur5(uv, v2(0.0, 5.0), R) + 0.75 * prev, 1.0));
}
@compute @workgroup_size(16, 16, 1)
fn blur3_h(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = textureDimensions(out); if (id.x >= dim.x || id.y >= dim.y) { return; }
    let R = v2(f32(dim.x), f32(dim.y)); let uv = (v2(id.xy) + 0.5) / R;
    textureStore(out, id.xy, v4(blur5(uv, v2(11.0, 0.0), R), 1.0));
}
@compute @workgroup_size(16, 16, 1)
fn blur3_v(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = textureDimensions(out); if (id.x >= dim.x || id.y >= dim.y) { return; }
    let R = v2(f32(dim.x), f32(dim.y)); let uv = (v2(id.xy) + 0.5) / R;
    let prev = textureSampleLevel(tex1, sam1, uv, 0.0).rgb;
    textureStore(out, id.xy, v4(blur5(uv, v2(0.0, 11.0), R) + 0.75 * prev, 1.0));
}


fn aces_scalar(x: f32) -> f32 {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

// for hue preserving 
fn tonemap(c: v3) -> v3 {
    let peak = max(max(c.r, c.g), max(c.b, 1e-5));
    var ratio = c / peak;
    let desat = clamp((peak - 1.5) / 3.5, 0.0, 0.85);
    ratio = mix(ratio, v3(1.0), desat);
    return ratio * aces_scalar(peak);
}

@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = textureDimensions(out);
    if (id.x >= dim.x || id.y >= dim.y) { return; }
    let R = v2(f32(dim.x), f32(dim.y));
    let uv = (v2(id.xy) + 0.5) / R;

    let scene = textureSampleLevel(tex0, sam0, uv, 0.0).rgb;

    // unsharp sharpen
    let e = 1.4 / R;
    let lo = (textureSampleLevel(tex0, sam0, uv + v2(e.x, 0.0), 0.0).rgb
            + textureSampleLevel(tex0, sam0, uv - v2(e.x, 0.0), 0.0).rgb
            + textureSampleLevel(tex0, sam0, uv + v2(0.0, e.y), 0.0).rgb
            + textureSampleLevel(tex0, sam0, uv - v2(0.0, e.y), 0.0).rgb) * 0.25;
    let sharp = max(v3(0.0), scene + (scene - lo) * p.sharpen);

    let bloom = textureSampleLevel(tex1, sam1, uv, 0.0).rgb;

    var color = (sharp + bloom * p.bloom) * p.exposure;
    color = pow(color, v3(p.gamma));

    let luma = dot(color, v3(0.2126, 0.7152, 0.0722));
    color = mix(v3(luma), color, 1.08);

    let q = uv - 0.5;
    color *= 1.0 - dot(q, q) * 0.55;

    textureStore(out, id.xy, v4(color, 1.0));
}

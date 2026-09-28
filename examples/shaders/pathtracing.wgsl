// Enes Altun, 2026;
// This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 Unported License.
//
// Real-time path tracer: NEE+MIS with per-point light importance, GGX VNDF sampling, Owen-scrambled Sobol,
// path regularization for mirror caustics, ReSTIR DI (temporal+spatial) while moving and plain NEE+MIS when still,
// motion-vector reprojection (camera + animated object) with variance clipping, and SVGF or FLR denoising.
// Refs: Dupuy & Benyoub 2023 (spherical-cap VNDF sampling), Burley 2020 (hash-based Owen-scrambled Sobol),
// Duff et al. 2017 (branchless ONB), Kaplanyan & Dachsbacher 2013 (path regularization),
// Bitterli et al. 2020 (ReSTIR DI), Schied et al. 2017 (SVGF), Salmi et al. 2024 (FLR, non-neural core),
// Salvi 2016 (variance clipping), Jimenez 2014 (13-tap / tent bloom), Karis 2014 (firefly-weighted bloom downsample).

struct TimeUniform { time: f32, delta: f32, frame: u32, _padding: u32 };
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

struct Params {
    cam_x: f32, cam_y: f32, cam_z: f32, fov: f32,
    tgt_x: f32, tgt_y: f32, tgt_z: f32, aperture: f32,
    pcam_x: f32, pcam_y: f32, pcam_z: f32, focus_dist: f32,
    ptgt_x: f32, ptgt_y: f32, ptgt_z: f32, exposure: f32,
    max_bounces: u32, diffuse_bounces: u32, accumulate: u32, cam_moved: u32,
    num_lights: u32, ris_candidates: u32, restir: u32, spatial_count: u32,
    spatial_radius: f32, c_cap: f32, hist_realtime: f32, hist_move: f32,
    denoise: u32, atrous_iters: u32, sigma_l: f32, sigma_n: f32,
    sigma_z: f32, firefly: f32, dispersion: f32, rotation_speed: f32,
    bloom: f32, use_hdri: u32, sky_strength: f32, open_roof: u32,
    regularize: f32, clip_gamma: f32, debug_view: u32, side_mode: u32,
    gamma: f32, obj_speed: f32, _p3: f32, _p4: f32,
}
@group(1) @binding(0) var out_tex: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: Params;

struct MouseUniform { position: vec2<f32>, click_position: vec2<f32>, wheel: vec2<f32>, buttons: vec2<u32> };
@group(2) @binding(0) var<uniform> mouse: MouseUniform;
@group(2) @binding(1) var channel0: texture_2d<f32>;
@group(2) @binding(2) var channel0_sampler: sampler;

@group(3) @binding(0) var tex0: texture_2d<f32>; @group(3) @binding(1) var sam0: sampler;
@group(3) @binding(2) var tex1: texture_2d<f32>; @group(3) @binding(3) var sam1: sampler;
@group(3) @binding(4) var tex2: texture_2d<f32>; @group(3) @binding(5) var sam2: sampler;
@group(3) @binding(6) var tex3: texture_2d<f32>; @group(3) @binding(7) var sam3: sampler;
@group(3) @binding(8) var tex4: texture_2d<f32>; @group(3) @binding(9) var sam4: sampler;

alias v4 = vec4<f32>; alias v3 = vec3<f32>; alias v2 = vec2<f32>; alias m3 = mat3x3<f32>; alias i2 = vec2<i32>;
const pi = 3.14159265359;
const INF = 1e30;
const NL_MAX = 32u;
const SKY_DEPTH = 1000.0;
const LUMA = v3(0.2126, 0.7152, 0.0722);

var<private> R: v2;
var<private> seed: u32;
var<private> px_seed: u32;
var<private> anim_time: f32;
var<private> wavelength: f32;

// ---------------------------------------------------------------- scene
struct Ray { o: v3, d: v3 };
struct Hit { p: v3, n: v3, t: f32, front: bool, mat: u32, hit: bool, light: i32 };
struct Material { albedo: v3, metallic: f32, roughness: f32 };
struct Sphere { c: v3, r: f32, mat: u32 };
struct Rect { c: v3, u: v3, v: v3, n: v3, mat: u32 };
struct Tri { a: v3, b: v3, c: v3, mat: u32 };
struct Light { pos: v3, r: f32, le: v3 };

const NSPH = 10u;
const SPHERES = array<Sphere, 9>(
    Sphere(v3(-1.9, 0.0, -1.0), 0.5, 6u), Sphere(v3(-2.4, -0.2, -2.3), 0.3, 5u), Sphere(v3(-1.4, -0.25, 0.3), 0.25, 2u),
    Sphere(v3(-2.7, -0.3, -0.3), 0.2, 12u), Sphere(v3(-1.0, -0.35, 0.7), 0.15, 9u), Sphere(v3(2.1, -0.1, -1.7), 0.4, 8u),
    Sphere(v3(1.4, -0.3, -0.3), 0.2, 10u), Sphere(v3(2.6, -0.35, -0.7), 0.15, 7u), Sphere(v3(0.55, -0.4, 0.8), 0.1, 1u),
);
const ROOM = array<Rect, 6>(
    Rect(v3(0.0, 5.0, 1.5), v3(8.0, 0.0, 0.0), v3(0.0, 0.0, 15.0), v3(0.0, -1.0, 0.0), 21u),
    Rect(v3(0.0, -0.5, 1.5), v3(8.0, 0.0, 0.0), v3(0.0, 0.0, 15.0), v3(0.0, 1.0, 0.0), 0u),
    Rect(v3(0.0, 2.25, -6.0), v3(8.0, 0.0, 0.0), v3(0.0, 5.5, 0.0), v3(0.0, 0.0, 1.0), 23u),
    Rect(v3(-4.0, 2.25, 1.5), v3(0.0, 0.0, 15.0), v3(0.0, 5.5, 0.0), v3(1.0, 0.0, 0.0), 26u),
    Rect(v3(4.0, 2.25, 1.5), v3(0.0, 0.0, 15.0), v3(0.0, 5.5, 0.0), v3(-1.0, 0.0, 0.0), 26u),
    Rect(v3(0.0, 4.985, 1.5), v3(8.0, 0.0, 0.0), v3(0.0, 0.0, 15.0), v3(0.0, -1.0, 0.0), 21u),
);
// glass prism (centre 0.3, 0.07, -1.3; size 1.3; half depth 0.55)
const PA0 = v3(0.3, 0.876, -0.75); const PB0 = v3(-0.415, -0.476, -0.75); const PC0 = v3(1.015, -0.476, -0.75);
const PA1 = v3(0.3, 0.876, -1.85); const PB1 = v3(-0.415, -0.476, -1.85); const PC1 = v3(1.015, -0.476, -1.85);
const TRIS = array<Tri, 8>(
    Tri(PA0, PB0, PC0, 30u), Tri(PA1, PC1, PB1, 30u), Tri(PA0, PA1, PB1, 30u), Tri(PA0, PB1, PB0, 30u),
    Tri(PB0, PB1, PC1, 30u), Tri(PB0, PC1, PC0, 30u), Tri(PC0, PC1, PA1, 30u), Tri(PC0, PA1, PA0, 30u),
);

// animated object: glossy ceramic ball bouncing across the floor (mat 15, unique -> gets its own motion vectors)
const OBJ_R = 0.3;
fn obj_center(t: f32) -> v3 { let s = t * params.obj_speed; return v3(sin(s * 0.9) * 1.8, -0.5 + OBJ_R + abs(sin(s * 2.2)) * 0.9, cos(s * 0.9) * 0.8 + 1.6); }
fn get_sphere(i: u32) -> Sphere {
    if (i == 9u) { return Sphere(obj_center(anim_time), OBJ_R, 15u); }
    return SPHERES[i];
}
// realtime mode animates; history then needs clipping (moving shadows/reflections can't be reprojected)
fn animating() -> bool { return params.accumulate == 0u; }
fn n_lights() -> u32 { return max(min(params.num_lights, NL_MAX), 1u); }
fn get_light(i: u32) -> Light {
    if (i == 0u) { return Light(v3(0.0, 1.6, 0.6), 0.6, light_col(0u)); }
    let f = f32(i); let spin = anim_time * 0.15 * params.rotation_speed;
    let a = f * (2.0 * pi / f32(max(n_lights() - 1u, 1u))) + spin;
    let pos = v3(cos(a) * 3.4, 2.2 + sin(f * 1.7 + anim_time * 0.4) * 1.4, sin(a) * 5.0 - 0.5);
    return Light(pos, 0.35, light_col(i));
}
// lamp colours: soft tints (not pure primaries), equal luminance so no hue is much brighter than another
fn light_col(i: u32) -> v3 {
    if (i == 0u) { let w = v3(1.0, 0.63, 0.34); return w / lum(w) * 6.0; }   // warm incandescent bulb (~2900K)
    let c = mix(v3(1.0), hue2rgb(fract(f32(i) * 0.618034 + 0.08)), 0.55);
    return c / lum(c) * 9.0;
}
// glowing-orb emitter: radiance ~ chord length through the sphere (brighter core, soft rim), mean 1 over the disc
fn orb(L: Light, o: v3, d: v3) -> f32 {
    let oc = L.pos - o; let tca = dot(oc, d); let b2 = max(dot(oc, oc) - tca * tca, 0.0);
    return 1.5 * sqrt(max(1.0 - b2 / (L.r * L.r), 0.0));
}
// side walls (3 = left, 4 = right): side_mode 0 = diffuse walls, 1 = mirrors, 2 = open
fn is_side(i: u32) -> bool { return i == 3u || i == 4u; }
fn room_on(i: u32) -> bool { return !(params.open_roof == 1u && (i == 0u || i == 5u)) && !(params.side_mode == 2u && is_side(i)); }
fn room_rect(i: u32) -> Rect { var r = ROOM[i]; if (params.side_mode == 0u && is_side(i)) { r.mat = select(25u, 24u, i == 3u); } return r; }

fn hash_u(a0: u32) -> u32 { var a = a0; a ^= a >> 16u; a *= 0x7feb352du; a ^= a >> 15u; a *= 0x846ca68bu; a ^= a >> 16u; return a; }
fn hash_f() -> f32 { seed = hash_u(seed); return f32(seed >> 8u) / 16777216.0; }
fn hue2rgb(h: f32) -> v3 { return clamp(v3(abs(h * 6.0 - 3.0) - 1.0, 2.0 - abs(h * 6.0 - 2.0), 2.0 - abs(h * 6.0 - 4.0)), v3(0.0), v3(1.0)); }
fn lum(c: v3) -> f32 { return dot(c, LUMA); }

fn hit_sphere(s: Sphere, ray: Ray, tmin: f32, tmax: f32, h: ptr<function, Hit>) -> bool {
    let oc = ray.o - s.c; let b = dot(oc, ray.d); let c = dot(oc, oc) - s.r * s.r; let disc = b * b - c;
    if (disc < 0.0) { return false; }
    let sq = sqrt(disc); var t = -b - sq; if (t < tmin || t > tmax) { t = -b + sq; if (t < tmin || t > tmax) { return false; } }
    let p = ray.o + t * ray.d; let out_n = (p - s.c) / s.r; let ff = dot(ray.d, out_n) < 0.0;
    (*h).t = t; (*h).p = p; (*h).n = select(-out_n, out_n, ff); (*h).front = ff; (*h).mat = s.mat; (*h).light = -1; return true;
}
fn hit_rect(q: Rect, ray: Ray, tmin: f32, tmax: f32, h: ptr<function, Hit>) -> bool {
    let dn = dot(q.n, ray.d); if (abs(dn) < 1e-6) { return false; }
    let t = dot(q.c - ray.o, q.n) / dn; if (t < tmin || t > tmax) { return false; }
    let p = ray.o + t * ray.d; let off = p - q.c;
    if (abs(dot(off, normalize(q.u))) > length(q.u) * 0.5 || abs(dot(off, normalize(q.v))) > length(q.v) * 0.5) { return false; }
    let ff = dn < 0.0; (*h).t = t; (*h).p = p; (*h).n = select(-q.n, q.n, ff); (*h).front = ff; (*h).mat = q.mat; (*h).light = -1; return true;
}
fn hit_tri(q: Tri, ray: Ray, tmin: f32, tmax: f32, h: ptr<function, Hit>) -> bool {
    let e1 = q.b - q.a; let e2 = q.c - q.a; let pv = cross(ray.d, e2); let det = dot(e1, pv); if (abs(det) < 1e-8) { return false; }
    let inv = 1.0 / det; let tv = ray.o - q.a; let u = dot(tv, pv) * inv; if (u < 0.0 || u > 1.0) { return false; }
    let qv = cross(tv, e1); let v = dot(ray.d, qv) * inv; if (v < 0.0 || u + v > 1.0) { return false; }
    let t = dot(e2, qv) * inv; if (t < tmin || t > tmax) { return false; }
    var ng = normalize(cross(e1, e2)); ng = select(-ng, ng, dot(ng, (q.a + q.b + q.c) / 3.0 - v3(0.3, 0.0, -1.3)) > 0.0);
    let ff = dot(ray.d, ng) < 0.0;
    (*h).t = t; (*h).p = ray.o + t * ray.d; (*h).n = select(-ng, ng, ff); (*h).front = ff; (*h).mat = q.mat; (*h).light = -1; return true;
}
fn intersect(ray: Ray) -> Hit {
    var h: Hit; h.hit = false; h.light = -1; var tmax = INF;
    for (var i = 0u; i < NSPH; i++) { if (hit_sphere(get_sphere(i), ray, 1e-3, tmax, &h)) { h.hit = true; tmax = h.t; } }
    for (var i = 0u; i < 6u; i++) { if (room_on(i) && hit_rect(room_rect(i), ray, 1e-3, tmax, &h)) { h.hit = true; tmax = h.t; } }
    for (var i = 0u; i < 8u; i++) { if (hit_tri(TRIS[i], ray, 1e-3, tmax, &h)) { h.hit = true; tmax = h.t; } }
    for (var i = 0u; i < n_lights(); i++) {
        let L = get_light(i);
        if (hit_sphere(Sphere(L.pos, L.r, 40u), ray, 1e-3, tmax, &h)) { h.hit = true; tmax = h.t; h.light = i32(i); h.n = normalize(h.p - L.pos); }
    }
    return h;
}
// shadow ray: any hit, lights don't occlude
fn occluded(o: v3, d: v3, tmax: f32) -> bool {
    let ray = Ray(o, d); var h: Hit;
    for (var i = 0u; i < NSPH; i++) { if (hit_sphere(get_sphere(i), ray, 1e-3, tmax, &h)) { return true; } }
    for (var i = 0u; i < 6u; i++) { if (room_on(i) && hit_rect(room_rect(i), ray, 1e-3, tmax, &h)) { return true; } }
    for (var i = 0u; i < 8u; i++) { if (hit_tri(TRIS[i], ray, 1e-3, tmax, &h)) { return true; } }
    return false;
}

fn cycle(n: v3, speed: f32) -> v3 { let c = anim_time * speed + dot(n, v3(0.5, 0.3, 1.2)); return 0.5 + 0.5 * sin(v3(c, c + 2.1, c + 4.2)); }
fn material(id: u32, p: v3, n: v3) -> Material {
    var m = Material(v3(0.75), 0.0, 0.9);
    switch (id) {
        case 0u: { m.albedo = select(v3(0.7), v3(0.02), fract((floor(p.x) + floor(p.z)) * 0.5) < 0.5); }
        case 1u: { m = Material(v3(0.95, 0.7, 0.3), 0.9, 0.1); }
        case 2u: { m = Material(v3(0.95, 0.95, 1.0), 1.0, 0.0); }
        case 5u: { m = Material(cycle(n, 1.3), 1.0, 0.0); }
        case 6u: { m = Material(v3(0.95, 0.9, 0.9), 1.0, 0.0); }
        case 7u: { m = Material(v3(0.1, 0.8, 0.2), 0.1, 0.7); }
        case 8u: { m = Material(v3(0.95, 0.64, 0.54), 0.85, 0.2); }
        case 9u: { m = Material(v3(0.9, 0.1, 0.2), 0.0, 0.6); }
        case 10u: { m = Material(v3(0.8, 0.8, 0.9), 1.0, 0.0); }
        case 12u: { m = Material(cycle(n, 10.3), 1.0, 0.0); }
        case 14u: { m = Material(cycle(n, 1.3), 1.0, 0.0); }
        case 15u: { m = Material(v3(0.85, 0.25, 0.12), 0.0, 0.25); }
        case 21u: { m.albedo = v3(0.75, 0.75, 0.73); }
        case 23u: { m.albedo = v3(0.20, 0.55, 0.52); }
        case 24u: { m.albedo = v3(0.66, 0.54, 0.42); }
        case 25u: { m.albedo = v3(0.46, 0.52, 0.64); }
        case 26u: { m = Material(v3(0.9, 0.9, 0.92), 1.0, 0.0); }
        case 30u: { m = Material(v3(0.99), 0.0, 0.0); }
        default: {}
    }
    return m;
}
fn is_glass(id: u32) -> bool { return id == 30u; }
fn is_mirror(m: Material) -> bool { return m.metallic > 0.9 && m.roughness < 0.03; }
fn is_delta(id: u32, m: Material) -> bool { return is_glass(id) || is_mirror(m); }
// ReSTIR handles rough surfaces; glossy ones keep NEE+MIS (light sampling alone is poor on tight lobes)
fn restir_ok(m: Material) -> bool { return m.roughness >= 0.2; }

fn sky(dir: v3) -> v3 {
    if (params.use_hdri == 1u) {
        let uv = v2((atan2(dir.z, dir.x) + pi) / (2.0 * pi), 1.0 - (asin(clamp(dir.y, -1.0, 1.0)) + pi * 0.5) / pi);
        return textureSampleLevel(channel0, channel0_sampler, uv, 0.0).rgb * params.sky_strength;
    }
    return mix(v3(0.045, 0.05, 0.06), v3(0.10, 0.11, 0.13), clamp(dir.y * 0.5 + 0.5, 0.0, 1.0)) * params.sky_strength;
}

// ---------------------------------------------------------------- sampling: Owen-scrambled Sobol (Burley 2020)
const SOBOL_DIR = array<u32, 96>(
    0x80000000u, 0xc0000000u, 0xa0000000u, 0xf0000000u, 0x88000000u, 0xcc000000u, 0xaa000000u, 0xff000000u, 0x80800000u, 0xc0c00000u, 0xa0a00000u, 0xf0f00000u, 0x88880000u, 0xcccc0000u, 0xaaaa0000u, 0xffff0000u, 0x80008000u, 0xc000c000u, 0xa000a000u, 0xf000f000u, 0x88008800u, 0xcc00cc00u, 0xaa00aa00u, 0xff00ff00u, 0x80808080u, 0xc0c0c0c0u, 0xa0a0a0a0u, 0xf0f0f0f0u, 0x88888888u, 0xccccccccu, 0xaaaaaaaau, 0xffffffffu,
    0x80000000u, 0xc0000000u, 0x60000000u, 0x90000000u, 0xe8000000u, 0x5c000000u, 0x8e000000u, 0xc5000000u, 0x68800000u, 0x9cc00000u, 0xee600000u, 0x55900000u, 0x80680000u, 0xc09c0000u, 0x60ee0000u, 0x90550000u, 0xe8808000u, 0x5cc0c000u, 0x8e606000u, 0xc5909000u, 0x6868e800u, 0x9c9c5c00u, 0xeeee8e00u, 0x5555c500u, 0x8000e880u, 0xc0005cc0u, 0x60008e60u, 0x9000c590u, 0xe8006868u, 0x5c009c9cu, 0x8e00eeeeu, 0xc5005555u,
    0x80000000u, 0xc0000000u, 0x20000000u, 0x50000000u, 0xf8000000u, 0x74000000u, 0xa2000000u, 0x93000000u, 0xd8800000u, 0x25400000u, 0x59e00000u, 0xe6d00000u, 0x78080000u, 0xb40c0000u, 0x82020000u, 0xc3050000u, 0x208f8000u, 0x51474000u, 0xfbea2000u, 0x75d93000u, 0xa0858800u, 0x914e5400u, 0xdbe79e00u, 0x25db6d00u, 0x58800080u, 0xe54000c0u, 0x79e00020u, 0xb6d00050u, 0x800800f8u, 0xc00c0074u, 0x200200a2u, 0x50050093u,
);
fn lk_perm(x0: u32, s: u32) -> u32 { var x = x0 + s; x ^= x * 0x6c50b47cu; x ^= x * 0xb82f1e52u; x ^= x * 0xc7afe638u; x ^= x * 0x8d22f6e6u; return x; }
fn owen(x: u32, s: u32) -> u32 { return reverseBits(lk_perm(reverseBits(x), s)); }
fn sobol(i0: u32, d: u32) -> u32 {
    if (d == 0u) { return reverseBits(i0); }
    var x = 0u; var i = i0; var b = 0u;
    loop { if (i == 0u) { break; } if ((i & 1u) != 0u) { x ^= SOBOL_DIR[(d - 1u) * 32u + b]; } i >>= 1u; b += 1u; }
    return x;
}
// 4D sample for dimension group g; the index shuffle decorrelates groups (padding)
fn rng4(g: u32) -> v4 {
    let s = hash_u(px_seed ^ hash_u(g * 0x9e3779b9u + 0x632be5abu));
    let i = owen(time_data.frame, s);
    var r: v4;
    for (var d = 0u; d < 4u; d++) { r[d] = f32(owen(sobol(i, d), hash_u(s + d * 0x68bc21ebu)) >> 8u) / 16777216.0; }
    return r;
}

// ---------------------------------------------------------------- BSDF: Lambert + GGX (height-correlated Smith), VNDF sampling
fn onb(n: v3) -> m3 {
    let s = select(-1.0, 1.0, n.z >= 0.0); let a = -1.0 / (s + n.z); let b = n.x * n.y * a;
    return m3(v3(1.0 + s * n.x * n.x * a, s * b, -s * n.x), v3(b, s + n.y * n.y * a, -n.y), n);
}
fn ggx_D(noh: f32, a: f32) -> f32 { let a2 = a * a; let d = noh * noh * (a2 - 1.0) + 1.0; return a2 / (pi * d * d + 1e-9); }
fn smith_G1(nv: f32, a: f32) -> f32 { let a2 = a * a; return 2.0 * nv / (nv + sqrt(a2 + (1.0 - a2) * nv * nv)); }
fn smith_V(nv: f32, nl: f32, a: f32) -> f32 { let a2 = a * a; let gv = nl * sqrt(nv * nv * (1.0 - a2) + a2); let gl = nv * sqrt(nl * nl * (1.0 - a2) + a2); return 0.5 / max(gv + gl, 1e-6); }
fn fresnel(voh: f32, f0: v3) -> v3 { return f0 + (1.0 - f0) * pow(clamp(1.0 - voh, 0.0, 1.0), 5.0); }
fn alpha_of(m: Material) -> f32 { return max(m.roughness * m.roughness, 2e-3); }
fn spec_prob(m: Material) -> f32 { let es = lum(mix(v3(0.04), m.albedo, m.metallic)); let ed = lum((1.0 - m.metallic) * m.albedo); return clamp(es / max(es + ed, 1e-4), 0.1, 0.9); }
fn bsdf_eval(m: Material, n: v3, wo: v3, wi: v3) -> v3 {
    let nl = dot(n, wi); let nv = dot(n, wo); if (nl <= 0.0 || nv <= 0.0) { return v3(0.0); }
    let h = normalize(wo + wi); let a = alpha_of(m); let f = fresnel(max(dot(wo, h), 0.0), mix(v3(0.04), m.albedo, m.metallic));
    return (1.0 - m.metallic) * m.albedo / pi * (1.0 - f) + ggx_D(max(dot(n, h), 0.0), a) * smith_V(nv, nl, a) * f;
}
fn bsdf_pdf(m: Material, n: v3, wo: v3, wi: v3) -> f32 {
    let nl = dot(n, wi); let nv = dot(n, wo); if (nl <= 0.0 || nv <= 0.0) { return 0.0; }
    let h = normalize(wo + wi); let a = alpha_of(m); let ps = spec_prob(m);
    let vndf = smith_G1(nv, a) * ggx_D(max(dot(n, h), 0.0), a) / (4.0 * nv);
    return ps * vndf + (1.0 - ps) * nl / pi;
}
// spherical-cap VNDF (Dupuy & Benyoub 2023)
fn sample_vndf(wo_l: v3, a: f32, u: v2) -> v3 {
    let wi = normalize(v3(wo_l.xy * a, wo_l.z));
    let phi = 2.0 * pi * u.x; let z = (1.0 - u.y) * (1.0 + wi.z) - wi.z; let st = sqrt(clamp(1.0 - z * z, 0.0, 1.0));
    let h = v3(st * cos(phi), st * sin(phi), z) + wi;
    return normalize(v3(h.xy * a, max(h.z, 0.0)));
}
fn bsdf_sample(m: Material, n: v3, wo: v3, u: v3) -> v3 {
    let T = onb(n);
    if (u.z < spec_prob(m)) {
        let wo_l = transpose(T) * wo; let h = T * sample_vndf(wo_l, alpha_of(m), u.xy);
        return reflect(-wo, h);
    }
    let r = sqrt(u.x); let phi = 2.0 * pi * u.y;
    return T * v3(r * cos(phi), r * sin(phi), sqrt(max(1.0 - u.x, 0.0)));
}
fn schlick(cosv: f32, ior: f32) -> f32 { var r0 = (1.0 - ior) / (1.0 + ior); r0 *= r0; return r0 + (1.0 - r0) * pow(clamp(1.0 - cosv, 0.0, 1.0), 5.0); }
fn spectral_rgb(w: f32) -> v3 {
    var c = v3(1.0, 0.0, 0.0);
    if (w < 440.0) { c = v3(-(w - 440.0) / 60.0, 0.0, 1.0); } else if (w < 490.0) { c = v3(0.0, (w - 440.0) / 50.0, 1.0); }
    else if (w < 510.0) { c = v3(0.0, 1.0, -(w - 510.0) / 20.0); } else if (w < 580.0) { c = v3((w - 510.0) / 70.0, 1.0, 0.0); }
    else if (w < 645.0) { c = v3(1.0, -(w - 645.0) / 65.0, 0.0); }
    return c / v3(0.52, 0.49, 0.33);
}
// glass: returns new direction; tints throughput on refraction (dispersion)
fn glass_step(d: v3, h: Hit, u: f32, thr: ptr<function, v3>) -> v3 {
    let ior = 1.5 + params.dispersion * (540.0 - wavelength) / 100.0;
    let ct = min(dot(-d, h.n), 1.0); let eta = select(ior, 1.0 / ior, h.front);
    if (eta * sqrt(max(1.0 - ct * ct, 0.0)) > 1.0 || schlick(ct, ior) > u) { return reflect(d, h.n); }
    *thr *= spectral_rgb(wavelength);
    return refract(d, h.n, eta);
}

// ---------------------------------------------------------------- lights: power selection + spherical-cap sampling
// selection weight = power (independent of position, so no trig here)
fn light_w(i: u32) -> f32 { return lum(light_col(i)) * select(0.1225, 0.36, i == 0u); }
// selection by estimated contribution at x: power / d^2 * facing (cos widened by the cap, so a light is 0 only
// when it's entirely below the horizon). exact discrete pdf -> MIS stays unbiased
fn light_imp(i: u32, x: v3, n: v3) -> f32 {
    let L = get_light(i); let to = L.pos - x; let d2 = max(dot(to, to), L.r * L.r); let d = sqrt(d2);
    return light_w(i) * clamp(dot(n, to / d) + L.r / d, 0.0, 1.0) / d2;
}
fn imp_total(x: v3, n: v3) -> f32 { var s = 0.0; for (var i = 0u; i < n_lights(); i++) { s += light_imp(i, x, n); } return s; }
fn pick_light(u: f32, x: v3, n: v3, tot: f32) -> u32 {
    var acc = 0.0; let tgt = u * tot;
    for (var i = 0u; i < n_lights(); i++) { acc += light_imp(i, x, n); if (tgt < acc) { return i; } }
    return n_lights() - 1u;
}
fn cap_cos(x: v3, L: Light) -> f32 { let d2 = dot(L.pos - x, L.pos - x); return sqrt(max(1.0 - L.r * L.r / max(d2, 1e-8), 0.0)); }
// solid-angle pdf of NEE choosing light i from x
// solid-angle pdf of NEE choosing light i from (x, n)
fn light_pdf(x: v3, n: v3, i: u32) -> f32 {
    let tot = imp_total(x, n); if (tot <= 0.0) { return 0.0; }
    let L = get_light(i); return light_imp(i, x, n) / tot / (2.0 * pi * max(1.0 - cap_cos(x, L), 1e-7));
}
struct LS { y: v3, ny: v3, wi: v3, d: f32, pdf_w: f32 };
fn sample_cap(x: v3, L: Light, u: v2) -> LS {
    var s: LS; let to = L.pos - x; let d = length(to); let ax = to / max(d, 1e-8);
    let cmax = cap_cos(x, L); let ct = 1.0 - u.x * (1.0 - cmax); let st = sqrt(max(1.0 - ct * ct, 0.0)); let phi = 2.0 * pi * u.y;
    s.wi = onb(ax) * v3(st * cos(phi), st * sin(phi), ct);
    let b = dot(x - L.pos, s.wi); let c = dot(x - L.pos, x - L.pos) - L.r * L.r; let t = -b - sqrt(max(b * b - c, 0.0));
    s.d = max(t, 0.0); s.y = x + s.wi * s.d; s.ny = normalize(s.y - L.pos); s.pdf_w = 1.0 / (2.0 * pi * max(1.0 - cmax, 1e-7));
    return s;
}
fn power_h(a: f32, b: f32) -> f32 { let a2 = a * a; return a2 / max(a2 + b * b, 1e-20); }

// ---------------------------------------------------------------- camera (deterministic per pixel+frame, shared by all passes)
fn cam_basis(pos: v3, tgt: v3) -> m3 { let w = normalize(pos - tgt); let u = normalize(cross(v3(0.0, 1.0, 0.0), w)); return m3(u, cross(w, u), w); }
fn cam_pos() -> v3 { return v3(params.cam_x, params.cam_y, params.cam_z); }
fn cam_tgt() -> v3 { return v3(params.tgt_x, params.tgt_y, params.tgt_z); }
fn camera_ray(pix: vec2<u32>) -> Ray {
    let r = rng4(0u); let B = cam_basis(cam_pos(), cam_tgt());
    let vph = 2.0 * tan(params.fov * pi / 360.0); let vpw = vph * R.x / R.y; let f = max(params.focus_dist, 0.01);
    let uv = (v2(pix) + r.xy) / R;
    let fp = cam_pos() + f * (-B[2] + (uv.x - 0.5) * vpw * B[0] + (0.5 - uv.y) * vph * B[1]);
    let a = 2.0 * pi * r.z; let rl = sqrt(r.w) * params.aperture * 0.5;
    let o = cam_pos() + B[0] * (rl * cos(a)) + B[1] * (rl * sin(a));
    return Ray(o, normalize(fp - o));
}
// stable pixel-centre pinhole ray for the G-buffer: identical every frame when still, so history always validates
fn center_ray(pix: vec2<u32>) -> Ray {
    let B = cam_basis(cam_pos(), cam_tgt());
    let vph = 2.0 * tan(params.fov * pi / 360.0); let vpw = vph * R.x / R.y;
    let uv = (v2(pix) + 0.5) / R;
    return Ray(cam_pos(), normalize(-B[2] + (uv.x - 0.5) * vpw * B[0] + (0.5 - uv.y) * vph * B[1]));
}
// unjittered pinhole projection -> uv
fn project(P: v3, pos: v3, tgt: v3) -> v2 {
    let B = cam_basis(pos, tgt); let d = P - pos; let z = max(-dot(d, B[2]), 1e-5);
    let vph = 2.0 * tan(params.fov * pi / 360.0); let vpw = vph * R.x / R.y;
    return v2(0.5 + dot(d, B[0]) / z / vpw, 0.5 - dot(d, B[1]) / z / vph);
}

// primary walk through mirrors/glass (deterministic) to the first non-delta surface
// kind: 0 = sky, 1 = direct surface, 2 = seen through mirrors/glass, 3 = emitter (mat = light id)
struct Walk { kind: u32, depth: f32, p: v3, n: v3, mat: u32, wo: v3 };
fn primary_walk(ray0: Ray) -> Walk {
    var w: Walk; w.kind = 0u; w.depth = SKY_DEPTH; w.n = v3(0.0, 1.0, 0.0); w.mat = 0u; w.p = v3(0.0); w.wo = -ray0.d;
    var ray = ray0; var dist = 0.0;
    for (var k = 0u; k < 6u; k++) {
        let h = intersect(ray);
        if (!h.hit) { return w; }
        dist += h.t;
        if (h.light >= 0) { w.kind = 3u; w.mat = u32(h.light); w.depth = dist; w.n = h.n; return w; }
        let m = material(h.mat, h.p, h.n);
        if (is_delta(h.mat, m)) {
            var d = reflect(ray.d, h.n);
            if (is_glass(h.mat)) { let ct = min(dot(-ray.d, h.n), 1.0); let eta = select(1.5, 1.0 / 1.5, h.front); if (eta * sqrt(max(1.0 - ct * ct, 0.0)) <= 1.0) { d = refract(ray.d, h.n, eta); } }
            ray = Ray(h.p + d * 2e-3, d); continue;
        }
        w.kind = select(2u, 1u, k == 0u); w.depth = dist; w.p = h.p; w.n = h.n; w.mat = h.mat; w.wo = -ray.d; return w;
    }
    return w;
}
fn is_surface(kind: u32) -> bool { return kind == 1u || kind == 2u; }
fn albedo_of(w: Walk) -> v3 { if (!is_surface(w.kind)) { return v3(1.0); } return max(material(w.mat, w.p, w.n).albedo, v3(1e-3)); }

// ---------------------------------------------------------------- pass setup + texture helpers
fn setup(gid: vec3<u32>, salt: u32) -> bool {
    let dims = textureDimensions(out_tex); if (gid.x >= dims.x || gid.y >= dims.y) { return false; }
    R = v2(dims); px_seed = hash_u(gid.x ^ hash_u(gid.y + 0x9e3779b9u));
    seed = hash_u(px_seed ^ hash_u(time_data.frame * 0x85ebca6bu + salt));
    anim_time = select(time_data.time, 0.0, params.accumulate > 0u);
    wavelength = 400.0 + hash_f() * 300.0;
    return true;
}
fn clampi(p: i2) -> i2 { return clamp(p, i2(0), i2(R) - 1); }
fn ld0(p: i2) -> v4 { return textureLoad(tex0, clampi(p), 0); }
fn ld1(p: i2) -> v4 { return textureLoad(tex1, clampi(p), 0); }
fn ld2(p: i2) -> v4 { return textureLoad(tex2, clampi(p), 0); }
fn ld3(p: i2) -> v4 { return textureLoad(tex3, clampi(p), 0); }
fn ld4(p: i2) -> v4 { return textureLoad(tex4, clampi(p), 0); }
fn store(gid: vec3<u32>, c: v4) { textureStore(out_tex, vec2<i32>(gid.xy), c); }
fn safe(c: v3) -> v3 { return select(v3(0.0), clamp(c, v3(0.0), v3(6e4)), all(c == c)); }

fn oct_enc(n: v3) -> v2 { var p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z)); if (n.z < 0.0) { p = (1.0 - abs(p.yx)) * select(v2(-1.0), v2(1.0), p >= v2(0.0)); } return p; }
fn oct_dec(e: v2) -> v3 { var n = v3(e, 1.0 - abs(e.x) - abs(e.y)); if (n.z < 0.0) { n = v3((1.0 - abs(n.yx)) * select(v2(-1.0), v2(1.0), n.xy >= v2(0.0)), n.z); } return normalize(n); }
// G-buffer texel: (oct normal, depth, kind*64 + mat)
struct G { n: v3, z: f32, kind: u32, mat: u32 };
fn gdec(t: v4) -> G { let c = u32(round(t.w)); return G(oct_dec(t.xy), t.z, c / 64u, c % 64u); }

// ---------------------------------------------------------------- reservoirs: (light*32 + M, u, v, W); sample = point on the light sphere
struct Res { light: u32, M: f32, uv: v2, W: f32 };
fn res_dec(t: v4) -> Res { let c = u32(round(t.x)); return Res(c / 32u, f32(c % 32u), t.yz, t.w); }
fn res_enc(r: Res) -> v4 { return v4(f32(r.light * 32u + u32(clamp(r.M, 0.0, 31.0))), r.uv, r.W); }
fn sph_uv(n: v3) -> v2 { return v2(n.z * 0.5 + 0.5, atan2(n.y, n.x) / (2.0 * pi) + 0.5); }
fn uv_sph(uv: v2) -> v3 { let z = uv.x * 2.0 - 1.0; let a = (uv.y - 0.5) * 2.0 * pi; let r = sqrt(max(1.0 - z * z, 0.0)); return v3(r * cos(a), r * sin(a), z); }
// visibility-free target (area measure) of light point (light,uv) seen from x
fn res_target(x: v3, n: v3, m: Material, wo: v3, light: u32, uv: v2) -> f32 {
    if (light >= n_lights()) { return 0.0; }
    let L = get_light(light); let ny = uv_sph(uv); let y = L.pos + L.r * ny;
    let to = y - x; let d2 = max(dot(to, to), 1e-8); let wi = to * inverseSqrt(d2);
    let cx = dot(n, wi); let cy = dot(ny, -wi); if (cx <= 0.0 || cy <= 0.0) { return 0.0; }
    return lum(bsdf_eval(m, n, wo, wi) * L.le) * orb(L, x, wi) * cx * cy / d2;
}
// ReSTIR for motion / realtime; still + converging uses plain NEE+MIS (unbiased, uncorrelated -> clean convergence)
fn restir_active() -> bool { return params.restir == 1u && (params.accumulate == 0u || params.cam_moved == 1u); }
fn res_valid(r: Res) -> bool { return r.W > 0.0 && r.W < 1e6 && r.light < n_lights(); }

// ---------------------------------------------------------------- reprojection (SVGF-style 2x2 bilinear with per-tap validity)
struct Rep { p: array<i2, 4>, w: array<f32, 4>, ok: bool };
// cur G from tex_g (current gbuf), prev gbuf in ldp
fn reproject(gid: vec3<u32>, g: G, gp_tex: u32) -> Rep {
    var r: Rep; r.ok = false; for (var k = 0; k < 4; k++) { r.w[k] = 0.0; r.p[k] = i2(0); }
    let ray = center_ray(gid.xy); let P = ray.o + ray.d * g.z;
    // object motion: where this surface point was last frame
    var Pp = P;
    if (animating() && g.kind == 1u && g.mat == 15u) { Pp = P - (obj_center(anim_time) - obj_center(anim_time - time_data.delta)); }
    let moved = params.cam_moved == 1u || any(Pp != P);
    let pc = (v2(gid.xy) + 0.5);
    let pcam = v3(params.pcam_x, params.pcam_y, params.pcam_z); let ptgt = v3(params.ptgt_x, params.ptgt_y, params.ptgt_z);
    var q = pc;
    if (moved) { q = pc + (project(Pp, pcam, ptgt) - project(P, cam_pos(), cam_tgt())) * R; }
    let zexp = select(g.z, length(Pp - pcam), moved);
    let b = q - 0.5; let i0 = i2(floor(b)); let f = b - floor(b);
    let bw = array<f32, 4>((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
    let off = array<i2, 4>(i2(0, 0), i2(1, 0), i2(0, 1), i2(1, 1));
    var sw = 0.0;
    for (var k = 0; k < 4; k++) {
        let t = i0 + off[k];
        if (any(t < i2(0)) || any(t >= i2(R))) { continue; }
        var gpt: v4; if (gp_tex == 3u) { gpt = ld3(t); } else { gpt = ld4(t); }
        let gp = gdec(gpt);
        if (gp.kind != g.kind || gp.mat != g.mat) { continue; }
        if (g.kind != 0u && (abs(gp.z - zexp) > 0.08 * zexp || dot(gp.n, g.n) < 0.9)) { continue; }
        r.p[k] = t; r.w[k] = bw[k]; sw += bw[k];
    }
    if (sw > 1e-3) { for (var k = 0; k < 4; k++) { r.w[k] /= sw; } r.ok = true; }
    return r;
}
// converge: true progressive mean (count kept as hi/lo, exact to millions); realtime: short moving average
fn hist_max() -> f32 { return select(max(params.hist_realtime, 1.0), 1e7, params.accumulate > 0u); }
fn q16(v: v3) -> v3 { return v3(unpack2x16float(pack2x16float(v.xy)), unpack2x16float(pack2x16float(v2(v.z, 0.0))).x); }
fn q16s(x: f32) -> f32 { return unpack2x16float(pack2x16float(v2(x, 0.0))).x; }

// ================================================================ PASSES
// gprev [gbuf]: last frame's G-buffer, kept for disocclusion tests
@compute @workgroup_size(16, 16, 1)
fn gprev(@builtin(global_invocation_id) gid: vec3<u32>) { if (!setup(gid, 0u)) { return; } store(gid, ld0(i2(gid.xy))); }

// gbuf []
@compute @workgroup_size(16, 16, 1)
fn gbuf(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!setup(gid, 1u)) { return; }
    let w = primary_walk(center_ray(gid.xy));
    store(gid, v4(oct_enc(w.n), w.depth, f32(w.kind * 64u + w.mat)));
}

// galb []: demodulation albedo + kind
@compute @workgroup_size(16, 16, 1)
fn galb(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!setup(gid, 2u)) { return; }
    let w = primary_walk(center_ray(gid.xy));
    store(gid, v4(albedo_of(w), f32(w.kind)));
}

// ris []: initial light candidates (power-picked, cap-sampled), then one visibility test
@compute @workgroup_size(16, 16, 1)
fn ris(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!setup(gid, 3u)) { return; }
    var out = Res(0u, 0.0, v2(0.0), 0.0);
    if (!restir_active()) { store(gid, res_enc(out)); return; }
    let w = primary_walk(camera_ray(gid.xy));
    if (w.kind != 1u) { store(gid, res_enc(out)); return; }
    let m = material(w.mat, w.p, w.n); if (!restir_ok(m)) { store(gid, res_enc(out)); return; }
    let tot = imp_total(w.p, w.n); let M = max(params.ris_candidates, 1u);
    if (tot <= 0.0) { store(gid, res_enc(out)); return; }
    var wsum = 0.0; var ph_sel = 0.0; var y_uv = v2(0.0); var y_l = 0u;
    for (var k = 0u; k < M; k++) {
        let li = pick_light(hash_f(), w.p, w.n, tot); let L = get_light(li);
        let s = sample_cap(w.p, L, v2(hash_f(), hash_f()));
        let cy = dot(s.ny, -s.wi); if (cy <= 0.0) { continue; }
        let uv = sph_uv(s.ny);
        let ph = res_target(w.p, w.n, m, w.wo, li, uv);
        let pA = light_imp(li, w.p, w.n) / tot * s.pdf_w * cy / max(s.d * s.d, 1e-8);
        let wk = ph / max(pA, 1e-12);
        wsum += wk; if (hash_f() * wsum < wk) { ph_sel = ph; y_uv = uv; y_l = li; }
    }
    if (ph_sel > 0.0) {
        let L = get_light(y_l); let y = L.pos + L.r * uv_sph(y_uv); let to = y - w.p; let d = length(to);
        if (!occluded(w.p + w.n * 2e-3, to / d, d - L.r * 0.05)) { out = Res(y_l, 1.0, y_uv, wsum / (f32(M) * ph_sel)); }
        else { out = Res(y_l, 1.0, y_uv, 0.0); }
    }
    store(gid, res_enc(out));
}

// rtemp [ris, rspat(last frame), gbuf, gprev]: temporal reuse at the reprojected pixel
@compute @workgroup_size(16, 16, 1)
fn rtemp(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!setup(gid, 4u)) { return; }
    let p = i2(gid.xy); var rc = res_dec(ld0(p));
    let g = gdec(ld2(p));
    if (!restir_active() || g.kind != 1u || time_data.frame == 0u) { store(gid, res_enc(rc)); return; }
    let w = primary_walk(camera_ray(gid.xy)); let m = material(w.mat, w.p, w.n);
    let rep = reproject(gid, g, 3u);
    if (!rep.ok) { store(gid, res_enc(rc)); return; }
    // nearest valid tap of the bilinear footprint
    var bk = 0; for (var k = 1; k < 4; k++) { if (rep.w[k] > rep.w[bk]) { bk = k; } }
    let rp = res_dec(ld1(rep.p[bk]));
    if (!res_valid(rp)) { store(gid, res_enc(rc)); return; }
    let mc = select(0.0, rc.M, rc.W > 0.0 || rc.M > 0.0); let mp = min(rp.M, params.c_cap);
    let phc = res_target(w.p, w.n, m, w.wo, rc.light, rc.uv); let php = res_target(w.p, w.n, m, w.wo, rp.light, rp.uv);
    let wc = phc * rc.W * mc; let wp = php * rp.W * mp; let ws = wc + wp;
    var sel = rc; var phs = phc; if (hash_f() * ws < wp) { sel = rp; phs = php; }
    let Mt = mc + mp;
    sel.M = Mt; sel.W = select(0.0, ws / (Mt * phs), phs > 0.0 && Mt > 0.0);
    store(gid, res_enc(sel));
}

// rspat [rtemp, gbuf]: spatial reuse over K geometry-similar neighbours
@compute @workgroup_size(16, 16, 1)
fn rspat(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!setup(gid, 5u)) { return; }
    let p = i2(gid.xy); let r0 = res_dec(ld0(p)); let g = gdec(ld1(p));
    if (!restir_active() || g.kind != 1u || params.spatial_count == 0u) { store(gid, res_enc(r0)); return; }
    let w = primary_walk(camera_ray(gid.xy)); let m = material(w.mat, w.p, w.n);
    var ph0 = res_target(w.p, w.n, m, w.wo, r0.light, r0.uv);
    var ws = ph0 * r0.W * r0.M; var sel = r0; var phs = ph0; var Mt = r0.M;
    for (var k = 0u; k < min(params.spatial_count, 8u); k++) {
        let a = hash_f() * 2.0 * pi; let rr = params.spatial_radius * sqrt(hash_f());
        let q = p + i2(round(v2(cos(a), sin(a)) * rr));
        if (any(q < i2(0)) || any(q >= i2(R)) || all(q == p)) { continue; }
        let gq = gdec(ld1(q));
        if (gq.kind != 1u || dot(gq.n, g.n) < 0.9 || abs(gq.z - g.z) > 0.1 * g.z) { continue; }
        let rq = res_dec(ld0(q)); if (!res_valid(rq)) { continue; }
        let ph = res_target(w.p, w.n, m, w.wo, rq.light, rq.uv);
        let wq = ph * rq.W * rq.M; ws += wq; Mt += rq.M;
        if (hash_f() * ws < wq) { sel = rq; phs = ph; }
    }
    sel.M = min(Mt, 31.0); sel.W = select(0.0, ws / (Mt * phs), phs > 0.0 && Mt > 0.0);
    store(gid, res_enc(sel));
}

// trace [rspat, galb]: one path per pixel, output demodulated by galb's albedo
fn add_c(L: ptr<function, v3>, c: v3, b: u32) {
    var v = safe(c);
    if (b > 0u && params.firefly > 0.0) { let l = lum(v); if (l > params.firefly) { v *= params.firefly / l; } }
    *L += v;
}
@compute @workgroup_size(16, 16, 1)
fn trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!setup(gid, 6u)) { return; }
    let p = i2(gid.xy); let res = res_dec(ld0(p));
    var ray = camera_ray(gid.xy);
    var thr = v3(1.0); var L = v3(0.0); var prev_delta = true; var prev_pdf = 0.0; var prev_p = ray.o; var prev_n = v3(0.0, 1.0, 0.0); var skip_le = false;
    var nd = 0u;
    for (var b = 0u; b < params.max_bounces; b++) {
        let h = intersect(ray);
        if (!h.hit) { add_c(&L, thr * sky(ray.d), b); break; }
        if (h.light >= 0) {
            let Lh = get_light(u32(h.light)); let Le = Lh.le * orb(Lh, ray.o, ray.d);
            if (prev_delta) { add_c(&L, thr * Le, b); }
            else if (!skip_le) { add_c(&L, thr * Le * power_h(prev_pdf, light_pdf(prev_p, prev_n, u32(h.light))), b); }
            break;
        }
        var m = material(h.mat, h.p, h.n); let r1 = rng4(1u + 2u * b); let r2 = rng4(2u + 2u * b);
        // path regularization (Kaplanyan & Dachsbacher 2013): after a diffuse bounce mirrors turn slightly rough,
        // so NEE can reach lights through them instead of rare caustic fireflies
        if (nd > 0u && params.regularize > 0.0 && !is_glass(h.mat)) { m.roughness = max(m.roughness, params.regularize); }
        if (is_glass(h.mat)) { let d = glass_step(ray.d, h, r2.w, &thr); thr *= m.albedo; ray = Ray(h.p + d * 2e-3, d); prev_delta = true; skip_le = false; continue; }
        if (is_mirror(m)) { let d = reflect(ray.d, h.n); thr *= m.albedo; ray = Ray(h.p + h.n * 2e-3, d); prev_delta = true; skip_le = false; continue; }
        nd += 1u; if (nd > params.diffuse_bounces) { break; }
        let wo = -ray.d; let x = h.p + h.n * 2e-3;
        skip_le = false;
        if (b == 0u && restir_active() && restir_ok(m) && res_valid(res)) {
            // ReSTIR direct at the camera vertex (replaces NEE there)
            let Lr = get_light(res.light); let ny = uv_sph(res.uv); let y = Lr.pos + Lr.r * ny;
            let to = y - h.p; let d2 = dot(to, to); let d = sqrt(d2); let wi = to / d;
            let cx = dot(h.n, wi); let cy = dot(ny, -wi);
            if (cx > 0.0 && cy > 0.0 && !occluded(x, wi, d - Lr.r * 0.05)) { add_c(&L, thr * bsdf_eval(m, h.n, wo, wi) * Lr.le * orb(Lr, h.p, wi) * cx * cy / d2 * res.W, b); }
            skip_le = true;
        } else if (!(b == 0u && restir_active() && restir_ok(m))) {
            // NEE + MIS, light chosen by its contribution here
            let tot = imp_total(h.p, h.n);
            if (tot > 0.0) {
                let li = pick_light(r1.x, h.p, h.n, tot); let Ll = get_light(li); let s = sample_cap(h.p, Ll, r1.yz);
                let cx = dot(h.n, s.wi);
                if (cx > 0.0 && !occluded(x, s.wi, s.d - Ll.r * 0.05)) {
                    let pl = light_imp(li, h.p, h.n) / tot * s.pdf_w; let pb = bsdf_pdf(m, h.n, wo, s.wi);
                    add_c(&L, thr * bsdf_eval(m, h.n, wo, s.wi) * Ll.le * orb(Ll, h.p, s.wi) * cx / pl * power_h(pl, pb), b);
                }
            }
        } else { skip_le = true; }
        let wi = bsdf_sample(m, h.n, wo, r2.xyz); let cw = dot(h.n, wi); if (cw <= 0.0) { break; }
        let pdf = bsdf_pdf(m, h.n, wo, wi); if (pdf <= 1e-7) { break; }
        thr *= bsdf_eval(m, h.n, wo, wi) * cw / pdf; prev_pdf = pdf; prev_delta = false; prev_p = h.p; prev_n = h.n;
        ray = Ray(x, wi);
        if (b >= 2u) { let q = clamp(max(thr.r, max(thr.g, thr.b)), 0.05, 0.95); if (r2.w > q) { break; } thr /= q; }
    }
    store(gid, v4(safe(L), 1.0));
}

// hprev [accum], lprev [accum_lo]: last frame's history (hi/lo halves), so both updates read the same input
@compute @workgroup_size(16, 16, 1)
fn hprev(@builtin(global_invocation_id) gid: vec3<u32>) { if (!setup(gid, 7u)) { return; } store(gid, ld0(i2(gid.xy))); }
@compute @workgroup_size(16, 16, 1)
fn lprev(@builtin(global_invocation_id) gid: vec3<u32>) { if (!setup(gid, 8u)) { return; } store(gid, ld0(i2(gid.xy))); }

// accum/accum_lo [trace, hprev, lprev, gbuf, gprev]: reprojected running mean; hi + lo = ~fp32 precision
struct Acc { c: v3, n: f32 };
fn accum_update(gid: vec3<u32>) -> Acc {
    let p = i2(gid.xy); let cur = ld0(p).rgb; let g = gdec(ld3(p));
    var a = Acc(cur, 1.0);
    if (time_data.frame == 0u) { return a; }
    let rep = reproject(gid, g, 4u); if (!rep.ok) { return a; }
    var hist = v3(0.0); var n = 0.0;
    for (var k = 0; k < 4; k++) { if (rep.w[k] > 0.0) { let hi = ld1(rep.p[k]); let lo = ld2(rep.p[k]); hist += rep.w[k] * (hi.rgb + lo.rgb); n += rep.w[k] * (hi.w + lo.w); } }
    if (n < 0.5) { return a; }
    if (params.cam_moved == 1u) { n = min(n, max(params.hist_move, 1.0)); }
    if (params.cam_moved == 1u || animating()) {
        // variance clipping: history outside the current neighbourhood (swept edges, moving shadows/reflections) is rejected
        var m1 = v3(0.0); var m2 = v3(0.0);
        for (var y = -1; y <= 1; y++) { for (var x = -1; x <= 1; x++) { let s = ld0(p + i2(x, y)).rgb; m1 += s; m2 += s * s; } }
        m1 /= 9.0; let sd = sqrt(max(m2 / 9.0 - m1 * m1, v3(0.0)));
        hist = clamp(hist, m1 - params.clip_gamma * sd, m1 + params.clip_gamma * sd);
    }
    let nn = min(floor(n + 0.5) + 1.0, hist_max());
    a.c = mix(hist, cur, 1.0 / nn); a.n = nn; return a;
}
@compute @workgroup_size(16, 16, 1)
fn accum(@builtin(global_invocation_id) gid: vec3<u32>) { if (!setup(gid, 9u)) { return; } let a = accum_update(gid); store(gid, v4(a.c, a.n)); }
@compute @workgroup_size(16, 16, 1)
fn accum_lo(@builtin(global_invocation_id) gid: vec3<u32>) { if (!setup(gid, 10u)) { return; } let a = accum_update(gid); store(gid, v4(a.c - q16(a.c), a.n - q16s(a.n))); }

// moments [trace, moments, gbuf, gprev]: luminance 1st/2nd moments for SVGF variance
@compute @workgroup_size(16, 16, 1)
fn moments(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!setup(gid, 11u)) { return; }
    let p = i2(gid.xy); let l = lum(ld0(p).rgb); let g = gdec(ld2(p));
    var m = v3(l, l * l, 1.0);
    if (time_data.frame > 0u) {
        let rep = reproject(gid, g, 3u);
        if (rep.ok) {
            var h = v3(0.0); for (var k = 0; k < 4; k++) { if (rep.w[k] > 0.0) { h += rep.w[k] * ld1(rep.p[k]).xyz; } }
            if (h.z >= 0.5) {
                var n = h.z; if (params.cam_moved == 1u) { n = min(n, max(params.hist_move, 1.0)); }
                let nn = min(floor(n + 0.5) + 1.0, hist_max());
                m = v3(mix(h.xy, v2(l, l * l), 1.0 / nn), nn);
            }
        }
    }
    store(gid, v4(m, 0.0));
}

// svar [accum, accum_lo, moments, gbuf]: colour + variance of the mean (-> 0 as it converges, so the filter backs off)
@compute @workgroup_size(16, 16, 1)
fn svar(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!setup(gid, 12u)) { return; }
    let p = i2(gid.xy); let hi = ld0(p); let c = hi.rgb + ld1(p).rgb; let mo = ld2(p); let g = gdec(ld3(p));
    var v = max(mo.y - mo.x * mo.x, 0.0);
    if (mo.z < 4.0 && is_surface(g.kind)) {
        var s1 = 0.0; var s2 = 0.0; var sw = 0.0;
        for (var y = -2; y <= 2; y++) { for (var x = -2; x <= 2; x++) {
            let q = p + i2(x, y); let gq = gdec(ld3(q)); if (gq.kind != g.kind) { continue; }
            let wq = pow(max(dot(gq.n, g.n), 0.0), 16.0) * exp(-abs(gq.z - g.z) / (0.05 * g.z + 1e-3));
            let l = lum(ld0(q).rgb + ld1(q).rgb); s1 += wq * l; s2 += wq * l * l; sw += wq;
        } }
        s1 /= max(sw, 1e-6); s2 /= max(sw, 1e-6); v = max(v, max(s2 - s1 * s1, 0.0) * 4.0 / max(mo.z, 1.0));
    }
    store(gid, v4(c, v / max(hi.w + ld1(p).w, 1.0)));
}

// a1..a4 [prev, gbuf, galb]: edge-aware a-trous (B3 5x5), luminance stop scaled by filtered std-dev
fn atrous(gid: vec3<u32>, level: u32) -> v4 {
    let p = i2(gid.xy); let c = ld0(p);
    if (params.denoise != 1u || level >= params.atrous_iters) { return c; }
    let g = gdec(ld1(p)); if (!is_surface(g.kind)) { return c; }
    var vf = 0.0; var vw = 0.0;
    for (var y = -1; y <= 1; y++) { for (var x = -1; x <= 1; x++) { let k = select(0.25, 0.5, x == 0) * select(0.25, 0.5, y == 0); vf += k * ld0(p + i2(x, y)).w; vw += k; } }
    let sd = sqrt(max(vf / vw, 0.0)); let lc = lum(c.rgb); let step = i32(1u << level); let ac = ld2(p).rgb;
    let h = array<f32, 5>(1.0 / 16.0, 0.25, 0.375, 0.25, 1.0 / 16.0);
    var sum = v3(0.0); var var_s = 0.0; var sw = 0.0;
    for (var y = -2; y <= 2; y++) { for (var x = -2; x <= 2; x++) {
        let q = p + i2(x, y) * step;
        if (any(q < i2(0)) || any(q >= i2(R))) { continue; }
        let gq = gdec(ld1(q)); if (gq.kind != g.kind) { continue; }
        let s = ld0(q);
        let dpx = length(v2(f32(x), f32(y))) * f32(step);
        let wz = exp(-abs(gq.z - g.z) / (params.sigma_z * 0.004 * g.z * max(dpx, 1.0) + 1e-4));
        let wn = pow(max(dot(gq.n, g.n), 0.0), params.sigma_n);
        let wl = exp(-abs(lum(s.rgb) - lc) / (params.sigma_l * sd + 1e-4));
        let wa = exp(-length(ld2(q).rgb - ac) * 10.0);
        let wk = h[x + 2] * h[y + 2] * wz * wn * wl * wa;
        sum += wk * s.rgb; var_s += wk * wk * s.w; sw += wk;
    } }
    if (sw < 1e-6) { return c; }
    return v4(sum / sw, var_s / (sw * sw));
}
@compute @workgroup_size(16, 16, 1)
fn a1(@builtin(global_invocation_id) gid: vec3<u32>) { if (!setup(gid, 13u)) { return; } store(gid, atrous(gid, 0u)); }
@compute @workgroup_size(16, 16, 1)
fn a2(@builtin(global_invocation_id) gid: vec3<u32>) { if (!setup(gid, 14u)) { return; } store(gid, atrous(gid, 1u)); }
@compute @workgroup_size(16, 16, 1)
fn a3(@builtin(global_invocation_id) gid: vec3<u32>) { if (!setup(gid, 15u)) { return; } store(gid, atrous(gid, 2u)); }
@compute @workgroup_size(16, 16, 1)
fn a4(@builtin(global_invocation_id) gid: vec3<u32>) { if (!setup(gid, 16u)) { return; } store(gid, atrous(gid, 3u)); }

// (Salmi et al. 2024, non neural core)
@compute @workgroup_size(16, 16, 1)
fn flr(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!setup(gid, 19u)) { return; }
    let p = i2(gid.xy); let c = ld0(p);
    if (params.denoise != 2u) { store(gid, c); return; }
    let g = gdec(ld1(p)); if (!is_surface(g.kind)) { store(gid, c); return; }
    let ac = lum(ld2(p).rgb);
    var M: array<f32, 36>; var B: array<f32, 18>;
    for (var i = 0; i < 36; i++) { M[i] = 0.0; } for (var i = 0; i < 18; i++) { B[i] = 0.0; }
    for (var y = -5; y <= 5; y++) { for (var x = -5; x <= 5; x++) {
        let q = p + i2(x, y) * 2;
        if (any(q < i2(0)) || any(q >= i2(R))) { continue; }
        let gq = gdec(ld1(q)); if (gq.kind != g.kind) { continue; }
        let w = exp(-f32(x * x + y * y) / 18.0) * pow(max(dot(gq.n, g.n), 0.0), 8.0) * exp(-abs(gq.z - g.z) / (0.1 * g.z + 1e-3));
        if (w < 1e-4) { continue; }
        let f = array<f32, 6>(1.0, gq.n.x - g.n.x, gq.n.y - g.n.y, gq.n.z - g.n.z, (gq.z - g.z) / max(g.z, 1e-3), lum(ld2(q).rgb) - ac);
        let yq = ld0(q).rgb;
        for (var i = 0; i < 6; i++) {
            let wi = w * f[i];
            for (var j = 0; j < 6; j++) { M[i * 6 + j] += wi * f[j]; }
            B[i * 3] += wi * yq.x; B[i * 3 + 1] += wi * yq.y; B[i * 3 + 2] += wi * yq.z;
        }
    } }
    let lam = 1e-3 * M[0]; for (var i = 1; i < 6; i++) { M[i * 6 + i] += lam; } M[0] += 1e-6;
    // Cholesky M = L L^T, solve M w = e0, intercept = w . B
    var Lm: array<f32, 36>; for (var i = 0; i < 36; i++) { Lm[i] = 0.0; }
    for (var j = 0; j < 6; j++) {
        var s = M[j * 6 + j]; for (var k = 0; k < j; k++) { s -= Lm[j * 6 + k] * Lm[j * 6 + k]; }
        if (s <= 1e-12) { store(gid, c); return; }
        let d = sqrt(s); Lm[j * 6 + j] = d;
        for (var i = j + 1; i < 6; i++) { var t = M[i * 6 + j]; for (var k = 0; k < j; k++) { t -= Lm[i * 6 + k] * Lm[j * 6 + k]; } Lm[i * 6 + j] = t / d; }
    }
    var u: array<f32, 6>;
    for (var i = 0; i < 6; i++) { var t = select(0.0, 1.0, i == 0); for (var k = 0; k < i; k++) { t -= Lm[i * 6 + k] * u[k]; } u[i] = t / Lm[i * 6 + i]; }
    var wv: array<f32, 6>;
    for (var ii = 0; ii < 6; ii++) { let i = 5 - ii; var t = u[i]; for (var k = i + 1; k < 6; k++) { t -= Lm[k * 6 + i] * wv[k]; } wv[i] = t / Lm[i * 6 + i]; }
    var yh = v3(0.0); for (var i = 0; i < 6; i++) { yh += wv[i] * v3(B[i * 3], B[i * 3 + 1], B[i * 3 + 2]); }
    let rel = sqrt(max(c.w, 0.0)) / max(lum(c.rgb), 1e-3);
    store(gid, v4(mix(c.rgb, max(yh, v3(0.0)), smoothstep(0.002, 0.03, rel)), c.w));
}

// comp [a4, flr]: HDR radiance for bloom + display, from the selected denoiser
@compute @workgroup_size(16, 16, 1)
fn comp(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!setup(gid, 17u)) { return; }
    let p = i2(gid.xy); store(gid, v4(select(ld0(p).rgb, ld1(p).rgb, params.denoise == 2u), 1.0));
}

// ---------------------------------------------------------------- bloom: 13-tap down / tent up pyramid (Jimenez 2014)
fn ts0() -> v2 { return 1.0 / vec2<f32>(textureDimensions(tex0)); }
fn bf0(uv: v2) -> v3 { let h = 0.5 * ts0(); return textureSampleLevel(tex0, sam0, clamp(uv, h, 1.0 - h), 0.0).rgb; }
fn bf1(uv: v2) -> v3 { let h = 0.5 / vec2<f32>(textureDimensions(tex1)); return textureSampleLevel(tex1, sam1, clamp(uv, h, 1.0 - h), 0.0).rgb; }
fn prefilter(c: v3) -> v3 { let l = max(c.r, max(c.g, c.b)); let soft = clamp(l - 0.5, 0.0, 1.0); return c * max(soft * soft * 0.5, l - 1.0) / max(l, 1e-4); }
fn tap(uv: v2, t: v2, o: v2, pre: bool) -> v3 { let c = bf0(uv + o * t); return select(c, prefilter(c), pre); }
fn kw(c: v3) -> f32 { return 1.0 / (1.0 + lum(c)); }
fn down13(uv: v2, pre: bool) -> v3 {
    let t = ts0();
    let a = tap(uv, t, v2(-2.0, -2.0), pre); let b = tap(uv, t, v2(0.0, -2.0), pre); let c = tap(uv, t, v2(2.0, -2.0), pre);
    let d = tap(uv, t, v2(-2.0, 0.0), pre);  let e = tap(uv, t, v2(0.0, 0.0), pre);  let f = tap(uv, t, v2(2.0, 0.0), pre);
    let g = tap(uv, t, v2(-2.0, 2.0), pre);  let h = tap(uv, t, v2(0.0, 2.0), pre);  let i = tap(uv, t, v2(2.0, 2.0), pre);
    let j = tap(uv, t, v2(-1.0, -1.0), pre); let k = tap(uv, t, v2(1.0, -1.0), pre);
    let l = tap(uv, t, v2(-1.0, 1.0), pre);  let m = tap(uv, t, v2(1.0, 1.0), pre);
    let g0 = (j + k + l + m) * 0.25; let g1 = (a + b + d + e) * 0.25; let g2 = (b + c + e + f) * 0.25;
    let g3 = (d + e + g + h) * 0.25; let g4 = (e + f + h + i) * 0.25;
    if (!pre) { return g0 * 0.5 + (g1 + g2 + g3 + g4) * 0.125; }
    let w0 = kw(g0) * 0.5; let w1 = kw(g1) * 0.125; let w2 = kw(g2) * 0.125; let w3 = kw(g3) * 0.125; let w4 = kw(g4) * 0.125;
    return (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4);
}
fn up_tent(uv: v2) -> v3 {
    let o = ts0(); var s = bf0(uv) * 4.0;
    s += (bf0(uv + v2(o.x, 0.0)) + bf0(uv - v2(o.x, 0.0)) + bf0(uv + v2(0.0, o.y)) + bf0(uv - v2(0.0, o.y))) * 2.0;
    s += bf0(uv + o) + bf0(uv - o) + bf0(uv + v2(o.x, -o.y)) + bf0(uv + v2(-o.x, o.y));
    return s / 16.0;
}
fn buv(gid: vec3<u32>) -> v2 { return (v2(gid.xy) + 0.5) / vec2<f32>(textureDimensions(out_tex)); }
fn inb(gid: vec3<u32>) -> bool { let d = textureDimensions(out_tex); return gid.x < d.x && gid.y < d.y; }
@compute @workgroup_size(16, 16, 1)
fn bloom_pre(@builtin(global_invocation_id) gid: vec3<u32>) { if (!inb(gid)) { return; } store(gid, v4(down13(buv(gid), true), 1.0)); }
@compute @workgroup_size(16, 16, 1)
fn bd2(@builtin(global_invocation_id) gid: vec3<u32>) { if (!inb(gid)) { return; } store(gid, v4(down13(buv(gid), false), 1.0)); }
@compute @workgroup_size(16, 16, 1)
fn bd3(@builtin(global_invocation_id) gid: vec3<u32>) { if (!inb(gid)) { return; } store(gid, v4(down13(buv(gid), false), 1.0)); }
@compute @workgroup_size(16, 16, 1)
fn bd4(@builtin(global_invocation_id) gid: vec3<u32>) { if (!inb(gid)) { return; } store(gid, v4(down13(buv(gid), false), 1.0)); }
@compute @workgroup_size(16, 16, 1)
fn bd5(@builtin(global_invocation_id) gid: vec3<u32>) { if (!inb(gid)) { return; } store(gid, v4(down13(buv(gid), false), 1.0)); }
@compute @workgroup_size(16, 16, 1)
fn bu4(@builtin(global_invocation_id) gid: vec3<u32>) { if (!inb(gid)) { return; } let uv = buv(gid); store(gid, v4(up_tent(uv) + bf1(uv), 1.0)); }
@compute @workgroup_size(16, 16, 1)
fn bu3(@builtin(global_invocation_id) gid: vec3<u32>) { if (!inb(gid)) { return; } let uv = buv(gid); store(gid, v4(up_tent(uv) + bf1(uv), 1.0)); }
@compute @workgroup_size(16, 16, 1)
fn bu2(@builtin(global_invocation_id) gid: vec3<u32>) { if (!inb(gid)) { return; } let uv = buv(gid); store(gid, v4(up_tent(uv) + bf1(uv), 1.0)); }
@compute @workgroup_size(16, 16, 1)
fn bu1(@builtin(global_invocation_id) gid: vec3<u32>) { if (!inb(gid)) { return; } let uv = buv(gid); store(gid, v4((up_tent(uv) + bf1(uv)) * 0.2, 1.0)); }

fn aces(x: f32) -> f32 { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }
fn tonemap(c: v3) -> v3 {
    let pk = max(max(c.r, c.g), max(c.b, 1e-5));
    return mix(c / pk, v3(1.0), 0.9 * smoothstep(0.5, 12.0, pk)) * aces(pk);
}
// main_image [comp, bu1, accum]; energy-conserving bloom mix
@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!setup(gid, 18u)) { return; }
    // debug: per-pixel history length, log scale, blue (1) -> white (65536)
    if (params.debug_view == 1u) {
        let t = clamp(log2(max(ld2(i2(gid.xy)).w, 1.0)) / 16.0, 0.0, 1.0);
        store(gid, v4(mix(v3(0.05, 0.15, 0.9), v3(1.0), t), 1.0)); return;
    }
    let hdr = mix(ld0(i2(gid.xy)).rgb, bf1(buv(gid)), params.bloom);
    let c = pow(tonemap(hdr * params.exposure), v3(1.0 / max(params.gamma, 0.1)));

    let dn = f32(hash_u(px_seed ^ (time_data.frame * 0x9e3779b9u)) >> 8u) / 16777216.0 - 0.5;
    store(gid, v4(c + dn / 255.0, 1.0));
}

// Pinscreen: an image as ray-traced pins placed by weighted Voronoi stippling, or as LEGO-style bricks — Enes Altun 2026, CC BY-NC-SA 3.0
// Stippling: Lloyd relaxation toward darkness-weighted centroids (Secord 2002); cells by jump flooding, centroids by atomic sums.

struct TimeUniform { time: f32, delta: f32, frame: u32, _padding: u32 };
@group(0) @binding(0) var<uniform> u_time: TimeUniform;

struct Params {
    n: u32, colored: u32, invert: u32, relax: f32,
    gamma: f32, pin_h: f32, yaw: f32, elev: f32,
    zoom: f32, light_az: f32, light_el: f32, rough: f32,
    pin_r: f32, expo: f32, gam: f32, bloom: f32,
    orb_spd: f32, light_spd: f32, wave_amp: f32, wave_spd: f32,
    lan_n: f32, lan_pow: f32, lan_spd: f32, key: f32,
    focus: f32, aperture: f32, bokeh_hi: f32, blades: f32,
    refl: f32, coat: f32, reseed: u32, glow: f32,
    metal: f32, aniso: f32, flake: f32, dome: f32,
    shape: u32, studs: f32, plates: f32, palette: u32,
    twinkle: f32, holo: f32, fl_size: f32, fl_dens: f32,
    fl_all: u32, _f0: f32, _f1: f32, _f2: f32,
};
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> p: Params;
@group(1) @binding(2) var input_texture: texture_2d<f32>;
@group(1) @binding(3) var input_sampler: sampler;

// per point: [2i] position, darkness, cell count; [2i+1] mean colour, eased height
@group(3) @binding(0) var<storage, read_write> pts: array<vec4<f32>>;
// nearest point per grid cell (jump flooding ping-pong); gb holds brick cells in bricks mode
@group(3) @binding(1) var<storage, read_write> ga: array<u32>;
@group(3) @binding(2) var<storage, read_write> gb: array<u32>;
// per point sums: x, y, weight, count, r, g, b
@group(3) @binding(3) var<storage, read_write> acc: array<atomic<u32>>;
@group(3) @binding(4) var<storage, read_write> hdr: array<vec2<u32>>;
@group(3) @binding(5) var<storage, read_write> b1: array<vec2<u32>>;
@group(3) @binding(6) var<storage, read_write> b2: array<vec2<u32>>;
// half-res depth of field layers: prefilter, gather, tent
@group(3) @binding(7) var<storage, read_write> dof: array<vec2<u32>>;
const MAXPIX: u32 = 3840u * 2400u;

alias v2 = vec2<f32>; alias v3 = vec3<f32>; alias v4 = vec4<f32>;
const G: u32 = 1024u;
const NONE: u32 = 0xffffffffu;
const MAXN: u32 = 131072u;

fn hu(a0: u32) -> u32 { var a=a0; a^=a>>16u; a*=0x7feb352du; a^=a>>15u; a*=0x846ca68bu; a^=a>>16u; return a; }
fn hf(s: ptr<function, u32>) -> f32 { *s = hu(*s); return f32(*s >> 8u) / 16777216.0; }

// the image in the grid: xy offset, zw size
fn img_rect() -> v4 {
    let d = textureDimensions(input_texture);
    var a = 1.0;
    if (d.x > 1u && d.y > 1u) { a = f32(d.x) / f32(d.y); }
    let g = f32(G);
    let sz = select(v2(g * a, g), v2(g, g / a), a >= 1.0);
    return v4((v2(g) - sz) * 0.5, sz);
}

fn image_uv(gp: v2) -> v2 { let r = img_rect(); return (gp - r.xy) / r.zw; }

fn image_rgb(uv: v2) -> v3 {
    let d = textureDimensions(input_texture);
    if (d.x > 1u && d.y > 1u) { return textureSampleLevel(input_texture, input_sampler, uv, 0.0).rgb; }
    let q = uv - 0.5;
    let r = length(q);
    let v = 0.55 + 0.35 * cos(r * 28.0 - 1.0) * exp(-r * 2.5) - 0.4 * exp(-dot(q - v2(0.12, -0.1), q - v2(0.12, -0.1)) * 40.0);
    return v3(clamp(v, 0.0, 1.0));
}

// stipple density, -1 outside the image
fn weight(gp: v2) -> f32 {
    let uv = image_uv(gp);
    if (any(uv < v2(0.0)) || any(uv > v2(1.0))) { return -1.0; }
    let l = dot(image_rgb(uv), v3(0.299, 0.587, 0.114));
    let tone = select(1.0 - l, l, p.invert != 0u);
    return 0.02 + 0.98 * pow(clamp(tone, 0.0, 1.0), p.gamma);
}

// rejection-sampled position by density
fn sample_pos(s: ptr<function, u32>) -> v2 {
    let r = img_rect();
    var best = r.xy + r.zw * 0.5;
    for (var k = 0; k < 24; k++) {
        let c = r.xy + v2(hf(s), hf(s)) * r.zw;
        best = c;
        if (hf(s) < weight(c)) { break; }
    }
    return best;
}

@compute @workgroup_size(256, 1, 1)
fn init_points(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= MAXN || p.reseed == 0u) { return; }
    var s = hu(i * 747796405u + u_time.frame * 2891336453u + 1u);
    pts[2u * i] = v4(sample_pos(&s), 0.0, 1.0);
    pts[2u * i + 1u] = v4(0.0);
}

@compute @workgroup_size(256, 1, 1)
fn clear_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < G * G && p.shape == 0u) { ga[id.x] = NONE; }
}

@compute @workgroup_size(256, 1, 1)
fn seed(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= p.n || p.shape != 0u) { return; }
    let c = vec2<u32>(clamp(pts[2u * i].xy, v2(0.0), v2(f32(G) - 1.0)));
    ga[c.y * G + c.x] = i;
}

fn jfa(id: u32, step: i32, from_a: bool) {
    if (id >= G * G || p.shape != 0u) { return; }
    let c = vec2<i32>(i32(id % G), i32(id / G));
    let cp = v2(c) + 0.5;
    var best = NONE;
    var bd = 1e30;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let q = c + vec2<i32>(x, y) * step;
            if (any(q < vec2<i32>(0)) || any(q >= vec2<i32>(i32(G)))) { continue; }
            let k = u32(q.y) * G + u32(q.x);
            let o = select(gb[k], ga[k], from_a);
            if (o == NONE) { continue; }
            let d = dot(pts[2u * o].xy - cp, pts[2u * o].xy - cp);
            if (d < bd) { bd = d; best = o; }
        }
    }
    if (from_a) { gb[id] = best; } else { ga[id] = best; }
}

// steps 512 .. 1, ending in ga
@compute @workgroup_size(256, 1, 1) fn jfa_0(@builtin(global_invocation_id) id: vec3<u32>) { jfa(id.x, 512, true); }
@compute @workgroup_size(256, 1, 1) fn jfa_1(@builtin(global_invocation_id) id: vec3<u32>) { jfa(id.x, 256, false); }
@compute @workgroup_size(256, 1, 1) fn jfa_2(@builtin(global_invocation_id) id: vec3<u32>) { jfa(id.x, 128, true); }
@compute @workgroup_size(256, 1, 1) fn jfa_3(@builtin(global_invocation_id) id: vec3<u32>) { jfa(id.x, 64, false); }
@compute @workgroup_size(256, 1, 1) fn jfa_4(@builtin(global_invocation_id) id: vec3<u32>) { jfa(id.x, 32, true); }
@compute @workgroup_size(256, 1, 1) fn jfa_5(@builtin(global_invocation_id) id: vec3<u32>) { jfa(id.x, 16, false); }
@compute @workgroup_size(256, 1, 1) fn jfa_6(@builtin(global_invocation_id) id: vec3<u32>) { jfa(id.x, 8, true); }
@compute @workgroup_size(256, 1, 1) fn jfa_7(@builtin(global_invocation_id) id: vec3<u32>) { jfa(id.x, 4, false); }
@compute @workgroup_size(256, 1, 1) fn jfa_8(@builtin(global_invocation_id) id: vec3<u32>) { jfa(id.x, 2, true); }
@compute @workgroup_size(256, 1, 1) fn jfa_9(@builtin(global_invocation_id) id: vec3<u32>) { jfa(id.x, 1, false); }

@compute @workgroup_size(256, 1, 1)
fn accumulate(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= G * G || p.shape != 0u) { return; }
    let o = ga[id.x];
    if (o == NONE) { return; }
    let gp = v2(f32(id.x % G), f32(id.x / G)) + 0.5;
    let w = weight(gp);
    if (w < 0.0) { return; }
    let wi = u32(w * 255.0 + 0.5);
    let col = image_rgb(image_uv(gp));
    let b = o * 7u;
    atomicAdd(&acc[b], wi * (id.x % G));
    atomicAdd(&acc[b + 1u], wi * (id.x / G));
    atomicAdd(&acc[b + 2u], wi);
    atomicAdd(&acc[b + 3u], 1u);
    atomicAdd(&acc[b + 4u], u32(col.r * 255.0));
    atomicAdd(&acc[b + 5u], u32(col.g * 255.0));
    atomicAdd(&acc[b + 6u], u32(col.b * 255.0));
}

// Lloyd step; a point that owns no cell respawns
@compute @workgroup_size(256, 1, 1)
fn relax(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= p.n || p.shape != 0u) { return; }
    let b = i * 7u;
    let sx = f32(atomicLoad(&acc[b]));
    let sy = f32(atomicLoad(&acc[b + 1u]));
    let sw = f32(atomicLoad(&acc[b + 2u]));
    let cnt = f32(atomicLoad(&acc[b + 3u]));
    let rgb = v3(f32(atomicLoad(&acc[b + 4u])), f32(atomicLoad(&acc[b + 5u])), f32(atomicLoad(&acc[b + 6u])));
    for (var k = 0u; k < 7u; k++) { atomicStore(&acc[b + k], 0u); }

    var pos = pts[2u * i].xy;
    if (cnt < 0.5 || sw <= 0.0) {
        var s = hu(i * 1664525u + u_time.frame * 1013904223u);
        pos = sample_pos(&s);
        pts[2u * i] = v4(pos, 0.0, 0.0);
        pts[2u * i + 1u] = v4(0.0);
        return;
    }
    let c = v2(sx, sy) / sw + 0.5;
    pos = mix(pos, c, p.relax);
    let tone = sw / (255.0 * cnt);
    let ph = pts[2u * i + 1u].w;
    pts[2u * i] = v4(pos, tone, cnt);
    pts[2u * i + 1u] = v4(rgb / (255.0 * cnt), ph + (tone - ph) * 0.12);
}

fn owner(gp: v2) -> u32 {
    let c = vec2<i32>(clamp(gp, v2(0.0), v2(f32(G) - 1.0)));
    return ga[u32(c.y) * G + u32(c.x)];
}

struct Hit { t: f32, n: v3, pin: u32, side: bool };

fn pin_r(o: u32) -> f32 { return p.pin_r * 0.62 * sqrt(max(pts[2u * o].w, 1.0) / 3.14159); }
fn pin_h(o: u32) -> f32 {
    let r = img_rect();
    let w = 1.0 + p.wave_amp * sin(length(pts[2u * o].xy - (r.xy + r.zw * 0.5)) * 0.035 - u_time.time * p.wave_spd);
    return p.pin_h * (0.06 + 0.94 * clamp(pts[2u * o + 1u].w, 0.0, 1.0)) * max(w, 0.0);
}

// pin: cylinder up to h, capped by a sphere cap
fn hit_pin(ro: v3, rd: v3, o: u32, best: ptr<function, Hit>) {
    let c = pts[2u * o].xy;
    let r = pin_r(o);
    let h = pin_h(o);
    let oc = ro.xy - c;
    let a = dot(rd.xy, rd.xy);
    let b = dot(oc, rd.xy);
    let k = dot(oc, oc) - r * r;
    let disc = b * b - a * k;
    if (a < 1e-8 || disc < 0.0) { return; }
    let t1 = (-b - sqrt(disc)) / a;
    let z1 = ro.z + rd.z * t1;
    if (t1 > 0.0 && z1 >= 0.0 && z1 <= h) {
        if (t1 < (*best).t) { *best = Hit(t1, v3((ro.xy + rd.xy * t1 - c) / r, 0.0), o, true); }
        return;
    }
    let dh = clamp(p.dome, 0.05, 1.0) * r;
    let rs = (r * r + dh * dh) / (2.0 * dh);
    let sc = v3(c, h + dh - rs);
    let so = ro - sc;
    let sb = dot(so, rd);
    let sk = dot(so, so) - rs * rs;
    let sd = sb * sb - sk;
    if (sd < 0.0) { return; }
    let ts = -sb - sqrt(sd);
    let hp = ro + rd * ts;
    if (ts > 0.0 && hp.z >= h && ts < (*best).t) { *best = Hit(ts, normalize(hp - sc), o, false); }
}

fn trace_pins(ro: v3, rd: v3, top: f32, first_only: bool, skip: u32) -> Hit {
    var best = Hit(1e30, v3(0.0, 0.0, 1.0), NONE, false);
    if (abs(rd.z) < 1e-5) { return best; }
    var t0 = (top - ro.z) / rd.z;
    var t1 = -ro.z / rd.z;
    if (t0 > t1) { let tt = t0; t0 = t1; t1 = tt; }
    t0 = max(t0, 0.0);
    if (t1 <= t0) { return best; }
    let lxy = max(length(rd.xy), 1e-4);
    let steps = min(u32(ceil((t1 - t0) * lxy / 0.7)) + 2u, 400u);
    let dt = (t1 - t0) / f32(steps - 1u);
    var last = NONE;
    for (var s = 0u; s < steps; s++) {
        let t = t0 + dt * f32(s);
        if (t > best.t + 4.0 * dt) { break; }
        let q = ro.xy + rd.xy * t;
        if (any(q < v2(0.0)) || any(q >= v2(f32(G)))) { continue; }
        let o = owner(q);
        if (o == NONE || o == last || o == skip) { continue; }
        last = o;
        hit_pin(ro, rd, o, &best);
        if (first_only && best.pin != NONE) { break; }
    }
    return best;
}

fn studio(d: v3, l: v3, rough: f32) -> v3 {
    var c = mix(v3(0.02, 0.02, 0.025), v3(0.25, 0.28, 0.34), smoothstep(-0.2, 0.9, d.z));
    let sharp = mix(220.0, 6.0, rough);
    let e = (sharp + 1.0) / 221.0;
    c += v3(1.0, 0.97, 0.92) * 4.0 * e * pow(max(dot(d, l), 0.0), sharp);
    let f = normalize(v3(-l.xy, 0.35));
    c += v3(1.0, 0.7, 0.45) * 0.8 * e * pow(max(dot(d, f), 0.0), sharp * 0.5);
    return c;
}

struct Cam { ro: v3, rd: v3, f: v3 };
fn camera(gid: vec2<u32>, D: v2) -> Cam {
    let r = img_rect();
    let aim = v3(r.xy + r.zw * 0.5, 0.0);
    let el = radians(clamp(p.elev, 5.0, 85.0));
    let yw = radians(p.yaw + u_time.time * p.orb_spd);
    let dist = 1500.0 / max(p.zoom, 0.1);
    let ro = aim + dist * v3(sin(yw) * cos(el), cos(yw) * cos(el), sin(el));
    let f = normalize(aim - ro);
    let rt = normalize(cross(v3(0.0, 0.0, 1.0), f));
    let up = cross(f, rt);
    let sp = (v2(gid) + 0.5 - 0.5 * D) / D.y * 2.0 * 0.32;
    return Cam(ro, normalize(f + sp.x * rt - sp.y * up), f);
}
fn key_light() -> v3 {
    let laz = radians(p.light_az + u_time.time * p.light_spd);
    let el = radians(p.light_el);
    return normalize(v3(cos(laz) * cos(el), sin(laz) * cos(el), sin(el)));
}

// bricks in LEGO proportions of the stud pitch: plate 0.4, stud radius 0.3, stud height 0.2125 at the default
const PLATE: f32 = 0.4;
const STUD_R: f32 = 0.3;
const GAP: f32 = 0.012;
fn stud_h() -> f32 { return clamp(p.dome, 0.05, 1.0) * 0.47; }
struct BGrid { org: v2, cs: f32, n: vec2<i32> };
fn bgrid() -> BGrid {
    let r = img_rect();
    let cs = max(r.z, r.w) / clamp(p.studs, 8.0, 256.0);
    return BGrid(r.xy, cs, vec2<i32>(ceil(r.zw / cs - 0.001)));
}
// cell: sRGB bytes, plates in the top byte
fn cell_data(c: vec2<i32>, g: BGrid) -> u32 { return gb[u32(c.y * g.n.x + c.x)]; }
fn cell_rgb(c: vec2<i32>, g: BGrid) -> v3 {
    let d = cell_data(c, g);
    return pow(v3(f32(d & 255u), f32((d >> 8u) & 255u), f32((d >> 16u) & 255u)) / 255.0, v3(2.2));
}
fn cell_h(c: vec2<i32>, g: BGrid) -> f32 {
    let r = img_rect();
    let ctr = g.org + (v2(c) + 0.5) * g.cs;
    let w = 1.0 + p.wave_amp * sin(length(ctr - (r.xy + r.zw * 0.5)) * 0.035 - u_time.time * p.wave_spd);
    return max(round(f32(cell_data(c, g) >> 24u) * w), 1.0) * PLATE * g.cs;
}
fn scene_top() -> f32 {
    if (p.shape == 1u) {
        let g = bgrid();
        return (clamp(p.plates, 1.0, 24.0) * PLATE * (1.0 + p.wave_amp) + stud_h()) * g.cs + 1.0;
    }
    return p.pin_h * 1.5 * (1.0 + p.wave_amp) + 14.0 * p.pin_r;
}

const NPAL: u32 = 36u;
const LEGO = array<v3, 36>(
    v3(0.949, 0.953, 0.949),
    v3(0.627, 0.647, 0.663),
    v3(0.424, 0.431, 0.408),
    v3(0.020, 0.075, 0.114),
    v3(0.788, 0.102, 0.035),
    v3(0.447, 0.055, 0.059),
    v3(0.996, 0.541, 0.094),
    v3(0.949, 0.804, 0.216),
    v3(1.000, 0.941, 0.227),
    v3(0.894, 0.804, 0.620),
    v3(0.584, 0.541, 0.451),
    v3(0.345, 0.165, 0.071),
    v3(0.208, 0.129, 0.000),
    v3(0.667, 0.490, 0.333),
    v3(0.816, 0.569, 0.408),
    v3(0.965, 0.843, 0.702),
    v3(0.137, 0.471, 0.255),
    v3(0.294, 0.624, 0.290),
    v3(0.733, 0.914, 0.043),
    v3(0.094, 0.275, 0.196),
    v3(0.608, 0.604, 0.353),
    v3(0.627, 0.737, 0.675),
    v3(0.000, 0.333, 0.749),
    v3(0.039, 0.204, 0.388),
    v3(0.353, 0.576, 0.859),
    v3(0.027, 0.545, 0.788),
    v3(0.212, 0.682, 0.749),
    v3(0.624, 0.765, 0.914),
    v3(0.376, 0.455, 0.631),
    v3(0.675, 0.471, 0.729),
    v3(0.573, 0.224, 0.471),
    v3(0.784, 0.439, 0.627),
    v3(0.894, 0.678, 0.784),
    v3(1.000, 0.412, 0.561),
    v3(0.663, 0.333, 0.000),
    v3(0.973, 0.733, 0.239)
);
// nearest LEGO colour (redmean distance)
fn lego(c: v3) -> v3 {
    var pal = LEGO;
    var best = pal[0];
    var bd = 1e9;
    for (var i = 0u; i < NPAL; i++) {
        let q = pal[i];
        let rm = 0.5 * (c.r + q.r);
        let d = c - q;
        let e = (2.0 + rm) * d.r * d.r + 4.0 * d.g * d.g + (3.0 - rm) * d.b * d.b;
        if (e < bd) { bd = e; best = q; }
    }
    return best;
}

@compute @workgroup_size(256, 1, 1)
fn brick_cells(@builtin(global_invocation_id) id: vec3<u32>) {
    if (p.shape != 1u) { return; }
    let g = bgrid();
    let i = id.x;
    if (i >= u32(g.n.x * g.n.y)) { return; }
    let c = vec2<i32>(i32(i % u32(g.n.x)), i32(i / u32(g.n.x)));
    var sum = v3(0.0);
    for (var y = 0; y < 6; y++) {
        for (var x = 0; x < 6; x++) {
            let gp = g.org + (v2(c) + (v2(f32(x), f32(y)) + 0.5) / 6.0) * g.cs;
            sum += image_rgb(clamp(image_uv(gp), v2(0.0), v2(1.0)));
        }
    }
    let avg = sum / 36.0;
    let l = dot(avg, v3(0.299, 0.587, 0.114));
    let tone = select(1.0 - l, l, p.invert != 0u);
    let k = 1.0 + round(pow(clamp(tone, 0.0, 1.0), p.gamma) * (clamp(p.plates, 1.0, 24.0) - 1.0));
    var sc = pow(avg, v3(1.0 / 2.2));
    if (p.palette >= 1u) {
        if (p.palette == 2u) {
            var bm = array<f32, 16>(0.0, 8.0, 2.0, 10.0, 12.0, 4.0, 14.0, 6.0, 3.0, 11.0, 1.0, 9.0, 15.0, 7.0, 13.0, 5.0);
            sc += ((bm[(c.y & 3) * 4 + (c.x & 3)] + 0.5) / 16.0 - 0.5) * 0.16;
        }
        sc = lego(clamp(sc, v3(0.0), v3(1.0)));
    }
    let q = vec3<u32>(clamp(sc, v3(0.0), v3(1.0)) * 255.0 + 0.5);
    gb[i] = q.x | (q.y << 8u) | (q.z << 16u) | (u32(k) << 24u);
}

// (t, normal), t < 0 on a miss
fn box_hit(ro: v3, rd: v3, lo: v3, hi: v3) -> v4 {
    let inv = 1.0 / select(rd, v3(1e-8), abs(rd) < v3(1e-8));
    let t1 = (lo - ro) * inv;
    let t2 = (hi - ro) * inv;
    let tn = min(t1, t2);
    let tf = max(t1, t2);
    let t0 = max(max(tn.x, tn.y), tn.z);
    let t3 = min(min(tf.x, tf.y), tf.z);
    if (t3 < t0 || t0 <= 0.0) { return v4(-1.0); }
    var n = v3(0.0, 0.0, -sign(rd.z));
    if (tn.x >= tn.y && tn.x >= tn.z) { n = v3(-sign(rd.x), 0.0, 0.0); } else if (tn.y >= tn.z) { n = v3(0.0, -sign(rd.y), 0.0); }
    return v4(t0, n);
}

fn hit_cell(ro: v3, rd: v3, c: vec2<i32>, g: BGrid, best: ptr<function, Hit>) {
    let lo = g.org + (v2(c) + GAP) * g.cs;
    let hi = g.org + (v2(c) + 1.0 - GAP) * g.cs;
    let hh = cell_h(c, g);
    let idx = u32(c.y * g.n.x + c.x);
    let b = box_hit(ro, rd, v3(lo, 0.0), v3(hi, hh));
    if (b.x > 0.0 && b.x < (*best).t) { *best = Hit(b.x, b.yzw, idx, b.w < 0.5); }
    let ctr = g.org + (v2(c) + 0.5) * g.cs;
    let sr = STUD_R * g.cs;
    let st = hh + stud_h() * g.cs;
    let oc = ro.xy - ctr;
    let a = dot(rd.xy, rd.xy);
    let bb = dot(oc, rd.xy);
    let disc = bb * bb - a * (dot(oc, oc) - sr * sr);
    if (a > 1e-8 && disc >= 0.0) {
        let ts = (-bb - sqrt(disc)) / a;
        let z = ro.z + rd.z * ts;
        if (ts > 0.0 && z >= hh && z <= st && ts < (*best).t) { *best = Hit(ts, v3((ro.xy + rd.xy * ts - ctr) / sr, 0.0), idx, true); }
    }
    if (rd.z < 0.0) {
        let tc = (st - ro.z) / rd.z;
        let q = ro.xy + rd.xy * tc - ctr;
        if (tc > 0.0 && dot(q, q) <= sr * sr && tc < (*best).t) { *best = Hit(tc, v3(0.0, 0.0, 1.0), idx, false); }
    }
}

// 2D DDA; everything stays inside its cell, so the first hit is the nearest
fn trace_bricks(ro: v3, rd: v3, top: f32, skip: u32) -> Hit {
    var best = Hit(1e30, v3(0.0, 0.0, 1.0), NONE, false);
    let g = bgrid();
    let lo = v3(g.org, 0.0);
    let hi = v3(g.org + v2(g.n) * g.cs, top);
    let inv = 1.0 / select(rd, v3(1e-8), abs(rd) < v3(1e-8));
    let tn = min((lo - ro) * inv, (hi - ro) * inv);
    let tf = max((lo - ro) * inv, (hi - ro) * inv);
    let te = max(max(max(tn.x, tn.y), tn.z), 0.0);
    let tx = min(min(tf.x, tf.y), tf.z);
    if (tx <= te) { return best; }
    let q = ro.xy + rd.xy * (te + 1e-3);
    var c = clamp(vec2<i32>(floor((q - g.org) / g.cs)), vec2<i32>(0), g.n - 1);
    let stp = vec2<i32>(select(vec2<i32>(-1), vec2<i32>(1), rd.xy > v2(0.0)));
    let nb = g.org + (v2(c) + select(v2(0.0), v2(1.0), rd.xy > v2(0.0))) * g.cs;
    var tm = (nb - ro.xy) * inv.xy;
    let dt = abs(g.cs * inv.xy);
    for (var i = 0; i < 600; i++) {
        if (any(c < vec2<i32>(0)) || any(c >= g.n)) { break; }
        if (u32(c.y * g.n.x + c.x) != skip) {
            hit_cell(ro, rd, c, g, &best);
            if (best.pin != NONE) { break; }
        }
        if (min(tm.x, tm.y) > tx) { break; }
        if (tm.x < tm.y) { tm.x += dt.x; c.x += stp.x; } else { tm.y += dt.y; c.y += stp.y; }
    }
    return best;
}

fn trace(ro: v3, rd: v3, top: f32, first_only: bool, skip: u32) -> Hit {
    if (p.shape == 1u) { return trace_bricks(ro, rd, top, skip); }
    return trace_pins(ro, rd, top, first_only, skip);
}

// lanterns: objects that glow for a while and light their neighbours, then hand over
struct Lamp { pos: v3, col: v3, pin: u32 };
fn lamp(i: u32) -> Lamp {
    let ph = u_time.time * p.lan_spd * 0.1 + f32(i) * 0.618;
    var s = hu(i * 9781u + u32(floor(ph)) * 6271u + 17u);
    let e = sin(3.14159 * fract(ph));
    if (p.shape == 1u) {
        let g = bgrid();
        var bc = vec2<i32>(-1);
        var bh = -1.0;
        for (var j = 0; j < 3; j++) {
            let c = clamp(vec2<i32>(v2(g.n) * (0.1 + 0.8 * v2(hf(&s), hf(&s)))), vec2<i32>(0), g.n - 1);
            if (cell_h(c, g) > bh) { bh = cell_h(c, g); bc = c; }
        }
        let ic = cell_rgb(bc, g);
        let col = mix(ic / max(max(ic.r, max(ic.g, ic.b)), 0.05), v3(1.0, 0.8, 0.6), 0.2);
        return Lamp(v3(g.org + (v2(bc) + 0.5) * g.cs, bh + stud_h() * g.cs + 1.0), col * e * e * p.lan_pow, u32(bc.y * g.n.x + bc.x));
    }
    let r = img_rect();
    // the tallest of three random picks
    var best = NONE;
    var bh = -1.0;
    for (var j = 0; j < 3; j++) {
        let q = owner(r.xy + r.zw * (0.1 + 0.8 * v2(hf(&s), hf(&s))));
        if (q != NONE && pin_h(q) > bh) { bh = pin_h(q); best = q; }
    }
    if (best == NONE) { return Lamp(v3(0.0), v3(0.0), NONE); }
    var col = v3(1.0, 0.62, 0.28);
    if (p.colored != 0u) {
        let ic = pts[2u * best + 1u].rgb;
        col = mix(ic / max(max(ic.r, max(ic.g, ic.b)), 0.05), v3(1.0, 0.8, 0.6), 0.2);
    }
    return Lamp(v3(pts[2u * best].xy, bh + clamp(p.dome, 0.05, 1.0) * pin_r(best) + 1.0), col * e * e * p.lan_pow, best);
}
fn lamp_cnt() -> u32 { return u32(clamp(p.lan_n, 0.0, 32.0) + 0.5); }

// shadowed; far points skip the shadow ray
fn lamp_light(lm: Lamp, pos: v3, n: v3, rd: v3, diff: v3, spec: v3, rough: f32, top: f32) -> v3 {
    let dv = lm.pos - pos;
    let d2 = dot(dv, dv);
    let lc = lm.col * 1500.0 / (d2 + 100.0);
    let l = dv * inverseSqrt(d2);
    let ndl = max(dot(n, l), 0.0);
    if (ndl <= 0.0 || max(lc.r, max(lc.g, lc.b)) < 0.01) { return v3(0.0); }
    let sh = trace(pos + n * 0.05, l, top, true, lm.pin);
    if (sh.pin != NONE && sh.t * sh.t < d2) { return v3(0.0); }
    let hv = normalize(l - rd);
    let a2 = rough * rough * rough * rough;
    let nh = max(dot(n, hv), 0.0);
    let dd = nh * nh * (a2 - 1.0) + 1.0;
    return lc * ndl * (diff + spec * min(a2 / (3.14159 * dd * dd + 1e-5), 50.0) * 0.3);
}

fn ggx_aniso(n: v3, h: v3, t: v3, b: v3, ax: f32, ay: f32) -> f32 {
    let nh = dot(n, h);
    if (nh <= 0.0) { return 0.0; }
    let x = dot(t, h) / ax;
    let y = dot(b, h) / ay;
    let d = x * x + y * y + nh * nh;
    return 1.0 / (3.14159 * ax * ay * d * d + 1e-6);
}

// planar depth of the last view() pixel, for depth of field
var<private> g_depth: f32;

fn cell_of(id: u32, g: BGrid) -> vec2<i32> { return vec2<i32>(i32(id % u32(g.n.x)), i32(id / u32(g.n.x))); }
fn obj_col(id: u32) -> v3 {
    if (p.shape == 1u) { let g = bgrid(); return cell_rgb(cell_of(id, g), g); }
    return pts[2u * id + 1u].rgb;
}
fn obj_base(id: u32) -> v3 {
    if (p.shape == 1u) { return obj_col(id); }
    return select(v3(0.78, 0.79, 0.82), mix(v3(0.78, 0.79, 0.82), obj_col(id), 0.85), p.colored != 0u);
}

// shared finish; cap = a pin's hat or a brick's top, fs = flake cell, pxw = a pixel's world size here
fn material(pos: v3, n: v3, rd: v3, l: v3, base: v3, rough: f32, ao: f32, lit: f32, top: f32, env: f32,
            lamps: ptr<function, array<Lamp, 32>>, id: u32, cap: bool, fs: f32, pxw: f32) -> v3 {
    let v = -rd;
    let hv = normalize(l + v);
    let nh = max(dot(n, hv), 0.0);
    let ndl = max(dot(n, l), 0.0);
    let m = clamp(p.metal, 0.0, 1.0);
    let f0 = mix(v3(0.04), base, m);
    let fres = f0 + (1.0 - f0) * pow(1.0 - max(dot(n, v), 0.0), 5.0);
    let tr = cross(v3(0.0, 0.0, 1.0), n);
    let an = clamp(p.aniso, 0.0, 1.0) * min(length(tr) * 4.0, 1.0);
    let t = select(v3(1.0, 0.0, 0.0), normalize(tr), length(tr) > 1e-4);
    let b = cross(n, t);
    let ra = rough * rough;
    let asp = sqrt(1.0 - 0.9 * an);
    let spec = ggx_aniso(n, hv, t, b, max(ra * asp, 0.002), max(ra / asp, 0.002));
    let bent = normalize(mix(n, cross(cross(b, v), b), an * 0.8));
    let rr = reflect(rd, n);
    var rh = Hit(1e30, v3(0.0, 0.0, 1.0), NONE, false);
    if (p.refl > 0.0) { rh = trace(pos + n * 0.05, rr, top, true, id); }
    var nbc = v3(0.0);
    let mirrored = rh.pin != NONE;
    if (mirrored) {
        let qb = obj_base(rh.pin);
        nbc = qb * (0.1 * env + 0.6 * max(dot(rh.n, l), 0.0) * p.key) + qb * studio(reflect(rr, rh.n), l, 0.6) * 0.3 * env;
        for (var i = 0u; i < lamp_cnt(); i++) {
            if ((*lamps)[i].pin == rh.pin) { nbc += (*lamps)[i].col * select(3.0, 0.8, rh.side); }
        }
    }
    let sref = studio(reflect(rd, bent), l, rough) * env;
    var col = fres * select(sref, mix(sref, nbc, clamp(p.refl, 0.0, 1.0)), mirrored) * ao;
    col += fres * min(spec, 50.0) * ndl * lit * 0.35 * p.key;
    let diff = base * (1.0 - m);
    let wrap = max((dot(n, l) + 0.25) / 1.25, 0.0);
    col += diff * (0.9 * wrap * lit * p.key + studio(n, l, 1.0) * 0.35 * env) * ao;
    col += base * 0.08 * ndl * lit * p.key;
    for (var i = 0u; i < lamp_cnt(); i++) {
        let lm = (*lamps)[i];
        if (lm.pin == id) {
            col += lm.col * select(1.2, 3.0, cap);
        } else {
            col += lamp_light(lm, pos, n, rd, diff * 0.9 + fres * 0.08, fres, rough, top) * ao;
        }
    }
    let back = normalize(v3(-l.xy, 0.3));
    col += v3(1.0, 0.55, 0.3) * pow(1.0 - max(dot(n, v), 0.0), 3.0) * max(dot(n, back), 0.0) * 0.6 * ao * p.key;
    // a sub-pixel flake grows to a pixel with its energy kept, so it can't flicker
    if (p.flake > 0.0 && (cap || p.fl_all != 0u)) {
        let cs = fs * clamp(p.fl_size, 0.2, 4.0);
        let cp = floor(pos / cs);
        let q = bitcast<vec3<u32>>(vec3<i32>(cp));
        var sd = hu((q.x * 73856093u) ^ (q.y * 19349663u) ^ (q.z * 83492791u) ^ (id * 2654435761u));
        let keep = hf(&sd) < p.fl_dens;
        let fc = (cp + 0.25 + 0.5 * v3(hf(&sd), hf(&sd), hf(&sd))) * cs;
        let rf = 0.3 * cs;
        let re = max(rf, pxw * 0.7);
        let inside = smoothstep(re, re * 0.55, length(pos - fc)) * (rf * rf) / (re * re);
        if (keep && inside > 0.0) {
            let ph = hf(&sd) * 6.2832;
            let sp = (0.6 + 1.4 * hf(&sd)) * u_time.time;
            let wob = v3(sin(sp + ph), sin(sp * 1.31 + ph * 2.1), sin(sp * 0.77 + ph * 0.6)) * 0.35 * p.twinkle;
            let fn_ = normalize(n + (v3(hf(&sd), hf(&sd), hf(&sd)) - 0.5) * 0.9 + wob);
            var fl = v3(1.0, 0.97, 0.92) * pow(max(dot(fn_, hv), 0.0), 500.0) * 60.0 * lit * p.key;
            for (var i = 0u; i < lamp_cnt(); i++) {
                let lm = (*lamps)[i];
                if (lm.pin == id) { continue; }
                let dl = lm.pos - pos;
                fl += lm.col * 1500.0 / (dot(dl, dl) + 100.0) * pow(max(dot(fn_, normalize(normalize(dl) + v)), 0.0), 500.0) * 60.0;
            }
            fl += studio(reflect(rd, fn_), l, 0.02) * env * 0.5;
            let hue = hf(&sd) + dot(fn_, v) * 3.0;
            let rainbow = 0.5 + 0.5 * cos(6.2832 * (hue + v3(0.0, 0.33, 0.67)));
            let tint = mix(mix(v3(1.0), base, 0.5), rainbow * 1.4, clamp(p.holo, 0.0, 1.0));
            col += fl * tint * inside * p.flake * ao;
        }
    }
    if (cap) {
        if (p.glow > 0.0) {
            let ic = obj_col(id);
            let lum = dot(ic, v3(0.299, 0.587, 0.114));
            col += ic * lum * lum * p.glow * 4.0;
        }
    }
    if (p.coat > 0.0) {
        let fc = (0.04 + 0.96 * pow(1.0 - max(dot(n, v), 0.0), 5.0)) * p.coat;
        let ca2 = 0.0016;
        let cd = nh * nh * (ca2 - 1.0) + 1.0;
        let cs = studio(rr, l, 0.04) * env;
        let cref = select(cs, mix(cs, nbc, clamp(p.refl, 0.0, 1.0)), mirrored);
        col = col * (1.0 - fc) + fc * cref * ao + p.coat * 0.04 * min(ca2 / (3.14159 * cd * cd), 2000.0) * ndl * lit * p.key;
    }
    return col;
}

fn pins_view(gid: vec2<u32>, D: v2) -> v3 {
    g_depth = 1e4;
    let cam = camera(gid, D);
    let ro = cam.ro;
    let rd = cam.rd;
    let f = cam.f;
    let l = key_light();
    let top = scene_top();
    let h = trace_pins(ro, rd, top, false, NONE);
    var lamps: array<Lamp, 32>;
    for (var i = 0u; i < lamp_cnt(); i++) { lamps[i] = lamp(i); }
    let env = 0.3 + 0.7 * p.key;

    var pos: v3;
    var n: v3;
    if (h.pin == NONE) {
        if (rd.z >= 0.0) { return studio(rd, l, 0.6) * 0.3 * env; }
        let tb = -ro.z / rd.z;
        pos = ro + rd * tb;
        g_depth = tb * dot(rd, f);
        let uv = image_uv(pos.xy);
        if (any(uv < v2(0.0)) || any(uv > v2(1.0))) { return studio(rd, l, 0.6) * 0.3 * env; }
        n = v3(0.0, 0.0, 1.0);
    } else {
        pos = ro + rd * h.t;
        g_depth = h.t * dot(rd, f);
        n = h.n;
    }

    let sh = trace_pins(pos + n * 0.05, l, top, true, NONE);
    let lit = select(1.0, 0.0, sh.pin != NONE);
    let ndl = max(dot(n, l), 0.0);

    if (h.pin == NONE) {
        let o = owner(pos.xy);
        var ao = 1.0;
        if (o != NONE) {
            let d = length(pos.xy - pts[2u * o].xy);
            let pr = pin_r(o);
            ao = mix(0.35, 1.0, smoothstep(pr, pr * 2.4, d));
        }
        let board = v3(0.035, 0.034, 0.038);
        let fb = 0.04 + 0.96 * pow(1.0 - max(dot(n, -rd), 0.0), 5.0);
        let kc = board * (0.25 * env + 1.4 * ndl * lit * p.key) + board * studio(n, l, 1.0) * 0.4 * env + fb * studio(reflect(rd, n), l, 0.35) * 0.5 * env;
        var lc = v3(0.0);
        for (var i = 0u; i < lamp_cnt(); i++) { lc += lamp_light(lamps[i], pos, n, rd, v3(0.5), v3(0.0), 1.0, top); }
        return (kc + lc) * ao;
    }

    let vr = f32(hu(h.pin * 2654435761u) >> 8u) / 16777216.0;
    let base = obj_base(h.pin) * (0.92 + 0.16 * vr);
    let rough = clamp(p.rough * (0.8 + 0.4 * vr), 0.04, 1.0);
    let ph = pin_h(h.pin);
    let pr = pin_r(h.pin);
    let pc = pts[2u * h.pin].xy;

    var ao = 1.0;
    if (h.side) { ao = mix(0.3, 1.0, smoothstep(0.0, ph, pos.z)); }
    var occ = 0.0;
    for (var k = 0; k < 6; k++) {
        let a = f32(k) * 1.0472;
        let q = owner(pc + v2(cos(a), sin(a)) * pr * 2.2);
        if (q != NONE && q != h.pin) { occ += smoothstep(0.0, ph * 0.5 + 1.0, pin_h(q) - pos.z); }
    }
    ao *= 1.0 - 0.1 * occ;

    var col = material(pos, n, rd, l, base, rough, ao, lit, top, env, &lamps, h.pin, !h.side, pr * 0.09, h.t * 0.64 / D.y);
    if (!h.side) {
        let e = smoothstep(0.78, 1.0, length(pos.xy - pc) / pr);
        col += mix(v3(0.04), base, clamp(p.metal, 0.0, 1.0)) * e * (0.6 * ndl * lit * p.key + 0.25 * env) * studio(reflect(rd, n), l, rough * 0.5);
    }
    return col;
}

fn bricks_view(gid: vec2<u32>, D: v2) -> v3 {
    g_depth = 1e4;
    let cam = camera(gid, D);
    let ro = cam.ro;
    let rd = cam.rd;
    let l = key_light();
    let top = scene_top();
    let g = bgrid();
    let h = trace_bricks(ro, rd, top, NONE);
    var lamps: array<Lamp, 32>;
    for (var i = 0u; i < lamp_cnt(); i++) { lamps[i] = lamp(i); }
    let env = 0.3 + 0.7 * p.key;

    if (h.pin == NONE) {
        if (rd.z >= 0.0) { return studio(rd, l, 0.6) * 0.3 * env; }
        let tb = -ro.z / rd.z;
        let bp = ro + rd * tb;
        g_depth = tb * dot(rd, cam.f);
        let uv = image_uv(bp.xy);
        if (any(uv < v2(-0.01)) || any(uv > v2(1.01))) { return studio(rd, l, 0.6) * 0.3 * env; }
        return v3(0.008) * env;
    }
    let pos = ro + rd * h.t;
    g_depth = h.t * dot(rd, cam.f);
    let c = cell_of(h.pin, g);
    let hh = cell_h(c, g);
    let base = cell_rgb(c, g);
    let ctr = g.org + (v2(c) + 0.5) * g.cs;
    let loc = pos.xy - ctr;
    let sr = STUD_R * g.cs;
    let stud = select(pos.z > hh + 0.01 * g.cs, length(loc) < sr * 1.05, h.n.z < 0.5);
    let st = hh + stud_h() * g.cs;
    let half = (0.5 - GAP) * g.cs;
    let bev = 0.05 * g.cs;

    var n = h.n;
    if (!stud) {
        if (n.z > 0.5) {
            let e = half - abs(loc);
            let w = v2(1.0) - smoothstep(v2(0.0), v2(bev), e);
            n = normalize(n + v3(sign(loc) * w, 0.0) * 0.9);
        } else {
            let tg = v2(-n.y, n.x);
            let al = dot(loc, tg);
            let wc = 1.0 - smoothstep(0.0, bev, half - abs(al));
            let wt = 1.0 - smoothstep(0.0, bev, hh - pos.z);
            n = normalize(n + v3(tg * sign(al) * wc * 0.9, wt * 0.9));
        }
    } else if (h.side) {
        n = normalize(n + v3(0.0, 0.0, (1.0 - smoothstep(0.0, bev * 0.6, st - pos.z)) * 0.9));
    } else {
        // stud top: rounded rim and the logo ring
        let rl = length(loc);
        let dir = loc / max(rl, 1e-4);
        let rim = 1.0 - smoothstep(0.0, bev * 0.6, sr - rl);
        let d = (rl / sr - 0.62) / 0.07;
        n = normalize(n + v3(dir * (rim * 0.9 - 2.0 * d * exp(-d * d) * 0.25), 0.0));
    }

    let sh = trace_bricks(pos + h.n * 0.05, l, top, NONE);
    let lit = select(1.0, 0.0, sh.pin != NONE);

    var ao = 1.0;
    if (!stud && h.n.z > 0.5) {
        let dirs = array<vec2<i32>, 4>(vec2<i32>(1, 0), vec2<i32>(-1, 0), vec2<i32>(0, 1), vec2<i32>(0, -1));
        var occ = 0.0;
        for (var k = 0; k < 4; k++) {
            let cn = c + dirs[k];
            if (any(cn < vec2<i32>(0)) || any(cn >= g.n)) { continue; }
            let edge = half - dot(loc, v2(dirs[k]));
            occ += smoothstep(0.0, 0.6 * g.cs, cell_h(cn, g) - hh) * (1.0 - smoothstep(0.0, 0.5 * g.cs, edge));
        }
        ao *= 1.0 - 0.45 * min(occ, 1.5);
        ao *= mix(0.6, 1.0, smoothstep(0.0, 0.14 * g.cs, length(loc) - sr));
    } else if (!stud) {
        let cn = c + vec2<i32>(round(h.n.xy));
        var hn = 0.0;
        if (all(cn >= vec2<i32>(0)) && all(cn < g.n)) { hn = cell_h(cn, g); }
        ao *= mix(0.35, 1.0, smoothstep(0.0, 0.6 * g.cs, pos.z - hn));
        let pz = PLATE * g.cs;
        let fz = fract(pos.z / pz);
        ao *= mix(0.55, 1.0, smoothstep(0.004 * g.cs, 0.02 * g.cs, min(fz, 1.0 - fz) * pz));
    }

    var col = material(pos, n, rd, l, base, clamp(p.rough, 0.04, 1.0), ao, lit, top, env, &lamps, h.pin, stud || h.n.z > 0.5, 0.05 * g.cs, h.t * 0.64 / D.y);
    return col;
}

fn view(gid: vec2<u32>, D: v2) -> v3 {
    if (p.shape == 1u) { return bricks_view(gid, D); }
    return pins_view(gid, D);
}

fn pack_px(c: v3, d: f32) -> vec2<u32> { return vec2<u32>(pack2x16float(c.rg), pack2x16float(v2(c.b, d))); }
fn unpack_px(q: vec2<u32>) -> v3 { return v3(unpack2x16float(q.x), unpack2x16float(q.y).x); }
fn aces(x: v3) -> v3 { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), v3(0.0), v3(1.0)); }
fn qdims(d: vec2<u32>) -> vec2<u32> { return (d + 3u) / 4u; }

@compute @workgroup_size(16, 16, 1)
fn pins_hdr(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output);
    if (gid.x >= dims.x || gid.y >= dims.y || dims.x * dims.y > MAXPIX) { return; }
    // no NaN / Inf: one bad pixel blurs into a square
    var c = view(gid.xy, v2(dims));
    c = select(c, v3(0.0), c != c);
    hdr[gid.y * dims.x + gid.x] = pack_px(clamp(c, v3(0.0), v3(1000.0)), min(g_depth, 6e4));
}

// signed circle of confusion in half-res pixels, positive behind focus
fn coc(d: f32, D: v2) -> f32 {
    let fd = 1500.0 / max(p.zoom, 0.1) * exp(p.focus);
    let m = min(p.aperture * D.y * 0.02, D.y * 0.025);
    return clamp(p.aperture * D.y * 0.02 * (1.0 - fd / max(d, 1.0)), -m, m);
}
fn hdims(d: vec2<u32>) -> vec2<u32> { return (d + 1u) / 2u; }
const HALF: u32 = MAXPIX / 4u + 4096u;

// in-focus pixels fade out here and come back sharp in the composite
@compute @workgroup_size(16, 16, 1)
fn dof_pre(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output);
    let hd = hdims(dims);
    if (gid.x >= hd.x || gid.y >= hd.y || dims.x * dims.y > MAXPIX || p.aperture <= 0.0) { return; }
    let D = v2(dims);
    var c = v3(0.0);
    var cmin = 1e9;
    var cmax = -1e9;
    for (var k = 0u; k < 4u; k++) {
        let px = min(gid.xy * 2u + vec2<u32>(k & 1u, k >> 1u), dims - 1u);
        let q = hdr[px.y * dims.x + px.x];
        c += unpack_px(q) * 0.25;
        let cc = coc(unpack2x16float(q.y).y, D);
        cmin = min(cmin, cc);
        cmax = max(cmax, cc);
    }
    let cc = select(cmax, cmin, -cmin > cmax);
    c *= 1.0 + p.bokeh_hi * smoothstep(1.0, 4.0, max(c.r, max(c.g, c.b)) * p.expo) * smoothstep(1.0, 3.0, abs(cc));
    dof[gid.y * hd.x + gid.x] = pack_px(c * smoothstep(0.0, 2.0, abs(cc)), cc);
}

// background taps capped by the centre's circle (no bleed onto sharp objects); foreground taps carry alpha
@compute @workgroup_size(16, 16, 1)
fn dof_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output);
    let hd = hdims(dims);
    if (gid.x >= hd.x || gid.y >= hd.y || dims.x * dims.y > MAXPIX || p.aperture <= 0.0) { return; }
    let D = v2(dims);
    let rmax = min(p.aperture * D.y * 0.02, D.y * 0.025);
    const N = 80u;
    let margin = max(rmax / sqrt(f32(N)) * 1.5, 1.0);
    let c0 = unpack2x16float(dof[gid.y * hd.x + gid.x].y).y;
    let nb = floor(p.blades);
    var bg = v4(0.0);
    var fg = v4(0.0);
    for (var i = 0u; i < N; i++) {
        let r = sqrt((f32(i) + 0.5) / f32(N)) * rmax;
        let a = f32(i) * 2.39996;
        var k = 1.0;
        if (nb >= 3.0) {
            let seg = 6.28318 / nb;
            k = cos(0.5 * seg) / cos(a + 0.3 - floor((a + 0.3) / seg) * seg - 0.5 * seg);
        }
        let sp = clamp(vec2<i32>(floor(v2(gid.xy) + 0.5 + v2(cos(a), sin(a)) * r * k)), vec2<i32>(0), vec2<i32>(hd) - 1);
        let q = dof[u32(sp.y) * hd.x + u32(sp.x)];
        let sc = unpack2x16float(q.y).y;
        let s = unpack_px(q);
        let bw = clamp((max(min(c0, sc), 0.0) - r + margin) / margin, 0.0, 1.0);
        let fw = clamp((-sc - r + margin) / margin, 0.0, 1.0) * step(1.0, -sc);
        bg += v4(s, 1.0) * bw;
        fg += v4(s, 1.0) * fw;
    }
    let bgc = bg.rgb / max(bg.a, 1e-4);
    let fgc = fg.rgb / max(fg.a, 1e-4);
    let fa = clamp(fg.a * 3.14159 / f32(N), 0.0, 1.0);
    dof[HALF + gid.y * hd.x + gid.x] = pack_px(mix(bgc, fgc, fa), fa);
}

@compute @workgroup_size(16, 16, 1)
fn dof_post(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output);
    let hd = hdims(dims);
    if (gid.x >= hd.x || gid.y >= hd.y || dims.x * dims.y > MAXPIX || p.aperture <= 0.0) { return; }
    var c = v4(0.0);
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let sp = clamp(vec2<i32>(gid.xy) + vec2<i32>(x, y), vec2<i32>(0), vec2<i32>(hd) - 1);
            let q = dof[HALF + u32(sp.y) * hd.x + u32(sp.x)];
            let w = f32((2 - abs(x)) * (2 - abs(y)));
            c += v4(unpack_px(q), unpack2x16float(q.y).y) * w;
        }
    }
    c /= 16.0;
    dof[2u * HALF + gid.y * hd.x + gid.x] = pack_px(c.rgb, c.a);
}

fn final_px(px: vec2<u32>, dims: vec2<u32>) -> v3 {
    let q = hdr[px.y * dims.x + px.x];
    let sharp = unpack_px(q);
    if (p.aperture <= 0.0) { return sharp; }
    let hd = hdims(dims);
    let x = clamp((v2(px) + 0.5) / 2.0 - 0.5, v2(0.0), v2(hd) - 1.0);
    let i = vec2<u32>(floor(x));
    let f = fract(x);
    let j = min(i + 1u, hd - 1u);
    let o = 2u * HALF;
    let qa = dof[o + i.y * hd.x + i.x];
    let qb = dof[o + i.y * hd.x + j.x];
    let qc = dof[o + j.y * hd.x + i.x];
    let qd = dof[o + j.y * hd.x + j.x];
    let b = mix(mix(v4(unpack_px(qa), unpack2x16float(qa.y).y), v4(unpack_px(qb), unpack2x16float(qb.y).y), f.x),
                mix(v4(unpack_px(qc), unpack2x16float(qc.y).y), v4(unpack_px(qd), unpack2x16float(qd.y).y), f.x), f.y);
    let ffa = smoothstep(1.0, 2.0, coc(unpack2x16float(q.y).y, v2(dims)));
    return mix(sharp, b.rgb, ffa + b.a - ffa * b.a);
}

// bloom source: 4x4 average, above white only
@compute @workgroup_size(16, 16, 1)
fn bloom_down(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output);
    let q = qdims(dims);
    if (gid.x >= q.x || gid.y >= q.y || dims.x * dims.y > MAXPIX) { return; }
    var c = v3(0.0);
    var ws = 0.0;
    for (var y = 0u; y < 4u; y++) {
        for (var x = 0u; x < 4u; x++) {
            let px = min(gid.xy * 4u + vec2<u32>(x, y), dims - 1u);
            let s = final_px(px, dims);
            let w = 1.0 / (1.0 + max(s.r, max(s.g, s.b)));
            c += s * w;
            ws += w;
        }
    }
    c /= ws;
    let br = max(c.r, max(c.g, c.b)) * p.expo;
    let k = smoothstep(0.7, 1.4, br);
    b1[gid.y * q.x + gid.x] = pack_px(c * k, 0.0);
}

// separable gaussian, sigma 5 quarter pixels
fn blur(src_b1: bool, gid: vec2<u32>, dir: vec2<i32>) {
    let dims = textureDimensions(output);
    let q = qdims(dims);
    if (gid.x >= q.x || gid.y >= q.y || dims.x * dims.y > MAXPIX) { return; }
    var c = v3(0.0);
    var ws = 0.0;
    for (var k = -12; k <= 12; k++) {
        let s = clamp(vec2<i32>(gid) + dir * k, vec2<i32>(0), vec2<i32>(q) - 1);
        let i = u32(s.y) * q.x + u32(s.x);
        let w = exp(-f32(k * k) / 50.0);
        c += unpack_px(select(b2[i], b1[i], src_b1)) * w;
        ws += w;
    }
    let o = gid.y * q.x + gid.x;
    if (src_b1) { b2[o] = pack_px(c / ws, 0.0); } else { b1[o] = pack_px(c / ws, 0.0); }
}
@compute @workgroup_size(16, 16, 1) fn bloom_h(@builtin(global_invocation_id) gid: vec3<u32>) { blur(true, gid.xy, vec2<i32>(1, 0)); }
@compute @workgroup_size(16, 16, 1) fn bloom_v(@builtin(global_invocation_id) gid: vec3<u32>) { blur(false, gid.xy, vec2<i32>(0, 1)); }

fn bloom_at(qp: v2, dims: vec2<u32>) -> v3 {
    let q = qdims(dims);
    let x = clamp(qp - 0.5, v2(0.0), v2(q) - 1.0);
    let i = vec2<u32>(floor(x));
    let f = fract(x);
    let j = min(i + 1u, q - 1u);
    let a = unpack_px(b1[i.y * q.x + i.x]);
    let b = unpack_px(b1[i.y * q.x + j.x]);
    let c = unpack_px(b1[j.y * q.x + i.x]);
    let d = unpack_px(b1[j.y * q.x + j.x]);
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let D = v2(dims);
    var c: v3;
    if (dims.x * dims.y > MAXPIX) {
        // bigger than the HDR buffer: no post
        c = view(gid.xy, D);
    } else {
        c = final_px(gid.xy, dims);
        if (p.bloom > 0.0) { c += bloom_at((v2(gid.xy) + 0.5) / 4.0, dims) * p.bloom; }
    }
    c = aces(c * p.expo);
    textureStore(output, gid.xy, v4(pow(c, v3(1.0 / max(p.gam, 0.1))), 1.0));
}

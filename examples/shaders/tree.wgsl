// Tree, Enes Altun, 2026. CC BY-NC-SA 3.0
// Light-walk rendering after wyatt; tree fractal after https://www.shadertoy.com/view/dsyczD

struct TimeUniform { time: f32, delta: f32, frame: u32, _padding: u32 };
@group(0) @binding(0) var<uniform> u_time: TimeUniform;

struct Params {
    eye_x: f32, eye_y: f32, eye_z: f32, fov: f32,
    fwd_x: f32, fwd_y: f32, fwd_z: f32, aperture: f32,
    rt_x: f32, rt_y: f32, rt_z: f32, focus: f32,
    up_x: f32, up_y: f32, up_z: f32, exposure: f32,
    n: u32, view_n: u32, walkers: u32, iterations: u32,
    steps: u32, zoom: f32, pan_x: f32, pan_y: f32,
    power: f32, offset: f32, bend: f32, align: f32,
    light: f32, light_dir: f32, phase: f32, phase_off: f32,
    darken: f32, gamma: f32, jewel: f32, jsat: f32,
    shine: f32, bloom: f32, twinkle: f32, _q0: f32,
    amp: f32, gem_h: f32, _q1: f32, _q2: f32,
    metal: f32, sun_az: f32, sun_el: f32, sun_power: f32,
};
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> p: Params;

@group(3) @binding(0) var tex0: texture_2d<f32>;
@group(3) @binding(1) var sam0: sampler;
@group(3) @binding(2) var tex1: texture_2d<f32>;
@group(3) @binding(3) var sam1: sampler;
@group(3) @binding(4) var tex2: texture_2d<f32>;
@group(3) @binding(5) var sam2: sampler;

alias v2 = vec2<f32>; alias v3 = vec3<f32>; alias v4 = vec4<f32>;
const PI = 3.14159265;

var<private> R: v2;

// Shadertoy frame: y up
fn frag(id: vec2<u32>) -> v2 { return v2(f32(id.x) + 0.5, R.y - (f32(id.y) + 0.5)); }
// clamped half a texel in (the sampler repeats)
fn uv_of(U: v2) -> v2 { return clamp(v2(U.x / R.x, 1.0 - U.y / R.y), 0.5 / R, 1.0 - 0.5 / R); }
fn A(U: v2) -> v4 { return textureSampleLevel(tex0, sam0, uv_of(U), 0.0); }
fn B(U: v2) -> v4 { return textureSampleLevel(tex1, sam1, uv_of(U), 0.0); }

fn hash(p4i: v4) -> v4 {
    var p4 = fract(p4i * v4(0.1031, 0.1030, 0.0973, 0.1099));
    p4 += dot(p4, p4.wzxy + 33.33);
    return fract((p4.xxyz + p4.yzzw) * p4.zywx);
}

fn complex_power(z: v2, x: f32) -> v2 {
    let r = pow(length(z), x);
    let a = x * atan2(z.y, z.x);
    return r * v2(cos(a), sin(a));
}

// buffer A; only on the first two frames (both ping-pong sides)
@compute @workgroup_size(16, 16, 1)
fn relief(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y || p.n >= 2u) { return; }
    R = v2(dims);
    let U = frag(id.xy);
    var z = ((U - 0.5 * R) / min(R.x, R.y) + v2(p.pan_x, p.pan_y)) * p.zoom;
    var dz = v2(0.1, 0.0);
    var i = 0u;
    for (; i < p.iterations; i++) {
        dz = p.power * pow(length(z), p.power - 1.0) * dz;
        z = complex_power(z, p.power) - v2(p.offset, 0.0);
        if (dot(z, z) > 77.0) { break; }
    }
    let ratio = f32(i) / f32(p.iterations);
    let sharp = dot(z, z) / max(dot(dz, dz), 1e-30);
    let c1 = 0.5 + 0.5 * cos(1.0 + v3(0.0, 0.5, 1.0) + PI * v3(2.0 * sharp));
    let c2 = 0.5 + 0.5 * cos(4.1 + PI * v3(sharp));
    let c = sqrt(mix(c1, c2, ratio));
    textureStore(output, id.xy, v4((c.r + c.g + c.b) / 3.0, 0.0, 0.0, 1.0));
}

// buffer B
@compute @workgroup_size(16, 16, 1)
fn slope(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y || p.n >= 2u) { return; }
    R = v2(dims);
    let U = frag(id.xy);
    let n = A(U + v2(0.0, 1.0)).x;
    let e = A(U + v2(1.0, 0.0)).x;
    let s = A(U - v2(0.0, 1.0)).x;
    let w = A(U - v2(1.0, 0.0)).x;
    textureStore(output, id.xy, v4(0.5 * (e - w), 0.5 * (n - s), A(U).x, 1.0));
}

// buffer C, as a running average (f16 would drown a running sum)
@compute @workgroup_size(16, 16, 1)
fn walk(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    R = v2(dims);
    let U0 = frag(id.xy);
    let ld = v2(cos(radians(p.light_dir)), sin(radians(p.light_dir)));
    var q = v4(0.0);
    for (var k = 0u; k < p.walkers; k++) {
        var U = U0;
        let h = hash(v4(U0, f32(u_time.frame), f32(k) + 1.0));
        var d = v2(cos(2.0 * PI * h.x), sin(2.0 * PI * h.x));
        for (var i = 0u; i < p.steps; i++) {
            U += d;
            let b = B(U);
            d = normalize(d + (1.0 + h.z) * p.bend * b.xy);
            q += p.light * exp(-p.align * length(d - ld)) * max(sin(p.phase_off + p.phase * h.z + v4(1.0, 2.0, 3.0, 4.0)), v4(0.0));
            q -= v4(1.0, 2.0, 3.0, 4.0) * p.darken * b.z;
        }
    }
    q /= f32(max(p.walkers, 1u));
    let prev = textureLoad(tex0, vec2<i32>(id.xy), 0);
    textureStore(output, id.xy, select(mix(prev, q, 1.0 / f32(p.n + 1u)), q, p.n == 0u));
}

fn cuv(uv: v2, d: v2) -> v2 { return clamp(uv, 0.5 / d, 1.0 - 0.5 / d); }
fn tap0(uv: v2) -> v4 { let d = v2(textureDimensions(tex0)); return textureSampleLevel(tex0, sam0, cuv(uv, d), 0.0); }
// the emitted share of a view pixel
fn glow_of(uv: v2) -> v3 { let c = tap0(uv); return c.rgb * clamp(c.a / max(lum(c.rgb), 1e-4), 0.0, 1.0); }
fn tap1(uv: v2) -> v4 { let d = v2(textureDimensions(tex1)); return textureSampleLevel(tex1, sam1, cuv(uv, d), 0.0); }
fn px_uv(id: vec2<u32>) -> v2 { return (v2(id) + 0.5) / v2(textureDimensions(output)); }
fn hf(q: v2) -> f32 { return fract(sin(dot(q, v2(12.9898, 78.233))) * 43758.5453); }
fn lum(c: v3) -> f32 { return dot(c, v3(0.2126, 0.7152, 0.0722)); }

var<private> seed: u32;
fn hu(a0: u32) -> u32 { var a = a0; a ^= a >> 16u; a *= 0x7feb352du; a ^= a >> 15u; a *= 0x846ca68bu; a ^= a >> 16u; return a; }
fn rnd() -> f32 { seed = hu(seed); return f32(seed >> 8u) / 16777216.0; }

// x: smoothed brightness, y: jewel (more colourful and brighter than its ring)
@compute @workgroup_size(16, 16, 1)
fn height(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let uv = px_uv(id.xy);
    let e = 1.0 / v2(dims);
    let c = max(tap0(uv).rgb * p.exposure, v3(0.0));
    var hs = 0.0;
    var ring = v3(0.0);
    for (var k = 0; k < 8; k++) {
        let d = v2(cos(f32(k) * 0.785398), sin(f32(k) * 0.785398));
        let a = max(tap0(uv + d * e * 3.0).rgb * p.exposure, v3(0.0));
        let b = max(tap0(uv + d * e * 7.0).rgb * p.exposure, v3(0.0));
        hs += min(lum(a), lum(b) * 1.5) + lum(b);
        ring += max(tap0(uv + d * e * 6.0).rgb * p.exposure, v3(0.0));
    }
    ring *= 0.125;
    let dc = max(c - ring, v3(0.0));
    let dm = max(dc.r, max(dc.g, dc.b));
    let sat = (dm - min(dc.r, min(dc.g, dc.b))) / (dm + 1e-3);
    // a fresh walk is speckle, and every speck would pass as a jewel: they fade in as it converges
    let gem = smoothstep(p.jsat, p.jsat + 0.2, sat) * smoothstep(0.02, 0.15, lum(dc)) * smoothstep(4.0, 48.0, f32(p.n));
    textureStore(output, id.xy, v4(hs / 16.0, gem, 0.0, 1.0));
}

// x in [-aspect, aspect], z in [-1, 1]
var<private> ASP: f32;
fn relief_uv(xz: v2) -> v2 { return v2(xz.x / ASP * 0.5 + 0.5, xz.y * 0.5 + 0.5); }
fn surf(xz: v2) -> v2 {
    let t = tap1(relief_uv(xz));
    return v2(p.amp * t.x + p.gem_h * t.y * t.y, t.y);
}
fn surf_n(xz: v2) -> v3 {
    let e = 2.0 / f32(textureDimensions(tex1).y);
    let hx = surf(xz + v2(e, 0.0)).x - surf(xz - v2(e, 0.0)).x;
    let hz = surf(xz + v2(0.0, e)).x - surf(xz - v2(0.0, e)).x;
    return normalize(v3(-hx, 2.0 * e, -hz));
}
fn sun_dir() -> v3 {
    let az = radians(p.sun_az);
    let el = radians(p.sun_el);
    return v3(cos(el) * sin(az), sin(el), cos(el) * cos(az));
}
fn sky(d: v3) -> v3 {
    var c = mix(v3(0.004, 0.005, 0.008), v3(0.05, 0.055, 0.07), smoothstep(-0.1, 0.8, d.y));
    let s = dot(d, sun_dir());
    c += v3(1.0, 0.94, 0.85) * p.sun_power * (8.0 * smoothstep(0.996, 0.999, s) + 0.08 * pow(max(s, 0.0), 24.0));
    return c;
}
fn box_t(ro: v3, rd: v3, lo: v3, hi: v3) -> v2 {
    let inv = 1.0 / select(rd, v3(1e-8), abs(rd) < v3(1e-8));
    let a = (lo - ro) * inv;
    let b = (hi - ro) * inv;
    return v2(max(max(min(a.x, b.x), min(a.y, b.y)), min(a.z, b.z)), min(min(max(a.x, b.x), max(a.y, b.y)), max(a.z, b.z)));
}

fn radiance(px: v2, D: v2) -> v4 {
    let eye = v3(p.eye_x, p.eye_y, p.eye_z);
    let f = v3(p.fwd_x, p.fwd_y, p.fwd_z);
    let rt = v3(p.rt_x, p.rt_y, p.rt_z);
    let up = v3(p.up_x, p.up_y, p.up_z);
    let sp = (px - 0.5 * D) / D.y * 2.0 * tan(radians(p.fov) * 0.5);
    let d0 = normalize(f + sp.x * rt - sp.y * up);
    let a = 6.2832 * rnd();
    let ro = eye + (rt * cos(a) + up * sin(a)) * sqrt(rnd()) * p.aperture;
    let rd = normalize(eye + d0 * (p.focus / dot(d0, f)) - ro);

    let bt = box_t(ro, rd, v3(-ASP, 0.0, -1.0), v3(ASP, p.amp + p.gem_h + 1e-3, 1.0));
    if (bt.x >= bt.y || bt.y <= 0.0) { return v4(sky(rd), 0.0); }
    let t0 = max(bt.x, 0.0);
    let steps = 160;
    let dt = (bt.y - t0) / f32(steps);
    var ta = t0;
    var th = -1.0;
    for (var i = 1; i <= steps; i++) {
        let t = t0 + dt * f32(i);
        let q = ro + rd * t;
        if (q.y <= surf(q.xz).x) {
            var lo = ta;
            var hi = t;
            for (var k = 0; k < 6; k++) {
                let m = 0.5 * (lo + hi);
                let qm = ro + rd * m;
                if (qm.y <= surf(qm.xz).x) { hi = m; } else { lo = m; }
            }
            th = hi;
            break;
        }
        ta = t;
    }
    if (th < 0.0) { return v4(sky(rd), 0.0); }
    let hp = ro + rd * th;
    let n = surf_n(hp.xz);
    let uv = relief_uv(hp.xz);
    let art = max(tap0(uv).rgb * p.exposure, v3(0.0));
    let gem = surf(hp.xz).y;
    let v = -rd;
    let l = sun_dir();
    let hv = normalize(l + v);
    let spec = pow(max(dot(n, hv), 0.0), 60.0) * 8.0;
    var c = art * (0.75 + 0.5 * max(dot(n, l), 0.0)) + art * p.metal * (spec * p.sun_power * 0.1 + sky(reflect(rd, n)) * 2.0);
    c += art * smoothstep(0.6, 1.1, lum(art)) * p.shine;
    var em = v3(0.0);
    if (gem > 0.02) {
        let cell = floor(uv * v2(textureDimensions(tex0)) / 5.0);
        let tw = mix(1.0, 0.3 + 0.7 * pow(0.5 + 0.5 * sin(u_time.time * (1.5 + 2.5 * hf(cell)) + 6.2832 * hf(cell + 7.0)), 3.0), p.twinkle);
        em = art * p.jewel * tw * smoothstep(0.02, 0.4, gem);
    }
    return v4(c + em, lum(em));
}

@compute @workgroup_size(16, 16, 1)
fn view(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let D = v2(dims);
    ASP = D.x / D.y;
    seed = hu(id.x + id.y * 7919u * dims.x + hu(u_time.frame * 3u + 1u));
    var c = radiance(v2(id.xy) + v2(rnd(), rnd()), D);
    if (any(c != c) || any(abs(c) > v4(1e5))) { c = v4(0.0); }
    // firefly clamp
    c *= min(1.0, 16.0 / max(lum(c.rgb), 1e-4));
    // never average longer than the walk has converged
    var k = max(1.0 / f32(p.view_n + 1u), 2.0 / f32(p.n + 2u));
    k = max(k, select(0.0, 0.12, p.twinkle > 0.0));
    let prev = textureLoad(tex2, vec2<i32>(id.xy), 0);
    let fresh = p.view_n == 0u || any(prev != prev);
    textureStore(output, id.xy, select(mix(prev, c, k), c, fresh));
}

// bloom
fn karis(c: v3) -> f32 { return 1.0 / (1.0 + max(c.r, max(c.g, c.b))); }
fn down13(uv: v2, first: bool) -> v3 {
    let e = 1.0 / v2(textureDimensions(tex0));
    let a = tap0(uv + e * v2(-2.0, -2.0)).rgb; let b = tap0(uv + e * v2(0.0, -2.0)).rgb; let c = tap0(uv + e * v2(2.0, -2.0)).rgb;
    let d = tap0(uv + e * v2(-1.0, -1.0)).rgb; let f = tap0(uv + e * v2(1.0, -1.0)).rgb;
    let g = tap0(uv + e * v2(-2.0, 0.0)).rgb; let h = tap0(uv).rgb; let i = tap0(uv + e * v2(2.0, 0.0)).rgb;
    let j = tap0(uv + e * v2(-1.0, 1.0)).rgb; let k = tap0(uv + e * v2(1.0, 1.0)).rgb;
    let l = tap0(uv + e * v2(-2.0, 2.0)).rgb; let m = tap0(uv + e * v2(0.0, 2.0)).rgb; let n = tap0(uv + e * v2(2.0, 2.0)).rgb;
    let q0 = (d + f + j + k) * 0.25; let q1 = (a + b + g + h) * 0.25; let q2 = (b + c + h + i) * 0.25;
    let q3 = (g + h + l + m) * 0.25; let q4 = (h + i + m + n) * 0.25;
    if (first) {
        let w0 = karis(q0) * 0.5; let w1 = karis(q1) * 0.125; let w2 = karis(q2) * 0.125; let w3 = karis(q3) * 0.125; let w4 = karis(q4) * 0.125;
        return (q0 * w0 + q1 * w1 + q2 * w2 + q3 * w3 + q4 * w4) / (w0 + w1 + w2 + w3 + w4);
    }
    return q0 * 0.5 + (q1 + q2 + q3 + q4) * 0.125;
}
fn up9(uv: v2) -> v3 {
    let e = 1.0 / v2(textureDimensions(tex0));
    var c = tap0(uv).rgb * 4.0;
    c += (tap0(uv + v2(e.x, 0.0)).rgb + tap0(uv - v2(e.x, 0.0)).rgb + tap0(uv + v2(0.0, e.y)).rgb + tap0(uv - v2(0.0, e.y)).rgb) * 2.0;
    c += tap0(uv + e).rgb + tap0(uv - e).rgb + tap0(uv + v2(e.x, -e.y)).rgb + tap0(uv + v2(-e.x, e.y)).rgb;
    return c / 16.0 + tap1(uv).rgb;
}
fn store_level(id: vec2<u32>, c: v3) {
    let d = textureDimensions(output);
    if (id.x < d.x && id.y < d.y) { textureStore(output, id, v4(c, 1.0)); }
}
// jewels only, Karis-weighted
fn down_glow(uv: v2) -> v3 {
    let e = 1.0 / v2(textureDimensions(tex0));
    var c = v3(0.0);
    var ws = 0.0;
    for (var k = 0; k < 4; k++) {
        let s = glow_of(uv + e * v2(f32(k & 1) - 0.5, f32(k >> 1u) - 0.5));
        let w = karis(s);
        c += s * w;
        ws += w;
    }
    return c / ws;
}
@compute @workgroup_size(16, 16, 1) fn bloom_d1(@builtin(global_invocation_id) id: vec3<u32>) { store_level(id.xy, down_glow(px_uv(id.xy))); }
@compute @workgroup_size(16, 16, 1) fn bloom_d2(@builtin(global_invocation_id) id: vec3<u32>) { store_level(id.xy, down13(px_uv(id.xy), false)); }
@compute @workgroup_size(16, 16, 1) fn bloom_d3(@builtin(global_invocation_id) id: vec3<u32>) { store_level(id.xy, down13(px_uv(id.xy), false)); }
@compute @workgroup_size(16, 16, 1) fn bloom_d4(@builtin(global_invocation_id) id: vec3<u32>) { store_level(id.xy, down13(px_uv(id.xy), false)); }
@compute @workgroup_size(16, 16, 1) fn bloom_u3(@builtin(global_invocation_id) id: vec3<u32>) { store_level(id.xy, up9(px_uv(id.xy))); }
@compute @workgroup_size(16, 16, 1) fn bloom_u2(@builtin(global_invocation_id) id: vec3<u32>) { store_level(id.xy, up9(px_uv(id.xy))); }
@compute @workgroup_size(16, 16, 1) fn bloom_u1(@builtin(global_invocation_id) id: vec3<u32>) { store_level(id.xy, up9(px_uv(id.xy))); }

@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let uv = px_uv(id.xy);
    var c = textureLoad(tex0, vec2<i32>(id.xy), 0).rgb;
    c += tap1(uv).rgb * p.bloom * 0.5;
    // soft shoulder above 0.8
    c = select(c, 0.8 + 0.2 * (1.0 - exp(-(c - 0.8) / 0.2)), c > v3(0.8));
    textureStore(output, id.xy, v4(pow(max(c, v3(0.0)), v3(1.0 / max(p.gamma, 0.1))), 1.0));
}

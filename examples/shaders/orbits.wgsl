// Orbits: orbit-trap Mandelbrot as a lit relief with GGX metal shading
// Enes Altun, 2026; CC 4.0
// Trap technique: https://iquilezles.org/articles/ftrapsgeometric/
// Deep zoom by perturbation around one reference orbit, with rebasing (Zhuoran, 2021).
// Normals are exact: the height is built from the potential and the distance estimate, and both
// have analytic gradients from the iteration derivative dz.
struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
const PI = 3.14159265;
@group(0) @binding(0) var<uniform> u_time: TimeUniform;
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: Params;

struct MouseUniform {
    position: vec2<f32>,
    click_position: vec2<f32>,
    wheel: vec2<f32>,
    buttons: vec2<u32>,
};
@group(2) @binding(0) var<uniform> u_mouse: MouseUniform;
// reference orbit Z_n at the view centre, computed in f64 on the host
@group(2) @binding(1) var<storage, read_write> ref_orbit: array<vec2<f32>>;
// per-pixel surface slope in screen pixels (dh/dx, dh/dy with y down), packed as two halves
@group(2) @binding(2) var<storage, read_write> slope_buf: array<u32>;

@group(3) @binding(0) var input_texture0: texture_2d<f32>;
@group(3) @binding(1) var input_sampler0: sampler;

struct Params {
    base_color: vec3<f32>,
    // complex-plane size of one pixel
    pixel_size: f32,
    palette_a: vec3<f32>,
    gamma_correction: f32,
    palette_b: vec3<f32>,
    iteration: i32,
    light_color: vec3<f32>,
    aa: i32,
    rim_color: vec3<f32>,
    ref_len: i32,
    col_ext: f32,
    trap_pow: f32,
    trap_x: f32,
    trap_y: f32,
    trap_c1: f32,
    trap_s1: f32,
    wave_speed: f32,
    fold_intensity: f32,
    // light azimuth and elevation, degrees
    light_az: f32,
    light_el: f32,
    spec_str: f32,
    rim_str: f32,
    ao_str: f32,
    relief: f32,
    ridge_amp: f32,
    ridge_freq: f32,
    // plateau thickness on the set's filaments
    plateau: f32,
    // soft shadow hardness and reach in pixels
    shadow_soft: f32,
    shadow_len: f32,
    bounce_str: f32,
    roughness: f32,
    metallic: f32,
    reflection: f32,
    // relief inside the set, from the orbit's closest pass by trap 2
    interior: f32,
};

// the distance unit the original look was tuned in, in pixels
const DE_UNIT: f32 = 0.00271;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
fn cdiv(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return vec2(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / max(dot(b, b), 1e-30); }

struct Orbit {
    // smooth iteration count, distance estimate in pixels, trap 1 average, trap 2 minimum
    s: f32,
    de: f32,
    t1: f32,
    t2: f32,
    inside: bool,
    // height and its gradient per pixel (y up)
    h: f32,
    grad: vec2<f32>,
};

// z_n = Z_n + d_n around the reference: d' = 2 Z d + d^2 + dc. When z comes closer to 0 than d,
// or the reference runs out, continue from z itself on the reference start (rebasing)
fn orbit(dc: vec2<f32>, t1: vec2<f32>, t2: vec2<f32>) -> Orbit {
    var dl = vec2(0.0);
    var z = vec2(0.0);
    var dz = vec2(0.0);
    var ri = 0;
    let rl = max(params.ref_len, 2);
    var n = 0.0;
    var r2 = 0.0;
    var escaped = false;
    var t1s = 0.0;
    var t1c = 0.0;
    var t2m = 1e20;
    var w_t = vec2(1.0, 0.0);
    var dz_t = vec2(1.0, 0.0);

    for (var i = 0; i < params.iteration; i++) {
        let dzn = 2.0 * cmul(z, dz) + vec2(1.0, 0.0);
        dl = cmul(2.0 * ref_orbit[ri] + dl, dl) + dc;
        ri++;
        z = ref_orbit[ri] + dl;
        dz = dzn;
        n += 1.0;
        r2 = dot(z, z);

        // orbit traps
        let d1 = length(z - t1);
        let f = 1.0 - smoothstep(0.6, 1.4, d1);
        t1c += f;
        t1s += f * d1;
        let w = z - t2;
        if (dot(w, w) < t2m) {
            t2m = dot(w, w);
            w_t = w;
            dz_t = dz;
        }

        if (r2 > 65536.0) {
            escaped = true;
            break;
        }
        if (r2 < dot(dl, dl) || ri >= rl - 1) {
            dl = z;
            ri = 0;
        }
    }

    var o: Orbit;
    o.t1 = t1s / max(t1c, 1.0);
    o.t2 = t2m;
    o.inside = !escaped;
    let px = params.pixel_size;

    if (escaped) {
        // log|z| and its gradient per pixel: grad Re(log z) = conj(dz / z)
        let lz = 0.5 * log(r2);
        let q = cdiv(dz, z);
        let g_l = vec2(q.x, -q.y) * px;
        o.s = n + 1.0 - log2(lz);
        o.de = sqrt(r2) * lz / max(length(dz), 1e-30) / px;
        let g_s = -g_l / (lz * 0.6931472);
        // the distance grows along the potential, one pixel per pixel
        let g_d = g_l / max(length(g_l), 1e-30);

        // plateau on the filaments plus iteration terraces near the set
        let D = o.de;
        let ep = exp(-0.9485 * D);
        let x = clamp(D / 1.845, 0.0, 1.0);
        let so = x * x * (3.0 - 2.0 * x);
        let dso = select(0.0, 6.0 * x * (1.0 - x) / 1.845, D < 1.845);
        let nr = exp(-0.0542 * D);
        let fr = params.ridge_freq;
        let sn = sin(o.s * fr);
        o.h = params.plateau * ep + params.ridge_amp * sn * so * nr;
        let dh_d = -0.9485 * params.plateau * ep + params.ridge_amp * sn * nr * (dso - 0.0542 * so);
        let dh_s = params.ridge_amp * fr * cos(o.s * fr) * so * nr;
        o.grad = dh_d * g_d + dh_s * g_s;
    } else {
        o.s = n;
        o.de = 0.0;
        // how far, in pixels, from where the orbit hits trap 2: 1 / |grad log|w||
        let q = cdiv(dz_t, w_t);
        let g_w = vec2(q.x, -q.y) * px;
        let lw = max(length(g_w), 1e-6);
        let e = exp(-0.05 / lw);
        o.h = params.plateau + params.interior * (1.0 - e);
        o.grad = params.interior * 0.05 * e * g_w / lw;
    }
    return o;
}

// orbit trap coloring: based tech https://iquilezles.org/articles/ftrapsgeometric/
fn get_base_color(o: Orbit) -> vec3<f32> {
    if (o.inside) {
        return params.base_color * 0.35 + vec3<f32>(0.03);
    }
    let ir = o.s / f32(params.iteration);
    let c1 = pow(clamp(2.0 * o.de * DE_UNIT, 0.0, 1.0), 0.5);
    let c2 = pow(clamp(1.5 * o.t1, 0.0, 1.0), 2.0);
    let c3 = pow(clamp(0.4 * o.t2, 0.0, 1.0), 0.25);

    let cl1 = 0.5 + 0.5 * sin(vec3(3.0) + 4.0 * c2 + params.palette_a);
    let cl2 = 0.5 + 0.5 * sin(vec3(4.1) + 2.0 * c3 + params.palette_b);
    let bc = 2.0 * sqrt(c1 * cl1 * cl2);

    let ec = 0.5 + 0.5 * cos(params.col_ext * o.t2 + o.t1 + params.base_color + PI * 6.0 * ir + params.trap_pow);
    let pc = mix(bc, ec, smoothstep(0.2, params.trap_s1, ir));
    let vc = pc * (0.5 + 0.5 * sin(PI * vec3(0.5, 0.7, 0.9) * ir + 1.0));
    return mix(pc, vc, params.trap_c1);
}

// pass 1: albedo + height to the texture, slope to the buffer
@compute @workgroup_size(16, 16, 1)
fn compute_fractal(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ss = vec2<f32>(textureDimensions(output));
    let coords = global_id.xy;
    if (coords.x >= u32(ss.x) || coords.y >= u32(ss.y)) { return; }
    // y up, like the complex plane
    let frag = vec2(f32(coords.x) + 0.5, ss.y - f32(coords.y) - 0.5);
    let t = u_time.time;
    let t1 = vec2(0.0, params.fold_intensity);
    let t2 = vec2(params.trap_x, params.trap_y) + params.wave_speed * vec2(cos(0.3 * t), sin(0.3 * t));

    let AA = max(params.aa, 1);
    var col = vec3(0.0);
    var h = 0.0;
    var g = vec2(0.0);
    for (var m = 0; m < AA; m++) {
        for (var k = 0; k < AA; k++) {
            let so = (vec2(f32(m), f32(k)) + 0.5) / f32(AA) - 0.5;
            let dc = (frag + so - 0.5 * ss) * params.pixel_size;
            let o = orbit(dc, t1, t2);
            col += get_base_color(o);
            h += o.h;
            g += o.grad;
        }
    }
    let inv = 1.0 / f32(AA * AA);
    textureStore(output, coords, vec4(col * inv, h * inv));
    slope_buf[coords.y * u32(ss.x) + coords.x] = pack2x16float(vec2(g.x, -g.y) * inv);
}

// gamma
fn gam(c: vec3<f32>, g: f32) -> vec3<f32> {
    return pow(c, vec3(1.0 / g));
}

fn aces(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3(0.0), vec3(1.0));
}

// ggx ndf
fn D_ggx(NoH: f32, a: f32) -> f32 {
    let a2 = a * a;
    let d = NoH * NoH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d + 1e-6);
}
// smith geometry
fn G_smith(NoV: f32, NoL: f32, a: f32) -> f32 {
    let k = a * 0.5;
    let gv = NoV / (NoV * (1.0 - k) + k);
    let gl = NoL / (NoL * (1.0 - k) + k);
    return gv * gl;
}
// schlick fresnel
fn F_schlick(VoH: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (vec3(1.0) - F0) * pow(clamp(1.0 - VoH, 0.0, 1.0), 5.0);
}

fn light_dir() -> vec3<f32> {
    let az = radians(params.light_az);
    let el = radians(params.light_el);
    return vec3(cos(el) * cos(az), cos(el) * sin(az), sin(el));
}

// studio environment: sky gradient, a key softbox along the light, two coloured strips and a
// horizon glow. Rougher surfaces see broader, dimmer lobes, so the energy stays about the same
fn env(d: vec3<f32>, rough: f32) -> vec3<f32> {
    let L = light_dir();
    var c = mix(vec3(0.05, 0.045, 0.04), vec3(0.45, 0.52, 0.65), smoothstep(-0.1, 0.9, d.z));
    let sharp = mix(160.0, 5.0, rough);
    let energy = (sharp + 1.0) / 161.0;
    c += params.light_color * 3.0 * energy * pow(max(dot(d, L), 0.0), sharp);
    let az = radians(params.light_az);
    for (var i = 0; i < 2; i++) {
        let a = az + select(-2.1, 2.1, i == 1);
        let s = normalize(vec3(cos(a) * 0.85, sin(a) * 0.85, 0.5));
        let tint = select(vec3(0.85, 0.9, 1.0), params.rim_color, i == 1);
        c += tint * 1.2 * energy * pow(max(dot(d, s), 0.0), sharp * 0.6);
    }
    let hz = (d.z - 0.06) / (0.04 + 0.3 * rough);
    c += vec3(0.6, 0.55, 0.5) * 0.5 * exp(-hz * hz);
    return c;
}

fn height_at(p: vec2<f32>, R: vec2<f32>) -> f32 {
    // the pass sampler repeats, so stay half a texel inside
    let uv = clamp(p / R, 0.5 / R, 1.0 - 0.5 / R);
    return textureSampleLevel(input_texture0, input_sampler0, uv, 0.0).a;
}

// pass 2: soft shadows, AO and PBR shading. in0 = albedo + height
@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let R = vec2<f32>(textureDimensions(output));
    let coords = global_id.xy;
    if (coords.x >= u32(R.x) || coords.y >= u32(R.y)) { return; }
    let pc = vec2<f32>(coords) + 0.5;

    let base = textureLoad(input_texture0, coords, 0);
    let g = unpack2x16float(slope_buf[coords.y * u32(R.x) + coords.x]);
    // height in pixels, the same scale for normals, shadows and AO
    let hs = params.relief * 50.0;
    let H0 = base.a * hs;
    let N = normalize(vec3(-g.x * hs, g.y * hs, 1.0));
    let L = light_dir();
    let V = vec3(0.0, 0.0, 1.0);
    let H = normalize(L + V);
    let NoL = max(dot(N, L), 0.0);
    let NoV = max(dot(N, V), 0.0);
    let NoH = max(dot(N, H), 0.0);
    let VoH = max(dot(V, H), 0.0);

    // soft shadow: march toward the light over the height field, keep the closest pass
    let sdir = normalize(vec2(L.x, -L.y));
    let rise = L.z / max(length(L.xy), 1e-3);
    var shadow = 1.0;
    for (var i = 1; i <= 32; i++) {
        let t = max(params.shadow_len * pow(f32(i) / 32.0, 1.6), f32(i) * 0.5);
        let p = pc + sdir * t;
        if (any(p < vec2(0.0)) || any(p > R)) { break; }
        let clearance = H0 + t * rise - height_at(p, R) * hs;
        shadow = min(shadow, params.shadow_soft * clearance / t);
        if (shadow <= 0.0) { break; }
    }
    shadow = clamp(shadow, 0.0, 1.0);

    // ambient occlusion: how much the surroundings rise above this point
    var occ = 0.0;
    var wsum = 0.0;
    for (var k = 0; k < 8; k++) {
        let ang = f32(k) * 0.7853982;
        let dir = vec2(cos(ang), sin(ang));
        for (var s = 0; s < 3; s++) {
            let r = 3.0 + f32(s) * 5.0;
            let w = 1.0 / (1.0 + r * 0.15);
            occ += clamp((height_at(pc + dir * r, R) * hs - H0) / r, 0.0, 1.0) * w;
            wsum += w;
        }
    }
    let ao = 1.0 - clamp(occ / wsum * 2.0, 0.0, 1.0) * params.ao_str;

    let alb = max(base.rgb, vec3(0.02));
    let rough = clamp(params.roughness, 0.04, 1.0);
    let a = rough * rough;
    let F0 = mix(vec3(0.04), alb, params.metallic);
    let spec = min(D_ggx(NoH, a) * G_smith(NoV, NoL, a) * F_schlick(VoH, F0) / max(4.0 * NoV * NoL, 1e-4) * params.spec_str, vec3(6.0));

    // key light, sky from the environment, and albedo-tinted light bleeding into the shadows
    let direct = NoL * shadow * params.light_color;
    let bounce = alb * params.light_color * params.bounce_str * 0.12 * (1.0 - shadow) * (0.5 + 0.5 * dot(N, L));
    var col = alb * (env(N, 1.0) * ao * 0.6 + direct + bounce) * mix(1.0, 0.4, params.metallic);
    col += spec * NoL * shadow * params.light_color;

    // environment reflection
    let refl = reflect(-V, N);
    col += F_schlick(NoV, F0) * env(refl, rough) * params.reflection * ao;

    // back rim: glow on edges facing away from the key
    let rim_dir = normalize(vec3(-L.xy, 0.4));
    col += pow(1.0 - NoV, 3.0) * max(dot(N, rim_dir), 0.0) * (1.0 - NoL) * params.rim_str * params.rim_color;

    // post
    col = gam(aces(col), params.gamma_correction);
    let uv = pc / R;
    col *= 0.7 + 0.3 * pow(16.0 * uv.x * uv.y * (1.0 - uv.x) * (1.0 - uv.y), 0.15);

    textureStore(output, coords, vec4(col, 1.0));
}

// Orbits 3D orbit trap Mandelbrot with PBR metal shading
// Enes Altun, 2026; CC 4.0
// Trap technique: https://iquilezles.org/articles/ftrapsgeometric/
struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
const PI = 3.14159;
@group(0) @binding(0) var<uniform> u_time: TimeUniform;
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: Params;

@group(3) @binding(0) var input_texture0: texture_2d<f32>;
@group(3) @binding(1) var input_sampler0: sampler;
@group(3) @binding(2) var input_texture1: texture_2d<f32>;
@group(3) @binding(3) var input_sampler1: sampler;

struct MouseUniform {
    position: vec2<f32>,
    click_position: vec2<f32>,
    wheel: vec2<f32>,
    buttons: vec2<u32>,
};
@group(2) @binding(0) var<uniform> u_mouse: MouseUniform;

struct Params {
    base_color: vec3<f32>,
    x: f32,
    rim_color: vec3<f32>,
    y: f32,
    accent_color: vec3<f32>,
    gamma_correction: f32,
    travel_speed: f32,
    iteration: i32,
    col_ext: f32,
    zoom: f32,
    trap_pow: f32,
    trap_x: f32,
    trap_y: f32,
    trap_c1: f32,
    aa: i32,
    trap_s1: f32,
    wave_speed: f32,
    fold_intensity: f32,
    lightdir_x: f32,
    lightdir_y: f32,
    spec_pow: f32,
    spec_str: f32,
    rim_str: f32,
    ao_str: f32,
    height_scale: f32,
    light_r: f32,
    light_g: f32,
    light_b: f32,
    ridge_amp: f32,
    ridge_freq: f32,
    _pad2: f32,
    plateau_height: f32,
    shadow_str: f32,
    shadow_dist: f32,
    bounce_str: f32,
    roughness: f32,
    metallic: f32,
    reflection: f32,
    rim_r: f32,
    rim_g: f32,
    rim_b: f32,
    _pad1: f32,
};

fn op(mn: f32, mx: f32, i: f32, p: f32, t: f32) -> f32 {
    let c = 2. * i + p;
    let m = t % c;
    if(m < i) {
        return mix(mx, mn, .5 - .5 * cos(PI * m / i));
    } else if(m < i + p) {
        return mn;
    } else {
        return mix(mn, mx, .5 - .5 * cos(PI * (m - i - p) / i));
    }
}

// mandelbrot iteration: returns (smooth iter, distance, trap1, trap2)
fn im(c: vec2<f32>, t1: vec2<f32>, t2: vec2<f32>, t: f32) -> vec4<f32> {
    var z = vec2(0.);
    var dz = vec2(0., 0.);
    var exterior = false;
    var r2 = 0.0;

    var t1s = 0.;
    var t1c = 0.;
    var t2m = 1e20;

    let mi = params.iteration;
    var n = 0.0;

    for(var i = 0; i < mi; i++) {
        dz = 2.0 * vec2(z.x * dz.x - z.y * dz.y, z.x * dz.y + z.y * dz.x) + vec2(1.0, 0.0);
        z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;

        n += 1.0;
        r2 = dot(z, z);

        // orbit traps
        let d1 = length(z - t1);
        let f = 1. - smoothstep(.6, 1.4, d1);
        t1c += f;
        t1s += f * d1;
        t2m = min(t2m, dot(z - t2, z - t2));
        // dist estimate
        if(r2 > 65536.0) {
            exterior = true;
            break;
        }
    }

    var d = 0.0;
    var smooth_i = n;

    if (exterior) {
        let base = 0.5 * sqrt(r2 / dot(dz, dz));
        if (n < 128.0) {
            // IQ estimator 1
            let en = exp2(n);
            d = base * en * (1.0 - pow(r2, -1.0 / en));
        } else {
            // exp2(n) overflows f32 at n >= 128; its large-n limit is the log estimator
            d = base * log(r2);
        }
        smooth_i = n + 1.0 - log2(log(r2) * 0.5);
    }

    return vec4(smooth_i, d, t1s / max(t1c, 1.), t2m);
}

// height field from distance + traps
fn hmap(zd: vec4<f32>, zl: f32) -> f32 {
    let smooth_iter = zd.x;
    let de_norm = zd.y / (zl + 1e-5);

    // plateau: high on the set, falls off outside
    let h_plateau = exp(-de_norm * 350.0) * params.plateau_height;

    // iteration terraces, confined to the structure so the open exterior doesn't ring
    // well at least the best solution from my side so far :/
    let is_outside = smoothstep(0.0, 0.005, de_norm);
    let near = exp(-de_norm * 20.0);
    let h_ridges = sin(smooth_iter * params.ridge_freq) * params.ridge_amp * is_outside * near;


    return h_plateau + h_ridges;
}

// orbit trap coloring: based tech https://iquilezles.org/articles/ftrapsgeometric/
fn get_base_color(zd: vec4<f32>, t01: f32, zl: f32) -> vec3<f32> {
    let ir = zd.x / f32(params.iteration);
    if(ir >= 0.99) {
        return params.base_color * 0.35 + vec3<f32>(0.03);
    }
    
    let c1 = pow(clamp(2. * zd.y / zl, 0., 1.), .5);
    let c2 = pow(clamp(1.5 * zd.z, 0., 1.), 2.);
    let c3 = pow(clamp(.4 * zd.w, 0., 1.), .25);
    
    let cl1 = .5 + .5 * sin(vec3(3.) + 4. * c2 + params.rim_color);
    let cl2 = .5 + .5 * sin(vec3(4.1) + 2. * c3 + params.accent_color);
    
    let bc = 2. * sqrt(c1 * cl1 * cl2);
    let te = op(params.trap_pow, params.trap_pow, 6., 0., t01);
    
    let ec = .5 + .5 * cos(params.col_ext * zd.w + zd.z + params.base_color + PI * 6. * ir + te);
    let bf = smoothstep(.2, params.trap_s1, ir);
    let pc = mix(bc, ec, bf);
    let tce = op(1., 1., 10., 5., t01);
    
    let vc = pc * (.5 + .5 * sin(PI * vec3(.5, .7, .9) * ir + tce));
    return mix(pc, vc, params.trap_c1);
}

// pass 1: fractal color + height (in alpha)
@compute @workgroup_size(16, 16, 1)
fn compute_fractal(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ss = vec2<f32>(textureDimensions(output));
    let coords = vec2<u32>(global_id.xy);
    if (coords.x >= u32(ss.x) || coords.y >= u32(ss.y)) { return; }
    let frag = vec2(f32(coords.x), ss.y - f32(coords.y));
    let AA = params.aa;
    let t = u_time.time;
    let t01 = t * .4;
    let cp = vec2(sin(.0002 * t * params.travel_speed / 10.), cos(.0002 * t * params.travel_speed / 10.));
    var p = vec2(.8085, .2607);
    if(t > 117.) { p.y += .00001 * (t - 1.); }
    let zl = op(.0005, .0005, 10., 5., t01);
    let t1 = vec2(0., params.fold_intensity);
    let t2 = vec2(params.trap_x, params.trap_y) + params.wave_speed * vec2(cos(.3 * t), sin(.3 * t));
    var col_acc = vec3(0.);
    var h_acc = 0.0;
    for(var m = 0; m < AA; m++) {
        for(var n = 0; n < AA; n++) {
            let so = vec2(f32(m), f32(n)) / f32(AA);
            let mr = min(ss.x, ss.y);
            let uv = ((frag + so - .5 * ss) / mr * params.zoom + p + cp) * 2.033 - vec2(params.x, params.y);
            let zd = im(uv, t1, t2, t01);
            col_acc += get_base_color(zd, t01, zl);
            h_acc += hmap(zd, zl);
        }
    }
    let final_col = col_acc / f32(AA * AA);
    let final_h = h_acc / f32(AA * AA);
    // dither to hide 16bit banding..
    let dn = 0.002 + abs(final_h) * 0.002;
    let dither = (fract(sin(dot(vec2<f32>(coords), vec2(12.9898, 78.233))) * 43758.5453) - 0.5) * dn;
    textureStore(output, coords, vec4(final_col, final_h + dither));
}

// pass 2: normal gradient (sobel) + multi-scale AO -> (dx, dy, ao, height)
@compute @workgroup_size(16, 16, 1)
fn prep(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let R = vec2<f32>(textureDimensions(output));
    let coords = vec2<u32>(global_id.xy);
    if (coords.x >= u32(R.x) || coords.y >= u32(R.y)) { return; }

    let uv = vec2<f32>(coords) / R;
    let px = 1.0 / R;
    let h_c = textureSampleLevel(input_texture0, input_sampler0, uv, 0.0).a;

    // 3x3 sobel
    let off = 2.0;
    let dxu = vec2(px.x * off, 0.0);
    let dyu = vec2(0.0, px.y * off);
    let h00 = textureSampleLevel(input_texture0, input_sampler0, uv - dxu - dyu, 0.0).a;
    let h10 = textureSampleLevel(input_texture0, input_sampler0, uv - dyu, 0.0).a;
    let h20 = textureSampleLevel(input_texture0, input_sampler0, uv + dxu - dyu, 0.0).a;
    let h01 = textureSampleLevel(input_texture0, input_sampler0, uv - dxu, 0.0).a;
    let h21 = textureSampleLevel(input_texture0, input_sampler0, uv + dxu, 0.0).a;
    let h02 = textureSampleLevel(input_texture0, input_sampler0, uv - dxu + dyu, 0.0).a;
    let h12 = textureSampleLevel(input_texture0, input_sampler0, uv + dyu, 0.0).a;
    let h22 = textureSampleLevel(input_texture0, input_sampler0, uv + dxu + dyu, 0.0).a;
    let dx = ((h20 + 2.0 * h21 + h22) - (h00 + 2.0 * h01 + h02)) / (8.0 * off);
    let dy = ((h02 + 2.0 * h12 + h22) - (h00 + 2.0 * h10 + h20)) / (8.0 * off);

    // multi scale AO
    var occ = 0.0;
    var wsum = 0.0;
    for (var k = 0; k < 8; k++) {
        let ang = f32(k) * 0.7853982;
        let dir = vec2(cos(ang), sin(ang));
        for (var s = 0; s < 3; s++) {
            let r = (3.0 + f32(s) * 5.0);
            let hs = textureSampleLevel(input_texture0, input_sampler0, uv + dir * r * px, 0.0).a;
            let w = 1.0 / (1.0 + r * 0.15);
            occ += max(0.0, hs - h_c) * w;
            wsum += w;
        }
    }
    let ao = 1.0 - clamp(occ / max(wsum, 1e-4) * 3.0, 0.0, 1.0);

    textureStore(output, coords, vec4(dx, dy, ao, h_c));
}

// gamma
fn g(c: vec3<f32>, g: f32) -> vec3<f32> {
    return pow(c, vec3(1. / g));
}

fn aces(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3(0.), vec3(1.));
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

// pass 3: PBR metal shading. in0 = prep geometry, in1 = albedo
@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let R = vec2<f32>(textureDimensions(output));
    let coords = vec2<u32>(global_id.xy);
    if (coords.x >= u32(R.x) || coords.y >= u32(R.y)) { return; }

    let uv = vec2<f32>(coords) / R;
    let px = 1.0 / R;

    let geo = textureSampleLevel(input_texture0, input_sampler0, uv, 0.0);
    let dx = geo.r;
    let dy = geo.g;
    let ao_raw = geo.b;
    let h_c = geo.a;
    let base_albedo = textureSampleLevel(input_texture1, input_sampler1, uv, 0.0).rgb;

    let amp = params.height_scale * 50.0;
    let N = normalize(vec3(-dx * amp, -dy * amp, 1.0));

    let light_color = vec3<f32>(params.light_r, params.light_g, params.light_b);
    let l_dir_2d = normalize(vec2(params.lightdir_x, params.lightdir_y));
    let L = normalize(vec3(params.lightdir_x, params.lightdir_y, 1.0));
    let V = vec3(0.0, 0.0, 1.0);
    let H = normalize(L + V);

    let NoL = max(dot(N, L), 0.0);
    let NoV = max(dot(N, V), 0.0);
    let NoH = max(dot(N, H), 0.0);
    let VoH = max(dot(V, H), 0.0);

    // contact shadows
    var shadow_acc = 0.0;
    for(var i = 1; i <= 14; i++) {
        let t = f32(i) * params.shadow_dist * 0.5;
        let h_sample = textureSampleLevel(input_texture0, input_sampler0, uv + l_dir_2d * t * px, 0.0).a;
        let ray_z = h_c + t * 0.02;
        if (h_sample > ray_z) {
            shadow_acc += (h_sample - ray_z) * params.shadow_str;
        }
    }
    let shadow = clamp(1.0 - shadow_acc, 0.0, 1.0);
    let ao = mix(1.0, ao_raw, params.ao_str);

    let alb = max(base_albedo, vec3(0.02));

    // ggx specular
    let rough = clamp(params.roughness, 0.04, 1.0);
    let a = rough * rough;
    let F0 = mix(vec3(0.04), alb, params.metallic);
    let D = D_ggx(NoH, a);
    let Gt = G_smith(NoV, NoL, a);
    let F = F_schlick(VoH, F0);
    let spec = min((D * Gt) * F / max(4.0 * NoV * NoL, 1e-4) * params.spec_str, vec3(6.0));

    // hemispheric env
    let sky = vec3(0.55, 0.65, 0.82);
    let gnd = vec3(0.12, 0.10, 0.09);
    let env = mix(gnd, sky, N.z * 0.5 + 0.5);

    // colored body
    let diff_amt = mix(1.0, 0.4, params.metallic);
    let direct = NoL * shadow * light_color;
    let ambient = env * ao;
    var col = alb * (ambient * 0.6 + direct) * diff_amt;

    // specular
    col += spec * NoL * shadow * light_color;

    // env reflection
    let fres = F_schlick(NoV, F0);
    col += fres * env * params.reflection * ao;

    // back rim: glow on shadowed edges facing away from the key
    let rim_dir = normalize(vec3(-l_dir_2d, 0.4));
    let rim = pow(1.0 - NoV, 3.0) * max(dot(N, rim_dir), 0.0) * (1.0 - NoL) * params.rim_str;
    col += rim * vec3<f32>(params.rim_r, params.rim_g, params.rim_b);

    // post
    col = aces(col);
    col = g(col, params.gamma_correction);
    let frag_uv = vec2(f32(coords.x), R.y - f32(coords.y)) / R;
    col *= .7 + .3 * pow(16. * frag_uv.x * frag_uv.y * (1. - frag_uv.x) * (1. - frag_uv.y), .15);

    textureStore(output, coords, vec4(col, 1.0));
}
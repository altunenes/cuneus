// Lichtenberg noise math inspired by: Lichtenberg figure by rory618 2018, https://www.shadertoy.com/view/3sl3WH

struct TimeUniform { time: f32, delta: f32, frame: u32, _padding: u32 };
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: LichParams;

@group(2) @binding(0) var<storage, read_write> atm: array<atomic<u32>>;

@group(3) @binding(0) var input_texture0: texture_2d<f32>;
@group(3) @binding(1) var input_sampler0: sampler;
@group(3) @binding(2) var input_texture1: texture_2d<f32>;
@group(3) @binding(3) var input_sampler1: sampler;

struct LichParams {
    cloud_density: f32,
    lightning_intensity: f32,
    branch_count: f32,
    feedback_decay: f32,
    base_color: vec3<f32>,
    glow_intensity: f32,
    specular_strength: f32,
    contrast: f32,
    gamma: f32,
    saturation: f32,
    color_shift: f32,
    spectrum_mix: f32,
    light_intensity: f32,
    morph: f32,
    breathe: f32,
    dynamic: f32,
    _pad3: f32,
    _pad4: f32,
};

const AS: f32 = 1024.0;
const AI: f32 = 1.0 / 1024.0;

// --- CIE 1931 chromaticity LUT ---
const spectrum = array<vec3<f32>, 45>(
    vec3<f32>(0.002362, 0.000253, 0.010482), vec3<f32>(0.019110, 0.002004, 0.086011),
    vec3<f32>(0.084736, 0.008756, 0.389366), vec3<f32>(0.204492, 0.021391, 0.972542),
    vec3<f32>(0.314679, 0.038676, 1.553480), vec3<f32>(0.383734, 0.062077, 1.967280),
    vec3<f32>(0.370702, 0.089456, 1.994800), vec3<f32>(0.302273, 0.128201, 1.745370),
    vec3<f32>(0.195618, 0.185190, 1.317560), vec3<f32>(0.080507, 0.253589, 0.772125),
    vec3<f32>(0.016172, 0.339133, 0.415254), vec3<f32>(0.003816, 0.460777, 0.218502),
    vec3<f32>(0.037465, 0.606741, 0.112044), vec3<f32>(0.117749, 0.761757, 0.060709),
    vec3<f32>(0.236491, 0.875211, 0.030451), vec3<f32>(0.376772, 0.961988, 0.013676),
    vec3<f32>(0.529826, 0.991761, 0.003988), vec3<f32>(0.705224, 0.997340, 0.000000),
    vec3<f32>(0.878655, 0.955552, 0.000000), vec3<f32>(1.014160, 0.868934, 0.000000),
    vec3<f32>(1.118520, 0.777405, 0.000000), vec3<f32>(1.123990, 0.658341, 0.000000),
    vec3<f32>(1.030480, 0.527963, 0.000000), vec3<f32>(0.856297, 0.398057, 0.000000),
    vec3<f32>(0.647467, 0.283493, 0.000000), vec3<f32>(0.431567, 0.179828, 0.000000),
    vec3<f32>(0.268329, 0.107633, 0.000000), vec3<f32>(0.152568, 0.060281, 0.000000),
    vec3<f32>(0.081261, 0.031800, 0.000000), vec3<f32>(0.040851, 0.015905, 0.000000),
    vec3<f32>(0.019941, 0.007749, 0.000000), vec3<f32>(0.009577, 0.003718, 0.000000),
    vec3<f32>(0.004553, 0.001768, 0.000000), vec3<f32>(0.002175, 0.000846, 0.000000),
    vec3<f32>(0.001045, 0.000407, 0.000000), vec3<f32>(0.000508, 0.000199, 0.000000),
    vec3<f32>(0.000251, 0.000098, 0.000000), vec3<f32>(0.000126, 0.000050, 0.000000),
    vec3<f32>(0.000065, 0.000025, 0.000000), vec3<f32>(0.000033, 0.000013, 0.000000),
    vec3<f32>(0.000018, 0.000007, 0.000000), vec3<f32>(0.000009, 0.000004, 0.000000),
    vec3<f32>(0.000005, 0.000002, 0.000000), vec3<f32>(0.000003, 0.000001, 0.000000),
    vec3<f32>(0.000002, 0.000001, 0.000000)
);
const xyz_to_rgb = mat3x3<f32>(
     3.2404542, -0.9692660,  0.0556434,
    -1.5371385,  1.8760108, -0.2040259,
    -0.4985314,  0.0415560,  1.0572252
);
fn wl_to_xyz(wl: f32) -> vec3<f32> {
    let x = (wl - 390.0) * 0.1;
    let index = u32(clamp(x, 0.0, 43.0));
    return mix(spectrum[index], spectrum[index + 1u], fract(x));
}
fn process_color(base_color: vec3<f32>, wave: f32, spectrum_mix: f32) -> vec3<f32> {
    let wl = 390.0 + 240.0 * wave;
    let spectral = max(vec3<f32>(0.0), xyz_to_rgb * wl_to_xyz(wl));
    return mix(base_color, spectral, spectrum_mix);
}

fn IHash(a: i32) -> i32 {
    var x = a;
    x = (x ^ 61) ^ (x >> 16);
    x = x + (x << 3);
    x = x ^ (x >> 4);
    x = x * 0x27d4eb;
    x = x ^ (x >> 15);
    return x;
}
fn Hash(a: i32) -> f32 { return f32(IHash(a)) / f32(0x7FFFFFFF); }
fn rand4(seed: i32) -> vec4<f32> {
    return vec4<f32>(Hash(seed^348593), Hash(seed^859375), Hash(seed^625384), Hash(seed^253625));
}
fn rand2(seed: i32) -> vec2<f32> { return vec2<f32>(Hash(seed^348593), Hash(seed^859375)); }
// uniform -> gaussian (Box-Muller)
fn randn(u: vec2<f32>) -> vec2<f32> {
    let r = sqrt(-2.0 * log(1e-9 + abs(u.x)));
    let a = u.y * 6.28318;
    return r * vec2<f32>(cos(a), sin(a));
}

fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn epoch_lt() -> vec3<f32> {
    let gt = 1.3 / (max(params.morph, 0.001) * 0.3);
    let cyc = gt + 2.0;
    let dyn = params.dynamic > 0.5;
    let ep = select(0.0, floor(time_data.time / cyc), dyn);
    return vec3<f32>(ep, time_data.time - ep * cyc, cyc);
}

fn dep(idx: u32, col: vec3<f32>, w: f32, st: u32) {
    atomicAdd(&atm[idx],        u32(col.r * w * AS));
    atomicAdd(&atm[idx + st],   u32(col.g * w * AS));
    atomicAdd(&atm[idx + 2u*st],u32(col.b * w * AS));
    atomicAdd(&atm[idx + 3u*st],u32(w * AS));
}

// Pass 1: one particle per threadd walk the branch process, splat the endpoint into the atomic field
@compute @workgroup_size(16, 16, 1)
fn splat(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let R = vec2<f32>(dims);
    let uw = dims.x; let st = dims.x * dims.y;
    let pid = i32(id.y * dims.x + id.x);

    let a0 = vec2<f32>(0.0);
    let b0 = R;
    let structSeed = IHash((i32(params.cloud_density) + i32(epoch_lt().x) * 7) * 61 + 3);
    let steps = i32(clamp(24.0 * params.branch_count, 4.0, 48.0));

    // several particles per thread; each stops at a random depth so the depth channel tags the tree root->tips
    for (var s = 0u; s < 4u; s = s + 1u) {
        var seed = structSeed;
        var seed2 = IHash((pid * 4 + i32(s)) ^ IHash(i32(time_data.frame) * 0x5da8d7));
        var a = a0; var b = b0;
        var c: vec2<f32>; var d: vec2<f32>;
        var col = vec3<f32>(1.0);
        for (var k = 0; k < steps; k = k + 1) {
            let l = length(b - a);
            c = (a + b) * 0.5 + l * randn(rand2(seed ^ bitcast<i32>(0x8593F4D5u))) / 6.0;
            d = (a + b) * 0.5 + l * randn(rand2(seed ^ bitcast<i32>(0x93D35DE5u)));
            let j = rand4(seed2 ^ IHash(pid * 4 + i32(s)));
            let d0 = length(a - c); let d1 = length(b - c); let d2 = length(c - d) * 0.25;
            let sm = d0 + d1 + d2 + 1e-6;
            if (j.x < d0 / sm) {
                b = c; seed = IHash(seed ^ 0x7d964ba9); seed2 = IHash(seed2 ^ 0x7d964ba9);
            } else if (j.x < (d0 + d1) / sm) {
                a = c; seed = IHash(seed ^ bitcast<i32>(0xb7798235u)); seed2 = IHash(seed2 ^ bitcast<i32>(0xb7798235u));
            } else {
                a = c; b = d; seed = IHash(seed ^ 0x5b2a74f5); seed2 = IHash(seed2 ^ 0x5b2a74f5);
                col *= vec3<f32>(0.95, 0.95, 0.956);
            }
        }
        var coord = mix(a, b, Hash(seed2)) + 0.5 * randn(rand2(seed2 ^ bitcast<i32>(0xAA91B4C3u)));
        let cx = coord.x; let cy = coord.y;
        if (cx > 0.0 && cy > 0.0 && cx < R.x - 1.0 && cy < R.y - 1.0) {
            let fx = fract(cx); let fy = fract(cy);
            let ix = u32(cx); let iy = u32(cy);
            dep(iy*uw + ix,           col, (1.0-fx)*(1.0-fy), st);
            dep(iy*uw + ix + 1u,      col, fx*(1.0-fy),       st);
            dep((iy+1u)*uw + ix,      col, (1.0-fx)*fy,       st);
            dep((iy+1u)*uw + ix + 1u, col, fx*fy,             st);
        }
    }
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(0.0));
}

@compute @workgroup_size(16, 16, 1)
fn resolve(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let st = dims.x * dims.y;
    let pi = id.y * dims.x + id.x;
    let r  = f32(atomicExchange(&atm[pi],        0u));
    let g  = f32(atomicExchange(&atm[pi + st],   0u));
    let bb = f32(atomicExchange(&atm[pi + 2u*st],0u));
    let w  = f32(atomicExchange(&atm[pi + 3u*st],0u));
    let fresh = vec4<f32>(r, g, bb, w) * AI;
    var prev = textureLoad(input_texture0, vec2<i32>(id.xy), 0);
    if (time_data.frame == 0u || (params.dynamic > 0.5 && epoch_lt().y < 0.1)) { prev = vec4<f32>(0.0); }
    textureStore(output, vec2<i32>(id.xy), prev * params.feedback_decay + fresh);
}

@compute @workgroup_size(16, 16, 1)
fn taa(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let R = vec2<f32>(dims);
    let uv = (vec2<f32>(id.xy) + 0.5) / R;
    let cur = textureSampleLevel(input_texture0, input_sampler0, uv, 0.0);
    var mn = cur; var mx = cur;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let n = textureSampleLevel(input_texture0, input_sampler0, uv + vec2<f32>(f32(x), f32(y)) / R, 0.0);
            mn = min(mn, n); mx = max(mx, n);
        }
    }
    let hist = textureSampleLevel(input_texture1, input_sampler1, uv, 0.0);
    let hc = clamp(hist, mn, mx);
    let blend = select(0.9, 0.0, time_data.frame < 4u || (params.dynamic > 0.5 && epoch_lt().y < 0.18));
    textureStore(output, vec2<i32>(id.xy), max(vec4<f32>(0.0), mix(cur, hc, blend)));
}

@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let uv = vec2<f32>(id.xy) / vec2<f32>(dims);

    let field = textureLoad(input_texture0, vec2<i32>(id.xy), 0);
    let energy = log(1.0 + field.a * params.lightning_intensity);
    let avg = field.rgb / max(field.a, 1e-3);
    let en = energy / (energy + 1.5);

    let el = epoch_lt();
    let dist = distance(uv, vec2<f32>(0.5, 0.16)) / 1.25;
    let front = clamp(el.y * max(params.morph, 0.001) * 0.3, 0.0, 1.3);
    let vis = smoothstep(front + 0.08, front - 0.05, dist);
    let fade = select(1.0, 1.0 - smoothstep(el.z - 0.6, el.z - 0.12, el.y), params.dynamic > 0.5);

    let wave = clamp(mix(0.05, 0.42, en) + params.color_shift * 0.01, 0.02, 0.6);
    var tint = process_color(params.base_color, wave, params.spectrum_mix) * avg;
    tint = mix(tint, vec3<f32>(1.0), smoothstep(0.55, 1.0, en) * 0.6);

    var col = tint * energy * params.light_intensity * vis * fade;

    var final_color = aces_tonemap(col);
    let gray = dot(final_color, vec3<f32>(0.2126, 0.7152, 0.0722));
    final_color = mix(vec3<f32>(gray), final_color, params.saturation);
    final_color = mix(final_color, smoothstep(vec3<f32>(0.0), vec3<f32>(1.0), final_color), params.contrast * 0.15);
    final_color = pow(max(final_color, vec3<f32>(0.0)), vec3<f32>(1.0 / max(params.gamma, 0.1)));
    let vig = uv * (1.0 - uv);
    final_color *= pow(vig.x * vig.y * 16.0, 0.12);

    textureStore(output, vec2<i32>(id.xy), vec4<f32>(final_color, 1.0));
}
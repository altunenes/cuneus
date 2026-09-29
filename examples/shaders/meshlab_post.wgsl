// Post for the meshlab example: HDR bloom chain over the rendered model, background, tonemap.
// channel0: MeshScene colour (linear HDR, alpha 0 where empty)
// channel1: MeshScene gbuffer (xyz world normal, w view depth), coverage weighted like channel0
// channel2: the HDRI loaded into the material (equirect), shown as background

struct TimeUniform { time: f32, delta: f32, frame: u32, _padding: u32 };
@group(0) @binding(0) var<uniform> u_time: TimeUniform;

struct PostParams {
    bloom: f32,
    threshold: f32,
    exposure: f32,
    gamma: f32,
    vignette: f32,
    bg_top_r: f32,
    bg_top_g: f32,
    bg_top_b: f32,
    bg_bot_r: f32,
    bg_bot_g: f32,
    bg_bot_b: f32,
    view_mode: f32,
    fog: f32,
    outline: f32,
    env_bg: f32,
    // camera: right * tan * aspect, up * tan, forward (for the background ray)
    crx: f32, cry: f32, crz: f32,
    cux: f32, cuy: f32, cuz: f32,
    cfx: f32, cfy: f32, cfz: f32,
};
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> p: PostParams;

@group(2) @binding(0) var channel0: texture_2d<f32>;
@group(2) @binding(1) var channel0_sampler: sampler;
@group(2) @binding(2) var channel1: texture_2d<f32>;
@group(2) @binding(3) var channel1_sampler: sampler;
@group(2) @binding(4) var channel2: texture_2d<f32>;
@group(2) @binding(5) var channel2_sampler: sampler;

@group(3) @binding(0) var input_texture0: texture_2d<f32>;
@group(3) @binding(1) var input_sampler0: sampler;
@group(3) @binding(2) var input_texture1: texture_2d<f32>;
@group(3) @binding(3) var input_sampler1: sampler;

alias v2 = vec2<f32>;
alias v3 = vec3<f32>;
alias v4 = vec4<f32>;

// keep bilinear taps half a texel inside t: the multipass sampler repeats, so taps at the edge
// would blend in the opposite side of the screen
fn cuv(uv: v2, t: texture_2d<f32>) -> v2 { let h = 0.5 / v2(textureDimensions(t)); return clamp(uv, h, 1.0 - h); }
fn ouv(id: vec3<u32>) -> v2 { return (v2(id.xy) + 0.5) / v2(textureDimensions(output)); }

// 4 bilinear taps covering 4x4 source texels
fn down(t: texture_2d<f32>, s: sampler, uv: v2) -> v3 {
    let o = 1.0 / v2(textureDimensions(t));
    return (textureSampleLevel(t, s, cuv(uv + v2(-o.x, -o.y), t), 0.0).rgb + textureSampleLevel(t, s, cuv(uv + v2(o.x, -o.y), t), 0.0).rgb
          + textureSampleLevel(t, s, cuv(uv + v2(-o.x, o.y), t), 0.0).rgb + textureSampleLevel(t, s, cuv(uv + v2(o.x, o.y), t), 0.0).rgb) * 0.25;
}

// same level + tent-filtered coarser level
fn up(id: vec3<u32>) -> v3 {
    let uv = ouv(id);
    let lp = 1.0 / v2(textureDimensions(input_texture1));
    var b = v3(0.0);
    for (var j = -1; j <= 1; j++) {
        for (var i = -1; i <= 1; i++) {
            b += textureSampleLevel(input_texture1, input_sampler1, cuv(uv + v2(f32(i), f32(j)) * lp, input_texture1), 0.0).rgb * f32((2 - abs(i)) * (2 - abs(j)));
        }
    }
    return textureSampleLevel(input_texture0, input_sampler0, cuv(uv, input_texture0), 0.0).rgb + b / 16.0;
}

// bright pass with a soft knee
@compute @workgroup_size(16, 16, 1)
fn bd1(@builtin(global_invocation_id) id: vec3<u32>) {
    let d = textureDimensions(output); if (id.x >= d.x || id.y >= d.y) { return; }
    let c = down(channel0, channel0_sampler, ouv(id)) * p.exposure;
    let m = max(c.r, max(c.g, c.b));
    let k = max(p.threshold * 0.5, 1e-4);
    let sk = clamp(m - p.threshold + k, 0.0, 2.0 * k);
    textureStore(output, id.xy, v4(c * max(sk * sk / (4.0 * k), m - p.threshold) / max(m, 1e-4), 1.0));
}
@compute @workgroup_size(16, 16, 1)
fn bd2(@builtin(global_invocation_id) id: vec3<u32>) { let d = textureDimensions(output); if (id.x >= d.x || id.y >= d.y) { return; } textureStore(output, id.xy, v4(down(input_texture0, input_sampler0, ouv(id)), 1.0)); }
@compute @workgroup_size(16, 16, 1)
fn bd3(@builtin(global_invocation_id) id: vec3<u32>) { let d = textureDimensions(output); if (id.x >= d.x || id.y >= d.y) { return; } textureStore(output, id.xy, v4(down(input_texture0, input_sampler0, ouv(id)), 1.0)); }
@compute @workgroup_size(16, 16, 1)
fn bd4(@builtin(global_invocation_id) id: vec3<u32>) { let d = textureDimensions(output); if (id.x >= d.x || id.y >= d.y) { return; } textureStore(output, id.xy, v4(down(input_texture0, input_sampler0, ouv(id)), 1.0)); }
@compute @workgroup_size(16, 16, 1)
fn bu3(@builtin(global_invocation_id) id: vec3<u32>) { let d = textureDimensions(output); if (id.x >= d.x || id.y >= d.y) { return; } textureStore(output, id.xy, v4(up(id), 1.0)); }
@compute @workgroup_size(16, 16, 1)
fn bu2(@builtin(global_invocation_id) id: vec3<u32>) { let d = textureDimensions(output); if (id.x >= d.x || id.y >= d.y) { return; } textureStore(output, id.xy, v4(up(id), 1.0)); }
@compute @workgroup_size(16, 16, 1)
fn bu1(@builtin(global_invocation_id) id: vec3<u32>) { let d = textureDimensions(output); if (id.x >= d.x || id.y >= d.y) { return; } textureStore(output, id.xy, v4(up(id), 1.0)); }

fn aces(x: v3) -> v3 { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), v3(0.0), v3(1.0)); }

@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let d = textureDimensions(output); if (id.x >= d.x || id.y >= d.y) { return; }
    let uv = ouv(id);
    let m = textureSampleLevel(channel0, channel0_sampler, uv, 0.0);
    let g = textureSampleLevel(channel1, channel1_sampler, uv, 0.0);
    let depth = g.w / max(m.a, 1e-3);

    // gbuffer debug views
    if p.view_mode > 0.5 {
        var dbg = normalize(g.xyz + 1e-6) * 0.5 + 0.5;
        if p.view_mode > 1.5 { dbg = v3(exp(-depth * 0.25)); }
        textureStore(output, id.xy, v4(dbg * step(1e-4, g.w), 1.0));
        return;
    }

    // background where the model is not: gradient, or the HDRI seen through the camera
    var bg = mix(v3(p.bg_top_r, p.bg_top_g, p.bg_top_b), v3(p.bg_bot_r, p.bg_bot_g, p.bg_bot_b), uv.y);
    if p.env_bg > 0.0 {
        let q = uv * 2.0 - 1.0;
        let dir = normalize(v3(p.cfx, p.cfy, p.cfz) + v3(p.crx, p.cry, p.crz) * q.x - v3(p.cux, p.cuy, p.cuz) * q.y);
        let e = v2(atan2(dir.z, dir.x) / 6.2831853 + 0.5, acos(clamp(dir.y, -1.0, 1.0)) / 3.14159265);
        bg = textureSampleLevel(channel2, channel2_sampler, e, 0.0).rgb * p.env_bg;
    }
    // depth fog towards the background, beyond the model's near side
    let f = (1.0 - exp(-max(depth - 1.5, 0.0) * p.fog)) * step(1e-4, g.w);
    // msaa resolve leaves edges premultiplied against the clear
    var col = bg * (1.0 - m.a) + mix(m.rgb * p.exposure, bg * m.a, f);
    col += textureSampleLevel(input_texture0, input_sampler0, cuv(uv, input_texture0), 0.0).rgb * p.bloom * 0.25;

    // outlines where depth or normal jumps
    if p.outline > 0.0 {
        let px = 1.0 / v2(d);
        var e = 0.0;
        for (var k = 0; k < 4; k++) {
            let o = select(v2(0.0, select(-1.0, 1.0, k == 3)), v2(select(-1.0, 1.0, k == 1), 0.0), k < 2);
            let gn = textureSampleLevel(channel1, channel1_sampler, cuv(uv + o * px, channel1), 0.0);
            e = max(e, abs(gn.w - g.w) / max(max(gn.w, g.w), 1e-3));
            if gn.w * g.w > 0.0 { e = max(e, 1.0 - dot(normalize(gn.xyz), normalize(g.xyz))); }
        }
        col *= 1.0 - smoothstep(0.05, 0.25, e) * p.outline;
    }

    col = aces(col);
    let vc = (uv - 0.5) * 2.0;
    col *= 1.0 - dot(vc, vc) * p.vignette * 0.25;
    textureStore(output, id.xy, v4(pow(max(col, v3(0.0)), v3(p.gamma)), 1.0));
}

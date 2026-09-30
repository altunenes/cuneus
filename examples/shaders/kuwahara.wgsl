// Kuwahara Filter, Enes Altun, 2025, MIT License
// Anisotropic Kuwahara filter after Kyprianidis, Kang & Döllner, "Image and Video Abstraction by
// Anisotropic Kuwahara Filtering" (2009), with the polynomial sector weights of Kyprianidis,
// "Anisotropic Kuwahara Filtering with Polynomial Weighting Functions" (2011).
// Sobel approach inspired by sofiene71: https://www.shadertoy.com/view/td3BzX
struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
}
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: KuwaharaParams;

@group(2) @binding(0) var channel0: texture_2d<f32>;
@group(2) @binding(1) var channel0_sampler: sampler;

@group(3) @binding(0) var input_texture0: texture_2d<f32>;
@group(3) @binding(1) var input_sampler0: sampler;
@group(3) @binding(2) var input_texture1: texture_2d<f32>;
@group(3) @binding(3) var input_sampler1: sampler;
@group(3) @binding(4) var input_texture2: texture_2d<f32>;
@group(3) @binding(5) var input_sampler2: sampler;

struct KuwaharaParams {
    // kernel radius in pixels
    radius: f32,
    // sharpness: how strongly low-variance sectors win
    q: f32,
    // anisotropy tuning: smaller stretches the ellipse more along the flow
    alpha: f32,
    // Gaussian sigma of the structure tensor smoothing
    tensor_sigma: f32,
    // mix with the original
    strength: f32,
    // 0 isotropic, 1 anisotropic
    mode: i32,
    // brush strokes: steps each way, blend, step length in pixels
    lic_length: f32,
    lic_strength: f32,
    lic_step: f32,
    saturation: f32,
    // 0 result, 1 original, 2 flow field
    view: i32,
    _pad: f32,
}

const PI: f32 = 3.14159265359;
const LUMA = vec3f(0.299, 0.587, 0.114);

fn has_input() -> bool {
    let d = textureDimensions(channel0);
    return d.x > 1u && d.y > 1u;
}

// the input, or a test pattern when nothing is loaded
fn get_input_color(uv: vec2f) -> vec3f {
    let u = clamp(uv, vec2f(0.0), vec2f(1.0));
    if (has_input()) {
        return textureSampleLevel(channel0, channel0_sampler, u, 0.0).rgb;
    }
    let circle = smoothstep(0.2, 0.21, distance(u, vec2f(0.5)));
    return mix(vec3f(0.8, 0.4, 0.2), vec3f(0.1, 0.1, 0.2), circle);
}

// pass textures are read with integer loads, clamped at the border (their sampler repeats)
fn load0(p: vec2i) -> vec4f {
    let d = vec2i(textureDimensions(input_texture0)) - 1;
    return textureLoad(input_texture0, clamp(p, vec2i(0), d), 0);
}

fn load1(p: vec2i) -> vec3f {
    let d = vec2i(textureDimensions(input_texture1)) - 1;
    return textureLoad(input_texture1, clamp(p, vec2i(0), d), 0).rgb;
}

// bilinear read of pass texture 1 at a pixel position, kept half a texel inside
fn sample1(pos: vec2f) -> vec3f {
    let d = vec2f(textureDimensions(input_texture1));
    let uv = clamp(pos / d, 0.5 / d, 1.0 - 0.5 / d);
    return textureSampleLevel(input_texture1, input_sampler1, uv, 0.0).rgb;
}

// the input resampled once to output size, so every later pass reads it with plain integer loads
@compute @workgroup_size(16, 16, 1)
fn source(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    textureStore(output, id.xy, vec4f(get_input_color((vec2f(id.xy) + 0.5) / vec2f(dims)), 1.0));
}

// structure tensor from Sobel gradients summed over RGB: (E, G, F) = (gx.gx, gy.gy, gx.gy).
// input_texture0 = source
@compute @workgroup_size(16, 16, 1)
fn structure_tensor(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let p = vec2i(id.xy);

    let c00 = load0(p + vec2i(-1, -1)).rgb;
    let c10 = load0(p + vec2i(0, -1)).rgb;
    let c20 = load0(p + vec2i(1, -1)).rgb;
    let c01 = load0(p + vec2i(-1, 0)).rgb;
    let c21 = load0(p + vec2i(1, 0)).rgb;
    let c02 = load0(p + vec2i(-1, 1)).rgb;
    let c12 = load0(p + vec2i(0, 1)).rgb;
    let c22 = load0(p + vec2i(1, 1)).rgb;
    let gx = (c20 + 2.0 * c21 + c22 - c00 - 2.0 * c01 - c02) * 0.25;
    let gy = (c02 + 2.0 * c12 + c22 - c00 - 2.0 * c10 - c20) * 0.25;

    textureStore(output, id.xy, vec4f(dot(gx, gx), dot(gy, gy), dot(gx, gy), 1.0));
}

// separable Gaussian smoothing of the tensor
fn blur_tensor(p: vec2i, axis: vec2i) -> vec3f {
    let s = max(params.tensor_sigma, 0.3);
    let r = min(i32(ceil(2.5 * s)), 16);
    var acc = vec3f(0.0);
    var tw = 0.0;
    for (var i = -r; i <= r; i++) {
        let w = exp(-0.5 * f32(i * i) / (s * s));
        acc += load0(p + axis * i).rgb * w;
        tw += w;
    }
    return acc / tw;
}

@compute @workgroup_size(16, 16, 1)
fn tensor_blur_h(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    textureStore(output, id.xy, vec4f(blur_tensor(vec2i(id.xy), vec2i(1, 0)), 1.0));
}

// vertical blur, then eigen-analysis: xy = edge tangent (minor eigenvector), z = anisotropy, w = angle
@compute @workgroup_size(16, 16, 1)
fn tensor_field(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let st = blur_tensor(vec2i(id.xy), vec2i(0, 1));
    let E = st.x;
    let G = st.y;
    let F = st.z;
    let root = sqrt(max((E - G) * (E - G) + 4.0 * F * F, 0.0));
    let l1 = 0.5 * (E + G + root);
    let l2 = 0.5 * (E + G - root);
    var t = vec2f(l1 - E, -F);
    t = select(vec2f(0.0, 1.0), normalize(t), length(t) > 1e-8);
    let A = select(0.0, (l1 - l2) / (l1 + l2), l1 + l2 > 1e-8);
    textureStore(output, id.xy, vec4f(t, A, atan2(t.y, t.x)));
}

// anisotropic Kuwahara: 8 smooth sectors in an ellipse aligned with the flow,
// each weighted by 1 / (1 + sigma^q) so the calm sectors dominate without hard switching.
// input_texture0 = tensor field, input_texture1 = source
@compute @workgroup_size(16, 16, 1)
fn kuwahara_filter(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let p = vec2i(id.xy);
    let orig = load1(p);

    let tf = load0(vec2i(id.xy));
    let aniso = params.mode != 0;
    let A = select(0.0, tf.z, aniso);
    let phi = select(0.0, tf.w, aniso);
    let r = max(params.radius, 1.0);
    let al = max(params.alpha, 0.05);
    // stretch capped at 1.5x the radius: most of the look, a fraction of the samples
    let a = r * clamp((al + A) / al, 0.1, 1.5);
    let b = r * clamp(al / (al + A), 0.1, 1.5);
    let cp = cos(phi);
    let sp = sin(phi);
    let mx = min(i32(ceil(sqrt(a * a * cp * cp + b * b * sp * sp))), 18);
    let my = min(i32(ceil(sqrt(a * a * sp * sp + b * b * cp * cp))), 18);

    let zeta = 2.0 / r;
    let sn = sin(PI / 8.0);
    let eta = (zeta + cos(PI / 8.0)) / (sn * sn);

    var m: array<vec4f, 8>;
    var s: array<vec3f, 8>;
    for (var j = -my; j <= my; j++) {
        for (var i = -mx; i <= mx; i++) {
            let o = vec2f(f32(i), f32(j));
            // rotate into the ellipse frame and scale it to a disc of radius 0.5
            var v = vec2f((cp * o.x + sp * o.y) * 0.5 / a, (-sp * o.x + cp * o.y) * 0.5 / b);
            if (dot(v, v) > 0.25) { continue; }
            let c = load1(p + vec2i(i, j));

            // polynomial sector weights, sectors 0 2 4 6, then the frame turned 45 degrees for 1 3 5 7
            var w: array<f32, 8>;
            var vxx = zeta - eta * v.x * v.x;
            var vyy = zeta - eta * v.y * v.y;
            var z = max(0.0, v.y + vxx); w[0] = z * z;
            z = max(0.0, -v.x + vyy); w[2] = z * z;
            z = max(0.0, -v.y + vxx); w[4] = z * z;
            z = max(0.0, v.x + vyy); w[6] = z * z;
            v = 0.70710678 * vec2f(v.x - v.y, v.x + v.y);
            vxx = zeta - eta * v.x * v.x;
            vyy = zeta - eta * v.y * v.y;
            z = max(0.0, v.y + vxx); w[1] = z * z;
            z = max(0.0, -v.x + vyy); w[3] = z * z;
            z = max(0.0, -v.y + vxx); w[5] = z * z;
            z = max(0.0, v.x + vyy); w[7] = z * z;

            var sum = 0.0;
            for (var k = 0; k < 8; k++) { sum += w[k]; }
            let g = exp(-3.125 * dot(v, v)) / max(sum, 1e-8);
            for (var k = 0; k < 8; k++) {
                let wk = w[k] * g;
                m[k] += vec4f(c * wk, wk);
                s[k] += c * c * wk;
            }
        }
    }

    var acc = vec4f(0.0);
    for (var k = 0; k < 8; k++) {
        if (m[k].w <= 0.0) { continue; }
        let mean = m[k].rgb / m[k].w;
        let va = abs(s[k] / m[k].w - mean * mean);
        let w = 1.0 / (1.0 + pow(255.0 * (va.r + va.g + va.b), 0.5 * params.q));
        acc += vec4f(mean * w, w);
    }
    let filtered = select(orig, acc.rgb / acc.w, acc.w > 0.0);
    textureStore(output, id.xy, vec4f(mix(orig, filtered, params.strength), 1.0));
}

// brush strokes: line integral convolution of the filtered image along the flow.
// input_texture0 = tensor field, input_texture1 = Kuwahara result
@compute @workgroup_size(16, 16, 1)
fn lic_edges(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let p0 = vec2f(id.xy) + 0.5;
    let c0 = sample1(p0);
    let tf = load0(vec2i(id.xy));
    // strokes only where there is a clear direction
    let blend = params.lic_strength * smoothstep(0.05, 0.3, tf.z);
    let steps = i32(params.lic_length);
    if (blend <= 0.001 || steps < 1) {
        textureStore(output, id.xy, vec4f(c0, 1.0));
        return;
    }

    let sigma = max(f32(steps) * 0.5, 0.5);
    var acc = c0;
    var wsum = 1.0;
    for (var dir = -1; dir <= 1; dir += 2) {
        var pos = p0;
        var prev = tf.xy * f32(dir);
        for (var i = 1; i <= steps; i++) {
            // the field has no sign, so keep each step pointing the way the stroke is going
            var t = load0(vec2i(floor(pos))).xy;
            if (dot(t, prev) < 0.0) { t = -t; }
            prev = t;
            pos += t * params.lic_step;
            if (any(pos < vec2f(0.0)) || any(pos >= vec2f(dims))) { break; }
            let w = exp(-0.5 * f32(i * i) / (sigma * sigma));
            acc += sample1(pos) * w;
            wsum += w;
        }
    }
    textureStore(output, id.xy, vec4f(mix(c0, acc / wsum, blend), 1.0));
}

// final: saturation, or the original, or the flow field (hue = orientation, brightness = anisotropy).
// input_texture0 = strokes, input_texture1 = tensor field, input_texture2 = source
@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    var col = textureLoad(input_texture0, vec2i(id.xy), 0).rgb;
    if (params.view == 1) {
        col = textureLoad(input_texture2, vec2i(id.xy), 0).rgb;
    } else if (params.view == 2) {
        let tf = textureLoad(input_texture1, vec2i(id.xy), 0);
        let h = tf.w / PI;
        let hue = 0.5 + 0.5 * cos(2.0 * PI * (h + vec3f(0.0, 0.33, 0.67)));
        col = hue * tf.z;
    } else {
        col = mix(vec3f(dot(col, LUMA)), col, params.saturation);
    }
    textureStore(output, id.xy, vec4f(clamp(col, vec3f(0.0), vec3f(1.0)), 1.0));
}

// 2D Neuron, by Enes Altun, 2025; MIT license
// Based on: https://www.shadertoy.com/view/3Xj3zz

struct TimeUniform { time: f32, delta: f32, frame: u32, _padding: u32 };
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

struct NeuronParams {
    pixel_offset: f32,
    pixel_offset2: f32,
    lights: f32,
    exp: f32,
    frame: f32,
    col1: f32,
    col2: f32,
    decay: f32,
};
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: NeuronParams;

// Multiple input textures for cross-buffer reading
@group(3) @binding(0) var input_texture0: texture_2d<f32>;
@group(3) @binding(1) var input_sampler0: sampler;
@group(3) @binding(2) var input_texture1: texture_2d<f32>;
@group(3) @binding(3) var input_sampler1: sampler;

const PI: f32 = 3.14159265;

fn S(a: f32, b: f32, k: f32) -> f32 {
    return max(0.0, k - abs(b - a)) / k;
}

fn SMIN(a: f32, b: f32, k: f32) -> f32 {
    let h = S(a, b, k);
    return min(a, b) - h * h * h * k / 6.0;
}

fn R2(a: f32) -> mat2x2<f32> {
    return mat2x2<f32>(cos(a), sin(a), -sin(a), cos(a));
}

fn C(p: vec2<f32>, r: f32) -> f32 {
    return length(p) - r;
}

fn CL(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, t1: f32, t2: f32, c: f32) -> f32 {
    let ba = b - a;
    let h = clamp(dot(p - a, ba) / dot(ba, ba), 0.0, 1.0);
    var pos = p - a - ba * h;
    pos.y -= sin(h * PI) * c;
    return length(pos) - mix(t1, t2, h);
}

fn sdNeuron(p: vec2<f32>) -> f32 {
    let soma = C(p, 0.1);
    let nuc = C(p - vec2<f32>(0.02, 0.01), 0.04);
    var d: f32 = 1.0;
    for(var i: f32 = 0.0; i < 12.0; i += 1.0) {
        let a = i / 6.0 * PI + sin(i) * 0.2;
        let rp = R2(a) * p;
        let l = 0.5 + sin(i * 1.7) * 0.2;
        let c = 0.1 * sin(i * 2.1);
        d = SMIN(d, CL(rp, vec2<f32>(0.0), vec2<f32>(l, 0.0), 0.02, 0.005, c), 0.1);
    }
    let ap = p - vec2<f32>(0.3, 0.0);
    let ax = CL(ap, vec2<f32>(0.0), vec2<f32>(0.8, -0.2), 0.04, 0.02, 0.1);
    let tb = vec2<f32>(1.1, -0.2);
    let t1 = CL(p - tb, vec2<f32>(0.0), R2(PI * 0.2) * vec2<f32>(0.3, 0.0), 0.015, 0.008, 0.05);
    let t2 = CL(p - tb, vec2<f32>(0.0), R2(-PI * 0.2) * vec2<f32>(0.3, 0.0), 0.015, 0.008, 0.05);
    let t3 = CL(p - tb, vec2<f32>(0.0), vec2<f32>(0.35, 0.0), 0.015, 0.008, 0.05);
    let t4 = CL(p - tb - vec2<f32>(0.25, 0.1), vec2<f32>(0.0), R2(PI * 0.3) * vec2<f32>(0.25, 0.0), 0.045, 0.028, 0.05);
    let t5 = CL(p - tb - vec2<f32>(0.0, -0.1), vec2<f32>(0.0), R2(-PI * 0.3) * vec2<f32>(0.25, 0.0), 0.045, 0.028, 0.05);
    var tm = SMIN(SMIN(t1, t2, 0.05), t3, 0.05);
    tm = SMIN(SMIN(tm, t4, 0.05), t5, 0.05);
    return SMIN(SMIN(max(soma, -nuc), d, 0.1), SMIN(ax, tm, 0.05), 0.1);
}

fn hash(p4: vec4<f32>) -> vec4<f32> {
    var p = fract(p4 * vec4<f32>(0.1031, 0.1030, 0.0973, 0.1099));
    p += dot(p, p.wzxy + 33.33);
    return fract((p.xxyz + p.yzzw) * p.zywx);
}

// Neuron geometry calculation
@compute @workgroup_size(16, 16, 1)
fn geometry(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }

    let R = vec2<f32>(dims);
    let U = vec2<f32>(id.xy);
    var p = (2.0 * U - R) / min(R.x, R.y);
    p = vec2<f32>(-p.y + 0.5, p.x);
    let d = sdNeuron(p);
    var c = vec3<f32>(0.0);
    let ng = smoothstep(0.02, -0.01, d);
    c += vec3<f32>(0.1, 0.95, 1.0) * 0.5 * ng;
    let sg = smoothstep(0.01, -0.01, C(p, 0.1));
    c += vec3<f32>(0.8, 0.85, 0.9) * 0.3 * sg;
    let nc = smoothstep(0.01, -0.01, C(p - vec2<f32>(0.02, 0.01), 0.04));
    c += vec3<f32>(0.7, 0.75, 0.8) * 0.6 * nc;
    let edge = smoothstep(-0.001, 0.001, d);
    c = mix(c, vec3<f32>(1.0), (1.0 - edge) * 0.7);
    
    var result = vec4<f32>(0.0);
    result.x = (c.r + c.g + c.b) / 4.0;
    
    textureStore(output, id.xy, result);
}

// Gradient computation from geometry pass
@compute @workgroup_size(16, 16, 1)
fn gradient(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }

    let R = vec2<f32>(dims);
    let U = vec2<f32>(id.xy);
    let pixel_offset = params.pixel_offset;
    let pixel_offset_2 = params.pixel_offset2;
    
    let n = textureLoad(input_texture0, vec2<i32>(U - vec2<f32>(pixel_offset_2, -pixel_offset)), 0);
    let e = textureLoad(input_texture0, vec2<i32>(U - vec2<f32>(pixel_offset, pixel_offset_2)), 0);
    let s = textureLoad(input_texture0, vec2<i32>(U - vec2<f32>(pixel_offset_2, pixel_offset)), 0);
    let w = textureLoad(input_texture0, vec2<i32>(U - vec2<f32>(-pixel_offset, pixel_offset_2)), 0);
    
    var result = vec4<f32>(0.0);
    result.x = 0.1 * (e.x - w.x);
    result.y = 0.1 * (s.x - n.x);
    result.z = textureLoad(input_texture0, vec2<i32>(U), 0).x;
    
    textureStore(output, id.xy, result);
}

// Particle tracing with self-feedback + gradient input
@compute @workgroup_size(16, 16, 1)
fn trace(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }

    let R = vec2<f32>(dims);
    var U = vec2<f32>(id.xy);
    var accumulated = textureLoad(input_texture0, vec2<i32>(U), 0);
    let frameCount = max(f32(time_data.frame), 1.0);
    let frameWeight = 1.0 / frameCount;
    let seed = vec4<f32>(U, f32(time_data.frame) * 0.1, 1.0);
    let h = hash(seed);
    var d = vec2<f32>(cos(2.0 * PI * h.x), sin(2.0 * PI * h.x));
    let amplitude = min(0.4 * params.frame, 0.1);
    var currentFrameContribution = vec4<f32>(0.0);
    var iter = params.col1;
    
    for(var i: f32 = 0.0; i < iter; i += 1.0) {
        U += d;
        
        let coords = clamp(vec2<i32>(U), vec2<i32>(0), vec2<i32>(R) - vec2<i32>(1));
        let b = textureLoad(input_texture1, coords, 0);
        
        d += (params.col2 + h.z) * 60.0 * b.xy;
        d = normalize(d);
        
        currentFrameContribution += amplitude * exp(-params.exp * length(d - vec2<f32>(0.0, 1.0))) * 
                                  max(sin(-2.0 + 6.0 * h.z + vec4<f32>(1.0, 2.0, 3.0, 4.0)), vec4<f32>(0.0));
        
        currentFrameContribution -= vec4<f32>(1.0, 2.0, 3.0, 4.0) * 0.0005 * b.z * params.frame;
    }
    
    let stabilizationFrames = 60.0;
    let blendFactor = min(frameCount, stabilizationFrames) / stabilizationFrames;
    let effectiveDecay = mix(params.decay, 1.0, blendFactor);
    var result = accumulated * effectiveDecay * (1.0 - frameWeight) + currentFrameContribution * frameWeight;
    
    textureStore(output, id.xy, result);
}

// Main image: Final gamma correction from trace pass
@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }

    let R = vec2<f32>(dims);
    let U = vec2<f32>(id.xy);
    
    let raw_result = textureLoad(input_texture0, vec2<i32>(U), 0);
    
    let result = pow(raw_result, vec4<f32>(params.lights));
    
    textureStore(output, id.xy, result);
}
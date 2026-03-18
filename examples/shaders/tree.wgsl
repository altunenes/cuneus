
struct TimeUniform { time: f32, delta: f32, frame: u32, _padding: u32 };
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

struct TreeParams {
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
@group(1) @binding(1) var<uniform> params: TreeParams;

@group(3) @binding(0) var input_texture0: texture_2d<f32>;
@group(3) @binding(1) var input_sampler0: sampler;
@group(3) @binding(2) var input_texture1: texture_2d<f32>;
@group(3) @binding(3) var input_sampler1: sampler;
@group(3) @binding(4) var input_texture2: texture_2d<f32>;
@group(3) @binding(5) var input_sampler2: sampler;

const PI: f32 = 3.14159265;
const ITER: i32 = 200;
const ZZ: f32 = 77.0;
const H: f32 = 0.1;

fn hash(p4: vec4<f32>) -> vec4<f32> {
    var p = fract(p4 * vec4<f32>(0.1031, 0.1030, 0.0973, 0.1099));
    p += dot(p, p.wzxy + 33.33);
    return fract((p.xxyz + p.yzzw) * p.zywx);
}

fn complex_power(z: vec2<f32>, x: f32) -> vec2<f32> {
    let r = length(z);
    let theta = atan2(z.y, z.x);
    let r_pow = pow(r, x);
    let x_theta = x * theta;
    return r_pow * vec2<f32>(cos(x_theta), sin(x_theta));
}

fn f(z: vec2<f32>) -> vec2<f32> {
    return complex_power(z, 1.5) - vec2<f32>(0.2, 0.0);
}

fn implicit(z_in: vec2<f32>) -> vec2<f32> {
    var z = z_in;
    var dz = vec2<f32>(H, 0.0);
    var i = 0;
    
    for (i = 0; i < ITER; i++) {
        dz = 1.5 * pow(length(z), 0.5) * dz;
        z = f(z);
        if (dot(z, z) > ZZ) { break; }
    }
    
    let dz_dot = max(dot(dz, dz), 1e-20);
    return vec2<f32>(f32(i), dot(z, z) / dz_dot);
}

// Buffer A: Fractal calculation with self-feedback
@compute @workgroup_size(16, 16, 1)
fn buffer_a(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }

    let R = vec2<f32>(dims);
    let U = vec2<f32>(id.xy);
    let zoom = 0.3;
    let pan = vec2<f32>(0.8, 1.7);
    
    let uv = ((vec2<f32>(U.x, R.y - U.y) - 0.5 * R) / min(R.y, R.x) + pan) * zoom;
    let z_and_i = implicit(uv);
    let iter_ratio = z_and_i.x / f32(ITER);
    let sharpness = clamp(z_and_i.y, 0.0, 100.0);
    
    let col1 = 0.5 + 0.5 * cos(params.col1 + vec3<f32>(0.0, 0.5, 1.0) + PI * vec3<f32>(2.0 * sharpness));
    let col2 = 0.5 + 0.5 * cos(params.col2 + PI * vec3<f32>(sharpness));
    let col = mix(col1, col2, iter_ratio);
    let col_sqrt = sqrt(col);
    
    var result = vec4<f32>(0.0);
    result.x = (col_sqrt.r + col_sqrt.g + col_sqrt.b) / 3.0;
    
    textureStore(output, id.xy, result);
}

// Buffer B: Gradient computation from Buffer A
@compute @workgroup_size(16, 16, 1)
fn buffer_b(@builtin(global_invocation_id) id: vec3<u32>) {
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
    result.x = 0.5 * (e.x - w.x);
    result.y = 0.5 * (s.x - n.x);
    result.z = textureLoad(input_texture0, vec2<i32>(U), 0).x;
    
    textureStore(output, id.xy, result);
}

// Buffer C: Particle tracing with self-feedback + Buffer B input
@compute @workgroup_size(16, 16, 1)
fn buffer_c(@builtin(global_invocation_id) id: vec3<u32>) {
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
        
        d += (1.0 + h.z) * 30.0 * b.xy;
        let d_len = length(d);
        d = select(d / d_len, vec2<f32>(0.0, 1.0), d_len < 1e-8);
        
        currentFrameContribution += amplitude * exp(-params.exp * length(d - vec2<f32>(0.0, 1.0))) * 
                                  max(sin(-2.0 + 6.0 * h.z + vec4<f32>(1.0, 2.0, 3.0, 4.0)), vec4<f32>(0.0));
        
        currentFrameContribution -= vec4<f32>(1.0, 2.0, 3.0, 4.0) * 0.0005 * b.z * params.frame;
    }
    
    let stabilizationFrames = 30.0;
    let blendFactor = min(frameCount, stabilizationFrames) / stabilizationFrames;
    let effectiveDecay = mix(params.decay, 1.0, blendFactor);
    var result = accumulated * effectiveDecay * (1.0 - frameWeight) + currentFrameContribution * frameWeight;
    
    textureStore(output, id.xy, result);
}

//gamma correction from Buffer C
@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }

    let R = vec2<f32>(dims);
    let U = vec2<f32>(id.xy);
    
    let raw_result = max(textureLoad(input_texture0, vec2<i32>(U), 0), vec4<f32>(0.0));
    let result = pow(raw_result, vec4<f32>(params.lights));
    let gamma = 1.4;
    let final_result = pow(result, vec4<f32>(gamma));
    
    textureStore(output, id.xy, final_result);
}
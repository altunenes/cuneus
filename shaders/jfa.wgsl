// Custom JFA (Jump Flooding Algorithm) with Clifford Attractor
// Based on Shadertoy implementation with multi-buffer ping-pong
// I also tried to use shadertoy term for that complex shader
// Backbone for JFA algorithm based on the : https://www.shadertoy.com/view/wcfSzs, wyatt, 2025 "JFA art 2", Shadertoy default license

struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
}
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

struct JfaParams {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    scale: f32,
    n: f32,
    gamma: f32,
    color_intensity: f32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
    color_w: f32,
    accumulation_speed: f32,
    fade_speed: f32,
    freeze_accumulation: f32,
    pattern_floor_add: f32,
    pattern_temp_add: f32,
    pattern_v_offset: f32,
    pattern_temp_mul1: f32,
    pattern_temp_mul2_3: f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
}
// Group 1: Output texture + Custom uniform
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: JfaParams;

@group(3) @binding(0) var input_texture0: texture_2d<f32>;
@group(3) @binding(1) var input_sampler0: sampler;
@group(3) @binding(2) var input_texture1: texture_2d<f32>;
@group(3) @binding(3) var input_sampler1: sampler;
@group(3) @binding(4) var input_texture2: texture_2d<f32>;  
@group(3) @binding(5) var input_sampler2: sampler;

alias v2 = vec2<f32>;
alias v3 = vec3<f32>;
alias v4 = vec4<f32>;
alias m2 = mat2x2<f32>;

var<private> R: v2;

fn hash(p4: v4) -> v4 {
    var p = fract(p4 * v4(0.1031, 0.1030, 0.0973, 0.1099));
    p += dot(p, p.wzxy + 33.33);
    return fract((p.xxyz + p.yzzw) * p.zywx);
}

fn ei(a: f32) -> m2 {
    let ca = cos(a);
    let sa = sin(a);
    return m2(ca, sa, -sa, ca);
}

fn cliffordAttractor(p: v2) -> v2 {
    let x = sin(params.a * p.y) + params.c * cos(params.a * p.x);
    let y = sin(params.b * p.x) + params.d * cos(params.b * p.y);
    return v2(x, y);
}

fn sample_input0(uv: v2) -> v4 {
    let coord = vec2<i32>(uv);
    let clamped_coord = clamp(coord, vec2<i32>(0), vec2<i32>(textureDimensions(input_texture0, 0)) - vec2<i32>(1));
    return textureLoad(input_texture0, clamped_coord, 0);
}

fn sample_input1(uv: v2) -> v4 {
    let coord = vec2<i32>(uv);
    let clamped_coord = clamp(coord, vec2<i32>(0), vec2<i32>(textureDimensions(input_texture1, 0)) - vec2<i32>(1));
    return textureLoad(input_texture1, clamped_coord, 0);
}

fn sample_input2(uv: v2) -> v4 {
    let coord = vec2<i32>(uv);
    let clamped_coord = clamp(coord, vec2<i32>(0), vec2<i32>(textureDimensions(input_texture2, 0)) - vec2<i32>(1));
    return textureLoad(input_texture2, clamped_coord, 0);
}

@compute @workgroup_size(16, 16, 1)
fn buffer_a(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output);
    R = v2(dims);
    
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    
    let U = v2(f32(gid.x), f32(gid.y));
    
    let frame_cycle = f32(time_data.frame / u32(params.n));
    
    let h0 = hash(v4(U, frame_cycle, 1.0)) - 1.0;
    var V = v3(h0.xy, h0.z);
    
    var j = 0.0;
    for (var i = 0.0; i < 24.0; i += 1.0) {
        let h = hash(v4(U, frame_cycle, i));
        let x = floor(h.x * 2.0) * 2.0 - params.pattern_floor_add;
        j += x;
        
        let temp_attractor = cliffordAttractor(V.xy);
        V.x = temp_attractor.x;
        V.y = temp_attractor.y;
        
        let temp_add = V.xy + params.pattern_temp_add * V.xy / dot(V.xy, V.xy);
        V.x = temp_add.x;
        V.y = temp_add.y;

        V.x -= params.pattern_v_offset;

        let temp_mul1 = V.xy * (params.pattern_temp_mul1 * ei(1.0 - 1.1 * x));
        V.x = temp_mul1.x;
        V.y = temp_mul1.y;
        
        let temp_mul2 = V.yz * (0.8 * ei(params.pattern_temp_mul2_3 + 1.1 * x));
        V.y = temp_mul2.x;
        V.z = temp_mul2.y;
        
        let temp_mul3 = V.xy * (0.8 * ei(params.pattern_temp_mul2_3 + 0.1 * 1.1 * x));
        V.x = temp_mul3.x;
        V.y = temp_mul3.y;
        
        if (length(V) > 4.0) { break; }
    }
    
    let temp_scale = V.xy * params.scale;
    V.x = temp_scale.x;
    V.y = temp_scale.y;
    
    var Q = (v4(V.y, V.x, 0.0, 0.0) * R.y * 1.5 + 0.5 * v4(R.x, R.y, R.x, R.y));
    Q.z = 11.2 + 0.12 * j;
    
    textureStore(output, gid.xy, Q);
}

// Buffer B - Jump Flooding Algorithm (ichannel0=bufferA, ichannel1=bufferB)
fn Y(inout_Q: ptr<function, v4>, U: v2, v: v2) {
    let x = sample_input1(U + v).xy; // Read from old "BufferB"
    // Compare distance to a point in the new point cloud ("BufferA")
    if (distance(U, sample_input0(x).xy) < distance(U, sample_input0((*inout_Q).xy).xy)) {
        (*inout_Q).x = x.x;
        (*inout_Q).y = x.y;
    }
}

@compute @workgroup_size(16, 16, 1)
fn buffer_b(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output);
    R = v2(dims);
    
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    
    let U = v2(f32(gid.x), f32(gid.y));
    let N = i32(params.n);
    
    var Q: v4;
    
    if (i32(time_data.frame) % N == 0) {
        // JFA Reset: Every pixel points to itself.
        Q = v4(U, 0.0, 1.0);
    } else {
        // JFA Propagation Step
        // Read previous state from BufferB
        Q = sample_input1(U);
        let k = exp2(f32(N - 1 - (i32(time_data.frame) % N)));
        Y(&Q, U, v2(0.0, k));
        Y(&Q, U, v2(k, 0.0));
        Y(&Q, U, v2(0.0, -k));
        Y(&Q, U, v2(-k, 0.0));
    }
    
    textureStore(output, gid.xy, Q);
}

// Buffer C - Color accumulation based on JFA results (ichannel0=bufferA, ichannel1=bufferB, ichannel2=bufferC)
@compute @workgroup_size(16, 16, 1)
fn buffer_c(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output);
    R = v2(dims);
    
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    
    let U = v2(f32(gid.x), f32(gid.y));
    let N = i32(params.n);
    let frame_in_cycle = i32(time_data.frame) % N;
    
    var Q: v4;

    // Only reset for the first few cycles, then allow accumulation to stabilize
    let cycle_count = i32(time_data.frame) / N;
     // Only reset for first 3 cycles
    let should_reset = (cycle_count < 3) && (frame_in_cycle == 0);
    
    if (should_reset) {
        Q = v4(0.0);
    } else {
        Q = sample_input2(U); // Read old color from "BufferC"
        // Apply fade to prevent infinite accumulation after stabilization
        // But only if accumulation is not frozen
        if (cycle_count >= 3 && params.freeze_accumulation < 0.5) {
            Q *= params.fade_speed;
        }
    }

    // On the *last* frame of the JFA cycle, add the new color information.
    // But only if accumulation is not frozen
    if (frame_in_cycle == (N - 1) && params.freeze_accumulation < 0.5) {
        // sample_input1(U) is the result of the completed JFA from BufferB
        // sample_input0(...) is looking up that point's data from the new BufferA
        let a = sample_input0(sample_input1(U).xy);
        
        let color_term = max(0.5 + 0.5 * sin(-2.0 + 3.0 * a.z + v4(params.color_r, params.color_g, params.color_b, params.color_w)), v4(0.0));
        let distance_term = exp(-length(U - a.xy));
        
        Q += color_term * distance_term * params.accumulation_speed;
    }
    
    textureStore(output, gid.xy, Q);
}

// (ichannel2=bufferC)
@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output);
    R = v2(dims);
    
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    
    let U = v2(f32(gid.x), f32(gid.y));
    

    var Q = sample_input2(U);
    
    Q *= 4.0 * params.color_intensity;
    Q = pow(max(Q, vec4<f32>(0.0)), vec4<f32>(params.gamma));
    
    textureStore(output, gid.xy, Q);
}
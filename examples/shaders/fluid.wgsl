// 2D Navier-Stokes fluid with position field tracking
// Enes Altun, 2026 License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
// Position field concept inspired by wyatt's Lattice Boltzmann shader (shadertoy.com/view/WdyGzy)
// Gradient-based lighting approach inspired by flockaroo (shadertoy.com/view/MsGSRd)
struct TimeUniform { time: f32, delta: f32, frame: u32, _padding: u32 };
@group(0) @binding(0) var<uniform> time_data: TimeUniform;
struct FluidParams {
    viscosity: f32,
    gravity: f32,
    pressure_scale: f32,
    vortex_strength: f32,
    turbulence: f32,
    flow_speed: f32,
    pos_diffusion: f32,
    texture_influence: f32,
    light_intensity: f32,
    spec_power: f32,
    spec_intensity: f32,
    color_vibrancy: f32,
    vortex_radius: f32,
    gamma: f32,
    feedback: f32,
    vortex_speed: f32,
    force_mode: f32,
    force_harmony: f32,
    force_count: f32,
    contrast: f32,
    warp_amount: f32,
    flow_intensity: f32,
    color_advect: f32,
    drift_decay: f32,
};
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: FluidParams;
@group(2) @binding(0) var channel0: texture_2d<f32>;
@group(2) @binding(1) var channel0_sampler: sampler;
@group(3) @binding(0) var input_texture0: texture_2d<f32>;
@group(3) @binding(1) var input_sampler0: sampler;
@group(3) @binding(2) var input_texture1: texture_2d<f32>;
@group(3) @binding(3) var input_sampler1: sampler;
@group(3) @binding(4) var input_texture2: texture_2d<f32>;
@group(3) @binding(5) var input_sampler2: sampler;

fn s0(px: vec2<f32>) -> vec4<f32> {
    let R = vec2<f32>(textureDimensions(input_texture0));
    return textureSampleLevel(input_texture0, input_sampler0, clamp((px + 0.5) / R, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0);
}
fn s1(px: vec2<f32>) -> vec4<f32> {
    let R = vec2<f32>(textureDimensions(input_texture1));
    return textureSampleLevel(input_texture1, input_sampler1, clamp((px + 0.5) / R, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0);
}
fn s2(px: vec2<f32>) -> vec4<f32> {
    let R = vec2<f32>(textureDimensions(input_texture2));
    return textureSampleLevel(input_texture2, input_sampler2, clamp((px + 0.5) / R, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0);
}

// Divergence-free initial seed via analytic curl of hash field
fn hash2(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}
fn seed_velocity(uv: vec2<f32>) -> vec2<f32> {
    let e = 0.005;
    let p = uv * 5.0;
    let dy = hash2(floor(p + vec2<f32>(0.0, e))) - hash2(floor(p - vec2<f32>(0.0, e)));
    let dx = hash2(floor(p + vec2<f32>(e, 0.0))) - hash2(floor(p - vec2<f32>(e, 0.0)));
    return vec2<f32>(dy, -dx) * 0.1;
}


// input_texture0 = self (prev velocity), input_texture1 = color_map
// Stores: vec4(vx, vy, curl, pressure)
@compute @workgroup_size(16, 16, 1)
fn fluid_sim(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let R = vec2<f32>(dims);
    let U = vec2<f32>(id.xy);
    let uv = U / R;

    // Advect (RK2)
    let dt = params.flow_speed;
    let C = s0(U);
    let v0 = C.xy;
    let mid = U - 0.5 * v0 * dt;
    let v_mid = s0(clamp(mid, vec2<f32>(0.0), R - 1.0)).xy;
    var vel = s0(clamp(U - v_mid * dt, vec2<f32>(0.0), R - 1.0)).xy;

    // Neighbors
    let N = s0(clamp(U + vec2<f32>(0.0, 1.0), vec2<f32>(0.0), R - 1.0));
    let S = s0(clamp(U - vec2<f32>(0.0, 1.0), vec2<f32>(0.0), R - 1.0));
    let E = s0(clamp(U + vec2<f32>(1.0, 0.0), vec2<f32>(0.0), R - 1.0));
    let W = s0(clamp(U - vec2<f32>(1.0, 0.0), vec2<f32>(0.0), R - 1.0));

    // Pressure Jacobi step 1
    let div = 0.5 * ((E.x - W.x) + (N.y - S.y));
    let p1 = (N.w + S.w + E.w + W.w - div) * 0.25;
    vel -= params.pressure_scale * 0.5 * vec2<f32>(E.w - W.w, N.w - S.w);

    // Viscosity
    vel = mix(vel, 0.25 * (N.xy + S.xy + E.xy + W.xy), params.viscosity * 0.01);

    // Vorticity confinement
    let curl_c = 0.5 * ((E.y - W.y) - (N.x - S.x));
    var eta = vec2<f32>(abs(E.z) - abs(W.z), abs(N.z) - abs(S.z));
    eta /= (length(eta) + 1e-5);
    vel += params.vortex_strength * curl_c * vec2<f32>(eta.y, -eta.x) * dt;

    // Texture buoyancy + edge flow
    let tex_w = params.texture_influence;
    let lc = dot(s1(U).rgb, vec3<f32>(0.299, 0.587, 0.114));
    let ln = dot(s1(U + vec2<f32>(0.0, 3.0)).rgb, vec3<f32>(0.299, 0.587, 0.114));
    let ls = dot(s1(U - vec2<f32>(0.0, 3.0)).rgb, vec3<f32>(0.299, 0.587, 0.114));
    let le = dot(s1(U + vec2<f32>(3.0, 0.0)).rgb, vec3<f32>(0.299, 0.587, 0.114));
    let lw = dot(s1(U - vec2<f32>(3.0, 0.0)).rgb, vec3<f32>(0.299, 0.587, 0.114));
    vel.y += (lc - 0.5) * params.gravity * tex_w;
    vel += vec2<f32>(-(ln - ls), le - lw) * params.gravity * 0.3 * tex_w;

    let t = time_data.time;
    let v_rad = params.vortex_radius;
    let v_spd = params.vortex_speed;
    let soft = mix(0.01, 0.08, params.force_harmony);
    let n_src = u32(clamp(params.force_count, 0.0, 8.0));

    var force = vec2<f32>(0.0);
    for (var s = 0u; s < 8u; s++) {
        if (s >= n_src) { break; }
        let fs = f32(s);
        let ang = fs * 1.257 + t * (v_spd + fs * v_spd * 0.4);
        let orbit_r = 0.12 + 0.06 * sin(t * v_spd * 0.6 + fs * 1.8);
        let cx = fract(fs * 0.618 + 0.1) * 0.6 + 0.2;
        let cy = fract(fs * 0.382 + 0.2) * 0.6 + 0.2;
        let center = vec2<f32>(cx + orbit_r * cos(ang), cy + orbit_r * sin(ang * 0.8 + fs));
        let d = uv - center;
        let dist2 = dot(d, d);
        let envelope = exp(-dist2 / v_rad);
        let pulse = sin(t * (v_spd * 2.5 + fs * v_spd) + fs * 2.1) * 0.5 + 0.5;
        let chirality = select(-1.0, 1.0, s % 2u == 0u);
        force += vec2<f32>(-d.y, d.x) / (dist2 + soft) * envelope * pulse * 0.03 * chirality;
    }

    vel += force;

    // Velocity dissipation + soft limit + boundary
    let dissipation = 1.0 / (1.0 + params.turbulence);
    let speed = length(vel);
    if (speed > 3.0) { vel *= 3.0 / speed; }
    vel *= dissipation;
    let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    vel *= smoothstep(0.0, 0.01, edge);

    if (time_data.frame < 2u) {
        vel = seed_velocity(uv);
    }
    textureStore(output, id.xy, vec4<f32>(vel, curl_c, p1));
}

// input_texture0 = fluid_sim, input_texture1 = self
// Stores: vec4(vx, vy, speed, pressure)
@compute @workgroup_size(16, 16, 1)
fn pressure_refine(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let R = vec2<f32>(dims);
    let U = vec2<f32>(id.xy);

    let C = s0(U);
    let N = s0(clamp(U + vec2<f32>(0.0, 1.0), vec2<f32>(0.0), R - 1.0));
    let S = s0(clamp(U - vec2<f32>(0.0, 1.0), vec2<f32>(0.0), R - 1.0));
    let E = s0(clamp(U + vec2<f32>(1.0, 0.0), vec2<f32>(0.0), R - 1.0));
    let W = s0(clamp(U - vec2<f32>(1.0, 0.0), vec2<f32>(0.0), R - 1.0));

    let div = 0.5 * ((E.x - W.x) + (N.y - S.y));
    let p2 = (N.w + S.w + E.w + W.w - div) * 0.25;

    // Temporal pressure accumulation
    let prev = s1(U);
    let prev_N = s1(clamp(U + vec2<f32>(0.0, 1.0), vec2<f32>(0.0), R - 1.0));
    let prev_S = s1(clamp(U - vec2<f32>(0.0, 1.0), vec2<f32>(0.0), R - 1.0));
    let prev_E = s1(clamp(U + vec2<f32>(1.0, 0.0), vec2<f32>(0.0), R - 1.0));
    let prev_W = s1(clamp(U - vec2<f32>(1.0, 0.0), vec2<f32>(0.0), R - 1.0));
    let p3 = (prev_N.w + prev_S.w + prev_E.w + prev_W.w - div) * 0.25;

    let p_final = mix(p2, p3, 0.5);

    var vel = C.xy;
    let grad_p = params.pressure_scale * 0.5 * vec2<f32>(
        mix(E.w, prev_E.w, 0.5) - mix(W.w, prev_W.w, 0.5),
        mix(N.w, prev_N.w, 0.5) - mix(S.w, prev_S.w, 0.5));
    vel -= grad_p * 0.5;

    let avg = 0.25 * (N.xy + S.xy + E.xy + W.xy);
    let smooth_vel = mix(vel, avg, 0.03);

    let spd = length(smooth_vel);
    textureStore(output, id.xy, vec4<f32>(smooth_vel, spd, p_final));
}

// input_texture0 = pressure_refine (corrected velocity.xy, speed.z)
// input_texture1 = self (previous position)
// input_texture2 = color_map
// Stores: vec4(pos.x, pos.y, vel.x, vel.y)
@compute @workgroup_size(16, 16, 1)
fn position_field(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let R = vec2<f32>(dims);
    let U = vec2<f32>(id.xy);
    let uv = U / R;

    let vel_data = s0(U);
    let vel = vel_data.xy * params.flow_speed * params.flow_intensity;

    // RK2 trace-back
    let mid = U - 0.5 * vel;
    let v_mid = s0(clamp(mid, vec2<f32>(0.0), R - 1.0)).xy * params.flow_speed * params.flow_intensity;
    let trace = clamp(U - v_mid, vec2<f32>(0.5), R - 1.5);

    var Q = s1(trace);

    // Neighbor healing
    let Nr = s1(U + vec2<f32>(0.0, 1.0));
    let Sr = s1(U - vec2<f32>(0.0, 1.0));
    let Er = s1(U + vec2<f32>(1.0, 0.0));
    let Wr = s1(U - vec2<f32>(1.0, 0.0));

    let max_diff = max(
        max(length(Nr.xy - Q.xy), length(Sr.xy - Q.xy)),
        max(length(Er.xy - Q.xy), length(Wr.xy - Q.xy)));
    let broken = smoothstep(2.0, 8.0, max_diff);

    let avg_pos = 0.25 * (Nr.xy + Sr.xy + Er.xy + Wr.xy);
    let diffuse = params.pos_diffusion * 0.04 + broken * 0.35;
    Q = vec4<f32>(mix(Q.xy, avg_pos, diffuse), Q.zw);

    // Reseed broken
    Q = vec4<f32>(mix(Q.xy, U, broken * 0.12), vel_data.xy);

    // Edge handling
    let edge_d = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    Q = vec4<f32>(mix(Q.xy, U, smoothstep(0.02, 0.0, edge_d) * 0.15), Q.zw);

    // Drift limit
    let disp = Q.xy - U;
    Q = vec4<f32>(Q.xy - disp * smoothstep(R.y * 0.3, R.y * 0.5, length(disp)) * 0.01, Q.zw);
    Q = vec4<f32>(clamp(Q.xy, vec2<f32>(0.5), R - 1.5), Q.zw);

    // Drift decay
    Q = vec4<f32>(mix(Q.xy, U, params.drift_decay), Q.zw);

    if (time_data.frame < 2u) { Q = vec4<f32>(U, 0.0, 0.0); }
    textureStore(output, id.xy, Q);
}

// input_texture0 = position_field (.xy=pos, .zw=vel)
// input_texture1 = self (prev color)
@compute @workgroup_size(16, 16, 1)
fn color_map(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let R = vec2<f32>(dims);
    let U = vec2<f32>(id.xy);

    let pos_data = s0(U);
    let uv = U / R;
    let vel = pos_data.zw;

    let raw_uv = pos_data.xy / R;
    let warped_uv = uv + (raw_uv - uv) * params.warp_amount;

    let original = textureSampleLevel(channel0, channel0_sampler, warped_uv, 0.0);

    // Semi-Lagrangian advection of previous color
    let trace = clamp(U - vel * params.flow_speed * params.color_advect, vec2<f32>(0.0), R - 1.0);
    var prev = s1(trace);

    // Feedback blend
    let fb = clamp(params.feedback, 0.0, 1.0);
    var Q = mix(original, prev, fb);

    // Energy conservation at high feedback
    let refresh = mix(0.0, 0.005, smoothstep(0.9, 1.0, fb));
    Q = mix(Q, original, refresh);

    Q = clamp(Q, vec4<f32>(0.0), vec4<f32>(1.0));

    if (time_data.frame < 4u) {
        Q = textureSampleLevel(channel0, channel0_sampler, U / R, 0.0);
    }
    textureStore(output, id.xy, Q);
}

// input_texture0 = color_map
// input_texture1 = pressure_refine (corrected velocity.xy, speed.z, pressure.w)
@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let R = vec2<f32>(dims);
    let U = vec2<f32>(id.xy);

    let base = s0(U);
    let fluid = s1(U);
    let fb = clamp(params.feedback, 0.0, 1.0);

    let cn = length(s0(U + vec2<f32>(0.0, 1.0)).rgb);
    let cs = length(s0(U - vec2<f32>(0.0, 1.0)).rgb);
    let ce = length(s0(U + vec2<f32>(1.0, 0.0)).rgb);
    let cw = length(s0(U - vec2<f32>(1.0, 0.0)).rgb);
    let fine = vec2<f32>(ce - cw, cn - cs);

    let cn3 = length(s0(U + vec2<f32>(0.0, 3.0)).rgb);
    let cs3 = length(s0(U - vec2<f32>(0.0, 3.0)).rgb);
    let ce3 = length(s0(U + vec2<f32>(3.0, 0.0)).rgb);
    let cw3 = length(s0(U - vec2<f32>(3.0, 0.0)).rgb);
    let coarse = vec2<f32>(ce3 - cw3, cn3 - cs3) / 3.0;

    // Velocity/pressure normals
    let fN = s1(clamp(U + vec2<f32>(0.0, 2.0), vec2<f32>(0.0), R - 1.0));
    let fS = s1(clamp(U - vec2<f32>(0.0, 2.0), vec2<f32>(0.0), R - 1.0));
    let fE = s1(clamp(U + vec2<f32>(2.0, 0.0), vec2<f32>(0.0), R - 1.0));
    let fW = s1(clamp(U - vec2<f32>(2.0, 0.0), vec2<f32>(0.0), R - 1.0));
    // Pressure as height field
    let vel_grad = vec2<f32>(fE.w - fW.w, fN.w - fS.w) * 0.5;

    // Blend color + pressure normals
    let color_grad = mix(coarse, fine, smoothstep(0.0, 0.02, length(fine)));
    let grad = color_grad + vel_grad * fb * 0.8;

    let z = mix(0.15, 0.5, smoothstep(0.0, 0.05, length(grad)));
    let normal = normalize(vec3<f32>(grad, z));

    // Two-light setup
    let t = time_data.time;

    // Key light
    let key = normalize(vec3<f32>(
        3.0 + 0.3 * sin(t * 0.3),
        3.0 + 0.3 * cos(t * 0.25),
        2.5
    ));
    let NdotL_key = max(dot(normal, key), 0.0);

    // Fill light
    let fill = normalize(vec3<f32>(
        -2.0 + 0.2 * cos(t * 0.2),
        -1.5,
        2.0
    ));
    let NdotL_fill = max(dot(normal, fill), 0.0);

    // Wrap diffuse
    let diff_key = NdotL_key * 0.7 + 0.3; // never fully dark
    let diff_fill = NdotL_fill * 0.3;
    let diffuse = diff_key + diff_fill;

    // GGX specular
    let V = vec3<f32>(0.0, 0.0, 1.0);
    let H = normalize(key + V);
    let NdotH = max(dot(normal, H), 0.0);
    let roughness = 1.0 / max(params.spec_power * 0.5, 1.0);
    let a2 = roughness * roughness;
    let denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    let D = a2 / (3.14159 * denom * denom + 1e-6);
    let spec = D * params.spec_intensity * 0.08 * NdotL_key;

    var col = base.rgb * clamp(diffuse * params.light_intensity, 0.5, 2.0) + vec3<f32>(spec);

    // Velocity tint
    col += vec3<f32>(fluid.x * 0.001, fluid.z * 0.0005, fluid.y * 0.001);

    let sat_boost = params.color_vibrancy;
    let lum = dot(col, vec3<f32>(0.299, 0.587, 0.114));
    col = mix(vec3<f32>(lum), col, sat_boost);

    // S-curve contrast
    let contrast_amt = params.contrast;
    col = mix(col, smoothstep(vec3<f32>(0.0), vec3<f32>(1.0), col), contrast_amt);

    col = pow(max(col, vec3<f32>(0.0)), vec3<f32>(params.gamma));
    textureStore(output, id.xy, vec4<f32>(col, 1.0));
}

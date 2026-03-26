// 2D Navier-Stokes with Helmholtz-Hodge projection
// Enes Altun, 2026 License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
// MacCormack advection · 12-iteration Jacobi pressure · curl-noise turbulence
// Pipeline: advect_forces → pressure(×12) → project → position_field → color_map → main
// Position field concept inspired by wyatt (shadertoy.com/view/WdyGzy)
// Gradient lighting inspired by flockaroo (shadertoy.com/view/MsGSRd)
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
    dye_intensity: f32,
    dye_radius: f32,
    bg_boil: f32,
    _padding: f32,
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

// Noise with analytical gradient for divergence-free curl noise

fn hash22(p: vec2<f32>) -> vec2<f32> {
    var p3 = fract(vec3<f32>(p.xyx) * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

fn value_noise_grad(p: vec2<f32>) -> vec3<f32> {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let du = 6.0 * f * (1.0 - f);
    let a = hash22(i).x;
    let b = hash22(i + vec2<f32>(1.0, 0.0)).x;
    let c = hash22(i + vec2<f32>(0.0, 1.0)).x;
    let d = hash22(i + vec2<f32>(1.0, 1.0)).x;
    let val = a + (b - a) * u.x + (c - a) * u.y + (a - b - c + d) * u.x * u.y;
    let gx = du.x * ((b - a) + (a - b - c + d) * u.y);
    let gy = du.y * ((c - a) + (a - b - c + d) * u.x);
    return vec3<f32>(gx, gy, val);
}

fn curl_noise_2d(p: vec2<f32>) -> vec2<f32> {
    let ng = value_noise_grad(p);
    return vec2<f32>(-ng.y, ng.x);
}

fn fbm_curl(p: vec2<f32>) -> vec2<f32> {
    var result = vec2<f32>(0.0);
    var amp = 1.0;
    var freq = 1.0;
    for (var i = 0; i < 3; i++) {
        result += curl_noise_2d(p * freq) * amp;
        freq *= 2.0;
        amp *= 0.5;
    }
    return result;
}


struct VortexData {
    force: vec2<f32>,
    dye: vec3<f32>,
};

fn compute_vortices(uv: vec2<f32>, t: f32) -> VortexData {
    var vd: VortexData;
    vd.force = vec2<f32>(0.0);
    vd.dye = vec3<f32>(0.0);

    let v_rad = params.vortex_radius;
    let v_spd = params.vortex_speed;
    let soft = mix(0.01, 0.08, params.force_harmony);
    let n_src = u32(clamp(params.force_count, 0.0, 18.0));
    let num_active = max(1.0, f32(n_src));
    let force_scale = min(1.0, 2.0 / sqrt(num_active));
    let dye_scale = min(1.0, 4.0 / num_active);

    for (var s = 0u; s < 18u; s++) {
        if (s >= n_src) { break; }
        let fs = f32(s);
        let time_s = t * v_spd * 2.0;
        let phase = fs * 1.618;

        let center = vec2<f32>(
            0.5 + 0.35 * sin(time_s * (0.7 + fs * 0.15) + phase)
                + 0.10 * cos(time_s * 0.43 - phase),
            0.5 + 0.35 * cos(time_s * (0.5 + fs * 0.22) + phase * 1.3)
                + 0.10 * sin(time_s * 0.57 + phase)
        );

        let d = uv - center;
        let dist2 = dot(d, d);
        let pulse = sin(t * (v_spd * 2.5 + fs * v_spd) + fs * 2.1) * 0.5 + 0.5;
        let chirality = select(-1.0, 1.0, s % 2u == 0u);

        let force_env = (v_rad / (dist2 + v_rad)) * exp(-dist2 / (v_rad * 8.0));
        vd.force += vec2<f32>(-d.y, d.x) / (dist2 + soft) * force_env * pulse * 0.03 * chirality * force_scale;

        let dye_rad = v_rad * params.dye_radius;
        let dye_env = exp(-dist2 / (dye_rad * 0.4));
        let src_color = vec3<f32>(
            0.5 + 0.5 * sin(fs * 2.1 + t * 0.5),
            0.5 + 0.5 * sin(fs * 3.7 + t * 0.6),
            0.5 + 0.5 * sin(fs * 1.3 + t * 0.7)
        );
        vd.dye += src_color * dye_env * pulse * params.dye_intensity * dye_scale;
    }
    return vd;
}

fn seed_velocity(uv: vec2<f32>) -> vec2<f32> {
    let e = 0.005;
    let p = uv * 5.0;
    let dy = hash22(floor(p + vec2<f32>(0.0, e))).x - hash22(floor(p - vec2<f32>(0.0, e))).x;
    let dx = hash22(floor(p + vec2<f32>(e, 0.0))).x - hash22(floor(p - vec2<f32>(e, 0.0))).x;
    return vec2<f32>(dy, -dx) * 0.1;
}

// Pass 1: advect_forces
// Reads: s0 = project (last frame: .xy=div-free velocity, .z=curl, .w=pressure)
// Writes: vec4(vx_raw, vy_raw, 0, 0)
@compute @workgroup_size(16, 16, 1)
fn advect_forces(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let R = vec2<f32>(dims);
    let U = vec2<f32>(id.xy);
    let uv = U / R;
    let dt = params.flow_speed;

    // Read last frame's projected velocity field
    let C = s0(U);
    let v0 = C.xy;

    // Sample 4 neighbors (reused for curl, vorticity, viscosity, MacCormack limiter)
    let vN = s0(clamp(U + vec2<f32>(0.0, 1.0), vec2<f32>(0.0), R - 1.0)).xy;
    let vS = s0(clamp(U - vec2<f32>(0.0, 1.0), vec2<f32>(0.0), R - 1.0)).xy;
    let vE = s0(clamp(U + vec2<f32>(1.0, 0.0), vec2<f32>(0.0), R - 1.0)).xy;
    let vW = s0(clamp(U - vec2<f32>(1.0, 0.0), vec2<f32>(0.0), R - 1.0)).xy;

    // MacCormack advection
    // Step 1: RK2 backward trace (standard semi-Lagrangian)
    let mid_back = U - 0.5 * v0 * dt;
    let v_mid = s0(clamp(mid_back, vec2<f32>(0.0), R - 1.0)).xy;
    let trace_back = clamp(U - v_mid * dt, vec2<f32>(0.0), R - 1.0);
    let v_hat = s0(trace_back).xy;

    // Step 2: Forward trace for error estimation
    let trace_fwd = clamp(trace_back + v_hat * dt, vec2<f32>(0.0), R - 1.0);
    let v_back = s0(trace_fwd).xy;

    // Step 3: MacCormack correction with limiter
    let v_mc = v_hat + 0.5 * (v0 - v_back);
    let v_min = min(min(vN, vS), min(vE, vW));
    let v_max = max(max(vN, vS), max(vE, vW));
    var vel = clamp(v_mc, min(v_min, v_hat), max(v_max, v_hat));

    // ── Viscosity (explicit diffusion) ──
    let avg_vel = 0.25 * (vN + vS + vE + vW);
    vel = mix(vel, avg_vel, params.viscosity * 0.01);

    // ── Vorticity confinement ──
    // Curl from last frame's velocity neighbors
    let curl_c = 0.5 * ((vE.y - vW.y) - (vN.x - vS.x));
    // Gradient of |curl| from stored curl in project buffer .z
    let curl_N = abs(s0(clamp(U + vec2<f32>(0.0, 1.0), vec2<f32>(0.0), R - 1.0)).z);
    let curl_S = abs(s0(clamp(U - vec2<f32>(0.0, 1.0), vec2<f32>(0.0), R - 1.0)).z);
    let curl_E = abs(s0(clamp(U + vec2<f32>(1.0, 0.0), vec2<f32>(0.0), R - 1.0)).z);
    let curl_W = abs(s0(clamp(U - vec2<f32>(1.0, 0.0), vec2<f32>(0.0), R - 1.0)).z);
    var eta = vec2<f32>(curl_E - curl_W, curl_N - curl_S);
    eta /= (length(eta) + 1e-5);
    vel += params.vortex_strength * curl_c * vec2<f32>(eta.y, -eta.x) * dt;

    // ── Texture buoyancy ──
    let tex_w = params.texture_influence;
    let lc = dot(textureSampleLevel(channel0, channel0_sampler, uv, 0.0).rgb, vec3<f32>(0.299, 0.587, 0.114));
    let ln = dot(textureSampleLevel(channel0, channel0_sampler, uv + vec2<f32>(0.0, 3.0) / R, 0.0).rgb, vec3<f32>(0.299, 0.587, 0.114));
    let ls = dot(textureSampleLevel(channel0, channel0_sampler, uv - vec2<f32>(0.0, 3.0) / R, 0.0).rgb, vec3<f32>(0.299, 0.587, 0.114));
    let le = dot(textureSampleLevel(channel0, channel0_sampler, uv + vec2<f32>(3.0, 0.0) / R, 0.0).rgb, vec3<f32>(0.299, 0.587, 0.114));
    let lw = dot(textureSampleLevel(channel0, channel0_sampler, uv - vec2<f32>(3.0, 0.0) / R, 0.0).rgb, vec3<f32>(0.299, 0.587, 0.114));
    vel.y += (lc - 0.5) * params.gravity * tex_w;
    vel += vec2<f32>(-(ln - ls), le - lw) * params.gravity * 0.3 * tex_w;

    let t = time_data.time;
    let vortices = compute_vortices(uv, t);
    vel += vortices.force;

    // Curl-noise turbulence
    let drift_uv = uv * 3.0 + vec2<f32>(sin(t * 0.1), cos(t * 0.15)) * 0.3;
    let boil = fbm_curl(drift_uv + t * 0.05);
    vel += boil * params.vortex_strength * params.bg_boil * 0.08;

    // ── Dissipation + speed limit + boundary ──
    let dissipation = 1.0 / (1.0 + params.turbulence);
    let speed = length(vel);
    if (speed > 3.0) { vel *= 3.0 / speed; }
    vel *= dissipation;
    let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    vel *= smoothstep(0.0, 0.01, edge);

    if (time_data.frame < 2u) {
        vel = seed_velocity(uv);
    }
    textureStore(output, id.xy, vec4<f32>(vel, 0.0, 0.0));
}

// Pass 2: pressure (Jacobi iteration, dispatched 12× via duplicate passes)
// Reads: s0 = advect_forces (.xy = raw velocity for divergence)
//        s1 = self (previous pressure iteration, .x = pressure)
// Writes: vec4(pressure, 0, 0, 0)
@compute @workgroup_size(16, 16, 1)
fn pressure(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let R = vec2<f32>(dims);
    let U = vec2<f32>(id.xy);

    // Velocity neighbors for divergence (from advect_forces)
    let vN = s0(clamp(U + vec2<f32>(0.0, 1.0), vec2<f32>(0.0), R - 1.0)).xy;
    let vS = s0(clamp(U - vec2<f32>(0.0, 1.0), vec2<f32>(0.0), R - 1.0)).xy;
    let vE = s0(clamp(U + vec2<f32>(1.0, 0.0), vec2<f32>(0.0), R - 1.0)).xy;
    let vW = s0(clamp(U - vec2<f32>(1.0, 0.0), vec2<f32>(0.0), R - 1.0)).xy;
    let div = 0.5 * ((vE.x - vW.x) + (vN.y - vS.y));

    // Pressure neighbors (from previous iteration)
    let pN = s1(clamp(U + vec2<f32>(0.0, 1.0), vec2<f32>(0.0), R - 1.0)).x;
    let pS = s1(clamp(U - vec2<f32>(0.0, 1.0), vec2<f32>(0.0), R - 1.0)).x;
    let pE = s1(clamp(U + vec2<f32>(1.0, 0.0), vec2<f32>(0.0), R - 1.0)).x;
    let pW = s1(clamp(U - vec2<f32>(1.0, 0.0), vec2<f32>(0.0), R - 1.0)).x;

    // Jacobi iteration: p_new = (sum_neighbors - divergence) / 4
    let p_new = (pN + pS + pE + pW - div) * 0.25;

    textureStore(output, id.xy, vec4<f32>(p_new, 0.0, 0.0, 0.0));
}

// Pass 3: project (subtract pressure gradient → divergence-free velocity)
// Reads: s0 = advect_forces (.xy = raw velocity)
//        s1 = pressure (.x = solved pressure)
// Writes: vec4(vx_proj, vy_proj, curl, pressure)
@compute @workgroup_size(16, 16, 1)
fn project(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let R = vec2<f32>(dims);
    let U = vec2<f32>(id.xy);

    let raw_vel = s0(U).xy;

    // Pressure gradient
    let pN = s1(clamp(U + vec2<f32>(0.0, 1.0), vec2<f32>(0.0), R - 1.0)).x;
    let pS = s1(clamp(U - vec2<f32>(0.0, 1.0), vec2<f32>(0.0), R - 1.0)).x;
    let pE = s1(clamp(U + vec2<f32>(1.0, 0.0), vec2<f32>(0.0), R - 1.0)).x;
    let pW = s1(clamp(U - vec2<f32>(1.0, 0.0), vec2<f32>(0.0), R - 1.0)).x;

    let grad_p = params.pressure_scale * 0.5 * vec2<f32>(pE - pW, pN - pS);
    var vel = raw_vel - grad_p;

    // Compute curl
    let vN = s0(clamp(U + vec2<f32>(0.0, 1.0), vec2<f32>(0.0), R - 1.0)).xy;
    let vS = s0(clamp(U - vec2<f32>(0.0, 1.0), vec2<f32>(0.0), R - 1.0)).xy;
    let vE = s0(clamp(U + vec2<f32>(1.0, 0.0), vec2<f32>(0.0), R - 1.0)).xy;
    let vW = s0(clamp(U - vec2<f32>(1.0, 0.0), vec2<f32>(0.0), R - 1.0)).xy;
    let curl = 0.5 * ((vE.y - vW.y) - (vN.x - vS.x));

    let p_local = s1(U).x;

    if (time_data.frame < 2u) {
        let uv = U / R;
        vel = seed_velocity(uv);
    }
    textureStore(output, id.xy, vec4<f32>(vel, curl, p_local));
}

// Pass 4: position_field (Lagrangian particle tracking)
// Reads: s0 = project (.xy = div-free velocity)
//        s1 = self (.xy = previous position, .zw = velocity)
//        s2 = color_map (reserved)
// Writes: vec4(pos.x, pos.y, vel.x, vel.y)
@compute @workgroup_size(16, 16, 1)
fn position_field(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let R = vec2<f32>(dims);
    let U = vec2<f32>(id.xy);
    let uv = U / R;

    let vel_data = s0(U);
    let vel = vel_data.xy * params.flow_speed * params.flow_intensity;

    // RK2 trace-back through divergence free velocity field
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

    // Reseed broken regions
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

// Pass 5: color_map (color advection + dye injection)
// Reads: s0 = position_field (.xy = pos, .zw = vel)
//        s1 = self (previous color)
// Writes: vec4(r, g, b, a)
@compute @workgroup_size(16, 16, 1)
fn color_map(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let R = vec2<f32>(dims);
    let U = vec2<f32>(id.xy);
    let uv = U / R;

    let pos_data = s0(U);
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

    // Dye injection
    let t = time_data.time;
    let vortices = compute_vortices(uv, t);
    Q = vec4<f32>(Q.rgb + vortices.dye, Q.a);

    Q = clamp(Q, vec4<f32>(0.0), vec4<f32>(1.0));

    if (time_data.frame < 4u) {
        Q = textureSampleLevel(channel0, channel0_sampler, U / R, 0.0);
    }
    textureStore(output, id.xy, Q);
}

// 
// Pass 6: main_image (gradient lighting + compositing)
// Reads: s0 = color_map
//        s1 = project (.xy = velocity, .z = curl, .w = pressure)
@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let R = vec2<f32>(dims);
    let U = vec2<f32>(id.xy);

    let base = s0(U);
    let fluid = s1(U);
    let fb = clamp(params.feedback, 0.0, 1.0);

    // Multi-scale color gradients for surface normals
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

    // Pressure based surface height
    let pN = s1(clamp(U + vec2<f32>(0.0, 2.0), vec2<f32>(0.0), R - 1.0)).w;
    let pS = s1(clamp(U - vec2<f32>(0.0, 2.0), vec2<f32>(0.0), R - 1.0)).w;
    let pE = s1(clamp(U + vec2<f32>(2.0, 0.0), vec2<f32>(0.0), R - 1.0)).w;
    let pW = s1(clamp(U - vec2<f32>(2.0, 0.0), vec2<f32>(0.0), R - 1.0)).w;
    let vel_grad = vec2<f32>(pE - pW, pN - pS) * 0.5;

    // Blend color + pressure normals
    let color_grad = mix(coarse, fine, smoothstep(0.0, 0.02, length(fine)));
    let grad = color_grad + vel_grad * fb * 0.8;

    let z = mix(0.15, 0.5, smoothstep(0.0, 0.05, length(grad)));
    let normal = normalize(vec3<f32>(grad, z));

    // Two-light setup
    let t = time_data.time;
    let key = normalize(vec3<f32>(
        3.0 + 0.3 * sin(t * 0.3),
        3.0 + 0.3 * cos(t * 0.25),
        2.5
    ));
    let NdotL_key = max(dot(normal, key), 0.0);

    let fill = normalize(vec3<f32>(
        -2.0 + 0.2 * cos(t * 0.2),
        -1.5,
        2.0
    ));
    let NdotL_fill = max(dot(normal, fill), 0.0);

    // Wrap diffuse
    let diff_key = NdotL_key * 0.7 + 0.3;
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

    // Saturation
    let lum = dot(col, vec3<f32>(0.299, 0.587, 0.114));
    col = mix(vec3<f32>(lum), col, params.color_vibrancy);

    // S-curve contrast
    col = mix(col, smoothstep(vec3<f32>(0.0), vec3<f32>(1.0), col), params.contrast);

    // Gamma
    col = pow(max(col, vec3<f32>(0.0)), vec3<f32>(params.gamma));

    textureStore(output, id.xy, vec4<f32>(col, 1.0));
}

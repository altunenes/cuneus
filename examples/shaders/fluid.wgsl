// 2D Navier-Stokes with Helmholtz-Hodge projection
// Enes Altun, 2026 License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
// MacCormack advection · 24-iteration Jacobi pressure · curl-noise turbulence
// Pipeline: advect_forces → pressure(×24) → project → position_field → color_map → main
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
    rolloff: f32,
    warp_amount: f32,
    _unused1: f32,
    _unused2: f32,
    drift_decay: f32,
    dye_intensity: f32,
    dye_radius: f32,
    bg_boil: f32,
    normal_amp: f32,
    rim_strength: f32,
    ink: f32,
    dye_diffusion: f32,
    stir: f32,
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
    return textureSampleLevel(input_texture0, input_sampler0, (clamp(px, vec2<f32>(0.0), R - 1.0) + 0.5) / R, 0.0);
}
fn s1(px: vec2<f32>) -> vec4<f32> {
    let R = vec2<f32>(textureDimensions(input_texture1));
    return textureSampleLevel(input_texture1, input_sampler1, (clamp(px, vec2<f32>(0.0), R - 1.0) + 0.5) / R, 0.0);
}
fn s2(px: vec2<f32>) -> vec4<f32> {
    let R = vec2<f32>(textureDimensions(input_texture2));
    return textureSampleLevel(input_texture2, input_sampler2, (clamp(px, vec2<f32>(0.0), R - 1.0) + 0.5) / R, 0.0);
}
// for the colour history only:
fn cr_w(t: f32) -> vec4<f32> {
    let t2 = t * t; let t3 = t2 * t;
    return vec4<f32>(-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1.0, -1.5 * t3 + 2.0 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2);
}
fn cubic(t: texture_2d<f32>, p: vec2<f32>) -> vec4<f32> {
    let Rm = vec2<i32>(textureDimensions(t)) - 1; let b = floor(p); let f = p - b; let o = vec2<i32>(b) - 1;
    let wx = cr_w(f.x); let wy = cr_w(f.y);
    var s = vec4<f32>(0.0);
    for (var j = 0; j < 4; j++) {
        var row = vec4<f32>(0.0);
        for (var i = 0; i < 4; i++) { row += wx[i] * textureLoad(t, clamp(o + vec2<i32>(i, j), vec2<i32>(0), Rm), 0); }
        s += wy[j] * row;
    }
    return s;
}
// MacCormack limiter bounds
struct Bounds { lo: vec4<f32>, hi: vec4<f32> };
fn src_bounds0(p: vec2<f32>) -> Bounds {
    let b = floor(p);
    let a = s0(b); let c = s0(b + vec2<f32>(1.0, 0.0)); let d = s0(b + vec2<f32>(0.0, 1.0)); let e = s0(b + vec2<f32>(1.0, 1.0));
    return Bounds(min(min(a, c), min(d, e)), max(max(a, c), max(d, e)));
}
fn src_bounds1(p: vec2<f32>) -> Bounds {
    let b = floor(p);
    let a = s1(b); let c = s1(b + vec2<f32>(1.0, 0.0)); let d = s1(b + vec2<f32>(0.0, 1.0)); let e = s1(b + vec2<f32>(1.0, 1.0));
    return Bounds(min(min(a, c), min(d, e)), max(max(a, c), max(d, e)));
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

fn src_center(fs: f32, t: f32) -> vec2<f32> {
    let time_s = t * params.vortex_speed * 2.0; let phase = fs * 1.618;
    return vec2<f32>(
        0.5 + 0.35 * sin(time_s * (0.7 + fs * 0.15) + phase) + 0.10 * cos(time_s * 0.43 - phase),
        0.5 + 0.35 * cos(time_s * (0.5 + fs * 0.22) + phase * 1.3) + 0.10 * sin(time_s * 0.57 + phase));
}

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
        let center = src_center(fs, t);

        let d = uv - center;
        let dist2 = dot(d, d);
        let chirality = select(-1.0, 1.0, s % 2u == 0u);

        let force_env = (v_rad / (dist2 + v_rad)) * exp(-dist2 / (v_rad * 8.0));

        // rot comp
        let rot = vec2<f32>(-d.y, d.x) * chirality;
        // Radial jet component: creates inflow/outflow → shear layers → Kelvin-Helmholtz
        let jet_phase = sin(t * (v_spd * 1.3 + fs * 0.7) + fs * 3.14159);
        let rad = d * jet_phase * 0.4;
        // stirring spoon: push along the source's path -> sheds a counter-rotating pair in its wake
        let path = src_center(fs, t + 0.05) - src_center(fs, t - 0.05);
        let stir_f = path / (length(path) + 1e-6) * force_env * 3.0 * params.stir;
        let combined = (rot + rad) / (dist2 + soft) * force_env + stir_f;

        let pulse = sin(t * (v_spd * 2.5 + fs * v_spd) + fs * 2.1) * 0.3 + 0.7;
        vd.force += combined * pulse * 0.03 * force_scale;

        let dye_rad = v_rad * params.dye_radius;
        let dye_env = exp(-dist2 / (dye_rad * 0.06));
        let src_color = vec3<f32>(
            0.5 + 0.5 * sin(fs * 2.1 + t * 0.5),
            0.5 + 0.5 * sin(fs * 3.7 + t * 0.6),
            0.5 + 0.5 * sin(fs * 1.3 + t * 0.7)
        );
        vd.dye += src_color * dye_env * params.dye_intensity * dye_scale;
    }
    return vd;
}

fn seed_velocity(uv: vec2<f32>) -> vec2<f32> {
    // curl noise
    return fbm_curl(uv * 6.0) * 0.08 + curl_noise_2d(uv * 12.0) * 0.03;
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

    // Step 3: MacCormack correction, limited to the source neighbourhood (keeps it stable without smearing)
    let v_mc = v_hat + 0.5 * (v0 - v_back);
    let vb = src_bounds0(trace_back);
    var vel = clamp(v_mc, vb.lo.xy, vb.hi.xy);

    // ── Viscosity (explicit diffusion) ──
    let avg_vel = 0.25 * (vN + vS + vE + vW);
    vel = mix(vel, avg_vel, clamp(params.viscosity, 0.0, 0.95));

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
    // curl forms... 
    let conf_force = params.vortex_strength * curl_c * vec2<f32>(eta.y, -eta.x) * dt;
    let conf_len = length(conf_force);
    vel += conf_force * min(1.0, 1.5 / (conf_len + 1e-6));

    // boussinesq: displaced warm fluid rises
    let l_rest = dot(textureSampleLevel(channel0, channel0_sampler, uv, 0.0).rgb, vec3<f32>(0.299, 0.587, 0.114));
    let l_now = dot(s1(U).rgb, vec3<f32>(0.299, 0.587, 0.114));
    vel.y -= (l_now - l_rest) * params.gravity * params.texture_influence;

    let t = time_data.time;
    let vortices = compute_vortices(uv, t);
    vel += vortices.force;

    // Curl-noise
    let drift_uv = uv * 4.0 + vec2<f32>(sin(t * 0.07), cos(t * 0.11)) * 0.2;
    let boil = fbm_curl(drift_uv + t * 0.03);
    vel += boil * params.bg_boil * 0.03;

    // dissipation + speed limit + boundary
    let dissipation = 1.0 / (1.0 + params.turbulence);
    let speed = length(vel);
    if (speed > 4.0) { vel *= 4.0 / speed; }
    vel *= dissipation;
    // free slip walls
    let wx = smoothstep(0.0, 2.0, min(U.x, R.x - 1.0 - U.x));
    let wy = smoothstep(0.0, 2.0, min(U.y, R.y - 1.0 - U.y));
    vel = vec2<f32>(vel.x * wx, vel.y * wy);

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
    let vel = vel_data.xy * params.flow_speed;

    // RK2 trace-back through divergence free velocity field
    let mid = U - 0.5 * vel;
    let v_mid = s0(clamp(mid, vec2<f32>(0.0), R - 1.0)).xy * params.flow_speed;
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
    let diffuse = params.pos_diffusion * 0.15 + broken * 0.35;
    Q = vec4<f32>(mix(Q.xy, avg_pos, diffuse), Q.zw);

    // Reseed broken regions
    Q = vec4<f32>(mix(Q.xy, U, broken * 0.12), vel_data.xy);

    // Edge handling
    let edge_px = min(min(U.x, R.x - 1.0 - U.x), min(U.y, R.y - 1.0 - U.y));
    Q = vec4<f32>(mix(Q.xy, U, smoothstep(2.0, 0.0, edge_px) * 0.15), Q.zw);

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
    let uv = (U + 0.5) / R;
    let pos_data = s0(U);
    let vel = pos_data.zw;

    // orign texture at warped position
    let raw_uv = (pos_data.xy + 0.5) / R;
    let warped_uv = uv + (raw_uv - uv) * params.warp_amount;
    let original = textureSampleLevel(channel0, channel0_sampler, warped_uv, 0.0);

    // MacCormack color advection
    let adv_vel = vel * params.flow_speed;
    let trace_back = clamp(U - adv_vel, vec2<f32>(0.0), R - 1.0);
    // dye diffusivity: 0 = sharp ink filaments (bicubic), 1 = soft spreading dye (bilinear)
    let prev_hat = mix(cubic(input_texture1, trace_back), s1(trace_back), clamp(params.dye_diffusion, 0.0, 1.0));

    // Forward trace for error est
    let vel_at_trace = s0(trace_back).zw * params.flow_speed;
    let trace_fwd = clamp(trace_back + vel_at_trace, vec2<f32>(0.0), R - 1.0);
    let prev_back = s1(trace_fwd);

    // MacCormack correction, limited to the source neighbourhood
    let prev_here = s1(U);
    var prev = prev_hat + 0.5 * (prev_here - prev_back);
    let cb = src_bounds1(trace_back);
    prev = clamp(prev, cb.lo, cb.hi);

    let fb = clamp(params.feedback, 0.0, 1.0);

    // Subtle dye tint at source positionsvia lum preserving
    let t = time_data.time;
    let vortices = compute_vortices(uv, t);
    let dye_lum = dot(vortices.dye, vec3<f32>(0.299, 0.587, 0.114));
    // ink (0 = natural, only the texture's own colours): sources also release their own coloured dye
    let ink_amt = clamp(max(vortices.dye.r, max(vortices.dye.g, vortices.dye.b)), 0.0, 1.0);
    let ink_col = vortices.dye / max(ink_amt, 1e-4);
    let advected = mix(prev.rgb * (1.0 + dye_lum * 0.15), ink_col, ink_amt * 0.35 * params.ink);

    // track what the fluid changed relative to the source texture
    let fluid_delta = advected - original.rgb;

    let delta_decay = fb * 0.997;
    var Q_rgb = original.rgb + fluid_delta * delta_decay;

    // Clean HDR compression
    let peak = max(max(Q_rgb.r, Q_rgb.g), Q_rgb.b);
    if (peak > 1.0) {
        Q_rgb /= peak;
    }
    Q_rgb = max(Q_rgb, vec3<f32>(0.0));

    var Q = vec4<f32>(Q_rgb, original.a);

    if (time_data.frame < 4u) {
        Q = textureSampleLevel(channel0, channel0_sampler, uv, 0.0);
    }
    textureStore(output, id.xy, Q);
}

//
// Pass 6: main_image (height field shading + compositing)
// Reads: s0 = color_map
//        s1 = project (.xy = velocity, .z = curl, .w = pressure)
@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let R = vec2<f32>(dims);
    let U = vec2<f32>(id.xy);
    let uv = U / R;
    let luma_w = vec3<f32>(0.299, 0.587, 0.114);

    let base = s0(U);
    let fluid = s1(U);

    // Height field = local luminance + pressure
    let pN = s1(clamp(U + vec2<f32>(0.0, 1.0), vec2<f32>(0.0), R - 1.0)).w;
    let pS = s1(clamp(U - vec2<f32>(0.0, 1.0), vec2<f32>(0.0), R - 1.0)).w;
    let pE = s1(clamp(U + vec2<f32>(1.0, 0.0), vec2<f32>(0.0), R - 1.0)).w;
    let pW = s1(clamp(U - vec2<f32>(1.0, 0.0), vec2<f32>(0.0), R - 1.0)).w;

    let hN = dot(s0(U + vec2<f32>(0.0, 1.0)).rgb, luma_w) + pN * 0.5;
    let hS = dot(s0(U - vec2<f32>(0.0, 1.0)).rgb, luma_w) + pS * 0.5;
    let hE = dot(s0(U + vec2<f32>(1.0, 0.0)).rgb, luma_w) + pE * 0.5;
    let hW = dot(s0(U - vec2<f32>(1.0, 0.0)).rgb, luma_w) + pW * 0.5;

    let dx = hE - hW;
    let dy = hN - hS;

    let amp = smoothstep(0.0, 0.3, sqrt(dx * dx + dy * dy)) * params.normal_amp;
    let nor = normalize(vec3<f32>(-dx * amp, -dy * amp, 1.0));

    // Three-light setup
    let l1 = normalize(vec3<f32>(0.6, 0.5, 1.0));
    let l2 = normalize(vec3<f32>(-0.7, -0.4, 0.8));
    let l3 = normalize(vec3<f32>(0.0, 0.8, 0.2));
    let vd = normalize(vec3<f32>(0.5 - uv.x, 0.5 - uv.y, 1.0));

    let df = max(dot(nor, l1), 0.0)
           + max(dot(nor, l2), 0.0) * 0.5
           + max(dot(nor, l3), 0.0) * 0.3;

    let h1d = normalize(l1 + vd);
    let h2d = normalize(l2 + vd);
    let nh1 = max(dot(nor, h1d), 0.0);
    let nh2 = max(dot(nor, h2d), 0.0);
    let tS = (pow(nh1, max(params.spec_power, 1.0))
            + pow(nh1, max(params.spec_power * 0.25, 1.0)) * 0.1
            + pow(nh2, max(params.spec_power * 0.5, 1.0)) * 0.1) * params.spec_intensity;


    let lum_here = dot(base.rgb, luma_w);
    let lum_wide = (dot(s0(U + vec2<f32>(0.0, 3.0)).rgb, luma_w)
                  + dot(s0(U - vec2<f32>(0.0, 3.0)).rgb, luma_w)
                  + dot(s0(U + vec2<f32>(3.0, 0.0)).rgb, luma_w)
                  + dot(s0(U - vec2<f32>(3.0, 0.0)).rgb, luma_w)) * 0.25;
    let ao = clamp(0.1 + 0.9 * (lum_here / (lum_wide + 0.01)), 0.0, 1.0);

    var col = base.rgb * (df * 0.4 + 0.3) * params.light_intensity * ao;
    col += vec3<f32>(1.0, 0.98, 0.95) * tS * .1 * ao;

    col += base.rgb * pow(1.0 - max(dot(nor, vd), 0.0), 3.0) * params.rim_strength * ao;

    let vel_mag = length(fluid.xy);
    let curl_mag = abs(fluid.z);
    let pressure_local = fluid.w;
    let spread = mix(3.0, 0.5, params.force_mode);
    let flow_factor = smoothstep(0.0, spread, vel_mag);
    let curl_factor = smoothstep(0.0, spread * 0.3, curl_mag);
    let p_factor = smoothstep(0.0, spread * 0.5, abs(pressure_local));
    let glow = max(flow_factor, max(curl_factor * 0.7, p_factor * 0.5)) * params.dye_intensity;
    col += col * glow * 0.15;

    // Saturation
    let lum = dot(col, luma_w);
    col = mix(vec3<f32>(lum), col, params.color_vibrancy);


    // highlight rolloff
    let rw = 0.4 * clamp(params.rolloff, 0.0, 1.0); let knee = 1.0 - rw;
    let pk = max(max(col.r, col.g), col.b);
    if (rw > 1e-3 && pk > knee) { col *= (knee + rw * (1.0 - exp(-(pk - knee) / rw))) / pk; }
    col = pow(max(col, vec3<f32>(0.0)), vec3<f32>(params.gamma));

    textureStore(output, id.xy, vec4<f32>(col, 1.0));
}

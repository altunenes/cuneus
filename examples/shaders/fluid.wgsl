// Lattice Boltzmann fluid + multi-scale rotation + position field tracking
// Lattice Boltzman method adapted from: wyatt: https://www.shadertoy.com/view/WdyGzy 2019; License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
// Multi scale rot inspiration: flockaroo, https://www.shadertoy.com/view/MsGSRd 2016; License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
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
    mixing: f32,
    gamma: f32,
    _pad1: f32,
    _pad2: f32,
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

fn bilerp0(px: vec2<f32>) -> vec4<f32> {
    let R = vec2<f32>(textureDimensions(input_texture0));
    return textureSampleLevel(input_texture0, input_sampler0, (px + 0.5) / R, 0.0);
}
fn bilerp1(px: vec2<f32>) -> vec4<f32> {
    let R = vec2<f32>(textureDimensions(input_texture1));
    return textureSampleLevel(input_texture1, input_sampler1, (px + 0.5) / R, 0.0);
}
fn bilerp2(px: vec2<f32>) -> vec4<f32> {
    let R = vec2<f32>(textureDimensions(input_texture2));
    return textureSampleLevel(input_texture2, input_sampler2, (px + 0.5) / R, 0.0);
}
// --- LB Helpers ---
fn vel(b: vec4<f32>) -> vec2<f32> {
    return vec2<f32>(b.x - b.y, b.z - b.w);
}
fn pres(b: vec4<f32>) -> f32 {
    return 0.25 * (b.x + b.y + b.z + b.w);
}
fn advect_fluid(U: vec2<f32>) -> vec4<f32> {
    var p = U;
    let s1 = bilerp0(p);
    p = U - 0.5 * vel(s1);
    let s2 = bilerp0(p);
    p = U - 0.5 * vel(s2);
    return bilerp0(p);
}
// Multi-scale rotation (flockaroo's curl vortex field)
const ROT_NUM = 5u;
const PI = 3.14159265;
const ANG = 4.0 * PI / f32(ROT_NUM);
fn get_rot_matrix() -> mat2x2<f32> {
    return mat2x2<f32>(
        vec2<f32>(cos(ANG), sin(ANG)),
        vec2<f32>(-sin(ANG), cos(ANG))
    );
}
// texture feedback with luminance based signal
fn rot_sample(pos: vec2<f32>, R: vec2<f32>) -> vec2<f32> {
    let uv = fract(pos / R);
    let px = uv * R;
    let distorted = bilerp2(px);
    let original = textureSampleLevel(channel0, channel0_sampler, uv, 0.0);
    let c = mix(distorted, original, 0.15);
    return vec2<f32>(
        dot(c.rgb, vec3<f32>(0.299, 0.587, 0.114)),
        dot(c.rgb, vec3<f32>(0.587, 0.114, 0.299))
    );
}
fn get_rot(pos: vec2<f32>, b: vec2<f32>, R: vec2<f32>) -> f32 {
    var p = b;
    var rot = 0.0;
    let m = get_rot_matrix();
    for (var i = 0u; i < ROT_NUM; i = i + 1u) {
        let s = rot_sample(pos + p, R);
        rot += dot(s - vec2<f32>(0.5), p.yx * vec2<f32>(1.0, -1.0));
        p = m * p;
    }
    return rot / f32(ROT_NUM) / dot(b, b);
}
fn hash2d(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}
fn smooth_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash2d(i), hash2d(i + vec2<f32>(1.0, 0.0)), u.x),
        mix(hash2d(i + vec2<f32>(0.0, 1.0)), hash2d(i + vec2<f32>(1.0, 1.0)), u.x),
        u.y
    );
}
fn fbm(p: vec2<f32>, t: f32) -> f32 {
    let drift = vec2<f32>(sin(t * 0.07) * 0.4, cos(t * 0.05) * 0.4);
    return smooth_noise(p + drift) * 0.5
         + smooth_noise(p * 2.13 + vec2<f32>(5.2, 1.3) + drift * 1.1) * 0.25
         + smooth_noise(p * 4.37 + vec2<f32>(2.8, 7.1) + drift * 1.5) * 0.125
         + smooth_noise(p * 8.71 + vec2<f32>(9.4, 3.7) + drift * 2.0) * 0.0625;
}
fn curl_noise(uv: vec2<f32>, t: f32) -> vec2<f32> {
    let e = 0.01;
    let f1 = fbm(uv + vec2<f32>(e, 0.0), t);
    let f2 = fbm(uv - vec2<f32>(e, 0.0), t);
    let f3 = fbm(uv + vec2<f32>(0.0, e), t);
    let f4 = fbm(uv - vec2<f32>(0.0, e), t);
    return vec2<f32>(f3 - f4, -(f1 - f2)) / (2.0 * e);
}

fn lb_curl(U: vec2<f32>) -> f32 {
    let vR = vel(bilerp0(U + vec2<f32>(1.0, 0.0)));
    let vL = vel(bilerp0(U - vec2<f32>(1.0, 0.0)));
    let vT = vel(bilerp0(U + vec2<f32>(0.0, 1.0)));
    let vB = vel(bilerp0(U - vec2<f32>(0.0, 1.0)));
    return 0.5 * ((vR.y - vL.y) - (vT.x - vB.x));
}

// input_texture0 = self (prev frame)
// input_texture1 = color_map (prev frame)
@compute @workgroup_size(16, 16, 1)
fn fluid_sim(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let R = vec2<f32>(dims);
    let U = vec2<f32>(id.xy);
    let uv = U / R;
    var Q = advect_fluid(U);
    let n = advect_fluid(U + vec2<f32>(0.0, 1.0));
    let e = advect_fluid(U + vec2<f32>(1.0, 0.0));
    let s = advect_fluid(U - vec2<f32>(0.0, 1.0));
    let w = advect_fluid(U - vec2<f32>(1.0, 0.0));
    let px = 0.25 * (pres(e) - pres(w));
    let py = 0.25 * (pres(n) - pres(s));
    let exchange = 0.25 * (n.w + e.y + s.z + w.x) - pres(Q);
    let gradient = vec4<f32>(px, -px, py, -py);
    Q += exchange - params.pressure_scale * gradient;
    let tex_color = bilerp1(U);
    let z = params.texture_influence * (0.8 - length(tex_color.xyz));
    let visc = params.viscosity * 0.01;
    Q = mix(Q, 0.25 * (n + e + s + w), visc);
    Q = mix(Q, vec4<f32>(pres(Q)), visc * clamp(1.0 - z, 0.0, 1.0));
    let curl_c = lb_curl(U);
    let curl_r = lb_curl(U + vec2<f32>(1.0, 0.0));
    let curl_l = lb_curl(U - vec2<f32>(1.0, 0.0));
    let curl_t = lb_curl(U + vec2<f32>(0.0, 1.0));
    let curl_b = lb_curl(U - vec2<f32>(0.0, 1.0));
    var eta = vec2<f32>(abs(curl_r) - abs(curl_l), abs(curl_t) - abs(curl_b)) * 0.5;
    let eta_len = length(eta) + 1e-5;
    eta = eta / eta_len;
    let vort_force = params.vortex_strength * curl_c * vec2<f32>(eta.y, -eta.x);
    Q.x += vort_force.x;
    Q.y -= vort_force.x;
    Q.z += vort_force.y;
    Q.w -= vort_force.y;
    let grav_sign = (uv.y - 0.5) * 2.0;
    let grav_force = params.gravity * z * grav_sign;
    Q.z -= grav_force;
    Q.w += grav_force;
    let curl = curl_noise(uv * 3.0, time_data.time) * params.turbulence;
    Q.x += curl.x * 0.5;
    Q.y -= curl.x * 0.5;
    Q.z += curl.y * 0.5;
    Q.w -= curl.y * 0.5;
    // gaussian falloffs for preventing disturbing calm areas
    let t = time_data.time;
    for (var seed = 0u; seed < 4u; seed = seed + 1u) {
        let fs = f32(seed);
        let orbit_angle = fs * 1.5708 + t * (0.06 + fs * 0.02);
        let orbit_r = 0.18 + 0.1 * sin(t * 0.04 + fs * 2.0);
        let center = vec2<f32>(
            0.5 + orbit_r * cos(orbit_angle),
            0.5 + orbit_r * sin(orbit_angle * 0.7 + fs)
        );
        let d = uv - center;
        let dist2 = dot(d, d);
        // Gaussian envelope nearby regions
        let envelope = exp(-dist2 / 0.02);
        let pulse = sin(t * (0.15 + fs * 0.08) + fs * 1.5) * 0.5 + 0.5;
        let dir = select(-1.0, 1.0, seed % 2u == 0u);
        let strength = params.turbulence * 10.0 * pulse * dir * envelope;
        let force = strength * vec2<f32>(-d.y, d.x) / (dist2 + 0.05);
        Q.x += force.x * 0.06;
        Q.y -= force.x * 0.06;
        Q.z += force.y * 0.06;
        Q.w -= force.y * 0.06;
    }
    // per pixel drift damping
    let v_current = vel(Q);
    let damp = 0.0003;
    Q.x -= v_current.x * damp;
    Q.y += v_current.x * damp;
    Q.z -= v_current.y * damp;
    Q.w += v_current.y * damp;
    // global energy dissipation
    Q *= 0.9998;
    Q = clamp(Q, vec4<f32>(-5.0), vec4<f32>(5.0));
    if (time_data.frame < 2u) {
        let tex = textureSampleLevel(channel0, channel0_sampler, uv, 0.0);
        Q = vec4<f32>(
            0.2 + 0.02 * tex.r,
            0.2 - 0.02 * tex.r,
            0.2 + 0.02 * tex.g,
            0.2 - 0.02 * tex.g
        );
    }
    textureStore(output, id.xy, Q);
}
// input_texture0 = fluid_sim (current frame) - LB velocity
// input_texture1 = self (prev frame) - position tracking
// input_texture2 = color_map (prev frame) - for rotation sampling
@compute @workgroup_size(16, 16, 1)
fn position_field(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let R = vec2<f32>(dims);
    let U = vec2<f32>(id.xy);
    var b = vec2<f32>(cos(ANG), sin(ANG));
    var rot_v = vec2<f32>(0.0);
    let m = get_rot_matrix();
    let bb_max_sq = (0.7 * R.y) * (0.7 * R.y);
    let min_scale_sq = 9.0;
    for (var l = 0u; l < 20u; l = l + 1u) {
        if (dot(b, b) > bb_max_sq) { break; }
        let scale_weight = smoothstep(min_scale_sq * 0.25, min_scale_sq, dot(b, b));
        var p = b;
        for (var i = 0u; i < ROT_NUM; i = i + 1u) {
            rot_v += p.yx * get_rot(U + p, b, R) * scale_weight;
            p = m * p;
        }
        b *= 2.0;
    }
    // rot displacement
    let rot_scale = mix(1.2, 0.4, params.mixing);
    let rot_disp = rot_v * vec2<f32>(-1.0, 1.0) * params.flow_speed * rot_scale;
    // LB velocity: carries positions with pressure driven flow
    let lb_scale = 0.4 + params.mixing * 0.6;
    let lb_c = vel(bilerp0(U));
    let lb_n = vel(bilerp0(U + vec2<f32>(0.0, 1.0)));
    let lb_e = vel(bilerp0(U + vec2<f32>(1.0, 0.0)));
    let lb_s = vel(bilerp0(U - vec2<f32>(0.0, 1.0)));
    let lb_w = vel(bilerp0(U - vec2<f32>(1.0, 0.0)));
    let lb_blur = 0.3 + params.mixing * 0.4;
    let lb_smooth = mix(lb_c, 0.25 * (lb_n + lb_e + lb_s + lb_w), lb_blur);
    let lb_vel = lb_smooth * params.flow_speed * lb_scale;
    var total_vel = rot_disp + lb_vel;
    let vel_len = length(total_vel);
    let max_vel = 8.0 + params.mixing * 6.0;
    if (vel_len > max_vel) {
        total_vel = total_vel * max_vel / vel_len;
    }
    // Two half-step trace-back
    var p2 = U - 0.5 * total_vel;
    let lb_mid = vel(bilerp0(p2)) * params.flow_speed * lb_scale;
    var total_vel2 = rot_disp + lb_mid;
    let vel_len2 = length(total_vel2);
    if (vel_len2 > max_vel) {
        total_vel2 = total_vel2 * max_vel / vel_len2;
    }
    p2 = U - 0.5 * total_vel2;
    p2 = p2 - R * floor(p2 / R);
    // Look up previous position field at traced back location
    var Q = bilerp1(p2);
    // Neighbor sampling
    let Nr = bilerp1(U + vec2<f32>(0.0, 1.0));
    let Er = bilerp1(U + vec2<f32>(1.0, 0.0));
    let Sr = bilerp1(U - vec2<f32>(0.0, 1.0));
    let Wr = bilerp1(U - vec2<f32>(1.0, 0.0));
    // in a "healthy field", neighbors should be within a few pixels of each other.
    let dN = length(Nr.xy - Q.xy);
    let dE = length(Er.xy - Q.xy);
    let dS = length(Sr.xy - Q.xy);
    let dW = length(Wr.xy - Q.xy);
    let max_neighbor_diff = max(max(dN, dE), max(dS, dW));
    // How "broken" is this region? >3px difference = something is wrong, >10px = shattered
    let broken = smoothstep(3.0, 12.0, max_neighbor_diff);
    let Nun = Q.xy + (Nr.xy - Q.xy - R * round((Nr.xy - Q.xy) / R));
    let Eun = Q.xy + (Er.xy - Q.xy - R * round((Er.xy - Q.xy) / R));
    let Sun = Q.xy + (Sr.xy - Q.xy - R * round((Sr.xy - Q.xy) / R));
    let Wun = Q.xy + (Wr.xy - Q.xy - R * round((Wr.xy - Q.xy) / R));
    let neighbor_avg = vec4<f32>(0.25 * (Nun + Eun + Sun + Wun), 0.0, 0.0);
    //Flux-weighted position mixing
    let fq = bilerp0(U);
    let f_n = bilerp0(U + vec2<f32>(0.0, 1.0));
    let f_e = bilerp0(U + vec2<f32>(1.0, 0.0));
    let f_s = bilerp0(U - vec2<f32>(0.0, 1.0));
    let f_w = bilerp0(U - vec2<f32>(1.0, 0.0));
    let flux_n = f_n.w - fq.z;
    let flux_e = f_e.y - fq.x;
    let flux_s = f_s.z - fq.w;
    let flux_w = f_w.x - fq.y;
    let flux_health = 1.0 - broken;
    let flux_scale = (0.2 + params.mixing * 0.5) * flux_health;
    Q = vec4<f32>(Q.xy + flux_scale * (
        flux_n * (Nun - Q.xy) +
        flux_e * (Eun - Q.xy) +
        flux_s * (Sun - Q.xy) +
        flux_w * (Wun - Q.xy)
    ), Q.zw);
    // Adaptive diffusion
    let dpdx = (Eun - Wun) * 0.5;
    let dpdy = (Nun - Sun) * 0.5;
    let stretch = max(length(dpdx), length(dpdy));
    let stretch_diffuse = smoothstep(3.0, 10.0, stretch) * 0.15;
    let heal_diffuse = broken * 0.3;
    // Higher mixing = more LB transport
    let mix_diffuse = params.mixing * params.mixing * 0.15;
    Q = mix(Q, neighbor_avg, 0.003 + stretch_diffuse + heal_diffuse + mix_diffuse);
    Q = mix(Q, neighbor_avg, params.pos_diffusion * 0.04);
    // Drift limit
    let disp = Q.xy - U - R * round((Q.xy - U) / R);
    let drift_dist = length(disp);
    let drift_pull = smoothstep(R.y * 0.35, R.y * 0.6, drift_dist) * 0.008;
    Q = vec4<f32>(Q.xy - disp * drift_pull, Q.zw);
    // Aggressive reseed for shattered regions
    let reseed = broken * 0.08;
    Q = mix(Q, vec4<f32>(U, 0.0, 0.0), reseed);
    Q = vec4<f32>(Q.x - R.x * floor(Q.x / R.x), Q.y - R.y * floor(Q.y / R.y), Q.zw);
    if (time_data.frame < 2u) {
        Q = vec4<f32>(U, 0.0, 0.0);
    }
    textureStore(output, id.xy, Q);
}
// input_texture0 = position_field
@compute @workgroup_size(16, 16, 1)
fn color_map(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let R = vec2<f32>(dims);
    let U = vec2<f32>(id.xy);
    let pos = bilerp0(U);
    let Q = textureSampleLevel(channel0, channel0_sampler, pos.xy / R, 0.0);
    textureStore(output, id.xy, Q);
}

// input_texture0 = color_map (current frame)
// input_texture1 = fluid_sim (current frame)
@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let R = vec2<f32>(dims);
    let U = vec2<f32>(id.xy);
    let uv = U / R;
    let base = bilerp0(U);
    // surface normal from color_map gradients
    let vn = length(bilerp0(U + vec2<f32>(0.0, 1.0)).xyz);
    let ve = length(bilerp0(U + vec2<f32>(1.0, 0.0)).xyz);
    let vs = length(bilerp0(U - vec2<f32>(0.0, 1.0)).xyz);
    let vw = length(bilerp0(U - vec2<f32>(1.0, 0.0)).xyz);
    let normal = normalize(vec3<f32>(ve - vw, vn - vs, 0.3));
    let light = normalize(vec3<f32>(
        3.0 + 0.2 * sin(time_data.time * 0.5),
        3.0 + 0.2 * cos(time_data.time * 0.5),
        2.0
    ));
    let diff = clamp(dot(normal, light), params.light_intensity, 1.0);
    let view_dir = vec3<f32>(0.0, 0.0, 1.0);
    let refl = reflect(-light, normal);
    let spec = pow(
        clamp(dot(refl, view_dir), 0.0, 1.0),
        params.spec_power
    ) * params.spec_intensity * 0.1;
    var final_color = base * vec4<f32>(vec3<f32>(diff), 1.0) + vec4<f32>(vec3<f32>(spec), 0.0);
    let fluid = bilerp1(U);
    let v = vel(fluid);
    let speed = length(v);
    final_color += vec4<f32>(v.x * 0.003, speed * 0.001, v.y * 0.003, 0.0);
    let lum = dot(final_color.rgb, vec3<f32>(0.299, 0.587, 0.114));
    final_color = vec4<f32>(mix(vec3<f32>(lum), final_color.rgb, params.color_vibrancy), 1.0);
    final_color = pow(max(final_color, vec4<f32>(0.0)), vec4<f32>(params.gamma));
    textureStore(output, id.xy, final_color);
}
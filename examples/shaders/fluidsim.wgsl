// Navier-Stokes Fluid Simulation
// Ported from Pavel Dobryakov's WebGL Fluid Simulation, MIT License
// https://github.com/PavelDoGreat/WebGL-Fluid-Simulation

struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
}

struct FluidParams {
    sim_width: u32,
    sim_height: u32,
    display_width: u32,
    display_height: u32,
    dt: f32,
    time: f32,
    velocity_dissipation: f32,
    density_dissipation: f32,
    pressure_val: f32,
    curl_strength: f32,
    splat_radius: f32,
    splat_x: f32,
    splat_y: f32,
    splat_dx: f32,
    splat_dy: f32,
    splat_force: f32,
    splat_color_r: f32,
    splat_color_g: f32,
    splat_color_b: f32,
    vel_ping: u32,
    prs_ping: u32,
    dye_ping: u32,
    do_splat: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> time_data: TimeUniform;
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: FluidParams;

// Storage buffer layout:
// velocity0: sim_width * sim_height * 2
// velocity1: sim_width * sim_height * 2
// pressure0: sim_width * sim_height
// pressure1: sim_width * sim_height
// divergence: sim_width * sim_height
// curl: sim_width * sim_height
// dye0: display_width * display_height * 4
// dye1: display_width * display_height * 4
@group(3) @binding(0) var<storage, read_write> fluid_data: array<f32>;

// Buffer offsets
fn velocity_offset(ping: u32) -> u32 {
    let sim_size = params.sim_width * params.sim_height;
    return select(0u, sim_size * 2u, ping == 1u);
}

fn pressure_offset(ping: u32) -> u32 {
    let sim_size = params.sim_width * params.sim_height;
    let base = sim_size * 4u;
    return base + select(0u, sim_size, ping == 1u);
}

fn divergence_offset() -> u32 {
    let sim_size = params.sim_width * params.sim_height;
    return sim_size * 6u;
}

fn curl_offset() -> u32 {
    let sim_size = params.sim_width * params.sim_height;
    return sim_size * 7u;
}

fn dye_offset(ping: u32) -> u32 {
    let sim_size = params.sim_width * params.sim_height;
    let display_size = params.display_width * params.display_height;
    let base = sim_size * 8u;
    return base + select(0u, display_size * 4u, ping == 1u);
}

// Helper functions for buffer access
fn get_velocity(x: i32, y: i32, ping: u32) -> vec2<f32> {
    let cx = clamp(x, 0, i32(params.sim_width) - 1);
    let cy = clamp(y, 0, i32(params.sim_height) - 1);
    let idx = velocity_offset(ping) + u32(cy) * params.sim_width * 2u + u32(cx) * 2u;
    return vec2<f32>(fluid_data[idx], fluid_data[idx + 1u]);
}

fn set_velocity(x: u32, y: u32, ping: u32, v: vec2<f32>) {
    let idx = velocity_offset(ping) + y * params.sim_width * 2u + x * 2u;
    fluid_data[idx] = v.x;
    fluid_data[idx + 1u] = v.y;
}

fn get_pressure(x: i32, y: i32, ping: u32) -> f32 {
    let cx = clamp(x, 0, i32(params.sim_width) - 1);
    let cy = clamp(y, 0, i32(params.sim_height) - 1);
    let idx = pressure_offset(ping) + u32(cy) * params.sim_width + u32(cx);
    return fluid_data[idx];
}

fn set_pressure(x: u32, y: u32, ping: u32, p: f32) {
    let idx = pressure_offset(ping) + y * params.sim_width + x;
    fluid_data[idx] = p;
}

fn get_divergence(x: i32, y: i32) -> f32 {
    let cx = clamp(x, 0, i32(params.sim_width) - 1);
    let cy = clamp(y, 0, i32(params.sim_height) - 1);
    let idx = divergence_offset() + u32(cy) * params.sim_width + u32(cx);
    return fluid_data[idx];
}

fn set_divergence(x: u32, y: u32, d: f32) {
    let idx = divergence_offset() + y * params.sim_width + x;
    fluid_data[idx] = d;
}

fn get_curl(x: i32, y: i32) -> f32 {
    let cx = clamp(x, 0, i32(params.sim_width) - 1);
    let cy = clamp(y, 0, i32(params.sim_height) - 1);
    let idx = curl_offset() + u32(cy) * params.sim_width + u32(cx);
    return fluid_data[idx];
}

fn set_curl(x: u32, y: u32, c: f32) {
    let idx = curl_offset() + y * params.sim_width + x;
    fluid_data[idx] = c;
}

fn get_dye(x: i32, y: i32, ping: u32) -> vec4<f32> {
    let cx = clamp(x, 0, i32(params.display_width) - 1);
    let cy = clamp(y, 0, i32(params.display_height) - 1);
    let idx = dye_offset(ping) + u32(cy) * params.display_width * 4u + u32(cx) * 4u;
    return vec4<f32>(fluid_data[idx], fluid_data[idx + 1u], fluid_data[idx + 2u], fluid_data[idx + 3u]);
}

fn set_dye(x: u32, y: u32, ping: u32, d: vec4<f32>) {
    let idx = dye_offset(ping) + y * params.display_width * 4u + x * 4u;
    fluid_data[idx] = d.x;
    fluid_data[idx + 1u] = d.y;
    fluid_data[idx + 2u] = d.z;
    fluid_data[idx + 3u] = d.w;
}

fn sample_velocity(uv: vec2<f32>, ping: u32) -> vec2<f32> {
    let pos = uv * vec2<f32>(f32(params.sim_width), f32(params.sim_height)) - 0.5;
    let ipos = vec2<i32>(floor(pos));
    let frac = fract(pos);

    let v00 = get_velocity(ipos.x, ipos.y, ping);
    let v10 = get_velocity(ipos.x + 1, ipos.y, ping);
    let v01 = get_velocity(ipos.x, ipos.y + 1, ping);
    let v11 = get_velocity(ipos.x + 1, ipos.y + 1, ping);

    let v0 = mix(v00, v10, frac.x);
    let v1 = mix(v01, v11, frac.x);
    return mix(v0, v1, frac.y);
}

fn sample_dye(uv: vec2<f32>, ping: u32) -> vec4<f32> {
    let pos = uv * vec2<f32>(f32(params.display_width), f32(params.display_height)) - 0.5;
    let ipos = vec2<i32>(floor(pos));
    let frac = fract(pos);

    let d00 = get_dye(ipos.x, ipos.y, ping);
    let d10 = get_dye(ipos.x + 1, ipos.y, ping);
    let d01 = get_dye(ipos.x, ipos.y + 1, ping);
    let d11 = get_dye(ipos.x + 1, ipos.y + 1, ping);

    let d0 = mix(d00, d10, frac.x);
    let d1 = mix(d01, d11, frac.x);
    return mix(d0, d1, frac.y);
}

@compute @workgroup_size(16, 16, 1)
fn clear_buffers(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < params.sim_width && id.y < params.sim_height) {
        set_velocity(id.x, id.y, 0u, vec2<f32>(0.0, 0.0));
        set_velocity(id.x, id.y, 1u, vec2<f32>(0.0, 0.0));
        set_pressure(id.x, id.y, 0u, 0.0);
        set_pressure(id.x, id.y, 1u, 0.0);
        set_divergence(id.x, id.y, 0.0);
        set_curl(id.x, id.y, 0.0);
    }
    if (id.x < params.display_width && id.y < params.display_height) {
        set_dye(id.x, id.y, 0u, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        set_dye(id.x, id.y, 1u, vec4<f32>(0.0, 0.0, 0.0, 0.0));
    }
}

@compute @workgroup_size(16, 16, 1)
fn splat_velocity(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.sim_width || id.y >= params.sim_height) { return; }
    if (params.do_splat == 0u) { return; }

    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(f32(params.sim_width), f32(params.sim_height));
    var p = uv - vec2<f32>(params.splat_x, params.splat_y);
    let aspect = f32(params.display_width) / f32(params.display_height);
    p.x = p.x * aspect;

    let radius = params.splat_radius / 100.0;
    let weight = exp(-dot(p, p) / radius);
    let splat = weight * vec2<f32>(params.splat_dx, params.splat_dy);
    let base = get_velocity(i32(id.x), i32(id.y), params.vel_ping);
    set_velocity(id.x, id.y, params.vel_ping, base + splat);
}

@compute @workgroup_size(16, 16, 1)
fn splat_dye(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.display_width || id.y >= params.display_height) { return; }
    if (params.do_splat == 0u) { return; }

    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(f32(params.display_width), f32(params.display_height));
    var p = uv - vec2<f32>(params.splat_x, params.splat_y);
    let aspect = f32(params.display_width) / f32(params.display_height);
    p.x = p.x * aspect;

    let radius = params.splat_radius / 100.0;
    let weight = exp(-dot(p, p) / radius);
    let splat = weight * vec3<f32>(params.splat_color_r, params.splat_color_g, params.splat_color_b);
    let base = get_dye(i32(id.x), i32(id.y), params.dye_ping);
    set_dye(id.x, id.y, params.dye_ping, vec4<f32>(base.rgb + splat, 1.0));
}

@compute @workgroup_size(16, 16, 1)
fn curl_compute(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.sim_width || id.y >= params.sim_height) { return; }

    let x = i32(id.x);
    let y = i32(id.y);

    let L = get_velocity(x - 1, y, params.vel_ping).y;
    let R = get_velocity(x + 1, y, params.vel_ping).y;
    let T = get_velocity(x, y + 1, params.vel_ping).x;
    let B = get_velocity(x, y - 1, params.vel_ping).x;

    let vorticity = R - L - T + B;
    set_curl(id.x, id.y, 0.5 * vorticity);
}

@compute @workgroup_size(16, 16, 1)
fn vorticity_apply(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.sim_width || id.y >= params.sim_height) { return; }

    let x = i32(id.x);
    let y = i32(id.y);

    let L = get_curl(x - 1, y);
    let R = get_curl(x + 1, y);
    let T = get_curl(x, y + 1);
    let B = get_curl(x, y - 1);
    let C = get_curl(x, y);

    var force = 0.5 * vec2<f32>(abs(T) - abs(B), abs(R) - abs(L));
    let len = length(force) + 0.0001;
    force = force / len;
    force = force * params.curl_strength * C;
    force.y = -force.y;

    var velocity = get_velocity(x, y, params.vel_ping);
    velocity = velocity + force * params.dt;
    velocity = clamp(velocity, vec2<f32>(-1000.0), vec2<f32>(1000.0));

    set_velocity(id.x, id.y, 1u - params.vel_ping, velocity);
}

@compute @workgroup_size(16, 16, 1)
fn divergence_compute(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.sim_width || id.y >= params.sim_height) { return; }

    let x = i32(id.x);
    let y = i32(id.y);

    var L = get_velocity(x - 1, y, params.vel_ping).x;
    var R = get_velocity(x + 1, y, params.vel_ping).x;
    var T = get_velocity(x, y + 1, params.vel_ping).y;
    var B = get_velocity(x, y - 1, params.vel_ping).y;

    let C = get_velocity(x, y, params.vel_ping);

    // Boundary conditions
    if (x == 0) { L = -C.x; }
    if (x == i32(params.sim_width) - 1) { R = -C.x; }
    if (y == i32(params.sim_height) - 1) { T = -C.y; }
    if (y == 0) { B = -C.y; }

    let div = 0.5 * (R - L + T - B);
    set_divergence(id.x, id.y, div);
}

// Clear pressure with dissipation
@compute @workgroup_size(16, 16, 1)
fn pressure_clear(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.sim_width || id.y >= params.sim_height) { return; }

    let p = get_pressure(i32(id.x), i32(id.y), params.prs_ping);
    set_pressure(id.x, id.y, 1u - params.prs_ping, params.pressure_val * p);
}

// Jacobi pressure iteration
@compute @workgroup_size(16, 16, 1)
fn pressure_iterate(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.sim_width || id.y >= params.sim_height) { return; }

    let x = i32(id.x);
    let y = i32(id.y);

    let L = get_pressure(x - 1, y, params.prs_ping);
    let R = get_pressure(x + 1, y, params.prs_ping);
    let T = get_pressure(x, y + 1, params.prs_ping);
    let B = get_pressure(x, y - 1, params.prs_ping);
    let divergence = get_divergence(x, y);

    let pressure = (L + R + B + T - divergence) * 0.25;
    set_pressure(id.x, id.y, 1u - params.prs_ping, pressure);
}

@compute @workgroup_size(16, 16, 1)
fn gradient_subtract(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.sim_width || id.y >= params.sim_height) { return; }

    let x = i32(id.x);
    let y = i32(id.y);

    // Read pressure from prs_ping (final pressure after iterations)
    let L = get_pressure(x - 1, y, params.prs_ping);
    let R = get_pressure(x + 1, y, params.prs_ping);
    let T = get_pressure(x, y + 1, params.prs_ping);
    let B = get_pressure(x, y - 1, params.prs_ping);

    // Read velocity from vel_ping
    var velocity = get_velocity(x, y, params.vel_ping);
    velocity = velocity - vec2<f32>(R - L, T - B);

    // Write to opposite velocity buffer
    set_velocity(id.x, id.y, 1u - params.vel_ping, velocity);
}

@compute @workgroup_size(16, 16, 1)
fn advect_velocity(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.sim_width || id.y >= params.sim_height) { return; }

    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(f32(params.sim_width), f32(params.sim_height));

    // Get velocity at current position
    let vel = sample_velocity(uv, params.vel_ping);

    // Trace back
    let rdx = 1.0 / vec2<f32>(f32(params.sim_width), f32(params.sim_height));
    let coord = uv - params.dt * vel * rdx;

    // Sample velocity at traced position
    var result = sample_velocity(coord, params.vel_ping);

    // Apply dissipation
    let dissipation = 1.0 / (1.0 + params.velocity_dissipation * params.dt);
    result = result * dissipation;

    set_velocity(id.x, id.y, 1u - params.vel_ping, result);
}

// Advect dye field
@compute @workgroup_size(16, 16, 1)
fn advect_dye(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.display_width || id.y >= params.display_height) { return; }

    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(f32(params.display_width), f32(params.display_height));

    // Sample velocity (uses vel_ping for current velocity state)
    let vel = sample_velocity(uv, params.vel_ping);

    // Trace back using sim resolution rdx
    let rdx = 1.0 / vec2<f32>(f32(params.sim_width), f32(params.sim_height));
    let coord = uv - params.dt * vel * rdx;

    // Sample dye at traced position (uses dye_ping)
    var result = sample_dye(coord, params.dye_ping);

    // Apply dissipation
    let dissipation = 1.0 / (1.0 + params.density_dissipation * params.dt);
    result = result * dissipation;

    set_dye(id.x, id.y, 1u - params.dye_ping, result);
}

@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }

    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(dims);
    let texel = 1.0 / vec2<f32>(dims);

    // Sample dye with bilinear interpolation
    var c = sample_dye(uv, params.dye_ping).rgb;
    let intensity = length(c);


    if (intensity > 0.0) {
        // Sample neighbors for gradient-based shading
        let L = sample_dye(uv - vec2<f32>(texel.x, 0.0), params.dye_ping).rgb;
        let R = sample_dye(uv + vec2<f32>(texel.x, 0.0), params.dye_ping).rgb;
        let T = sample_dye(uv + vec2<f32>(0.0, texel.y), params.dye_ping).rgb;
        let B = sample_dye(uv - vec2<f32>(0.0, texel.y), params.dye_ping).rgb;

        // Compute gradient for normal
        let dx = length(R) - length(L);
        let dy = length(T) - length(B);

        // Normal from gradient
        let n = normalize(vec3<f32>(dx, dy, length(texel)));
        let l = vec3<f32>(0.0, 0.0, 1.0);

        // Diffuse shading - blend toward 1.0 as intensity drops for smooth fade
        let raw_diffuse = clamp(dot(n, l) + 0.7, 0.7, 1.0);
        // Smoothly reduce shading effect as fluid fades (prevents sudden appearance changes)
        let shade_factor = smoothstep(0.0, 0.1, intensity);
        let diffuse = mix(1.0, raw_diffuse, shade_factor);

        // Cubic fade
        let r = reflect(-l, n);
        let spec_raw = pow(max(0.0, dot(r, vec3<f32>(0.0, 0.0, 1.0))), 200.0);
        let specular = spec_raw * intensity * intensity * intensity;

        c = c * diffuse + specular;
    }

    c = max(c, vec3<f32>(0.0));
    c = pow(c, vec3<f32>(1.1));
    textureStore(output, id.xy, vec4<f32>(c, 1.0));
}

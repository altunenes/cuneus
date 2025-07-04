struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

struct LorenzParams {
    sigma: f32,          
    rho: f32,            
    beta: f32,           
    step_size: f32,      
    motion_speed: f32,  
    rotation_x: f32,     
    rotation_y: f32,     
    click_state: i32,    
    brightness: f32,     
    color1_r: f32,       
    color1_g: f32,      
    color1_b: f32,       
    color2_r: f32,       
    color2_g: f32,     
    color2_b: f32,       
    scale: f32,          
    dof_amount: f32,     
    dof_focal_dist: f32,
    gamma: f32,
    exposure: f32,
    particle_count: f32,
}
@group(1) @binding(0) var<uniform> params: LorenzParams;

@group(2) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(3) @binding(0) var<storage, read_write> atomic_buffer: array<atomic<u32>>;

alias v4 = vec4<f32>;
alias v3 = vec3<f32>;
alias v2 = vec2<f32>;
alias m2 = mat2x2<f32>;
alias m3 = mat3x3<f32>;
alias m4 = mat4x4<f32>;
const pi = 3.14159265359;
const tau = 6.28318530718;

var<private> R: v2;
var<private> seed: u32;

fn rot(a: f32) -> m2 { 
    return m2(cos(a), -sin(a), sin(a), cos(a)); 
}

fn rotX(a: f32) -> m3 {
    let r = rot(a); 
    return m3(1.0, 0.0, 0.0, 0.0, r[0][0], r[0][1], 0.0, r[1][0], r[1][1]);
}

fn rotY(a: f32) -> m3 {
    let r = rot(a); 
    return m3(r[0][0], 0.0, r[0][1], 0.0, 1.0, 0.0, r[1][0], 0.0, r[1][1]);
}

fn rotZ(a: f32) -> m3 {
    let r = rot(a); 
    return m3(r[0][0], r[0][1], 0.0, r[1][0], r[1][1], 0.0, 0.0, 0.0, 1.0);
}

fn hash_u(_a: u32) -> u32 { 
    var a = _a; 
    a ^= a >> 16;
    a *= 0x7feb352du;
    a ^= a >> 15;
    a *= 0x846ca68bu;
    a ^= a >> 16;
    return a; 
}

fn hash_f() -> f32 { 
    var s = hash_u(seed); 
    seed = s;
    return (f32(s) / f32(0xffffffffu)); 
}

fn hash_v2() -> v2 { 
    return v2(hash_f(), hash_f()); 
}

fn hash_v3() -> v3 { 
    return v3(hash_f(), hash_f(), hash_f()); 
}

fn sin_add(a: f32) -> f32 {
    return sin(a) * 0.5 + 0.5;
}

fn sample_disk() -> v2 {
    let r = hash_v2();
    return v2(sin(r.x * tau), cos(r.x * tau)) * sqrt(r.y);
}

fn get_cyclic_params(t: f32) -> vec3<f32> {
    let speed = params.motion_speed * 0.1;
    
    let sigma = params.sigma * (1.0 + 0.1 * sin(t * 0.27 * speed));
    let rho = params.rho * (1.0 + 0.05 * sin(t * 0.31 * speed));
    let beta = params.beta * (1.0 + 0.05 * sin(t * 0.41 * speed));
    
    return vec3<f32>(sigma, rho, beta);
}

fn lorenz_step(p: v3, lorenz_params: vec3<f32>, dt: f32) -> v3 {
    let sigma = lorenz_params.x;
    let rho = lorenz_params.y;
    let beta = lorenz_params.z;
    
    let dx = sigma * (p.y - p.x);
    let dy = p.x * (rho - p.z) - p.y;
    let dz = p.x * p.y - beta * p.z;
    
    return p + vec3<f32>(dx, dy, dz) * dt;
}

fn projParticle(p: v3) -> v3 {
    var projected = p;
    projected.x /= R.x/R.y;
    return projected;
}

fn aces_tonemap(color: v3) -> v3 {
    const m1 = mat3x3<f32>(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777
    );
    const m2 = mat3x3<f32>(
        1.60475, -0.10208, -0.00327,
        -0.53108,  1.10813, -0.07276,
        -0.07367, -0.00605,  1.07602
    );

    var v = m1 * color;    
    var a = v * (v + 0.0245786) - 0.000090537;
    var b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return m2 * (a / b);
}

fn densityField(pos: v3) -> f32 {
    let screen_pos = projParticle(pos);
    let uv = screen_pos.xy * 0.5 + 0.5;
    
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return 0.0;
    }
    
    let coords = vec2<u32>(uv * R);
    if (coords.x >= u32(R.x) || coords.y >= u32(R.y)) {
        return 0.0;
    }
    
    let idx = coords.x + u32(R.x) * coords.y;
    let density1 = f32(atomicLoad(&atomic_buffer[idx]));
    let density2 = f32(atomicLoad(&atomic_buffer[idx + u32(R.x * R.y)]));
    
    let total_density = (density1 + density2) * 0.01;
    let distance_from_center = length(pos);
    let falloff = exp(-distance_from_center * 0.1);
    
    return total_density * falloff;
}

fn volumetricTrace(ray_pos: v3, ray_dir: v3) -> v3 {
    var color = v3(0.0);
    var pos = ray_pos;
    let step_size = 0.05;
    var accumulated_density = 0.0;
    
    for (var i = 0; i < 150; i++) {
        let density = densityField(pos);
        
        if (density > 0.001) {
            let depth_factor = f32(i) / 150.0;
            let light_color = mix(
                vec3<f32>(params.color1_r, params.color1_g, params.color1_b),
                vec3<f32>(params.color2_r, params.color2_g, params.color2_b),
                depth_factor
            );
            
            let scattering = density * (1.0 - accumulated_density) * step_size;
            color += light_color * scattering * params.brightness * 5.0;
            
            accumulated_density += density * step_size * 0.1;
            
            if (accumulated_density > 0.95) {
                break;
            }
        }
        
        pos += ray_dir * step_size;
        
        if (length(pos) > 30.0) {
            break;
        }
    }
    
    let distance_fade = exp(-length(ray_pos) * 0.02);
    return color * distance_fade;
}

@compute @workgroup_size(256, 1, 1)
fn Splat(@builtin(global_invocation_id) id: vec3<u32>) {
    let Ru = vec2<u32>(textureDimensions(output));
    R = v2(Ru);
    let muv = v2(params.rotation_x, params.rotation_y);
    seed = id.x + hash_u(time_data.frame);
    
    // Dynamic iterations for smooth particle density
    let iters = 40 + i32(sin_add(time_data.time * 0.1) * 80.0);
    let base_time = time_data.time * 0.3 + hash_f() * 0.2;
    
    // DOF setup
    var focusDist = params.dof_focal_dist * 2.0 - 1.0;
    if (params.click_state > 0) {
        focusDist = muv.y * 2.0;
    }
    let dofFac = 1.0 / v2(R.x/R.y, 1.0) * params.dof_amount;
    
    let lorenz_params = get_cyclic_params(base_time);
    
    // Proper Lorenz starting positions with small variations
    var p = v3(
        0.1 + (hash_f() * 2.0 - 1.0) * 0.1,
        0.001 + (hash_f() * 2.0 - 1.0) * 0.1,
        (hash_f() * 2.0 - 1.0) * 0.1
    );
    
    let dt = params.step_size;
    
    // Burn-in to get onto the attractor
    for(var i = 0; i < 100; i++) {
        p = lorenz_step(p, lorenz_params, dt);
    }
    
    // Main rendering loop
    for(var i = 0; i < iters; i++) {
        p = lorenz_step(p, lorenz_params, dt);
        
        // Skip some iterations for better distribution
        if (i % 2 != 0) {
            continue;
        }
        
        // Transform and project
        var view_p = p * params.scale;
        view_p = rotY(muv.x * 2.0) * view_p;
        view_p = rotX(muv.y * 1.0) * view_p;
        view_p = rotZ(sin(time_data.time * 0.05) * 0.2) * view_p;
        
        var q = projParticle(view_p);
        
        // DOF effect
        var k = q.xy;
        let d = q.z - focusDist;
        k += sample_disk() * abs(d) * 0.08 * dofFac;
        
        // Map to screen coordinates
        let uv = k.xy/2.0 + 0.5;
        let cc = vec2<u32>(uv.xy * R.xy);
        let idx = cc.x + Ru.x * cc.y;
        
        // Beautiful color variation based on position
        let color_shift = sin(p.x * 1.5) * cos(p.y * 1.5) * 0.5 + 0.5;
        
        // Store particle
        if (uv.x > 0.0 && uv.x < 1.0 && uv.y > 0.0 && uv.y < 1.0 && idx < u32(Ru.x*Ru.y)) {
            let base_intensity = 120.0 * params.exposure; // Use exposure parameter
            atomicAdd(&atomic_buffer[idx], u32((1.0 - color_shift) * base_intensity));
            atomicAdd(&atomic_buffer[idx + Ru.x*Ru.y], u32(color_shift * base_intensity * 1.5 + f32(i % 3) * 30.0));
        }
    }
}

@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let res = vec2<u32>(textureDimensions(output));
    if (id.x >= res.x || id.y >= res.y) { return; }
    let hist_id = id.x + u32(res.x) * id.y;
    var col1 = f32(atomicLoad(&atomic_buffer[hist_id])) * vec3<f32>(params.color1_r, params.color1_g, params.color1_b);
    var col2 = f32(atomicLoad(&atomic_buffer[hist_id + res.x*res.y])) * vec3<f32>(params.color2_r, params.color2_g, params.color2_b);
    var col = (col1 + col2) * params.brightness;
    
    let uv = vec2<f32>(id.xy) / vec2<f32>(res);
    let screen_pos = (uv - 0.5) * 2.0;
    
    let aspect = vec2<f32>(res).x / vec2<f32>(res).y;
    let ray_dir = normalize(vec3<f32>(screen_pos.x * aspect, screen_pos.y, 1.5));
    let ray_pos = vec3<f32>(0.0, 0.0, -15.0);
    
    let muv = v2(params.rotation_x, params.rotation_y);
    var transformed_ray_dir = ray_dir;
    transformed_ray_dir = rotY(muv.x * tau) * transformed_ray_dir;
    transformed_ray_dir = rotX(muv.y * pi) * transformed_ray_dir;
    
    var transformed_ray_pos = ray_pos;
    transformed_ray_pos = rotY(muv.x * tau) * transformed_ray_pos;
    transformed_ray_pos = rotX(muv.y * pi) * transformed_ray_pos;
    
    let volumetric_color = volumetricTrace(transformed_ray_pos, transformed_ray_dir);
    
    let particle_strength = length(col);
    let volumetric_strength = length(volumetric_color);
    let blend_factor = 0.7 + 0.3 * sin(time_data.time * 0.1);
    
    col = mix(col, col + volumetric_color, blend_factor);
    col += volumetric_color * 0.5;
    
    let color_shift = sin(time_data.time * 0.2) * 0.2;
    col = mix(
        col,
        col * vec3<f32>(1.0 + color_shift, 1.0, 1.0 - color_shift),
        0.3
    );
    
    col = aces_tonemap(col);
    
    col = pow(col, vec3<f32>(1.0 / params.gamma));
    
    col += vec3<f32>(0.001, 0.001, 0.003);
    
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(col, 1.0));
    
    atomicStore(&atomic_buffer[hist_id], 0u);
    atomicStore(&atomic_buffer[hist_id + res.x*res.y], 0u);
}
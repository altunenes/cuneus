
// Enes Altun, 2025; MIT License
// formula: https://paulbourke.net/fractals/3dmandelbrot/
struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

struct QuadraticBulbParams {
    mouse_x: f32,
    mouse_y: f32,
    max_bounces: u32,
    samples_per_pixel: u32,
    accumulate: u32,
    _padding1: u32,
    
    derivative_power: f32,
    derivative_offset: f32,
    transform_coeff_xy: f32,
    transform_coeff_xz: f32,
    
    animation_speed: f32,
    hold_duration: f32,
    transition_duration: f32,
    
    exposure: f32,
    focal_length: f32,
    dof_strength: f32,
    
    palette_a_r: f32,
    palette_a_g: f32,
    palette_a_b: f32,
    palette_b_r: f32,
    palette_b_g: f32,
    palette_b_b: f32,
    palette_c_r: f32,
    palette_c_g: f32,
    palette_c_b: f32,
    palette_d_r: f32,
    palette_d_g: f32,
    palette_d_b: f32,
    
    manual_rotation_x: f32,
    manual_rotation_y: f32,
    manual_rotation_z: f32,
    use_mouse_rotation: u32,
    
    gamma: f32,
    zoom: f32,
    
    background_r: f32,
    background_g: f32,
    background_b: f32,
    sun_color_r: f32,
    sun_color_g: f32,
    sun_color_b: f32,
    fog_color_r: f32,
    fog_color_g: f32,
    fog_color_b: f32,
    glow_color_r: f32,
    glow_color_g: f32,
    glow_color_b: f32,
}
@group(1) @binding(0) var<uniform> params: QuadraticBulbParams;

@group(2) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(3) @binding(0) var<storage, read_write> atomic_buffer: array<atomic<u32>>;

alias v4 = vec4<f32>;
alias v3 = vec3<f32>;
alias v2 = vec2<f32>;
alias m3 = mat3x3<f32>;
const pi = 3.14159265359;
const tau = 6.28318530718;
const sqrt3 = 1.7320508075688772;

var<private> R: v2;
var<private> seed: u32;

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

fn hash22(p: v2) -> v2 {
    var p_mod = p + 1.61803398875;
    var p3 = fract(v3(p_mod.x, p_mod.y, p_mod.x) * v3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

fn hash12(p: v2) -> f32 {
    var p3 = fract(v3(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn rotation_x(angle: f32) -> m3 {
    let c = cos(angle);
    let s = sin(angle);
    return m3(1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c);
}

fn rotation_y(angle: f32) -> m3 {
    let c = cos(angle);
    let s = sin(angle);
    return m3(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c);
}

fn rotation_z(angle: f32) -> m3 {
    let c = cos(angle);
    let s = sin(angle);
    return m3(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0);
}

fn pal(t: f32, a: v3, b: v3, c: v3, d: v3) -> v3 {
    return a + b * cos(tau * (c * t + d));
}

fn spectrum(n: f32) -> v3 {
    let a = v3(params.palette_a_r, params.palette_a_g, params.palette_a_b);
    let b = v3(params.palette_b_r, params.palette_b_g, params.palette_b_b);
    let c = v3(params.palette_c_r, params.palette_c_g, params.palette_c_b);
    let d = v3(params.palette_d_r, params.palette_d_g, params.palette_d_b);
    return pal(n, a, b, c, d);
}

fn smootherstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn quadratic_mandelbrot_3d(pos: v3) -> v4 {
    var z = pos;
    var dr = 1.0;
    var r = 0.0;
    var trap = v4(abs(z), dot(z, z));
    
    for (var i = 0; i < 12; i++) {
        r = length(z);
        if (r > 2.2) { break; }
        
        if (r < 0.0008) { break; }
        
        dr = params.derivative_power * r * dr + params.derivative_offset;
        
        let x = z.x;
        let y = z.y;
        let zz = z.z;
        
        z = v3(
            x * x - y * y - zz * zz,
            params.transform_coeff_xy * x * y,
            params.transform_coeff_xz * x * zz
        );
        
        z += pos;
        
        trap = min(trap, v4(abs(z), dot(z, z)));
    }
    
    let log_r = log(max(r, 0.0001));
    return v4(0.5 * log_r * r / max(dr, 0.0001), trap.yzw);
}

struct Model {
    d: f32,
    uvw: v3,
    albedo: v3,
    id: i32,
}

struct Material {
    albedo: v3,
    specular: f32,
    roughness: f32,
}

fn shade_model(model: Model) -> Material {
    let color_val = length(model.albedo.xy) * 0.4 + length(model.uvw) * 0.15;
    let col = spectrum(clamp(color_val, 0.0, 1.0)) * mix(0.7, 1.3, smoothstep(0.0, 0.6, color_val));
    return Material(col, 0.0, 0.0);
}

fn map_scene(p: v3, rotation: m3) -> Model {
    let rotated_pos = rotation * p;
    let scaled_pos = rotated_pos * 0.8;
    let res = quadratic_mandelbrot_3d(scaled_pos);
    let d = res.x / 0.8;
    let orbit_trap = res.yzw;
    return Model(d, p, orbit_trap, 1);
}

fn calc_normal(p: v3, rotation: m3) -> v3 {
    let eps = 0.1;
    let h = v2(eps, 0.0);
    return normalize(v3(
        map_scene(p + v3(h.x, h.y, h.y), rotation).d - map_scene(p - v3(h.x, h.y, h.y), rotation).d,
        map_scene(p + v3(h.y, h.x, h.y), rotation).d - map_scene(p - v3(h.y, h.x, h.y), rotation).d,
        map_scene(p + v3(h.y, h.y, h.x), rotation).d - map_scene(p - v3(h.y, h.y, h.x), rotation).d
    ));
}

struct Hit {
    model: Model,
    pos: v3,
    ray_length: f32,
}

fn march(origin: v3, ray_direction: v3, max_dist: f32, understep: f32, rotation: m3) -> Hit {
    var ray_position = origin;
    var ray_length = 0.0;
    var model: Model;

    for (var i = 0; i < 140; i++) {
        model = map_scene(ray_position, rotation);
        ray_length += model.d * understep;
        ray_position = origin + ray_direction * ray_length;

        if (model.d < 0.0015) { break; }

        if (ray_length > max_dist) {
            model.id = 0;
            break;
        }
    }
    
    return Hit(model, ray_position, ray_length);
}

fn ortho(a: v3) -> v3 {
    let b = cross(v3(-1.0, -1.0, 0.5), a);
    return b;
}

fn get_sample_biased(dir: v3, power: f32, rnd: v2) -> v3 {
    let normalized_dir = normalize(dir);
    let o1 = normalize(ortho(normalized_dir));
    let o2 = normalize(cross(normalized_dir, o1));
    var r = rnd;
    r.x = r.x * tau;
    r.y = pow(r.y, 1.0 / (power + 1.0));
    let oneminus = sqrt(1.0 - r.y * r.y);
    return cos(r.x) * oneminus * o1 + sin(r.x) * oneminus * o2 + r.y * normalized_dir;
}

fn get_cone_sample(dir: v3, extent: f32, rnd: v2) -> v3 {
    let normalized_dir = normalize(dir);
    let o1 = normalize(ortho(normalized_dir));
    let o2 = normalize(cross(normalized_dir, o1));
    var r = rnd;
    r.x = r.x * tau;
    r.y = 1.0 - r.y * extent;
    let oneminus = sqrt(1.0 - r.y * r.y);
    return cos(r.x) * oneminus * o1 + sin(r.x) * oneminus * o2 + r.y * normalized_dir;
}

fn rnd_unit2(rnd: v2) -> v2 {
    let h = rnd * v2(1.0, tau);
    let phi = h.y;
    let r = sqrt(h.x);
    return r * v2(sin(phi), cos(phi));
}

fn env(dir: v3) -> v3 {
    let glow_color = v3(params.glow_color_r, params.glow_color_g, params.glow_color_b);
    let background_color = v3(params.background_r, params.background_g, params.background_b);
    return mix(background_color * 0.15, glow_color, 
               smoothstep(-0.25, 0.25, dot(dir, normalize(v3(0.4, 1.0, -0.6)))));
}

fn sample_direct(hit: Hit, nor: v3, throughput: v3, rnd: v2, rotation: m3) -> v3 {
    let sun_pos = normalize(v3(-0.3, 1.0, -0.7)) * 100.0;
    let sun_color = v3(params.sun_color_r, params.sun_color_g, params.sun_color_b);
    
    var col = v3(0.0);
    let light_dir = sun_pos - hit.pos;
    let light_sample_dir = get_cone_sample(light_dir, 0.001, rnd);
    let diffuse = dot(nor, light_sample_dir);
    let shadow_origin = hit.pos + nor * (0.0002 / abs(dot(light_sample_dir, nor)));
    
    if (diffuse > 0.0) {
        let sh = march(shadow_origin, light_sample_dir, 1.0, 2.5, rotation);
        if (sh.model.id == 0) {
            col += throughput * sun_color * diffuse;
        }
    }
    return col;
}

fn aces_tonemap(x: v3) -> v3 {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), v3(0.0), v3(1.0));
}

fn draw(frag_coord: v2, frame: u32) -> v4 {
    let p = (-R + 2.0 * frag_coord) / R.y;
    
    seed = u32(frag_coord.x) + u32(frag_coord.y) * u32(R.x) + frame * 719393;
    let initial_seed = hash22(frag_coord + f32(frame) * sqrt3);
    
    let jitter = 2.0 * (initial_seed - 0.5) / R;
    let jittered_p = p + jitter;
    let scaled_p = jittered_p * 1.4;

    var col = v3(0.0);

    let base_cam_pos = v3(6.0, 1.2, 2.5);
    let cam_pos = base_cam_pos * params.zoom;
    let focus_point = v3(0.0, 0.0, 0.0);
    let cam_tar = focus_point;
    
    var current_rotation: v3;
    
    if (params.use_mouse_rotation > 0) {
        let mouse_sensitivity = 5.0;
        current_rotation = v3(
            (params.mouse_y - 0.5) * mouse_sensitivity,
            (params.mouse_x - 0.5) * mouse_sensitivity,
            0.0
        );
    } else {
        current_rotation = v3(params.manual_rotation_x, params.manual_rotation_y, params.manual_rotation_z);
    }
    
    let rotation = rotation_z(current_rotation.z) * rotation_y(current_rotation.y) * rotation_x(current_rotation.x);
    
    let ww = normalize(cam_tar - cam_pos);
    let uu = normalize(cross(v3(0.0, 1.0, 0.0), ww));
    let vv = normalize(cross(ww, uu));
    let cam_mat = m3(-uu, vv, ww);
    
    let ray_dir = normalize(cam_mat * v3(scaled_p, params.focal_length));
    var origin = cam_pos;
    var ray_direction = ray_dir;
    
    let focal_point_dist = distance(cam_pos, focus_point);
    let focal_plane_dist = dot(cam_mat[2], ray_dir) * focal_point_dist;
    
    var hit = march(origin, ray_direction, min(focal_plane_dist, 8.0), 0.55, rotation);
    if (hit.model.id == 0) {
        let ray_focal_point = origin + ray_direction * focal_plane_dist;
        origin += cam_mat * v3(rnd_unit2(initial_seed), 0.0) * params.dof_strength;
        ray_direction = normalize(ray_focal_point - origin);
        origin = hit.pos;
        hit = march(origin, ray_direction, 10.0, 0.55, rotation);
    }

    var throughput = v3(1.0);
    let bg_col = v3(params.background_r, params.background_g, params.background_b);
    var ray_length = 0.0;
    
    for (var bounce = 0; bounce < i32(params.max_bounces); bounce++) {
        if (bounce > 0) {
            hit = march(origin, ray_direction, 6.0, 1.2, rotation);
        }
       
        if (hit.model.id == 0) {
            if (bounce > 0) {
                col += env(ray_direction) * throughput;
            }
            if (bounce == 0) {
                col = bg_col;
            }
            break;
        }

        ray_length += hit.ray_length;

        let nor = calc_normal(hit.pos, rotation);
        let material = shade_model(hit.model);
        
        throughput *= material.albedo;

        let rnd1 = hash_v2();
        let diffuse_ray_dir = get_sample_biased(nor, 1.0, rnd1);

        let rnd2 = hash_v2();
        col += sample_direct(hit, nor, throughput, rnd2, rotation);
        ray_direction = diffuse_ray_dir;
    
        origin = hit.pos + nor * (0.0002 / abs(dot(ray_direction, nor)));
    }

    let fog_color = v3(params.fog_color_r, params.fog_color_g, params.fog_color_b);
    let fog = 1.0 - exp((ray_length - focal_point_dist) * -1.0);
    col = mix(col, fog_color, clamp(fog, 0.0, 1.0));

    return v4(col, 1.0);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(output);
    R = v2(f32(dimensions.x), f32(dimensions.y));
    
    if (global_id.x >= dimensions.x || global_id.y >= dimensions.y) {
        return;
    }
    
    let frag_coord = v2(f32(global_id.x), f32(global_id.y)) + 0.5;
    let pixel_idx = global_id.x + dimensions.x * global_id.y;
    
    var pixel_color = v3(0.0);
    
    for (var s: u32 = 0; s < params.samples_per_pixel; s++) {
        let sample_result = draw(frag_coord, time_data.frame * params.samples_per_pixel + s);
        pixel_color += sample_result.rgb;
    }
    
    pixel_color /= f32(params.samples_per_pixel);
    pixel_color *= params.exposure;
    
    let should_accumulate = params.accumulate > 0 && time_data.frame > 0;
    
    if (should_accumulate) {
        let old_r = f32(atomicLoad(&atomic_buffer[pixel_idx * 3])) / 1000.0;
        let old_g = f32(atomicLoad(&atomic_buffer[pixel_idx * 3 + 1])) / 1000.0;
        let old_b = f32(atomicLoad(&atomic_buffer[pixel_idx * 3 + 2])) / 1000.0;
        let old_color = v3(old_r, old_g, old_b);
        
        let max_blend_frames = 32.0;
        let effective_frame = min(f32(time_data.frame), max_blend_frames);
        let blend_factor = 1.0 / (effective_frame + 1.0);

        pixel_color = mix(old_color, pixel_color, max(blend_factor, 0.05));
    }
    
    atomicStore(&atomic_buffer[pixel_idx * 3], u32(pixel_color.r * 1000.0));
    atomicStore(&atomic_buffer[pixel_idx * 3 + 1], u32(pixel_color.g * 1000.0));
    atomicStore(&atomic_buffer[pixel_idx * 3 + 2], u32(pixel_color.b * 1000.0));
    
    pixel_color = aces_tonemap(pixel_color);
    pixel_color = pow(pixel_color, v3(1.0 / params.gamma));
    
    textureStore(output, vec2<i32>(global_id.xy), v4(pixel_color, 1.0));
}
// Path tracer for Mandelbulb, Enes Altun, 2025. CC 3.0 
// This shader uses various techniques, citations and inspirations:
// - Hash functions (hash22, hash12): Dave_Hoskins https://www.shadertoy.com/view/4djSRW
// - Random unit disk sampling (rnd_unit2): iq https://www.shadertoy.com/view/tl23Rm
// - Spectrum palette function: iq https://www.shadertoy.com/view/ll2GD3
// - Path tracing setup and lighting: yx https://www.shadertoy.com/view/ts2cWm
// - Biased sampling technique: yx http://blog.hvidtfeldts.net/index.php/2015/01/path-tracing-3d-fractals/
// - Additional path tracing reference: demofox https://www.shadertoy.com/view/WsBBR3
// - Surface offset technique: https://www.shadertoy.com/view/lsXGzH
// - The idea for path tracing for 3D fractals: Kleinian Seahorse: https://www.shadertoy.com/view/Ns2fzy by tdhooper;
// - http://blog.hvidtfeldts.net/index.php/2015/01/path-tracing-3d-fractals/

struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

struct MandelbulbParams {
    power: f32,
    max_bounces: u32,
    samples_per_pixel: u32,
    accumulate: u32,

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

    rotation_x: f32,
    rotation_y: f32,
    rotation_z: f32,
    _pad: f32,
}
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: MandelbulbParams;

// Group 2: Global Engine Resources (mouse)
struct MouseUniform {
    position: vec2<f32>,
    click_position: vec2<f32>,
    wheel: vec2<f32>,
    buttons: vec2<u32>,
};
@group(2) @binding(0) var<uniform> mouse: MouseUniform;

// Group 3: Multipass input textures
@group(3) @binding(0) var input_texture0: texture_2d<f32>;
@group(3) @binding(1) var input_sampler0: sampler;

alias v4 = vec4<f32>;
alias v3 = vec3<f32>;
alias v2 = vec2<f32>;
alias m3 = mat3x3<f32>;
const pi = 3.14159265359;
const tau = 6.28318530718;
const sqrt3 = 1.7320508075688772;

const UNDERSTEP = 0.5;
const BOUNCE_UNDERSTEP = 2.0;

// defaults for best look
const FOCUS_OFFSET = v3(0.0, -0.1, 0.2);
const MANDEL_ROT_X = 0.0;
const MANDEL_ROT_Y = -1.9;
const MANDEL_ROT_Z = 1.7;
const FOCUS_CAM_POS = v3(0.8, 0.2, 0.4);
const FOCUS_CAM_TARGET = v3(0.0, 0.0, 0.0);
const FOCUS_SCALE = 1.6;
const FOCUS_MANDEL_SCALE = 2.0;

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

fn mandelbulb(pos: v3, power: f32) -> v4 {
    var z = pos;
    var dr = 1.0;
    var r = 0.0;
    var trap = v4(abs(z), dot(z, z));

    for (var i = 0; i < 15; i++) {
        r = length(z);
        if (r > 2.0) { break; }

        let theta = acos(z.z / r);
        let phi = atan2(z.y, z.x);
        dr = pow(r, power - 1.0) * power * dr + 1.0;

        let zr = pow(r, power);
        let new_theta = theta * power;
        let new_phi = phi * power;

        z = zr * v3(sin(new_theta) * cos(new_phi), sin(new_phi) * sin(new_theta), cos(new_theta));
        z += pos;

        trap = min(trap, v4(abs(z), dot(z, z)));
    }

    return v4(0.5 * log(r) * r / dr, trap.yzw);
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
    let color_val = length(model.albedo.xy) * 0.3 + length(model.uvw) * 0.1;
    let col = spectrum(clamp(color_val, 0.0, 1.0)) * mix(0.8, 2.0, smoothstep(0.0, 0.5, color_val));
    return Material(col, 0.0, 0.0);
}

fn map_scene(p: v3, rotation: m3, power: f32) -> Model {
    let rotated_pos = rotation * p;
    let scaled_pos = (rotated_pos - FOCUS_OFFSET) * FOCUS_MANDEL_SCALE;
    let res = mandelbulb(scaled_pos, power);
    let d = res.x / FOCUS_MANDEL_SCALE;
    let orbit_trap = res.yzw;
    return Model(d, p, orbit_trap, 1);
}

fn calc_normal(p: v3, rotation: m3, power: f32) -> v3 {
    let eps = 0.0001;
    let h = v2(eps, 0.0);
    return normalize(v3(
        map_scene(p + v3(h.x, h.y, h.y), rotation, power).d - map_scene(p - v3(h.x, h.y, h.y), rotation, power).d,
        map_scene(p + v3(h.y, h.x, h.y), rotation, power).d - map_scene(p - v3(h.y, h.x, h.y), rotation, power).d,
        map_scene(p + v3(h.y, h.y, h.x), rotation, power).d - map_scene(p - v3(h.y, h.y, h.x), rotation, power).d
    ));
}

struct Hit {
    model: Model,
    pos: v3,
    ray_length: f32,
}

fn march(origin: v3, ray_direction: v3, max_dist: f32, understep: f32, rotation: m3, power: f32) -> Hit {
    var ray_position = origin;
    var ray_length = 0.0;
    var model: Model;

    for (var i = 0; i < 300; i++) {
        model = map_scene(ray_position, rotation, power);
        ray_length += model.d * understep;
        ray_position = origin + ray_direction * ray_length;

        if (model.d < 0.00005) { break; }

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
    return mix(background_color * 0.2, glow_color * 0.8,
               smoothstep(-0.2, 0.2, dot(dir, normalize(v3(0.5, 1.0, -0.5)))));
}

fn sample_direct(hit: Hit, nor: v3, throughput: v3, rnd: v2, rotation: m3, power: f32) -> v3 {
    let sun_pos = normalize(v3(-0.2, 1.2, -0.8)) * 100.0;
    let sun_color = v3(params.sun_color_r, params.sun_color_g, params.sun_color_b);

    var col = v3(0.0);
    let light_dir = sun_pos - hit.pos;
    let light_sample_dir = get_cone_sample(light_dir, 0.001, rnd);
    let diffuse = dot(nor, light_sample_dir);
    let shadow_origin = hit.pos + nor * (0.0002 / abs(dot(light_sample_dir, nor)));

    if (diffuse > 0.0) {
        let sh = march(shadow_origin, light_sample_dir, 1.0, BOUNCE_UNDERSTEP, rotation, power);
        if (sh.model.id == 0) {
            col += throughput * sun_color / 8.0 * diffuse;
        }
    }
    return col;
}

fn draw(frag_coord: v2, frame: u32) -> v4 {
    let p = (-R + 2.0 * frag_coord) / R.y;

    seed = u32(frag_coord.x) + u32(frag_coord.y) * u32(R.x) + frame * 719393;
    let initial_seed = hash22(frag_coord + f32(frame) * sqrt3);

    let jitter = 2.0 * (initial_seed - 0.5) / R;
    let jittered_p = p + jitter;
    let scaled_p = jittered_p * FOCUS_SCALE;

    var col = v3(0.0);

    let cam_pos = FOCUS_CAM_POS * params.zoom;
    let focus_point = FOCUS_CAM_TARGET;
    let cam_tar = focus_point;


    let mouse_rotation = rotation_y(params.rotation_x) * rotation_x(params.rotation_y) * rotation_z(params.rotation_z);
    let base_rotation = rotation_z(MANDEL_ROT_Z) * rotation_y(MANDEL_ROT_Y) * rotation_x(MANDEL_ROT_X);
    let rotation = mouse_rotation * base_rotation;
    let ww = normalize(cam_tar - cam_pos);
    let uu = normalize(cross(v3(0.0, 1.0, 0.0), ww));
    let vv = normalize(cross(ww, uu));
    let cam_mat = m3(-uu, vv, ww);

    let ray_dir = normalize(cam_mat * v3(scaled_p, params.focal_length));
    var origin = cam_pos;
    var ray_direction = ray_dir;

    // Depth of field
    let focal_point_dist = distance(cam_pos, focus_point);
    let focal_plane_dist = dot(cam_mat[2], ray_dir) * focal_point_dist;

    var hit = march(origin, ray_direction, focal_plane_dist, UNDERSTEP, rotation, params.power);
    if (hit.model.id == 0) {
        let ray_focal_point = origin + ray_direction * focal_plane_dist;
        origin += cam_mat * v3(rnd_unit2(initial_seed), 0.0) * params.dof_strength;
        ray_direction = normalize(ray_focal_point - origin);
        origin = hit.pos;
        hit = march(origin, ray_direction, 20.0, UNDERSTEP, rotation, params.power);
    }

    var throughput = v3(1.0);
    let bg_col = v3(params.background_r, params.background_g, params.background_b);
    var ray_length = 0.0;
    // Path tracing loop
    for (var bounce = 0; bounce < i32(params.max_bounces); bounce++) {
        if (bounce > 0) {
            hit = march(origin, ray_direction, 10.0, BOUNCE_UNDERSTEP, rotation, params.power);
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

        let nor = calc_normal(hit.pos, rotation, params.power);
        let material = shade_model(hit.model);

        throughput *= material.albedo;

        let rnd1 = hash_v2();
        let diffuse_ray_dir = get_sample_biased(nor, 1.0, rnd1);

        let rnd2 = hash_v2();
        col += sample_direct(hit, nor, throughput, rnd2, rotation, params.power);
        ray_direction = diffuse_ray_dir;

        origin = hit.pos + nor * (0.0002 / abs(dot(ray_direction, nor)));
    }

    let fog_color = v3(params.fog_color_r, params.fog_color_g, params.fog_color_b);
    let fog = 1.0 - exp((ray_length - focal_point_dist) * -1.2);
    col = mix(col, fog_color, clamp(fog, 0.0, 1.0));

    return v4(col, 1.0);
}

// Path tracing with self-feedback accumulation
@compute @workgroup_size(16, 16, 1)
fn buffer_a(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(output);
    R = v2(f32(dimensions.x), f32(dimensions.y));

    if (global_id.x >= dimensions.x || global_id.y >= dimensions.y) {
        return;
    }

    let frag_coord = v2(f32(global_id.x), f32(global_id.y)) + 0.5;

    var pixel_color = v3(0.0);

    // Render samples for this frame
    for (var s: u32 = 0; s < params.samples_per_pixel; s++) {
        let sample_result = draw(frag_coord, time_data.frame * params.samples_per_pixel + s);
        pixel_color += sample_result.rgb;
    }

    pixel_color /= f32(params.samples_per_pixel);

    // Accumulate with previous frame (self-feedback)
    // Alpha channel tracks frame count for averaging
    let should_accumulate = params.accumulate > 0 && time_data.frame > 0;

    var final_color = v4(pixel_color, 1.0);

    if (should_accumulate) {
        let last_col = textureLoad(input_texture0, vec2<i32>(global_id.xy), 0);
        final_color = v4(pixel_color, 1.0) + last_col;
    }

    textureStore(output, vec2<i32>(global_id.xy), final_color);
}

// Tonemapping and gamma correction
fn aces_tonemap(x: v3) -> v3 {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), v3(0.0), v3(1.0));
}

@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(output);

    if (global_id.x >= dimensions.x || global_id.y >= dimensions.y) {
        return;
    }

    // Read from buffer_a - alpha channel contains frame count
    // Divide by frame count for proper average
    let tex = textureLoad(input_texture0, vec2<i32>(global_id.xy), 0);
    var col = tex.rgb / tex.a;

    // Apply exposure
    col *= params.exposure;

    // ACES tonemapping
    col = aces_tonemap(col);

    // Gamma correction
    col = pow(col, v3(1.0 / params.gamma));

    textureStore(output, vec2<i32>(global_id.xy), v4(col, 1.0));
}

// Enes Altun, 2026;
// This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
@group(0) @binding(0) var<uniform> u_t: TimeUniform;

struct Params {
    col_bg: vec4<f32>,
    col_line: vec4<f32>,
    col_core: vec4<f32>,
    col_amber: vec4<f32>,
    ball_offset_x: f32,
    ball_offset_y: f32,
    ball_sink: f32,
    distortion_amt: f32,
    noise_amt: f32,
    stream_width: f32,
    scale: f32,
    angle: f32,
    
    line_freq: f32,
    cam_height: f32,
    cam_distance: f32,
    cam_fov: f32,
    
    ball_roughness: f32,
    ball_metalness: f32,
    gamma: f32,
    saturation: f32,
    
    exposure: f32,
    contrast: f32,
    max_bounces: u32,
    samples_per_pixel: u32,
    
    accumulate: u32,
    time_offset: f32,
    dof_strength: f32,
    focal_distance: f32,
    
    rotation_x: f32,
    rotation_y: f32,
    use_hdri: u32,
    animate_flow: u32,
};

@group(1) @binding(0) var out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> p: Params;

@group(2) @binding(0) var channel0: texture_2d<f32>;
@group(2) @binding(1) var channel0_sampler: sampler;

@group(3) @binding(0) var input_texture0: texture_2d<f32>;
@group(3) @binding(1) var input_sampler0: sampler;

alias v2 = vec2<f32>;
alias v3 = vec3<f32>;
alias v4 = vec4<f32>;
alias m3 = mat3x3<f32>;
const pi = 3.14159265359;
const tau = 6.28318530718;

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

fn hash_v2() -> v2 { return v2(hash_f(), hash_f()); }

fn hash22(p_in: v2) -> v2 {
    var p_mod = p_in + 1.61803398875;
    var p3 = fract(v3(p_mod.x, p_mod.y, p_mod.x) * v3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

fn random_unit_vector() -> v3 {
    let a = hash_f() * tau;
    let z = hash_f() * 2.0 - 1.0;
    let r = sqrt(1.0 - z*z);
    return v3(r * cos(a), r * sin(a), z);
}

fn rnd_unit2(rnd: v2) -> v2 {
    let h_val = rnd * v2(1.0, tau);
    let phi = h_val.y;
    let r = sqrt(h_val.x);
    return r * v2(sin(phi), cos(phi));
}

fn rot(v: v2, a: f32) -> v2 {
    let s = sin(a); let c = cos(a);
    return v2(v.x * c - v.y * s, v.x * s + v.y * c);
}

fn pal(t: f32) -> v3 {
    return v3(0.5) + v3(0.5) * cos(tau * (v3(1.0) * t + v3(0.0, 0.33, 0.67)));
}

fn h(u: v2) -> f32 { return fract(sin(dot(u, v2(12.9898, 78.233))) * 43758.5453); }
fn nz(u: v2) -> f32 {
    let i = floor(u); let f_val = fract(u); let w = f_val * f_val * (3.0 - 2.0 * f_val);
    return mix(mix(h(i + v2(0., 0.)), h(i + v2(1., 0.)), w.x),
               mix(h(i + v2(0., 1.)), h(i + v2(1., 1.)), w.x), w.y);
}
fn f(u: v2) -> f32 {
    var v = 0.0; var a = 0.5; var q = u; let m = mat2x2<f32>(0.8, 0.6, -0.6, 0.8);
    for(var i = 0; i < 4; i++) { v += a * nz(q); q = m * q * 2.0; a *= 0.5; }
    return v;
}

fn cosineDirection(n: v3) -> v3 {
    let r = hash_v2();
    let u = normalize(cross(n, v3(0.0, 1.0, 1.0)));
    let v_vec = cross(u, n);
    
    let ra = sqrt(r.y);
    let rx = ra * cos(tau * r.x); 
    let ry = ra * sin(tau * r.x);
    let rz = sqrt(1.0 - r.y);
    return normalize(rx * u + ry * v_vec + rz * n);
}

fn sample_background(dir: v3) -> v3 {
    let phi = atan2(dir.z, dir.x);
    let theta = asin(clamp(dir.y, -1.0, 1.0));
    let u = (phi + pi) / tau;
    let v = 1.0 - (theta + pi / 2.0) / pi; 
    return textureSampleLevel(channel0, channel0_sampler, v2(u, v), 0.0).rgb;
}

struct Hit {
    dist: f32,
    geom_normal: v3,
    id: i32, 
    pos: v3,
}

fn iPln(ro: v3, rd: v3, y: f32) -> f32 {
    if (abs(rd.y) < 1e-4) { return 1e5; }
    let t = (y - ro.y) / rd.y;
    return select(1e5, t, t >= 0.0);
}

fn iSph(ro: v3, rd: v3, c: v3, r: f32) -> f32 {
    let oc = ro - c;
    let b = dot(oc, rd);
    let h_val = b * b - (dot(oc, oc) - r * r);
    if (h_val < 0.0) { return 1e5; }
    let t = -b - sqrt(h_val);
    return select(1e5, t, t > 1e-4);
}

fn map_scene(ro: v3, rd: v3) -> Hit {
    var hit = Hit(1e5, v3(0.0), 0, v3(0.0));

    let t_pln = iPln(ro, rd, 0.0);
    if (t_pln < hit.dist) {
        hit.dist = t_pln;
        hit.id = 1;
        hit.pos = ro + rd * t_pln;
        hit.geom_normal = v3(0.0, 1.0, 0.0); 
    }

    let br = 0.25 * p.scale;
    let sink = sin(p.time_offset * 3.2) * 0.1 + 0.1;
    let bc = v3(0.3 + p.ball_offset_x * 0.5, br * mix(0.2, 0.6, sink) * p.ball_sink, 0.5 + p.ball_offset_y * 0.5);
    
    let t_sph = iSph(ro, rd, bc, br);
    if (t_sph < hit.dist) {
        hit.dist = t_sph;
        hit.id = 2;
        hit.pos = ro + rd * t_sph;
        hit.geom_normal = normalize(hit.pos - bc);
    }

    return hit;
}

struct Material {
    albedo: v3,
    roughness: f32,
    reflectance: f32, 
    mat_normal: v3, 
    emission: v3,
}

fn get_plane_material(hit_pos: v3) -> Material {
    var mat = Material(v3(0.0), 0.1, 0.0, v3(0.0, 1.0, 0.0), v3(0.0));
    
    let S = p.scale;
    let A = p.angle;
    let T = p.time_offset;
    
    let br = 0.25 * S;
    let sink = sin(T * 3.2) * 0.1 + 0.1;
    let ball_y = br * mix(0.2, 0.6, sink) * p.ball_sink;
    let bc = v3(0.3 + p.ball_offset_x * 0.5, ball_y, 0.5 + p.ball_offset_y * 0.5);
    let b2 = v2(bc.x, bc.z);
    
    var puv = rot(v2(hit_pos.x, hit_pos.z), A);
    let bp = rot(b2, A);
    let dst = puv - bp;
    let r2 = dot(dst, dst);
    let r = sqrt(r2);
    
    let wl_r = sqrt(max(0.0, br * br - bc.y * bc.y));
    let pr = wl_r;
    let dfo = smoothstep(pr * 8.0, pr * 1.5, r);
    let disp = dst * ((pr * pr) / max(r2, 1e-3) * dfo);
    var fuv = puv - disp;

    let iw = smoothstep(pr * 1.2, pr * 5.0, dst.x) * smoothstep(pr * 12.0, pr * 5.0, dst.x);
    let wm = iw * smoothstep(pr * 7.0, 0.0, abs(dst.y));
    let ed = smoothstep(0.0, 1.0, 1.0 - abs(fuv.y) * 0.3);
    
    var flow_time = T;
    if (p.animate_flow == 1u) {
        flow_time += f32(u_t.frame) * 0.033;
    }
    
    let nuv = fuv * 4.5 - v2(flow_time * 4.0, 0.0);
    let n1 = f(nuv);
    let n2 = f(nuv + v2(5.2, 1.3) + v2(flow_time * 0.3, flow_time * 0.1)); 

    let ts = p.distortion_amt * 0.02 * wm * ed;
    fuv += v2((n2 - 0.5) * 0.5, n1 - 0.5) * ts;

    let fr = p.line_freq / S;
    let line_val = sin(fuv.y * fr);
    let sl = smoothstep(-0.5, 0.8, line_val); 
    
    let oil = smoothstep(0.2, 0.8, abs(n1 - 0.5) * 2.0 * wm);
    var bCol = mix(p.col_bg.rgb, mix(p.col_line.rgb, pal(n1 + T * 0.1), oil * 0.6), sl);

    let swd = p.stream_width;
    let swW = swd + swd * 0.4 * smoothstep(0.0, 2.0, dst.x);
    let isT = 1.0 - smoothstep(swW * 0.8, swW, abs(fuv.y - bp.y));
    var em_str = 0.5;
    
    if (isT > 0.01) {
        let sc = mix(p.col_core.rgb, p.col_amber.rgb, smoothstep(0.0, 3.0, dst.x));
        bCol = mix(bCol, mix(sc * 0.2, sc * 1.2, sl), isT);
        em_str = 1.5;
    }

    let pt = rot(v2((n1 - 0.5) * 10.12 * wm, cos(fuv.y * fr) * 0.5 + (n2 - 0.5) * 10.12 * wm), -A);
    mat.mat_normal = normalize(v3(pt.x * 0.05, 1.0, pt.y * 0.05));

    mat.albedo = bCol;
    mat.roughness = mix(0.8, 0.2, sl); 
    mat.reflectance = 0.04; 
    mat.emission = bCol * sl * em_str;
    
    return mat;
}

fn env(dir: v3) -> v3 {
    if (p.use_hdri == 1u) {
        return sample_background(dir) * 1.5; 
    } else {
        let top_light = smoothstep(0.8, 0.99, dir.y) * 4.0 * v3(1.0, 0.95, 0.9);
        let rim_light = smoothstep(0.8, 0.99, -dir.z) * 1.5 * v3(0.5, 0.6, 1.0);
        let ambient = p.col_bg.rgb * mix(0.1, 0.4, dir.y * 0.5 + 0.5);
        return top_light + rim_light + ambient;
    }
}

fn draw(frag_coord: v2, frame_idx: u32, R: v2) -> v3 {
    seed = u32(frag_coord.x) + u32(frag_coord.y) * u32(R.x) + frame_idx * 719393u;
    let initial_seed = hash22(frag_coord + f32(frame_idx) * 1.732);

    let jitter = 2.0 * (initial_seed - 0.5) / R;
    let uv = v2(2.0 * frag_coord.x - R.x, R.y - 2.0 * frag_coord.y) / R.y + jitter;

    let cam_pos = v3(0.0, p.cam_height, -p.cam_distance * 0.5);
    let cam_tar = v3(0.3 + p.ball_offset_x * 0.5, 0.0, 0.5 + p.ball_offset_y * 0.5);

    let cx = cos(p.rotation_x); let sx = sin(p.rotation_x);
    let cy = cos(p.rotation_y); let sy = sin(p.rotation_y);
    let rotY = m3(cy, 0.0, sy, 0.0, 1.0, 0.0, -sy, 0.0, cy);
    let rotX = m3(1.0, 0.0, 0.0, 0.0, cx, -sx, 0.0, sx, cx);
    
    let ro = rotY * rotX * cam_pos;
    
    let ww = normalize(cam_tar - ro);
    let uu = normalize(cross(v3(0.0, 1.0, 0.0), ww));
    let vv = normalize(cross(ww, uu));
    let cam_mat = m3(uu, vv, ww);

    var rd = normalize(cam_mat * v3(uv, p.cam_fov));
    var origin = ro;

    if (p.dof_strength > 0.0) {
        let focal_point = origin + rd * p.focal_distance;
        origin += cam_mat * v3(rnd_unit2(initial_seed), 0.0) * p.dof_strength;
        rd = normalize(focal_point - origin);
    }

    var col = v3(0.0);
    var throughput = v3(1.0);

    for (var bounce = 0u; bounce < p.max_bounces; bounce++) {
        let hit = map_scene(origin, rd);

        if (hit.id == 0) { 
            col += throughput * env(rd);
            break;
        }

        var mat: Material;
        if (hit.id == 1) { 
            mat = get_plane_material(hit.pos);
        } else if (hit.id == 2) { 
            mat = Material(v3(0.9), p.ball_roughness, p.ball_metalness, hit.geom_normal, v3(0.0));
        }

        col += throughput * mat.emission;

        let fre = dot(rd, mat.mat_normal); 
        let rd0 = reflect(rd, mat.mat_normal);
        let rd1 = cosineDirection(mat.mat_normal);
        
        let refProb = mat.reflectance + (1.0 - mat.reflectance) * pow(clamp(1.0 + fre, 0.0, 1.0), 5.0);
        
        var scatter_dir: v3;
        
        if (hash_f() < refProb) {
            scatter_dir = normalize(rd0 + mat.roughness * random_unit_vector());
            let spec_color = mix(v3(1.0), mat.albedo, mat.reflectance);
            throughput *= spec_color; 
        } else {
            scatter_dir = rd1;
            let diffuse_color = mat.albedo * (1.0 - mat.reflectance);
            throughput *= diffuse_color;
        }

        if (dot(scatter_dir, hit.geom_normal) < 0.0) {
            scatter_dir = reflect(scatter_dir, hit.geom_normal);
        }

        origin = hit.pos + hit.geom_normal * 0.001;
        rd = scatter_dir;
        
        if (bounce > 2u) {
            let prob = max(throughput.x, max(throughput.y, throughput.z));
            if (hash_f() > prob) { break; }
            throughput /= prob;
        }
    }

    return col;
}

@compute @workgroup_size(16, 16, 1)
fn accumulate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dim = textureDimensions(out);
    let R = v2(f32(dim.x), f32(dim.y));

    if (global_id.x >= dim.x || global_id.y >= dim.y) { return; }
    let frag_coord = v2(f32(global_id.x), f32(global_id.y)) + 0.5;

    var pixel_color = v3(0.0);

    for (var s = 0u; s < p.samples_per_pixel; s++) {
        pixel_color += draw(frag_coord, u_t.frame * p.samples_per_pixel + s, R);
    }
    pixel_color /= f32(p.samples_per_pixel);

    var final_color = v4(pixel_color, 1.0);

    if (p.accumulate > 0u && u_t.frame > 0u) {
        let last_col = textureLoad(input_texture0, vec2<i32>(global_id.xy), 0);
        
        var blend_factor: f32;
        if (p.animate_flow == 1u) {
            blend_factor = 0.15; 
        } else {
            blend_factor = 1.0 / f32(u_t.frame + 1u); 
        }
        
        final_color = v4(mix(last_col.rgb, pixel_color, blend_factor), 1.0);
    }

    textureStore(out, vec2<i32>(global_id.xy), final_color);
}

fn aces_tonemap(x: v3) -> v3 {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), v3(0.0), v3(1.0));
}

@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dim = textureDimensions(out);
    if (global_id.x >= dim.x || global_id.y >= dim.y) { return; }

    let tex = textureLoad(input_texture0, vec2<i32>(global_id.xy), 0);
    var col = tex.rgb / max(tex.a, 1.0);

    col *= p.exposure;
    col = aces_tonemap(col);
    
    let lum = dot(col, v3(0.299, 0.587, 0.114));
    col = mix(v3(lum), col, p.saturation);
    
    col = (col - 0.5) * p.contrast + 0.5;
    col = max(col, v3(0.0));

    col = pow(col, v3(1.0 / p.gamma));

    textureStore(out, vec2<i32>(global_id.xy), v4(col, 1.0));
}
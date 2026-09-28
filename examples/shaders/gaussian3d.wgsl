struct GaussianParams {
    num_gaussians: u32,
    scale_modifier: f32,
    scene_scale: f32,
    gamma: f32,
    depth_shift: u32,
    up_mode: u32,
    sh_amt: f32,
    opacity_scale: f32,
    depth_near: f32,
    depth_far: f32,
    near_cull: f32,
    sh_degree: u32,
    focus_dist: f32,
    aperture: f32,
    view_mode: u32,
    oil_enable: u32,
    hardness: f32,
    bristle: f32,
    canvas: f32,
    impasto: f32,
    edge_rag: f32,
    edge_blur: f32,
    sharp_radius: f32,
    focus_x: f32,
    focus_y: f32,
    anim_prog: f32,
    anim_order: u32,
    anim_band: f32,
    anim_grow: f32,
    scene_cx: f32,
    scene_cy: f32,
    scene_cz: f32,
    scene_r: f32,
    _p0: f32,
    _p1: f32,
    _p2: f32,
};

struct Camera {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>,
};

struct Gaussian3D {
    position: vec3<f32>,
    _pad0: f32,
    cov: array<f32, 6>,
    _pad1: vec2<f32>,
    color: vec4<f32>,
};

struct Gaussian2D {
    mean: vec2<f32>,
    depth: f32,
    radius: f32,
    conic: vec3<f32>,
    opacity: f32,
    color: vec3<f32>,
    _pad: f32,
};

struct TimeUniform { time: f32, delta: f32, frame: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> time: TimeUniform;
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: GaussianParams;

@group(3) @binding(0) var<storage, read_write> gaussians: array<Gaussian3D>;
@group(3) @binding(1) var<storage, read_write> gaussian_2d: array<Gaussian2D>;
@group(3) @binding(2) var<storage, read_write> depth_keys: array<u32>;
@group(3) @binding(3) var<storage, read_write> sorted_indices: array<u32>;
@group(3) @binding(4) var<storage, read_write> camera_data: Camera;
// spherical harmonics, degree <= 2, 12 u32 of packed f16 per gaussian: coeff j x rgb
@group(3) @binding(5) var<storage, read_write> sh_data: array<u32>;

// scene up axis: 0 as stored, 1 flipped (colmap y-down), 2 z-up
fn orient() -> mat3x3<f32> {
    if params.up_mode == 1u { return mat3x3<f32>(vec3<f32>(1., 0., 0.), vec3<f32>(0., -1., 0.), vec3<f32>(0., 0., -1.)); }
    if params.up_mode == 2u { return mat3x3<f32>(vec3<f32>(1., 0., 0.), vec3<f32>(0., 0., -1.), vec3<f32>(0., 1., 0.)); }
    return mat3x3<f32>(vec3<f32>(1., 0., 0.), vec3<f32>(0., 1., 0.), vec3<f32>(0., 0., 1.));
}

fn sh_coef(idx: u32, j: u32) -> vec3<f32> {
    let b = idx * 12u;
    let h0 = j * 3u;
    let u0 = unpack2x16float(sh_data[b + h0 / 2u]);
    let u1 = unpack2x16float(sh_data[b + (h0 + 2u) / 2u]);
    if (h0 % 2u) == 0u { return vec3<f32>(u0.x, u0.y, u1.x); }
    return vec3<f32>(u0.y, u1.x, u1.y);
}

// view-dependent colour, degree 1 and 2 bands
fn sh_rest(idx: u32, d: vec3<f32>) -> vec3<f32> {
    let x = d.x; let y = d.y; let z = d.z;
    var c = 0.4886025 * (-y * sh_coef(idx, 0u) + z * sh_coef(idx, 1u) - x * sh_coef(idx, 2u));
    if params.sh_degree >= 2u {
        c += 1.0925484 * x * y * sh_coef(idx, 3u) - 1.0925484 * y * z * sh_coef(idx, 4u)
           + 0.3153916 * (2. * z * z - x * x - y * y) * sh_coef(idx, 5u)
           - 1.0925484 * x * z * sh_coef(idx, 6u) + 0.5462742 * (x * x - y * y) * sh_coef(idx, 7u);
    }
    return c;
}

// radius (in sigmas) where opacity*falloff drops below 1/255
fn vis_k(op: f32) -> f32 { return clamp(sqrt(max(2. * log(max(op, 1e-4) * 255.), 0.)), 0., 3.); }

fn cull(idx: u32) {
    gaussian_2d[idx].radius = 0.0;
    depth_keys[idx] = 0xFFFFFFFFu;
}

@compute @workgroup_size(256, 1, 1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.num_gaussians { return; }

    sorted_indices[idx] = idx;

    let g = gaussians[idx];
    let M = orient();
    let pos_world = vec4<f32>(M * g.position * params.scene_scale, 1.0);
    let pos_view = camera_data.view * pos_world;
    if pos_view.z <= params.near_cull { cull(idx); return; }

    let pos_clip = camera_data.proj * pos_view;
    let pos_ndc = pos_clip.xyz / pos_clip.w;
    if abs(pos_ndc.x) > 1.3 || abs(pos_ndc.y) > 1.3 { cull(idx); return; }

    let screen_pos = vec2<f32>(
        (pos_ndc.x * 0.5 + 0.5) * camera_data.viewport.x,
        (1.0 - (pos_ndc.y * 0.5 + 0.5)) * camera_data.viewport.y
    );

    // 2d covariance: J W (M S M^T) W^T J^T
    let t = pos_view.xyz;
    let focal = camera_data.focal;
    let limx = 0.65 * camera_data.viewport.x / focal.x;
    let limy = 0.65 * camera_data.viewport.y / focal.y;
    let txtz = clamp(t.x / t.z, -limx, limx);
    let tytz = clamp(t.y / t.z, -limy, limy);
    let J = mat3x2<f32>(
        vec2<f32>(focal.x / t.z, 0.0),
        vec2<f32>(0.0, focal.y / t.z),
        vec2<f32>(-focal.x * txtz / t.z, -focal.y * tytz / t.z)
    );
    let W = mat3x3<f32>(camera_data.view[0].xyz, camera_data.view[1].xyz, camera_data.view[2].xyz);
    let c3 = mat3x3<f32>(
        vec3<f32>(g.cov[0], g.cov[1], g.cov[2]),
        vec3<f32>(g.cov[1], g.cov[3], g.cov[4]),
        vec3<f32>(g.cov[2], g.cov[4], g.cov[5])
    ) * (params.scene_scale * params.scene_scale);
    let cw = M * c3 * transpose(M);
    let c2 = J * (W * cw * transpose(W)) * transpose(J);

    // animate: splats fade and grow in along the chosen order
    var gr = 1.;
    var op = clamp(g.color.a * params.opacity_scale, 0., 1.);
    if params.anim_prog >= 0. {
        let C = vec3<f32>(params.scene_cx, params.scene_cy, params.scene_cz);
        let R = max(params.scene_r, 1e-4);
        var key = fract(sin(f32(idx) * 12.9898) * 43758.5453);
        if params.anim_order == 0u { key = 1. - clamp(sqrt(max(cw[0][0], max(cw[1][1], cw[2][2]))) / (R * .03), 0., 1.); }
        if params.anim_order == 1u { key = clamp(length(pos_world.xyz - C) / R, 0., 1.); }
        if params.anim_order == 2u { key = clamp((pos_world.y - C.y + R) / (2. * R), 0., 1.); }
        let bd = max(params.anim_band, .01);
        let rv = smoothstep(key * (1. - bd), key * (1. - bd) + bd, params.anim_prog);
        if rv <= 0. { cull(idx); return; }
        op *= rv;
        gr = mix(1., rv, params.anim_grow);
    }

    let s2 = params.scale_modifier * params.scale_modifier * gr * gr;
    var a = c2[0][0] * s2 + 0.3;
    let b = c2[0][1] * s2;
    var c = c2[1][1] * s2 + 0.3;

    // depth of field: depth blur + radial blur around the focus point, widened with energy kept
    if params.aperture > 0. || params.edge_blur > 0. {
        let lens = params.aperture * 0.02 * (params.depth_far - params.depth_near);
        let vp = camera_data.viewport;
        let rr = length((screen_pos / vp - vec2<f32>(params.focus_x, params.focus_y)) * vec2<f32>(vp.x / vp.y, 1.));
        let rb = params.edge_blur * smoothstep(params.sharp_radius, params.sharp_radius + .6, rr) * 20. * vp.y / 1080.;
        let coc = min(lens * focal.y * abs(1. / max(params.focus_dist, 1e-3) - 1. / t.z) + rb, 30. * vp.y / 1080.);
        let d0 = a * c - b * b;
        let sc = coc * 0.5;
        a += sc * sc; c += sc * sc;
        op *= sqrt(max(d0, 1e-8) / max(a * c - b * b, 1e-8));
    }

    let det = a * c - b * b;
    if det <= 0.0 || op * 255. < 1. { cull(idx); return; }
    let conic = vec3<f32>(c / det, -b / det, a / det);

    let mid = 0.5 * (a + c);
    let lambda = mid + sqrt(max(0.1, mid * mid - det));
    let radius = ceil(vis_k(op) * sqrt(lambda));
    if radius <= 0.0 { cull(idx); return; }

    // colour: base + view-dependent bands, direction in the file's own frame
    var col = g.color.rgb;
    if params.sh_amt > 0. && params.sh_degree > 0u {
        let R = mat3x3<f32>(camera_data.view[0].xyz, camera_data.view[1].xyz, camera_data.view[2].xyz);
        let eye = -(transpose(R) * camera_data.view[3].xyz);
        let dir = transpose(M) * normalize(pos_world.xyz - eye);
        col += sh_rest(idx, dir) * params.sh_amt;
    }

    gaussian_2d[idx].mean = screen_pos;
    gaussian_2d[idx].depth = pos_view.z;
    gaussian_2d[idx].radius = radius;
    gaussian_2d[idx].conic = conic;
    gaussian_2d[idx].opacity = op;
    gaussian_2d[idx].color = max(col, vec3<f32>(0.));

    // back to front: linear depth over the scene's near..far range
    let kb = 32u - params.depth_shift;
    let tn = clamp((pos_view.z - params.depth_near) / max(params.depth_far - params.depth_near, 1e-4), 0., 1.);
    depth_keys[idx] = u32((1. - tn) * f32((1u << kb) - 2u));
}


@group(0) @binding(0) var<uniform> render_params: GaussianParams;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<storage, read> render_gaussian_2d: array<Gaussian2D>;
@group(0) @binding(3) var<storage, read> render_sorted_indices: array<u32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) local_pos: vec2<f32>,
    @location(1) color: vec3<f32>,
    @location(2) opacity: f32,
    @location(3) conic: vec3<f32>,
    // major axis, sqrt of the conic eigenvalues (major, minor), seed, radius, normalised depth
    @location(4) axis: vec2<f32>,
    @location(5) ev: vec2<f32>,
    @location(6) extra: vec3<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    var out: VertexOutput;

    let gaussian_idx = render_sorted_indices[instance_index];
    let g = render_gaussian_2d[gaussian_idx];

    if g.radius <= 0.0 {
        out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        out.opacity = 0.0;
        return out;
    }

    var offset: vec2<f32>;
    switch vertex_index {
        case 0u: { offset = vec2<f32>(-1.0, -1.0); }
        case 1u: { offset = vec2<f32>(1.0, -1.0); }
        case 2u: { offset = vec2<f32>(-1.0, 1.0); }
        case 3u: { offset = vec2<f32>(1.0, -1.0); }
        case 4u: { offset = vec2<f32>(1.0, 1.0); }
        case 5u: { offset = vec2<f32>(-1.0, 1.0); }
        default: { offset = vec2<f32>(0.0); }
    }

    var rad = g.radius;
    if render_params.view_mode == 3u { rad = 2.; }
    let screen_pos = g.mean + offset * rad;
    let ndc = (screen_pos / camera.viewport) * 2.0 - 1.0;
    out.position = vec4<f32>(ndc.x, -ndc.y, 0.5, 1.0);

    out.local_pos = offset * rad;
    out.color = g.color;
    out.opacity = g.opacity;
    out.conic = g.conic;

    // principal axes of the splat for stroke shading
    let q = g.conic;
    let th = 0.5 * atan2(2. * q.y, q.x - q.z);
    let cs = cos(th); let sn = sin(th);
    let l1 = q.x * cs * cs + 2. * q.y * cs * sn + q.z * sn * sn;
    let l2 = q.x * sn * sn - 2. * q.y * cs * sn + q.z * cs * cs;
    let e1 = vec2<f32>(cs, sn);
    let major_first = l1 <= l2;
    out.axis = select(vec2<f32>(-sn, cs), e1, major_first);
    out.ev = sqrt(max(select(vec2<f32>(l2, l1), vec2<f32>(l1, l2), major_first), vec2<f32>(0.)));
    let tn = clamp((g.depth - render_params.depth_near) / max(render_params.depth_far - render_params.depth_near, 1e-4), 0., 1.);
    out.extra = vec3<f32>(fract(sin(f32(gaussian_idx) * 12.9898) * 43758.5453) * 40., g.radius, tn);

    return out;
}

fn h21(p: vec2<f32>) -> f32 {
    var q = fract(p * vec2<f32>(.1031, .1173));
    q += dot(q, q.yx + 33.33);
    return fract((q.x + q.y) * q.x);
}
fn vn2(p: vec2<f32>) -> f32 {
    let i = floor(p); let f = fract(p); let u = f * f * (3. - 2. * f);
    return mix(mix(h21(i), h21(i + vec2<f32>(1., 0.)), u.x), mix(h21(i + vec2<f32>(0., 1.)), h21(i + vec2<f32>(1., 1.)), u.x), u.y);
}
fn to_linear(c: vec3<f32>) -> vec3<f32> {
    return select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045));
}
fn ramp(t: f32) -> vec3<f32> {
    return clamp(vec3<f32>(1.5 - abs(4. * t - 3.), 1.5 - abs(4. * t - 2.), 1.5 - abs(4. * t - 1.)), vec3<f32>(0.), vec3<f32>(1.));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if in.opacity <= 0.0 { discard; }

    let d = in.local_pos;
    let power = -0.5 * (in.conic.x * d.x * d.x + 2.0 * in.conic.y * d.x * d.y + in.conic.z * d.y * d.y);
    if power > 0.0 || power < -4.5 { discard; }

    var col = in.color;
    var alpha = min(0.99, in.opacity * exp(power));

    // shading: hardened dabs, bristle streaks, impasto relief; skipped on tiny splats
    let sf = smoothstep(3., 12., in.extra.y);
    if render_params.oil_enable != 0u && render_params.view_mode == 0u && sf > 0.001 {
        let nx = dot(d, in.axis) * in.ev.x;
        let ny = dot(d, vec2<f32>(-in.axis.y, in.axis.x)) * in.ev.y;
        let streak = vn2(vec2<f32>(nx * .6, ny * 7.) + in.extra.x) - .5;
        let esf = select(streak * .35, streak, streak < 0.);
        let pdf = -power * max(1. + render_params.edge_rag * 2. * esf * sf, 0.4);
        let hard = 1. + (render_params.hardness - 1.) * sf;
        alpha = min(0.99, in.opacity * exp(-pow(pdf, hard)));
        col *= 1. - render_params.bristle * .55 * max(0., -streak) * sf;
        col += in.color * render_params.bristle * .25 * max(0., streak) * sf;
        let relief = clamp(ny, -1., 1.);
        col *= 1. - render_params.impasto * .5 * max(0., relief) * sf;
        col += in.color * render_params.impasto * .5 * max(0., -relief) * sf;
        // canvas in screen space
        let cuv = in.position.xy;
        let weave = (sin(cuv.x * 1.3) * .5 + .5) * (sin(cuv.y * 1.3) * .5 + .5);
        col *= 1. + render_params.canvas * (.28 * (weave - .5) + .18 * (vn2(cuv * .5) - .5));
    }

    // debug views: depth, ellipse outlines, points
    if render_params.view_mode == 1u { col = ramp(1. - in.extra.z); }
    if render_params.view_mode == 2u {
        alpha = in.opacity * exp(-pow(sqrt(-2. * power) - 2., 2.) * 8.);
    }
    if render_params.view_mode == 3u { alpha = min(0.99, in.opacity * 2.) * step(dot(d, d), 1.5); }
    if alpha < 1.0 / 255.0 { discard; }

    // file colours are display-referred; the window encodes to srgb itself
    let c = to_linear(pow(max(col, vec3<f32>(0.)), vec3<f32>(render_params.gamma)));
    return vec4<f32>(c * alpha, alpha);
}

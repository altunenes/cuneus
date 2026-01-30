struct GaussianParams {
    num_gaussians: u32,
    scale_modifier: f32,
    scene_scale: f32,
    gamma: f32,
    depth_shift: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
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

fn compute_cov2d(
    cov3d: array<f32, 6>, 
    pos_view: vec3<f32>, 
    focal: vec2<f32>, 
    viewport: vec2<f32>, 
    view_mat: mat4x4<f32>,
    scale_mod: f32
) -> vec3<f32> {
    let t = pos_view;
    let tan_fovx = 0.5 * viewport.x / focal.x;
    let tan_fovy = 0.5 * viewport.y / focal.y;
    let limx = 1.3 * tan_fovx;
    let limy = 1.3 * tan_fovy;

    let txtz = clamp(t.x / t.z, -limx, limx);
    let tytz = clamp(t.y / t.z, -limy, limy);

    let J = mat3x2<f32>(
        vec2<f32>(focal.x / t.z, 0.0),
        vec2<f32>(0.0, focal.y / t.z),
        vec2<f32>(-focal.x * txtz / t.z, -focal.y * tytz / t.z)
    );

    let W = mat3x3<f32>(view_mat[0].xyz, view_mat[1].xyz, view_mat[2].xyz);
    let cov3d_mat = mat3x3<f32>(
        vec3<f32>(cov3d[0], cov3d[1], cov3d[2]),
        vec3<f32>(cov3d[1], cov3d[3], cov3d[4]),
        vec3<f32>(cov3d[2], cov3d[4], cov3d[5])
    );

    let cov_view = W * cov3d_mat * transpose(W);
    let cov2d = J * cov_view * transpose(J);

    let s2 = scale_mod * scale_mod;
    let a = cov2d[0][0] * s2 + 0.3;
    let b = cov2d[0][1] * s2;
    let c = cov2d[1][1] * s2 + 0.3;

    let det = a * c - b * b;
    if det <= 0.0 { return vec3<f32>(0.0); }

    return vec3<f32>(c / det, -b / det, a / det);
}

fn compute_radius(conic: vec3<f32>) -> f32 {
    let det = conic.x * conic.z - conic.y * conic.y;
    if det <= 0.0 { return 0.0; }
    let mid = 0.5 * (conic.z + conic.x) / det;
    let lambda = mid + sqrt(max(0.1, mid * mid - 1.0 / det));
    return ceil(3.0 * sqrt(lambda));
}

@compute @workgroup_size(256, 1, 1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.num_gaussians { return; }

    sorted_indices[idx] = idx;

    let g = gaussians[idx];
    let scaled_pos = g.position * params.scene_scale;
    let pos_world = vec4<f32>(scaled_pos, 1.0);
    let pos_view = camera_data.view * pos_world;

    if pos_view.z <= 0.2 {
        gaussian_2d[idx].radius = 0.0;
        depth_keys[idx] = 0xFFFFFFFFu;
        return;
    }

    let pos_clip = camera_data.proj * pos_view;
    let pos_ndc = pos_clip.xyz / pos_clip.w;

    if pos_ndc.x < -1.3 || pos_ndc.x > 1.3 || pos_ndc.y < -1.3 || pos_ndc.y > 1.3 {
        gaussian_2d[idx].radius = 0.0;
        depth_keys[idx] = 0xFFFFFFFFu;
        return;
    }

    let screen_pos = vec2<f32>(
        (pos_ndc.x * 0.5 + 0.5) * camera_data.viewport.x,
        (1.0 - (pos_ndc.y * 0.5 + 0.5)) * camera_data.viewport.y
    );

    let ss2 = params.scene_scale * params.scene_scale;
    var scaled_cov: array<f32, 6>;
    scaled_cov[0] = g.cov[0] * ss2;
    scaled_cov[1] = g.cov[1] * ss2;
    scaled_cov[2] = g.cov[2] * ss2;
    scaled_cov[3] = g.cov[3] * ss2;
    scaled_cov[4] = g.cov[4] * ss2;
    scaled_cov[5] = g.cov[5] * ss2;

    let conic = compute_cov2d(
        scaled_cov, 
        pos_view.xyz, 
        camera_data.focal, 
        camera_data.viewport, 
        camera_data.view,
        params.scale_modifier
    );

    if conic.x == 0.0 && conic.y == 0.0 && conic.z == 0.0 {
        gaussian_2d[idx].radius = 0.0;
        depth_keys[idx] = 0xFFFFFFFFu;
        return;
    }

    let radius = compute_radius(conic);
    
    if radius <= 0.0 {
        gaussian_2d[idx].radius = 0.0;
        depth_keys[idx] = 0xFFFFFFFFu;
        return;
    }

    gaussian_2d[idx].mean = screen_pos;
    gaussian_2d[idx].depth = pos_view.z;
    gaussian_2d[idx].radius = radius;
    gaussian_2d[idx].conic = conic;
    gaussian_2d[idx].opacity = g.color.a;
    gaussian_2d[idx].color = g.color.rgb;

    let depth_uint = bitcast<u32>(pos_view.z);
    depth_keys[idx] = (0xFFFFFFFFu - depth_uint) >> params.depth_shift;
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

    // Quad vertices
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

    let screen_pos = g.mean + offset * g.radius;
    let ndc = (screen_pos / camera.viewport) * 2.0 - 1.0;
    out.position = vec4<f32>(ndc.x, -ndc.y, 0.5, 1.0);

    out.local_pos = offset * g.radius;
    out.color = g.color;
    out.opacity = g.opacity;
    out.conic = g.conic;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if in.opacity <= 0.0 { discard; }

    let d = in.local_pos;
    let power = -0.5 * (in.conic.x * d.x * d.x + 2.0 * in.conic.y * d.x * d.y + in.conic.z * d.y * d.y);

    if power > 0.0 { discard; }

    if power < -4.5 { discard; }
    let gaussian = exp(power);
    var alpha = min(0.99, in.opacity * gaussian);
    if alpha < 1.0 / 255.0 { discard; }

    let color = pow(in.color, vec3<f32>(render_params.gamma));
    return vec4<f32>(color * alpha, alpha);
}
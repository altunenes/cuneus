//! glTF / GLB models drawn with your own WGSL materials into an HDR texture that compute passes
//! read. See the "3D Models (glTF)" section of usage.md.

mod anim;
mod data;

pub use anim::{AnimChannel, AnimInterpolation, AnimProperty, MeshAnimation, MeshNode, MeshSkinning, MorphSlot, Pose};
pub use data::{MeshAlpha, MeshData, MeshImage, MeshMaterial, MeshPrimitive, MeshSampler, MeshVertex, MeshWrap};

use crate::compute::ComputeShader;
use crate::math::{Mat4, Vec3};
use crate::{Core, OrbitCamera, OrbitPose, RenderKit};
use log::{error, info, warn};
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;

/// Bindings, vertex types and helpers prepended to every mesh material shader
pub const MESH_PRELUDE: &str = include_str!("prelude.wgsl");
const SKIN_SHADER: &str = include_str!("skin.wgsl");
const FALLBACK_FRAGMENT: &str = include_str!("fallback.wgsl");
const INSTANCES_SHADER: &str = include_str!("instances.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GlobalsU { time: f32, delta_time: f32, frame_count: u32, _pad: u32 }

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ViewU { clip_from_world: [f32; 16], view_from_world: [f32; 16], world_position: [f32; 3], _pad: f32, viewport: [f32; 4] }

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SunU {
    clip_from_world: [f32; 16], direction: [f32; 3], _pad: f32, color: [f32; 3], shadows: u32,
    softness: f32, normal_bias: f32, light_count: u32, light_texel: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LightU {
    position: [f32; 3], range: f32, color: [f32; 3], kind: u32, direction: [f32; 3], cos_outer: f32,
    cos_inner: f32, shadow: i32, _p0: f32, _p1: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TransformU {
    world_from_local: [f32; 16], normal_from_local: [f32; 16], data: [f32; 4],
    anim: [f32; 4], anim_b: [f32; 4], ids: [u32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CullU {
    culled: u32, count: u32, full: u32, object: u32, center: [f32; 3], radius: f32,
    cull: u32, prims: u32, _p0: u32, _p1: u32, planes: [[f32; 4]; 6],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GltfMaterialU {
    base_color: [f32; 4], emissive: [f32; 4],
    metallic: f32, roughness: f32, normal_scale: f32, occlusion_strength: f32,
    alpha_cutoff: f32, alpha_mode: u32, flags: u32, _pad: u32,
    uv_row0: [f32; 4], uv_row1: [f32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MeshId(pub usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MaterialId(pub usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ObjectId(pub usize);

/// How a material's output combines with what is behind it
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Blend {
    /// From the model: glTF BLEND → `Alpha`, otherwise `Opaque` (MASK is cut in the prepass)
    #[default]
    Auto,
    Opaque,
    /// Output alpha is opacity; sorted back to front
    Alpha,
    /// Adds light scaled by output alpha (glow shells, sparks); never hides what is behind
    Additive,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Cull {
    /// From the model: glTF doubleSided → `None`, otherwise `Back`
    #[default]
    Auto,
    None,
    Back,
    Front,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaterialOptions {
    pub blend: Blend,
    pub cull: Cull,
    /// Opaque only: shade each pixel once after a depth prepass; turn off if the fragment uses `discard`
    pub prepass: bool,
    /// Opaque only: cast sun shadows
    pub shadows: bool,
}

impl Default for MaterialOptions {
    fn default() -> Self { Self { blend: Blend::Auto, cull: Cull::Auto, prepass: true, shadows: true } }
}

/// The scene's directional light. Materials read it as `sun` and shadow with `directional_shadow`.
#[derive(Clone, Copy, Debug)]
pub struct Sun {
    /// Towards the light
    pub direction: Vec3,
    /// Linear colour x intensity
    pub color: [f32; 3],
    pub shadows: bool,
    /// Shadow map resolution
    pub shadow_size: u32,
    /// Penumbra radius in shadow map texels
    pub softness: f32,
    /// Push along the normal against self-shadow acne, in shadow map texels
    pub normal_bias: f32,
    /// World sphere the shadow map covers; `None` fits every shadow casting object
    pub bounds: Option<(Vec3, f32)>,
}

impl Sun {
    /// Aim from `yaw` around the vertical axis and `height` above the horizon, in radians
    pub fn set_angles(&mut self, yaw: f32, height: f32) {
        let (sy, cy, sh, ch) = (yaw.sin(), yaw.cos(), height.sin(), height.cos());
        self.direction = Vec3::new(ch * sy, sh, ch * cy);
    }
}

impl Default for Sun {
    fn default() -> Self {
        Self { direction: Vec3::new(0.4, 0.8, 0.45).normalize(), color: [3.0; 3], shadows: true, shadow_size: 2048, softness: 2.0, normal_bias: 1.5, bounds: None }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LightKind {
    Point,
    /// Cone half angles in radians: full light inside `inner`, none outside `outer`
    Spot { inner: f32, outer: f32 },
}

/// A point or spot light; materials read them with `light_count()` / `light_sample(i, ...)`
#[derive(Clone, Copy, Debug)]
pub struct Light {
    pub kind: LightKind,
    pub position: Vec3,
    /// Spot lights: where the cone points
    pub direction: Vec3,
    /// Linear colour x intensity
    pub color: [f32; 3],
    /// Distance where the light fades to zero
    pub range: f32,
    /// Spot: 1 shadow map, point: 6 (a cube)
    pub shadows: bool,
}

impl Light {
    pub fn point(position: Vec3, color: [f32; 3], range: f32) -> Self {
        Self { kind: LightKind::Point, position, direction: Vec3::new(0.0, -1.0, 0.0), color, range, shadows: false }
    }
    pub fn spot(position: Vec3, direction: Vec3, color: [f32; 3], range: f32, angle: f32) -> Self {
        Self { kind: LightKind::Spot { inner: angle * 0.8, outer: angle }, position, direction, color, range, shadows: false }
    }
    pub fn with_shadows(mut self) -> Self { self.shadows = true; self }

    // shadow cameras: one for a spot, six cube faces (+x -x +y -y +z -z) for a point
    fn shadow_views(&self) -> Vec<(Mat4, Vec3)> {
        let near = (self.range * 0.01).max(1e-3);
        let look = |dir: Vec3, fov: f32| {
            let up = if dir.y.abs() > 0.99 { Vec3::new(0.0, 0.0, 1.0) } else { Vec3::Y };
            (Mat4::perspective_rh(fov, 1.0, near, self.range.max(near * 2.0)) * Mat4::look_at_rh(self.position, self.position + dir, up), self.position)
        };
        match self.kind {
            LightKind::Spot { outer, .. } => vec![look(self.direction.normalize_or_zero(), (outer * 2.0 + 0.1).min(3.0))],
            LightKind::Point => [Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0), Vec3::Y, Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, -1.0)]
                .iter().map(|&d| look(d, std::f32::consts::FRAC_PI_2)).collect(),
        }
    }
}

// light shadow maps: one array layer per shadow view, each with its own camera uniform
struct LightShadows {
    size: u32,
    layers: u32,
    array: wgpu::TextureView,
    passes: Vec<(wgpu::TextureView, wgpu::Buffer, wgpu::BindGroup)>,
}

/// What `MeshScene::pick` found: the object, which of its copies, and the world point hit
#[derive(Clone, Copy, Debug)]
pub struct Pick {
    pub object: ObjectId,
    pub instance: u32,
    pub position: Vec3,
}

// Mapping flag: 0 waiting, 1 mapped, 2 failed
enum PickState { Idle, Copied, Mapping(Arc<AtomicU8>) }

struct PickTargets { size: [u32; 2], id: wgpu::Texture, pos: wgpu::Texture, id_view: wgpu::TextureView, pos_view: wgpu::TextureView, depth: wgpu::TextureView }

/// One copy: transform, 4 floats for `get_instance_data`, and its animation in crowd objects
#[derive(Clone, Copy, Debug)]
pub struct Instance {
    pub transform: Mat4,
    pub data: [f32; 4],
    pub anim: InstanceAnim,
}

impl From<Mat4> for Instance {
    fn from(transform: Mat4) -> Self { Self { transform, data: [0.0; 4], anim: InstanceAnim::default() } }
}

/// Crowd copy animation: `clip` at `time` (-1 = rest pose), blended to `next` at `next_time` by `fade`
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InstanceAnim {
    pub clip: i32,
    pub time: f32,
    pub next: i32,
    pub next_time: f32,
    pub fade: f32,
}

impl Default for InstanceAnim {
    fn default() -> Self { Self { clip: -1, time: 0.0, next: -1, next_time: 0.0, fade: 0.0 } }
}

impl InstanceAnim {
    pub fn clip(clip: usize, time: f32) -> Self { Self { clip: clip as i32, time, ..Default::default() } }
    /// Blend towards `next` (at its own time) by `fade`
    pub fn fade_to(self, next: usize, next_time: f32, fade: f32) -> Self { Self { next: next as i32, next_time, fade, ..self } }
    fn gpu(&self) -> ([f32; 4], [f32; 4]) {
        ([self.clip as f32, self.time, self.next as f32, self.next_time], [self.fade, 0.0, 0.0, 0.0])
    }
}

/// Copies from your storage buffer: `array<struct { transform: mat4x4<f32>, data: vec4<f32>, anim: vec4<f32>, anim_b: vec4<f32> }>`
/// (anim = clip, time, next clip, next time; anim_b.x = fade); culled on the GPU, `bounds` is a world sphere around all copies
#[derive(Clone, Debug)]
pub struct GpuInstances {
    pub buffer: wgpu::Buffer,
    pub count: u32,
    pub bounds: (Vec3, f32),
}

/// Bytes per instance in a `GpuInstances` buffer
pub const GPU_INSTANCE_SIZE: u64 = 112;

/// Camera for one frame
pub struct MeshView {
    pub clip_from_world: Mat4,
    pub view_from_world: Mat4,
    pub world_position: Vec3,
}

impl MeshView {
    /// View from an orbit camera pose (e.g. `camera.pose` or `camera.turntable(t)`)
    pub fn orbit(camera: &OrbitCamera, pose: OrbitPose, aspect: f32) -> Self {
        let view_from_world = pose.view_from_world();
        Self { clip_from_world: camera.clip_from_view(aspect) * view_from_world, view_from_world, world_position: pose.eye() }
    }
}

/// Material from a WGSL file next to the calling file, with hot reload (like `compute_shader!`)
#[macro_export]
macro_rules! mesh_material {
    ($scene:expr, $core:expr, $shader_path:literal, $params:ty) => {{
        let dir = std::path::Path::new(file!()).parent().map(|p| p.to_path_buf()).unwrap_or_default();
        let id = $scene.add_material($core, include_str!($shader_path), std::mem::size_of::<$params>() as u64);
        $scene.watch_material(id, dir.join($shader_path));
        id
    }};
}

const SAMPLES: u32 = 4;
pub const MESH_OUTPUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// xyz world normal, w view depth (0 where empty)
pub const MESH_GBUFFER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Stage { Shadow, Prepass, Opaque, Late, Alpha, Additive, Pick }

// shader, stage, cull, crowd (vertex shader skinning)
type PipelineKey = (usize, Stage, Option<wgpu::Face>, bool);

struct GpuMesh {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    primitives: Vec<MeshPrimitive>,
    // per glTF material
    groups: Vec<wgpu::BindGroup>,
    props: Vec<(MeshAlpha, bool)>,
    _textures: Vec<wgpu::Texture>,
    center: Vec3,
    radius: f32,
    skin: Option<GpuSkin>,
    crowd: Option<CrowdBake>,
    uid: u64,
}

// every clip sampled at CROWD_FPS into bone matrices, for vertex shader skinning of crowds
struct CrowdBake {
    palettes: wgpu::Buffer,
    clips: wgpu::Buffer,
}

const CROWD_FPS: f32 = 30.0;

struct GpuSkin {
    buf: wgpu::Buffer,
    deltas: wgpu::Buffer,
    skinning: MeshSkinning,
    rest: Pose,
}

// an animated object's own posed copy of its mesh
struct ObjectSkin {
    uid: u64,
    palette: wgpu::Buffer,
    morph_weights: wgpu::Buffer,
    out: wgpu::Buffer,
    group: wgpu::BindGroup,
    count: u32,
}

struct Shader {
    module: wgpu::ShaderModule,
    source: String,
    path: Option<PathBuf>,
    watch: Option<(RecommendedWatcher, Receiver<notify::Result<notify::Event>>)>,
}

struct Material {
    shader: usize,
    params: wgpu::Buffer,
    group: wgpu::BindGroup,
    size: u64,
    options: MaterialOptions,
    textures: [wgpu::TextureView; MATERIAL_TEXTURES],
    // textures the scene uploaded for a slot (replaced, not accumulated)
    owned: [Option<wgpu::Texture>; MATERIAL_TEXTURES],
}

/// Texture slots per material (`material_texture0..3` in WGSL)
pub const MATERIAL_TEXTURES: usize = 4;

// GPU-driven copies, their cull parameters and indirect draw arguments
struct GpuState {
    src: GpuInstances,
    params: wgpu::Buffer,
    args: wgpu::Buffer,
    // cull bind group, valid while the transforms buffer generation matches
    group: Option<(u64, wgpu::BindGroup)>,
}

struct Object {
    mesh: MeshId,
    material: MaterialId,
    instances: Vec<Instance>,
    gpu: Option<GpuState>,
    crowd: bool,
    pose: Option<Pose>,
    skin: Option<ObjectSkin>,
}

struct Targets {
    msaa: wgpu::TextureView,
    depth: wgpu::TextureView,
    output: wgpu::Texture,
    output_view: wgpu::TextureView,
    gbuffer_msaa: wgpu::TextureView,
    gbuffer_view: wgpu::TextureView,
    size: [u32; 2],
}

#[derive(Clone, Copy)]
struct Draw {
    key: PipelineKey,
    // byte offset of indirect draw arguments in the object's GPU state (GPU-culled copies)
    indirect: Option<u64>,
    object: usize,
    material: usize,
    mesh: usize,
    prim: usize,
    first: u32,
    count: u32,
    dist: f32,
}

/// Meshes, materials and objects drawn into one HDR texture with shared depth
pub struct MeshScene {
    layout: wgpu::PipelineLayout,
    view_layout: wgpu::BindGroupLayout,
    sun_buf: wgpu::Buffer,
    shadow_view_buf: wgpu::Buffer,
    shadow_group: wgpu::BindGroup,
    shadow_map: wgpu::TextureView,
    shadow_size: u32,
    _shadow_dummy: wgpu::TextureView,
    array_dummy: wgpu::TextureView,
    lights_buf: wgpu::Buffer,
    light_views_buf: wgpu::Buffer,
    light_shadows: LightShadows,
    shadow_sampler: wgpu::Sampler,
    skin_layout: wgpu::BindGroupLayout,
    skin_pipeline: wgpu::ComputePipeline,
    expand_layout: wgpu::BindGroupLayout,
    expand_pipeline: wgpu::ComputePipeline,
    next_uid: u64,
    params_layout: wgpu::BindGroupLayout,
    gltf_layout: wgpu::BindGroupLayout,
    transforms_layout: wgpu::BindGroupLayout,
    view_group: wgpu::BindGroup,
    globals: wgpu::Buffer,
    view_buf: wgpu::Buffer,
    transforms: wgpu::Buffer,
    transforms_group: wgpu::BindGroup,
    crowd_dummy: [wgpu::Buffer; 3],
    // per crowd mesh: transforms generation, mesh uid, group
    crowd_groups: HashMap<usize, (u64, u64, wgpu::BindGroup)>,
    // bumped whenever the transforms buffer is reallocated (cached groups point at it)
    transforms_gen: u64,
    finalize_pipeline: wgpu::ComputePipeline,
    capacity: usize,
    sampler: wgpu::Sampler,
    samplers: HashMap<MeshSampler, wgpu::Sampler>,
    white: wgpu::Texture,
    flat_normal: wgpu::Texture,
    shaders: Vec<Shader>,
    pipelines: HashMap<PipelineKey, Option<wgpu::RenderPipeline>>,
    materials: Vec<Material>,
    meshes: Vec<GpuMesh>,
    objects: Vec<Option<Object>>,
    targets: Targets,
    channel: Option<u32>,
    gbuffer_channel: Option<u32>,
    /// Linear clear colour; alpha 0 marks empty pixels for the post pass
    pub clear: [f32; 4],
    pub sun: Sun,
    /// Point and spot lights (any number; shadowed ones cost 1 or 6 extra depth passes each)
    pub lights: Vec<Light>,
    /// Resolution of each light shadow map
    pub light_shadow_size: u32,
    light_views_used: usize,
    pick_request: Option<[u32; 2]>,
    pick_state: PickState,
    pick_targets: Option<PickTargets>,
    pick_buf: wgpu::Buffer,
    picked: Option<Option<Pick>>,
    /// Skip copies outside the camera view (shadows still see every caster)
    pub culling: bool,
    /// Bounding sphere scale for culling; raise it when a vertex shader pushes geometry far out
    pub cull_margin: f32,
}

impl MeshScene {
    pub fn new(core: &Core) -> Self {
        let device = &core.device;
        let vf = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT;
        let ub = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vf,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let tex = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vf,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let bgl = |label: &str, entries: &[wgpu::BindGroupLayoutEntry]| device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries });
        let view_layout = bgl("Mesh View Layout", &[
            ub(0), ub(1), ub(2),
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: vf,
                ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                count: None,
            },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: vf, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison), count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: vf, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: vf,
                ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2Array, multisampled: false },
                count: None,
            },
            wgpu::BindGroupLayoutEntry { binding: 7, visibility: vf, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ]);
        let params_layout = bgl("Mesh Params Layout", &[
            ub(0), tex(1), tex(2), tex(3), tex(4),
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: vf, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
        ]);
        let gltf_layout = bgl("Mesh glTF Layout", &[
            ub(0), tex(1),
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: vf, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            tex(3), tex(4), tex(5), tex(6),
        ]);
        let ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vf,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        // transforms + crowd skin, baked bones, clip table
        let transforms_layout = bgl("Mesh Transforms Layout", &[ro(0), ro(1), ro(2), ro(3)]);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Mesh Pipeline Layout"),
            bind_group_layouts: &[Some(&view_layout), Some(&params_layout), Some(&gltf_layout), Some(&transforms_layout)],
            immediate_size: 0,
        });

        let globals = uniform_buffer(device, "Mesh Globals", std::mem::size_of::<GlobalsU>() as u64);
        let view_buf = uniform_buffer(device, "Mesh View", std::mem::size_of::<ViewU>() as u64);
        let shadow_view_buf = uniform_buffer(device, "Mesh Shadow View", std::mem::size_of::<ViewU>() as u64);
        let sun_buf = uniform_buffer(device, "Mesh Sun", std::mem::size_of::<SunU>() as u64);
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Mesh Shadow Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let shadow_dummy = depth_texture(device, 1).create_view(&Default::default());
        let array_dummy = depth_array(device, 1, 1).create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() });
        let sun = Sun::default();
        let shadow_map = depth_texture(device, sun.shadow_size).create_view(&Default::default());
        let storage = |label, size| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let lights_buf = storage("Mesh Lights", std::mem::size_of::<LightU>() as u64);
        let light_views_buf = storage("Mesh Light Views", 64);
        let lr = ViewResources { globals: &globals, sun: &sun_buf, sampler: &shadow_sampler, lights: &lights_buf, light_views: &light_views_buf };
        // shadow passes see a dummy map in place of the one they draw into
        let shadow_group = view_group(device, &view_layout, &lr, &shadow_view_buf, &shadow_dummy, &array_dummy);
        let view_group = view_group(device, &view_layout, &lr, &view_buf, &shadow_map, &array_dummy);
        let light_shadows = LightShadows { size: 0, layers: 0, array: array_dummy.clone(), passes: Vec::new() };
        let crowd_dummy = [48u64, 64, 16].map(|size| device.create_buffer(&wgpu::BufferDescriptor { label: Some("Mesh Crowd Dummy"), size, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false }));
        let (transforms, transforms_group) = Self::make_transforms(device, &transforms_layout, &crowd_dummy, 16);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Mesh Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            anisotropy_clamp: 16,
            ..Default::default()
        });
        let px = |c: [u8; 4]| MeshImage { width: 1, height: 1, pixels: c.to_vec() };
        let white = upload_image(core, &px([255; 4]), false);
        let flat_normal = upload_image(core, &px([128, 128, 255, 255]), false);
        let targets = Self::make_targets(core, core.size.width.max(1), core.size.height.max(1));

        let st = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let skin_layout = bgl("Mesh Skin Layout", &[st(0, true), st(1, true), st(2, true), st(3, false), st(4, true), st(5, true)]);
        let skin_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Mesh Skinning"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Mesh Skinning Layout"),
                bind_group_layouts: &[Some(&skin_layout)],
                immediate_size: 0,
            })),
            module: &device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Mesh Skinning"), source: wgpu::ShaderSource::Wgsl(SKIN_SHADER.into()) }),
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let expand_layout = bgl("Mesh Instances Layout", &[st(0, true), st(1, false), wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        }, st(3, false)]);
        let expand_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Mesh Instances"), source: wgpu::ShaderSource::Wgsl(INSTANCES_SHADER.into()) });
        let expand_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Mesh Instances Layout"),
            bind_group_layouts: &[Some(&expand_layout)],
            immediate_size: 0,
        });
        let compute = |entry: &str| device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Mesh Instances"),
            layout: Some(&expand_pipeline_layout),
            module: &expand_module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });
        let (expand_pipeline, finalize_pipeline) = (compute("main"), compute("finalize"));

        Self {
            skin_layout, skin_pipeline, expand_layout, expand_pipeline, next_uid: 0,
            layout, view_layout, sun_buf, shadow_view_buf, shadow_group, shadow_size: sun.shadow_size, shadow_map, _shadow_dummy: shadow_dummy, shadow_sampler, sun,
            array_dummy, lights_buf, light_views_buf, light_shadows, lights: Vec::new(), light_shadow_size: 1024, light_views_used: 0,
            pick_request: None, pick_state: PickState::Idle, pick_targets: None, picked: None,
            pick_buf: device.create_buffer(&wgpu::BufferDescriptor { label: Some("Mesh Pick Readback"), size: 512, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }),
            params_layout, gltf_layout, transforms_layout, view_group, globals, view_buf,
            transforms, transforms_group, crowd_dummy, crowd_groups: HashMap::new(), transforms_gen: 0, finalize_pipeline, capacity: 16, sampler, samplers: HashMap::new(), white, flat_normal,
            shaders: Vec::new(), pipelines: HashMap::new(), materials: Vec::new(), meshes: Vec::new(), objects: Vec::new(),
            targets, channel: None, gbuffer_channel: None, clear: [0.0; 4], culling: true, cull_margin: 1.2,
        }
    }

    fn resize_shadow_map(&mut self, device: &wgpu::Device, size: u32) {
        self.shadow_size = size.clamp(64, 8192);
        self.shadow_map = depth_texture(device, self.shadow_size).create_view(&Default::default());
        self.rebuild_view_groups(device);
    }

    // every group 0: main view, sun shadow pass, one per light shadow view
    fn rebuild_view_groups(&mut self, device: &wgpu::Device) {
        let lr = ViewResources { globals: &self.globals, sun: &self.sun_buf, sampler: &self.shadow_sampler, lights: &self.lights_buf, light_views: &self.light_views_buf };
        self.view_group = view_group(device, &self.view_layout, &lr, &self.view_buf, &self.shadow_map, &self.light_shadows.array);
        self.shadow_group = view_group(device, &self.view_layout, &lr, &self.shadow_view_buf, &self._shadow_dummy, &self.array_dummy);
        for (_, buf, group) in &mut self.light_shadows.passes {
            *group = view_group(device, &self.view_layout, &lr, buf, &self._shadow_dummy, &self.array_dummy);
        }
    }

    // upload lights and their shadow cameras; grows buffers and the shadow array when needed
    fn update_lights(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut views: Vec<(Mat4, Vec3)> = Vec::new();
        let data: Vec<LightU> = self.lights.iter().map(|l| {
            let shadow = if l.shadows { let s = views.len() as i32; views.extend(l.shadow_views()); s } else { -1 };
            let (kind, inner, outer) = match l.kind { LightKind::Point => (0, 0.0, 0.0), LightKind::Spot { inner, outer } => (1, inner, outer) };
            LightU {
                position: l.position.to_array(), range: l.range, color: l.color, kind,
                direction: l.direction.normalize_or_zero().to_array(), cos_outer: outer.cos(), cos_inner: inner.min(outer - 1e-3).cos(),
                shadow, _p0: 0.0, _p1: 0.0,
            }
        }).collect();
        let mut rebuild = false;
        let grow = |buf: &mut wgpu::Buffer, label, bytes: u64, rebuild: &mut bool| if bytes > buf.size() {
            *buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: bytes.next_power_of_two(), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            *rebuild = true;
        };
        grow(&mut self.lights_buf, "Mesh Lights", (data.len() * std::mem::size_of::<LightU>()) as u64, &mut rebuild);
        grow(&mut self.light_views_buf, "Mesh Light Views", (views.len() * 64) as u64, &mut rebuild);
        if !data.is_empty() { queue.write_buffer(&self.lights_buf, 0, bytemuck::cast_slice(&data)); }
        let flat: Vec<[f32; 16]> = views.iter().map(|(m, _)| m.to_cols_array()).collect();
        if !flat.is_empty() { queue.write_buffer(&self.light_views_buf, 0, bytemuck::cast_slice(&flat)); }

        let (size, layers) = (self.light_shadow_size.clamp(64, 4096), views.len() as u32);
        if layers > 0 && (layers > self.light_shadows.layers || size != self.light_shadows.size) {
            let tex = depth_array(device, size, layers);
            let passes = (0..layers).map(|i| {
                let layer = tex.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2), base_array_layer: i, array_layer_count: Some(1), ..Default::default() });
                (layer, uniform_buffer(device, "Mesh Light Shadow View", std::mem::size_of::<ViewU>() as u64), self.shadow_group.clone())
            }).collect();
            let array = tex.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() });
            self.light_shadows = LightShadows { size, layers, array, passes };
            rebuild = true;
        }
        if rebuild { self.rebuild_view_groups(device); }
        for (i, (m, eye)) in views.iter().enumerate() {
            let (_, buf, _) = &self.light_shadows.passes[i];
            queue.write_buffer(buf, 0, bytemuck::bytes_of(&ViewU {
                clip_from_world: m.to_cols_array(), view_from_world: Mat4::IDENTITY.to_cols_array(), world_position: eye.to_array(), _pad: 0.0,
                viewport: [0.0, 0.0, size as f32, size as f32],
            }));
        }
        self.light_views_used = views.len();
    }

    fn make_transforms(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, dummy: &[wgpu::Buffer; 3], capacity: usize) -> (wgpu::Buffer, wgpu::BindGroup) {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mesh Transforms"),
            size: (capacity * std::mem::size_of::<TransformU>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = transforms_group(device, layout, &buf, [&dummy[0], &dummy[1], &dummy[2]]);
        (buf, group)
    }

    // compile + test-build pipelines in an error scope; `fn vertex` (or `mesh_vertex`) is wrapped
    // by two entry points, the crowd one poses the vertex first
    fn compile(device: &wgpu::Device, layout: &wgpu::PipelineLayout, source: &str) -> Result<wgpu::ShaderModule, String> {
        let (body, call) = if strip_comments(source).contains("fn vertex(") { (source.replace("@vertex", ""), "vertex") } else { (source.to_string(), "mesh_vertex") };
        let wgsl = format!("{MESH_PRELUDE}\n{body}\n
@vertex fn cuneus_vertex(v: Vertex) -> VertexOutput {{ return {call}(v); }}
@vertex fn cuneus_crowd_vertex(v: Vertex, @builtin(vertex_index) vi: u32) -> VertexOutput {{ return {call}(crowd_pose(v, vi)); }}
");
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Mesh Material"), source: wgpu::ShaderSource::Wgsl(wgsl.into()) });
        for (stage, crowd) in [(Stage::Opaque, false), (Stage::Opaque, true), (Stage::Prepass, false), (Stage::Shadow, false), (Stage::Pick, false)] {
            let _ = Self::pipeline(device, layout, &module, stage, None, crowd);
        }
        match pollster::block_on(scope.pop()) {
            Some(e) => Err(e.to_string()),
            None => Ok(module),
        }
    }

    fn pipeline(device: &wgpu::Device, layout: &wgpu::PipelineLayout, module: &wgpu::ShaderModule, stage: Stage, cull: Option<wgpu::Face>, crowd: bool) -> wgpu::RenderPipeline {
        let vertex = if crowd { "cuneus_crowd_vertex" } else { "cuneus_vertex" };
        let (fragment, formats): (&str, &[wgpu::TextureFormat]) = match stage {
            Stage::Shadow => ("cuneus_shadow", &[]),
            Stage::Prepass => ("cuneus_prepass", &[MESH_GBUFFER_FORMAT]),
            Stage::Pick => ("cuneus_pick", &[wgpu::TextureFormat::Rg32Uint, wgpu::TextureFormat::Rgba32Float]),
            _ => ("fragment", &[MESH_OUTPUT_FORMAT]),
        };
        let targets: Vec<Option<wgpu::ColorTargetState>> = formats.iter().map(|&format| Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })).collect();
        let add = wgpu::BlendComponent { src_factor: wgpu::BlendFactor::SrcAlpha, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add };
        let blend = match stage {
            // premultiplied result: empty background ends as (rgb*a, a), what the post pass expects
            Stage::Alpha => Some(wgpu::BlendState {
                color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::SrcAlpha, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
                alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
            }),
            Stage::Additive => Some(wgpu::BlendState {
                color: add,
                alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Zero, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
            }),
            _ => None,
        };
        let (write, compare) = match stage {
            Stage::Shadow | Stage::Prepass | Stage::Late | Stage::Pick => (true, wgpu::CompareFunction::Less),
            Stage::Opaque => (false, wgpu::CompareFunction::Equal),
            Stage::Alpha | Stage::Additive => (false, wgpu::CompareFunction::LessEqual),
        };
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Mesh Pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module,
                entry_point: Some(vertex),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<MeshVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x4, 4 => Float32x4, 5 => Float32x2],
                })],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: Some(fragment),
                targets: &targets.iter().map(|t| t.clone().map(|t| wgpu::ColorTargetState { blend, ..t })).collect::<Vec<_>>(),
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: cull, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(write),
                depth_compare: Some(compare),
                stencil: Default::default(),
                bias: if stage == Stage::Shadow { wgpu::DepthBiasState { constant: 2, slope_scale: 1.5, clamp: 0.0 } } else { Default::default() },
            }),
            multisample: wgpu::MultisampleState { count: if matches!(stage, Stage::Shadow | Stage::Pick) { 1 } else { SAMPLES }, ..Default::default() },
            multiview_mask: None,
            cache: None,
        })
    }

    fn make_targets(core: &Core, w: u32, h: u32) -> Targets {
        let tex = |label: &str, format: wgpu::TextureFormat, samples: u32, usage: wgpu::TextureUsages| {
            core.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let rt = wgpu::TextureUsages::RENDER_ATTACHMENT;
        let rt_read = rt | wgpu::TextureUsages::TEXTURE_BINDING;
        let msaa = tex("Mesh MSAA", MESH_OUTPUT_FORMAT, SAMPLES, rt).create_view(&Default::default());
        let depth = tex("Mesh Depth", DEPTH_FORMAT, SAMPLES, rt).create_view(&Default::default());
        let output = tex("Mesh Output", MESH_OUTPUT_FORMAT, 1, rt_read);
        let output_view = output.create_view(&Default::default());
        let gbuffer_msaa = tex("Mesh GBuffer MSAA", MESH_GBUFFER_FORMAT, SAMPLES, rt).create_view(&Default::default());
        let gbuffer_view = tex("Mesh GBuffer", MESH_GBUFFER_FORMAT, 1, rt_read).create_view(&Default::default());
        Targets { msaa, depth, output, output_view, gbuffer_msaa, gbuffer_view, size: [w, h] }
    }

    // meshes

    /// Upload geometry; draw it with `spawn`, as many times as you like
    pub fn add_mesh(&mut self, core: &Core, data: &MeshData) -> MeshId {
        let m = self.upload_mesh(core, data);
        self.meshes.push(m);
        MeshId(self.meshes.len() - 1)
    }

    /// Swap the geometry behind `id`; every object using it follows
    pub fn replace_mesh(&mut self, core: &Core, id: MeshId, data: &MeshData) {
        let m = self.upload_mesh(core, data);
        if let Some(slot) = self.meshes.get_mut(id.0) { *slot = m; }
    }

    fn upload_mesh(&mut self, core: &Core, data: &MeshData) -> GpuMesh {
        use wgpu::util::DeviceExt;
        let device = &core.device;
        self.next_uid += 1;
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mesh Vertices"),
            contents: bytemuck::cast_slice(&data.vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
        });
        // joints + weights per vertex, read by the skinning pass
        let skin = data.skinning.as_ref()
            .filter(|sk| sk.joints.len() == data.vertices.len() && sk.morph.len() == data.vertices.len() && !sk.palette.is_empty())
            .map(|sk| {
                let packed: Vec<[u32; 12]> = sk.joints.iter().zip(&sk.weights).zip(&sk.morph)
                    .map(|((j, w), m)| [j[0], j[1], j[2], j[3], w[0].to_bits(), w[1].to_bits(), w[2].to_bits(), w[3].to_bits(), m[0], m[1], m[2], m[3]])
                    .collect();
                let storage = |label, contents: &[u8]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage: wgpu::BufferUsages::STORAGE });
                let deltas = if sk.morph_deltas.is_empty() { vec![[0.0f32; 4]; 2] } else { sk.morph_deltas.clone() };
                GpuSkin {
                    buf: storage("Mesh Skin", bytemuck::cast_slice(&packed)),
                    deltas: storage("Mesh Morph Deltas", bytemuck::cast_slice(&deltas)),
                    rest: sk.rest(),
                    skinning: sk.clone(),
                }
            });
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mesh Indices"),
            contents: bytemuck::cast_slice(&data.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // colour textures are sRGB, data textures (normal, metallic-roughness, occlusion) linear
        let mut srgb = vec![false; data.images.len()];
        for m in &data.materials {
            for i in [m.base_color_image, m.emissive_image].into_iter().flatten() {
                if let Some(s) = srgb.get_mut(i) { *s = true; }
            }
        }
        let textures: Vec<wgpu::Texture> = data.images.iter().zip(&srgb).map(|(i, &s)| upload_image(core, i, s)).collect();
        let view = |t: &wgpu::Texture, srgb: bool| t.create_view(&wgpu::TextureViewDescriptor {
            format: Some(if srgb { wgpu::TextureFormat::Rgba8UnormSrgb } else { wgpu::TextureFormat::Rgba8Unorm }),
            ..Default::default()
        });
        let pick = |i: Option<usize>, srgb: bool, fallback: &wgpu::Texture| view(i.and_then(|i| textures.get(i)).unwrap_or(fallback), srgb);

        let mut groups = Vec::with_capacity(data.materials.len());
        let mut props = Vec::with_capacity(data.materials.len());
        for m in &data.materials {
            let bit = |i: Option<usize>, b: u32| if i.is_some_and(|i| i < textures.len()) { b } else { 0 };
            let flags = bit(m.base_color_image, 1) | bit(m.normal_image, 2) | bit(m.metallic_roughness_image, 4)
                | bit(m.emissive_image, 8) | bit(m.occlusion_image, 16) | if m.double_sided { 32 } else { 0 };
            let u = GltfMaterialU {
                base_color: m.base_color,
                emissive: [m.emissive[0], m.emissive[1], m.emissive[2], 1.0],
                metallic: m.metallic,
                roughness: m.roughness,
                normal_scale: m.normal_scale,
                occlusion_strength: m.occlusion_strength,
                alpha_cutoff: m.alpha_cutoff,
                alpha_mode: match m.alpha { MeshAlpha::Opaque => 0, MeshAlpha::Mask => 1, MeshAlpha::Blend => 2 },
                flags,
                _pad: 0,
                uv_row0: [m.uv_transform[0][0], m.uv_transform[0][1], m.uv_transform[0][2], m.uv_set as f32],
                uv_row1: [m.uv_transform[1][0], m.uv_transform[1][1], m.uv_transform[1][2], 0.0],
            };
            let sampler = self.sampler_for(&core.device, m.sampler);
            let b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Mesh glTF Material"),
                contents: bytemuck::bytes_of(&u),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let views = [
                pick(m.base_color_image, true, &self.white),
                pick(m.normal_image, false, &self.flat_normal),
                pick(m.metallic_roughness_image, false, &self.white),
                pick(m.emissive_image, true, &self.white),
                pick(m.occlusion_image, false, &self.white),
            ];
            groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Mesh glTF Group"),
                layout: &self.gltf_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&views[0]) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&views[1]) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&views[2]) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&views[3]) },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&views[4]) },
                ],
            }));
            props.push((m.alpha, m.double_sided));
        }
        info!("Mesh: {} vertices, {} triangles, {} materials, {} textures",
            data.vertices.len(), data.indices.len() / 3, data.materials.len(), data.images.len());
        // animated parts can leave the rest bounds; margin for shadow fitting
        let radius = if skin.is_some() { data.radius * 1.5 } else { data.radius };
        GpuMesh { vertices, indices, primitives: data.primitives.clone(), groups, props, _textures: textures, center: data.center, radius, skin, crowd: None, uid: self.next_uid }
    }

    // one sampler per distinct glTF filtering / wrapping
    fn sampler_for(&mut self, device: &wgpu::Device, s: MeshSampler) -> wgpu::Sampler {
        self.samplers.entry(s).or_insert_with(|| {
            let wrap = |w| match w { MeshWrap::Repeat => wgpu::AddressMode::Repeat, MeshWrap::Clamp => wgpu::AddressMode::ClampToEdge, MeshWrap::Mirror => wgpu::AddressMode::MirrorRepeat };
            let f = |nearest| if nearest { wgpu::FilterMode::Nearest } else { wgpu::FilterMode::Linear };
            let smooth = !s.mag_nearest && !s.min_nearest && !s.mip_nearest;
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Mesh glTF Sampler"),
                address_mode_u: wrap(s.wrap_u),
                address_mode_v: wrap(s.wrap_v),
                mag_filter: f(s.mag_nearest),
                min_filter: f(s.min_nearest),
                mipmap_filter: if s.mip_nearest { wgpu::MipmapFilterMode::Nearest } else { wgpu::MipmapFilterMode::Linear },
                anisotropy_clamp: if smooth { 16 } else { 1 },
                ..Default::default()
            })
        }).clone()
    }

    // materials

    /// Material from WGSL source; `params_size` is the uniform's size (`@group(1) @binding(0)`), plain shading if it fails to compile
    pub fn add_material(&mut self, core: &Core, source: &str, params_size: u64) -> MaterialId {
        let module = match Self::compile(&core.device, &self.layout, source) {
            Ok(m) => m,
            Err(e) => {
                error!("Mesh shader failed, using fallback: {e}");
                Self::compile(&core.device, &self.layout, FALLBACK_FRAGMENT).expect("fallback mesh shader")
            }
        };
        self.shaders.push(Shader { module, source: source.to_string(), path: None, watch: None });
        self.push_material(core, self.shaders.len() - 1, params_size, MaterialOptions::default())
    }

    /// Same shader as `of`, with its own parameter values; starts with the same options and textures
    pub fn material_variant(&mut self, core: &Core, of: MaterialId) -> MaterialId {
        let m = &self.materials[of.0];
        let (shader, size, options, textures) = (m.shader, m.size, m.options, m.textures.clone());
        let id = self.push_material(core, shader, size, options);
        self.materials[id.0].textures = textures;
        self.rebuild_material_group(&core.device, id);
        id
    }

    fn push_material(&mut self, core: &Core, shader: usize, size: u64, options: MaterialOptions) -> MaterialId {
        let params = uniform_buffer(&core.device, "Mesh Params", size);
        let textures = std::array::from_fn(|_| self.white.create_view(&Default::default()));
        let group = self.material_group(&core.device, &params, &textures);
        self.materials.push(Material { shader, params, group, size, options, textures, owned: Default::default() });
        MaterialId(self.materials.len() - 1)
    }

    fn material_group(&self, device: &wgpu::Device, params: &wgpu::Buffer, textures: &[wgpu::TextureView; MATERIAL_TEXTURES]) -> wgpu::BindGroup {
        let t = |i: usize| wgpu::BindGroupEntry { binding: i as u32 + 1, resource: wgpu::BindingResource::TextureView(&textures[i]) };
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mesh Params Group"),
            layout: &self.params_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                t(0), t(1), t(2), t(3),
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        })
    }

    fn rebuild_material_group(&mut self, device: &wgpu::Device, id: MaterialId) {
        let Some(m) = self.materials.get(id.0) else { return };
        let group = self.material_group(device, &m.params, &m.textures);
        self.materials[id.0].group = group;
    }

    /// Bind any filterable 2D float texture view to `material_texture{slot}`
    pub fn set_texture(&mut self, core: &Core, id: MaterialId, slot: usize, view: &wgpu::TextureView) {
        let Some(m) = self.materials.get_mut(id.0) else { return };
        let Some(t) = m.textures.get_mut(slot) else { warn!("material texture slot {slot} out of range"); return };
        *t = view.clone();
        m.owned[slot] = None;
        self.rebuild_material_group(&core.device, id);
    }

    /// The view bound to `material_texture{slot}` (e.g. to show a loaded HDRI in a post pass)
    pub fn material_texture(&self, id: MaterialId, slot: usize) -> Option<&wgpu::TextureView> {
        self.materials.get(id.0)?.textures.get(slot)
    }

    /// Upload an RGBA8 image (mipmapped) to `material_texture{slot}`; `srgb` for colour images
    pub fn set_image(&mut self, core: &Core, id: MaterialId, slot: usize, image: &MeshImage, srgb: bool) {
        let tex = upload_image(core, image, srgb);
        let view = tex.create_view(&wgpu::TextureViewDescriptor {
            format: Some(if srgb { wgpu::TextureFormat::Rgba8UnormSrgb } else { wgpu::TextureFormat::Rgba8Unorm }),
            ..Default::default()
        });
        self.set_texture(core, id, slot, &view);
        if let Some(o) = self.materials.get_mut(id.0).and_then(|m| m.owned.get_mut(slot)) { *o = Some(tex); }
    }

    /// Load an image into `material_texture{slot}`; .hdr / .exr stay HDR, `srgb` for 8-bit colour images
    pub fn set_image_file<P: AsRef<Path>>(&mut self, core: &Core, id: MaterialId, slot: usize, path: P, srgb: bool) -> anyhow::Result<()> {
        let path = path.as_ref();
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or_default().to_ascii_lowercase();
        let img = image::open(path)?;
        if ext == "hdr" || ext == "exr" {
            let rgba = img.to_rgba32f();
            let tex = upload_hdr(core, rgba.width(), rgba.height(), rgba.as_raw());
            self.set_texture(core, id, slot, &tex.create_view(&Default::default()));
            if let Some(o) = self.materials.get_mut(id.0).and_then(|m| m.owned.get_mut(slot)) { *o = Some(tex); }
        } else {
            let rgba = img.to_rgba8();
            self.set_image(core, id, slot, &MeshImage { width: rgba.width(), height: rgba.height(), pixels: rgba.into_raw() }, srgb);
        }
        Ok(())
    }

    /// Upload a material's parameter values (`@group(1) @binding(0)`)
    pub fn set_params<T: bytemuck::Pod>(&self, queue: &wgpu::Queue, id: MaterialId, params: &T) {
        if let Some(m) = self.materials.get(id.0) {
            queue.write_buffer(&m.params, 0, bytemuck::bytes_of(params));
        }
    }

    pub fn set_options(&mut self, id: MaterialId, options: MaterialOptions) {
        if let Some(m) = self.materials.get_mut(id.0) { m.options = options; }
    }

    pub fn options(&self, id: MaterialId) -> MaterialOptions {
        self.materials.get(id.0).map(|m| m.options).unwrap_or_default()
    }

    /// Rebuild the material's shader when `path` is saved (a broken edit keeps the last good one)
    pub fn watch_material<P: AsRef<Path>>(&mut self, id: MaterialId, path: P) {
        let path = path.as_ref().to_path_buf();
        let Some(shader) = self.materials.get(id.0).and_then(|m| self.shaders.get_mut(m.shader)) else { return };
        let (tx, rx) = channel();
        match notify::recommended_watcher(tx) {
            Ok(mut w) => {
                if let Err(e) = w.watch(&path, RecursiveMode::NonRecursive) {
                    warn!("mesh hot reload: cannot watch {path:?}: {e}");
                    return;
                }
                shader.watch = Some((w, rx));
                shader.path = Some(path);
            }
            Err(e) => warn!("mesh hot reload: {e}"),
        }
    }

    /// Rebuild edited shaders; returns true when one changed. `render` calls this.
    pub fn check_hot_reload(&mut self, device: &wgpu::Device) -> bool {
        let mut any = false;
        for (i, s) in self.shaders.iter_mut().enumerate() {
            let Some((_, rx)) = &s.watch else { continue };
            let mut changed = false;
            while rx.try_recv().is_ok() { changed = true; }
            if !changed { continue; }
            let Some(src) = s.path.as_ref().and_then(|p| std::fs::read_to_string(p).ok()) else { continue };
            if src == s.source { continue; }
            s.source = src;
            match Self::compile(device, &self.layout, &s.source) {
                Ok(module) => {
                    s.module = module;
                    self.pipelines.retain(|k, _| k.0 != i);
                    any = true;
                    info!("Mesh shader reloaded: {:?}", s.path);
                }
                Err(e) => error!("Mesh shader error, keeping the previous one:\n{e}"),
            }
        }
        any
    }

    // objects

    /// Place a mesh with a material in the scene
    pub fn spawn(&mut self, mesh: MeshId, material: MaterialId, transform: Mat4) -> ObjectId {
        self.spawn_instances(mesh, material, vec![transform.into()])
    }

    /// Many copies of a mesh in one draw; edit them later with `instances_mut`
    pub fn spawn_instances(&mut self, mesh: MeshId, material: MaterialId, instances: Vec<Instance>) -> ObjectId {
        let o = Some(Object { mesh, material, instances, gpu: None, crowd: false, pose: None, skin: None });
        match self.objects.iter().position(|o| o.is_none()) {
            Some(i) => { self.objects[i] = o; ObjectId(i) }
            None => { self.objects.push(o); ObjectId(self.objects.len() - 1) }
        }
    }

    pub fn despawn(&mut self, id: ObjectId) {
        if let Some(o) = self.objects.get_mut(id.0) { *o = None; }
    }

    fn object_mut(&mut self, id: ObjectId) -> Option<&mut Object> {
        self.objects.get_mut(id.0).and_then(|o| o.as_mut())
    }

    /// Transform of the first instance
    pub fn set_transform(&mut self, id: ObjectId, transform: Mat4) {
        if let Some(o) = self.object_mut(id) {
            match o.instances.first_mut() {
                Some(i) => i.transform = transform,
                None => o.instances.push(transform.into()),
            }
        }
    }

    pub fn transform(&self, id: ObjectId) -> Option<Mat4> {
        self.objects.get(id.0)?.as_ref()?.instances.first().map(|i| i.transform)
    }

    pub fn instances_mut(&mut self, id: ObjectId) -> Option<&mut Vec<Instance>> {
        self.object_mut(id).map(|o| &mut o.instances)
    }

    /// Place the object's copies from your GPU buffer instead of `instances` (`None` goes back)
    pub fn set_gpu_instances(&mut self, core: &Core, id: ObjectId, source: Option<GpuInstances>) {
        let Some(o) = self.object_mut(id) else { return };
        let old = o.gpu.take();
        o.gpu = source.map(|src| match old {
            Some(g) => GpuState { src, group: None, ..g },
            None => GpuState {
                src,
                params: uniform_buffer(&core.device, "Mesh Cull Params", std::mem::size_of::<CullU>() as u64),
                args: indirect_buffer(&core.device, 1),
                group: None,
            },
        });
    }

    /// Each copy plays its own `Instance::anim` (or GPU anim fields), posed in the vertex shader; blend shapes stay at rest
    pub fn set_crowd(&mut self, id: ObjectId, on: bool) {
        if let Some(o) = self.object_mut(id) { o.crowd = on; }
    }

    /// Spawn an object whose copies come from your GPU buffer
    pub fn spawn_gpu_instances(&mut self, core: &Core, mesh: MeshId, material: MaterialId, source: GpuInstances) -> ObjectId {
        let id = self.spawn_instances(mesh, material, Vec::new());
        self.set_gpu_instances(core, id, Some(source));
        id
    }

    /// One clip at `time` (loops), `None` = rest pose; shared by all instances
    pub fn animate(&mut self, id: ObjectId, animation: Option<usize>, time: f32) {
        match animation {
            Some(a) => self.animate_layers(id, &[(a, time, 1.0)]),
            None => self.animate_layers(id, &[]),
        }
    }

    /// (clip, time, weight) layers, see `MeshSkinning::pose_layers`
    pub fn animate_layers(&mut self, id: ObjectId, layers: &[(usize, f32, f32)]) {
        let Some(o) = self.objects.get_mut(id.0).and_then(|o| o.as_mut()) else { return };
        let Some(skin) = self.meshes.get(o.mesh.0).and_then(|m| m.skin.as_ref()) else { return };
        o.pose = Some(skin.skinning.pose_layers(layers));
    }

    /// Your own pose (build it from `MeshSkinning::rest`, `pose_layers` or `palette_from_local`)
    pub fn set_pose(&mut self, id: ObjectId, pose: Pose) {
        if let Some(o) = self.object_mut(id) { o.pose = Some(pose); }
    }

    /// Nodes, skins and animations of a mesh, `None` for static meshes
    pub fn skinning(&self, mesh: MeshId) -> Option<&MeshSkinning> {
        self.meshes.get(mesh.0)?.skin.as_ref().map(|s| &s.skinning)
    }

    pub fn set_material(&mut self, id: ObjectId, material: MaterialId) {
        if let Some(o) = self.object_mut(id) { o.material = material; }
    }

    pub fn set_mesh(&mut self, id: ObjectId, mesh: MeshId) {
        if let Some(o) = self.object_mut(id) { o.mesh = mesh; }
    }

    /// Ask what is under pixel (x, y); the answer comes from `take_pick` a frame or two later
    pub fn pick(&mut self, x: u32, y: u32) {
        self.pick_request = Some([x, y]);
    }

    /// Once a pick finished: `Some(Some(hit))`, or `Some(None)` when nothing was there
    pub fn take_pick(&mut self) -> Option<Option<Pick>> {
        self.picked.take()
    }

    // readback: map the copied pixel once its frame was submitted, resolve it when mapped
    fn poll_pick(&mut self, device: &wgpu::Device) {
        let _ = device.poll(wgpu::PollType::Poll);
        match std::mem::replace(&mut self.pick_state, PickState::Idle) {
            PickState::Copied => {
                let done = Arc::new(AtomicU8::new(0));
                let flag = done.clone();
                self.pick_buf.map_async(wgpu::MapMode::Read, .., move |r| flag.store(if r.is_ok() { 1 } else { 2 }, Ordering::Release));
                self.pick_state = PickState::Mapping(done);
            }
            PickState::Mapping(done) if done.load(Ordering::Acquire) == 2 => {
                warn!("mesh pick readback failed");
                self.picked = Some(None);
            }
            PickState::Mapping(done) if done.load(Ordering::Acquire) == 1 => {
                // object + 1 and copy, then the world position
                let hit = self.pick_buf.get_mapped_range(..).ok().and_then(|b| {
                    let u = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
                    let object = u(0).checked_sub(1)? as usize;
                    Some(Pick { object: ObjectId(object), instance: u(4), position: Vec3::new(f32::from_bits(u(256)), f32::from_bits(u(260)), f32::from_bits(u(264))) })
                });
                self.pick_buf.unmap();
                self.picked = Some(hit);
            }
            other => self.pick_state = other,
        }
    }

    // output

    /// Feed the colour to `post` as `channel`; from then on it follows `post`'s size (resize, export)
    pub fn attach(&mut self, core: &Core, post: &mut ComputeShader, channel: u32) {
        self.channel = Some(channel);
        self.follow(core, post, true);
    }

    /// Feed the gbuffer (xyz world normal, w view depth, 0 where empty) as `channel`, coverage weighted like the colour
    pub fn attach_gbuffer(&mut self, core: &Core, post: &mut ComputeShader, channel: u32) {
        self.gbuffer_channel = Some(channel);
        self.follow(core, post, true);
    }

    // match post's output size, rebind on change
    fn follow(&mut self, core: &Core, post: &mut ComputeShader, force: bool) {
        let t = &post.get_output_texture().texture;
        let size = [t.width().max(1), t.height().max(1)];
        if !force && size == self.targets.size { return; }
        if size != self.targets.size { self.targets = Self::make_targets(core, size[0], size[1]); }
        if let Some(c) = self.channel {
            post.update_channel_texture(c, &self.targets.output_view, &self.sampler, &core.device, &core.queue);
        }
        if let Some(c) = self.gbuffer_channel {
            post.update_channel_texture(c, &self.targets.gbuffer_view, &self.sampler, &core.device, &core.queue);
        }
    }

    /// The HDR result (single sample)
    pub fn output_view(&self) -> &wgpu::TextureView { &self.targets.output_view }
    pub fn output_texture(&self) -> &wgpu::Texture { &self.targets.output }
    pub fn gbuffer_view(&self) -> &wgpu::TextureView { &self.targets.gbuffer_view }
    pub fn output_sampler(&self) -> &wgpu::Sampler { &self.sampler }
    pub fn size(&self) -> [u32; 2] { self.targets.size }

    fn get_pipeline(&mut self, device: &wgpu::Device, key: PipelineKey) {
        if self.pipelines.contains_key(&key) { return; }
        let s = &self.shaders[key.0];
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let p = Self::pipeline(device, &self.layout, &s.module, key.1, key.2, key.3);
        let p = match pollster::block_on(scope.pop()) {
            Some(e) => { error!("Mesh pipeline failed: {e}"); None }
            None => Some(p),
        };
        self.pipelines.insert(key, p);
    }

    /// Draw every object into `post`'s channel; dispatch `post` after
    pub fn render(&mut self, encoder: &mut wgpu::CommandEncoder, core: &Core, post: &mut ComputeShader, view: &MeshView) {
        self.check_hot_reload(&core.device);
        self.follow(core, post, false);
        self.poll_pick(&core.device);
        let picking = matches!(self.pick_state, PickState::Idle) && self.pick_request.is_some_and(|[x, y]| x < self.targets.size[0] && y < self.targets.size[1]);
        let queue = &core.queue;
        let [w, h] = self.targets.size;
        let t = post.time_uniform.data;
        queue.write_buffer(&self.globals, 0, bytemuck::bytes_of(&GlobalsU { time: t.time, delta_time: t.delta, frame_count: t.frame, _pad: 0 }));
        queue.write_buffer(&self.view_buf, 0, bytemuck::bytes_of(&ViewU {
            clip_from_world: view.clip_from_world.to_cols_array(),
            view_from_world: view.view_from_world.to_cols_array(),
            world_position: view.world_position.to_array(),
            _pad: 0.0,
            viewport: [0.0, 0.0, w as f32, h as f32],
        }));

        // animated objects: palette → compute pass into the object's own vertex buffer
        let mut skinned = Vec::new();
        for (oi, slot) in self.objects.iter_mut().enumerate() {
            let Some(o) = slot else { continue };
            if o.crowd { continue; }
            let Some((mesh, skin)) = self.meshes.get(o.mesh.0).and_then(|m| m.skin.as_ref().map(|s| (m, s))) else { o.skin = None; continue };
            if o.skin.as_ref().map(|s| s.uid) != Some(mesh.uid) {
                let count = skin.skinning.joints.len() as u32;
                let storage = |label, size: u64| core.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: size.max(16),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let palette = storage("Mesh Palette", (skin.rest.palette.len() * 64) as u64);
                let morph_weights = storage("Mesh Morph Weights", (skin.rest.morph_weights.len() * 4) as u64);
                let out = core.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Mesh Posed Vertices"),
                    size: count as u64 * std::mem::size_of::<MeshVertex>() as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
                    mapped_at_creation: false,
                });
                let group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Mesh Skin Group"),
                    layout: &self.skin_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: mesh.vertices.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: skin.buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: palette.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: skin.deltas.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: morph_weights.as_entire_binding() },
                    ],
                });
                o.skin = Some(ObjectSkin { uid: mesh.uid, palette, morph_weights, out, group, count });
            }
            // a pose from another mesh (or a hand-made one of the wrong size) falls back to rest
            let rest = &skin.rest;
            let pose = o.pose.as_ref();
            let pal = pose.map(|p| &p.palette).filter(|p| p.len() == rest.palette.len()).unwrap_or(&rest.palette);
            let mw = pose.map(|p| &p.morph_weights).filter(|m| m.len() == rest.morph_weights.len()).unwrap_or(&rest.morph_weights);
            let flat: Vec<[f32; 16]> = pal.iter().map(|m| m.to_cols_array()).collect();
            if let Some(os) = &o.skin {
                queue.write_buffer(&os.palette, 0, bytemuck::cast_slice(&flat));
                if !mw.is_empty() { queue.write_buffer(&os.morph_weights, 0, bytemuck::cast_slice(mw)); }
                skinned.push(oi);
            }
        }
        if !skinned.is_empty() {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Mesh Skinning"), timestamp_writes: None });
            cp.set_pipeline(&self.skin_pipeline);
            for &oi in &skinned {
                if let Some(os) = self.objects[oi].as_ref().and_then(|o| o.skin.as_ref()) {
                    cp.set_bind_group(0, &os.group, &[]);
                    let [x, y] = workgroups(os.count);
                    cp.dispatch_workgroups(x, y, 1);
                }
            }
        }

        // camera frustum planes (normal, offset), inside when dot(n, p) + d >= 0
        let m = view.clip_from_world.cols;
        let row = |i: usize| [m[0][i], m[1][i], m[2][i], m[3][i]];
        let plane = |a: [f32; 4], b: [f32; 4], k: f32| {
            let n = Vec3::new(a[0] + k * b[0], a[1] + k * b[1], a[2] + k * b[2]);
            let l = n.length().max(1e-12);
            (n * (1.0 / l), (a[3] + k * b[3]) / l)
        };
        let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
        let planes = [plane(r3, r0, 1.0), plane(r3, r0, -1.0), plane(r3, r1, 1.0), plane(r3, r1, -1.0), plane(r2, r2, 0.0), plane(r3, r2, -1.0)];

        self.update_lights(&core.device, queue);
        let any_shadows = self.sun.shadows || self.light_views_used > 0;

        // crowd meshes: bake their clips once
        let crowd_meshes: Vec<usize> = self.objects.iter().flatten().filter(|o| o.crowd).map(|o| o.mesh.0).collect();
        for mi in crowd_meshes {
            if let Some(m) = self.meshes.get_mut(mi) {
                if m.crowd.is_none() { m.crowd = m.skin.as_ref().map(|sk| bake_crowd(&core.device, &sk.skinning)); }
            }
        }

        // GPU copies first: packed (visible) region, plus a full region for shadows; CPU copies after
        let indirect = self.culling && core.device.features().contains(wgpu::Features::INDIRECT_FIRST_INSTANCE);
        let mut gpu_region: HashMap<usize, (u32, Option<u32>)> = HashMap::new();
        let mut gpu_total = 0u32;
        for (oi, o) in self.objects.iter().enumerate() {
            let Some(o) = o else { continue };
            let Some(g) = &o.gpu else { continue };
            let casts = any_shadows && self.materials.get(o.material.0).is_some_and(|m| m.options.shadows);
            let culled = gpu_total;
            gpu_total += g.src.count;
            let full = casts.then(|| { let f = gpu_total; gpu_total += g.src.count; f });
            gpu_region.insert(oi, (culled, full));
        }

        // CPU copies: visible ones for the camera, all of them again for shadows when some were culled
        let mut data: Vec<TransformU> = Vec::new();
        let mut draws: Vec<Draw> = Vec::new();
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        let tu = |object: usize, k: usize, i: &Instance| {
            let (anim, anim_b) = i.anim.gpu();
            TransformU {
                world_from_local: i.transform.to_cols_array(),
                normal_from_local: i.transform.normal_matrix().to_cols_array(),
                data: i.data, anim, anim_b, ids: [object as u32 + 1, k as u32, 0, 0],
            }
        };
        for (oi, o) in self.objects.iter().enumerate() {
            let Some(o) = o else { continue };
            let (Some(mat), Some(mesh)) = (self.materials.get(o.material.0), self.meshes.get(o.mesh.0)) else { continue };
            let casts = any_shadows && mat.options.shadows;
            let gpu = o.gpu.as_ref().map(|g| &g.src);
            let (first, count, sfirst, scount, dist) = if let Some(g) = gpu {
                if g.count == 0 { continue; }
                let (culled, full) = gpu_region[&oi];
                if casts {
                    let r = Vec3::splat(g.bounds.1);
                    (lo, hi) = (lo.min(g.bounds.0 - r), hi.max(g.bounds.0 + r));
                }
                (culled, g.count, full.unwrap_or(culled), g.count, (g.bounds.0 - view.world_position).length())
            } else {
                if o.instances.is_empty() { continue; }
                let visible = |i: &&Instance| !self.culling || {
                    let c = i.transform.transform_point3(mesh.center);
                    let r = mesh.radius * i.transform.max_scale() * self.cull_margin;
                    planes.iter().all(|&(n, d)| n.dot(c) + d >= -r)
                };
                let first = gpu_total + data.len() as u32;
                data.extend(o.instances.iter().enumerate().filter(|(_, i)| visible(i)).map(|(k, i)| tu(oi, k, i)));
                let count = gpu_total + data.len() as u32 - first;
                let (sfirst, scount) = if !casts || count == o.instances.len() as u32 { (first, count) } else {
                    let s = gpu_total + data.len() as u32;
                    data.extend(o.instances.iter().enumerate().map(|(k, i)| tu(oi, k, i)));
                    (s, gpu_total + data.len() as u32 - s)
                };
                if casts {
                    for i in &o.instances {
                        let c = i.transform.transform_point3(mesh.center);
                        let r = mesh.radius * i.transform.max_scale();
                        (lo, hi) = (lo.min(c - Vec3::splat(r)), hi.max(c + Vec3::splat(r)));
                    }
                }
                let p0 = o.instances[0].transform.transform_point3(Vec3::ZERO);
                (first, count, sfirst, scount, (p0 - view.world_position).length())
            };
            for (pi, p) in mesh.primitives.iter().enumerate() {
                let (alpha, double_sided) = mesh.props.get(p.material).copied().unwrap_or_default();
                let blend = match mat.options.blend {
                    Blend::Auto if alpha == MeshAlpha::Blend => Blend::Alpha,
                    Blend::Auto => Blend::Opaque,
                    b => b,
                };
                let cull = match mat.options.cull {
                    Cull::Auto if double_sided => None,
                    Cull::Auto | Cull::Back => Some(wgpu::Face::Back),
                    Cull::None => None,
                    Cull::Front => Some(wgpu::Face::Front),
                };
                // GPU copies: packed region, drawn indirectly
                let crowd = o.crowd && mesh.skin.is_some();
                let args = (gpu.is_some() && indirect).then_some(INDIRECT_HEADER + pi as u64 * INDIRECT_STRIDE);
                let mut push = |key, first, count, indirect| if count > 0 {
                    draws.push(Draw { key, indirect, object: oi, material: o.material.0, mesh: o.mesh.0, prim: pi, first, count, dist });
                };
                let k = |stage| (mat.shader, stage, cull, crowd);
                match blend {
                    Blend::Alpha => push(k(Stage::Alpha), first, count, args),
                    Blend::Additive => push(k(Stage::Additive), first, count, args),
                    _ if mat.options.prepass => { push(k(Stage::Prepass), first, count, args); push(k(Stage::Opaque), first, count, args); }
                    _ => push(k(Stage::Late), first, count, args),
                }
                if casts && matches!(blend, Blend::Opaque | Blend::Auto) {
                    push((mat.shader, Stage::Shadow, None, crowd), sfirst, scount, None);
                }
            }
        }

        // sun: orthographic shadow map around the casters (receivers behind them clamp to the far plane)
        if self.sun.shadow_size.clamp(64, 8192) != self.shadow_size { self.resize_shadow_map(&core.device, self.sun.shadow_size); }
        let dir = self.sun.direction.normalize_or_zero();
        let (center, radius) = self.sun.bounds.unwrap_or(if lo.x <= hi.x { ((lo + hi) * 0.5, ((hi - lo) * 0.5).length()) } else { (Vec3::ZERO, 1.0) });
        let radius = radius.max(1e-3);
        let up = if dir.y.abs() > 0.99 { Vec3::new(0.0, 0.0, 1.0) } else { Vec3::Y };
        let eye = center + dir * radius * 2.0;
        let sun_view = Mat4::look_at_rh(eye, center, up);
        let sun_clip = Mat4::orthographic_rh(-radius, radius, -radius, radius, radius * 0.5, radius * 3.5) * sun_view;
        let texel = 2.0 * radius / self.shadow_size as f32;
        queue.write_buffer(&self.sun_buf, 0, bytemuck::bytes_of(&SunU {
            clip_from_world: sun_clip.to_cols_array(),
            direction: dir.to_array(), _pad: 0.0,
            color: self.sun.color, shadows: self.sun.shadows as u32,
            softness: self.sun.softness / self.shadow_size as f32,
            normal_bias: self.sun.normal_bias * texel,
            light_count: self.lights.len() as u32,
            light_texel: 2.0 / self.light_shadows.size.max(1) as f32,
        }));
        queue.write_buffer(&self.shadow_view_buf, 0, bytemuck::bytes_of(&ViewU {
            clip_from_world: sun_clip.to_cols_array(),
            view_from_world: sun_view.to_cols_array(),
            world_position: eye.to_array(),
            _pad: 0.0,
            viewport: [0.0, 0.0, self.shadow_size as f32, self.shadow_size as f32],
        }));
        // grow by half again, never past what one storage binding may hold
        let total = gpu_total as usize + data.len();
        let limit = core.device.limits().max_storage_buffer_binding_size as usize / std::mem::size_of::<TransformU>();
        if total > limit {
            error!("mesh: {total} instances exceed the device limit of {limit}; extra copies are dropped");
            data.truncate(limit.saturating_sub(gpu_total as usize));
            draws.retain(|d| (d.first + d.count) as usize <= limit);
        }
        if total.min(limit) > self.capacity {
            self.capacity = (total + total / 2).clamp(16, limit);
            (self.transforms, self.transforms_group) = Self::make_transforms(&core.device, &self.transforms_layout, &self.crowd_dummy, self.capacity);
            self.transforms_gen += 1;
        }
        if !data.is_empty() {
            queue.write_buffer(&self.transforms, gpu_total as u64 * std::mem::size_of::<TransformU>() as u64, bytemuck::cast_slice(&data));
        }
        // GPU copies: cull and pack into their regions, then fill the indirect instance counts
        if gpu_total > 0 {
            let mut groups: Vec<(wgpu::BindGroup, u32)> = Vec::new();
            for (oi, slot) in self.objects.iter_mut().enumerate() {
                let Some(o) = slot else { continue };
                let (Some(g), Some(mesh)) = (o.gpu.as_mut(), self.meshes.get(o.mesh.0)) else { continue };
                if g.src.count == 0 { continue; }
                let (culled, full) = gpu_region[&oi];
                let prims = mesh.primitives.len();
                if g.args.size() < INDIRECT_HEADER + prims as u64 * INDIRECT_STRIDE { g.args = indirect_buffer(&core.device, prims); g.group = None; }
                // counter 0, then per primitive: index count, instance count (filled on the GPU), first index, base vertex, first instance
                let mut args: Vec<u32> = vec![0; (INDIRECT_HEADER / 4) as usize];
                for p in &mesh.primitives { args.extend([p.index_count, 0, p.first_index, 0, culled]); }
                queue.write_buffer(&g.args, 0, bytemuck::cast_slice(&args));
                queue.write_buffer(&g.params, 0, bytemuck::bytes_of(&CullU {
                    culled, count: g.src.count, full: full.unwrap_or(u32::MAX), object: oi as u32 + 1,
                    center: mesh.center.to_array(), radius: mesh.radius * self.cull_margin,
                    cull: indirect as u32, prims: prims as u32, _p0: 0, _p1: 0,
                    planes: planes.map(|(n, d)| [n.x, n.y, n.z, d]),
                }));
                if g.group.as_ref().map(|(g_gen, _)| *g_gen) != Some(self.transforms_gen) {
                    g.group = Some((self.transforms_gen, core.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Mesh Instances Group"),
                        layout: &self.expand_layout,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: g.src.buffer.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: self.transforms.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 2, resource: g.params.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 3, resource: g.args.as_entire_binding() },
                        ],
                    })));
                }
                if let Some((_, group)) = &g.group { groups.push((group.clone(), g.src.count)); }
            }
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Mesh Instances"), timestamp_writes: None });
            cp.set_pipeline(&self.expand_pipeline);
            for (group, count) in &groups {
                cp.set_bind_group(0, group, &[]);
                let [x, y] = workgroups(*count);
                cp.dispatch_workgroups(x, y, 1);
            }
            cp.set_pipeline(&self.finalize_pipeline);
            for (group, _) in &groups {
                cp.set_bind_group(0, group, &[]);
                cp.dispatch_workgroups(1, 1, 1);
            }
        }

        // crowd meshes read the shared transforms plus their own skin and baked clips
        for d in &draws {
            if !d.key.3 { continue; }
            let m = &self.meshes[d.mesh];
            if self.crowd_groups.get(&d.mesh).is_some_and(|(g_gen, uid, _)| *g_gen == self.transforms_gen && *uid == m.uid) { continue; }
            if let (Some(sk), Some(c)) = (&m.skin, &m.crowd) {
                let group = transforms_group(&core.device, &self.transforms_layout, &self.transforms, [&sk.buf, &c.palettes, &c.clips]);
                self.crowd_groups.insert(d.mesh, (self.transforms_gen, m.uid, group));
            }
        }
        let pick_draws: Vec<Draw> = if picking {
            draws.iter().filter(|d| matches!(d.key.1, Stage::Prepass | Stage::Late | Stage::Alpha | Stage::Additive))
                .map(|d| Draw { key: (d.key.0, Stage::Pick, d.key.2, d.key.3), ..*d }).collect()
        } else { Vec::new() };
        for d in draws.iter().chain(&pick_draws) { self.get_pipeline(&core.device, d.key); }

        // opaque grouped by shader, material, mesh (fewer state changes), then transparent far to near
        let rank = |s: Stage| match s { Stage::Shadow | Stage::Prepass => 0, Stage::Opaque => 1, Stage::Late => 2, _ => 3 };
        let group = |d: &Draw| (d.key.0, d.key.2.map(|f| f as u8), d.material, d.mesh, d.object);
        draws.sort_by(|a, b| rank(a.key.1).cmp(&rank(b.key.1)).then_with(|| {
            if rank(a.key.1) == 3 { b.dist.total_cmp(&a.dist) } else { group(a).cmp(&group(b)) }
        }));

        let clear = |c: [f32; 4]| wgpu::Color { r: c[0] as f64, g: c[1] as f64, b: c[2] as f64, a: c[3] as f64 };
        let depth = |load| Some(wgpu::RenderPassDepthStencilAttachment {
            view: &self.targets.depth,
            depth_ops: Some(wgpu::Operations { load, store: wgpu::StoreOp::Store }),
            stencil_ops: None,
        });
        if self.sun.shadows {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Mesh Shadow Pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_map,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            self.draw(&mut pass, &self.shadow_group, draws.iter().filter(|d| d.key.1 == Stage::Shadow));
        }
        for (layer, _, group) in self.light_shadows.passes.iter().take(self.light_views_used) {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Mesh Light Shadow Pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: layer,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            self.draw(&mut pass, group, draws.iter().filter(|d| d.key.1 == Stage::Shadow));
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Mesh Prepass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.targets.gbuffer_msaa,
                    resolve_target: Some(&self.targets.gbuffer_view),
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(clear([0.0; 4])), store: wgpu::StoreOp::Discard },
                    depth_slice: None,
                })],
                depth_stencil_attachment: depth(wgpu::LoadOp::Clear(1.0)),
                ..Default::default()
            });
            self.draw(&mut pass, &self.view_group, draws.iter().filter(|d| d.key.1 == Stage::Prepass));
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Mesh Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.targets.msaa,
                resolve_target: Some(&self.targets.output_view),
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(clear(self.clear)), store: wgpu::StoreOp::Discard },
                depth_slice: None,
            })],
            depth_stencil_attachment: depth(wgpu::LoadOp::Load),
            ..Default::default()
        });
        self.draw(&mut pass, &self.view_group, draws.iter().filter(|d| !matches!(d.key.1, Stage::Prepass | Stage::Shadow)));
        drop(pass);

        if let (true, Some([x, y])) = (picking, self.pick_request.take()) {
            self.render_pick(encoder, &core.device, &pick_draws, x, y);
            self.pick_state = PickState::Copied;
        }
    }

    // draw instance ids + world positions, copy the one pixel for readback
    fn render_pick(&mut self, encoder: &mut wgpu::CommandEncoder, device: &wgpu::Device, draws: &[Draw], x: u32, y: u32) {
        let size = self.targets.size;
        if self.pick_targets.as_ref().map(|t| t.size) != Some(size) {
            let tex = |format, usage| device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Mesh Pick"),
                size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format, usage, view_formats: &[],
            });
            let rt = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC;
            let (id, pos) = (tex(wgpu::TextureFormat::Rg32Uint, rt), tex(wgpu::TextureFormat::Rgba32Float, rt));
            let depth = tex(DEPTH_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT).create_view(&Default::default());
            self.pick_targets = Some(PickTargets { size, id_view: id.create_view(&Default::default()), pos_view: pos.create_view(&Default::default()), id, pos, depth });
        }
        let Some(t) = &self.pick_targets else { return };
        {
            let attach = |view| Some(wgpu::RenderPassColorAttachment {
                view, resolve_target: None, depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Mesh Pick Pass"),
                color_attachments: &[attach(&t.id_view), attach(&t.pos_view)],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &t.depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            self.draw(&mut pass, &self.view_group, draws.iter());
        }
        for (tex, offset) in [(&t.id, 0), (&t.pos, 256)] {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d { x, y, z: 0 }, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyBufferInfo { buffer: &self.pick_buf, layout: wgpu::TexelCopyBufferLayout { offset, bytes_per_row: Some(256), rows_per_image: Some(1) } },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            );
        }
    }

    fn draw<'a>(&self, pass: &mut wgpu::RenderPass<'_>, view_group: &wgpu::BindGroup, draws: impl Iterator<Item = &'a Draw>) {
        pass.set_bind_group(0, view_group, &[]);
        let (mut pipe, mut mat, mut mesh, mut group3) = (None, None, None, None);
        for d in draws {
            let Some(Some(p)) = self.pipelines.get(&d.key) else { continue };
            let crowd = d.key.3;
            // crowd meshes bind their baked clips with the transforms
            let g3 = if crowd { Some(d.mesh) } else { None };
            if group3 != Some(g3) {
                let Some(g) = (if crowd { self.crowd_groups.get(&d.mesh).map(|(_, _, g)| g) } else { Some(&self.transforms_group) }) else { continue };
                pass.set_bind_group(3, g, &[]);
                group3 = Some(g3);
            }
            if pipe != Some(d.key) { pass.set_pipeline(p); pipe = Some(d.key); }
            if mat != Some(d.material) { pass.set_bind_group(1, &self.materials[d.material].group, &[]); mat = Some(d.material); }
            let m = &self.meshes[d.mesh];
            let o = self.objects.get(d.object).and_then(|o| o.as_ref());
            // compute-posed copy for single animated objects, rest vertices for crowds
            let posed = if crowd { None } else { o.and_then(|o| o.skin.as_ref()).map(|s| &s.out) };
            let vb = (d.mesh, posed.map(|_| d.object));
            if mesh != Some(vb) {
                pass.set_vertex_buffer(0, posed.unwrap_or(&m.vertices).slice(..));
                pass.set_index_buffer(m.indices.slice(..), wgpu::IndexFormat::Uint32);
                mesh = Some(vb);
            }
            let prim = &m.primitives[d.prim];
            let Some(g) = m.groups.get(prim.material) else { continue };
            pass.set_bind_group(2, g, &[]);
            match (d.indirect, o.and_then(|o| o.gpu.as_ref())) {
                (Some(offset), Some(gpu)) => pass.draw_indexed_indirect(&gpu.args, offset),
                _ => pass.draw_indexed(prim.first_index..prim.first_index + prim.index_count, 0, d.first..d.first + d.count),
            }
        }
    }

    /// Use instead of `post.handle_export`; `frame(scene, time, aspect)` animates the scene and returns the camera
    pub fn handle_export(&mut self, core: &Core, post: &mut ComputeShader, base: &mut RenderKit, frame: impl FnOnce(&mut MeshScene, f32, f32) -> MeshView) {
        post.handle_export_dispatch(core, base, |cs, encoder, core| {
            let t = &cs.get_output_texture().texture;
            let (w, h) = (t.width(), t.height());
            let view = frame(self, cs.time_uniform.data.time, w as f32 / h as f32);
            self.render(encoder, core, cs, &view);
            cs.dispatch_at_resolution(encoder, core, w, h);
        });
    }
}

const INDIRECT_HEADER: u64 = 16;
const INDIRECT_STRIDE: u64 = 20;

fn indirect_buffer(device: &wgpu::Device, prims: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Mesh Indirect Args"),
        size: INDIRECT_HEADER + prims.max(1) as u64 * INDIRECT_STRIDE,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn transforms_group(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, transforms: &wgpu::Buffer, crowd: [&wgpu::Buffer; 3]) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Mesh Transforms Group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: transforms.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: crowd[0].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: crowd[1].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: crowd[2].as_entire_binding() },
        ],
    })
}

// every clip at CROWD_FPS (end frame included), then the rest pose as the last clip
fn bake_crowd(device: &wgpu::Device, sk: &MeshSkinning) -> CrowdBake {
    use wgpu::util::DeviceExt;
    let bones = sk.palette.len().max(1);
    let mut palettes: Vec<[f32; 16]> = Vec::new();
    let mut clips: Vec<[f32; 4]> = Vec::new();
    let mut push = |frames: Vec<Vec<Mat4>>, duration: f32, palettes: &mut Vec<[f32; 16]>| {
        clips.push([(palettes.len() / bones) as f32, frames.len() as f32, duration, bones as f32]);
        for f in frames { palettes.extend(f.iter().map(|m| m.to_cols_array())); }
    };
    for (ci, a) in sk.animations.iter().enumerate() {
        let n = ((a.duration * CROWD_FPS).ceil() as usize + 1).max(2);
        push((0..n).map(|f| sk.pose(Some(ci), a.duration * f as f32 / (n - 1) as f32).palette).collect(), a.duration, &mut palettes);
    }
    let rest = sk.rest().palette;
    push(vec![rest.clone(), rest], 0.0, &mut palettes);
    let storage = |label, contents: &[u8]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage: wgpu::BufferUsages::STORAGE });
    info!("Crowd: {} clips baked, {} bone matrices", sk.animations.len(), palettes.len());
    CrowdBake { palettes: storage("Mesh Crowd Palettes", bytemuck::cast_slice(&palettes)), clips: storage("Mesh Crowd Clips", bytemuck::cast_slice(&clips)) }
}

// 64-wide groups for n items, spilling into y past the 65535 per-dimension limit
// (skin.wgsl / instances.wgsl rebuild the index from num_workgroups)
fn workgroups(n: u32) -> [u32; 2] {
    let g = n.div_ceil(64).max(1);
    [g.min(65535), g.div_ceil(65535)]
}

// WGSL without // and /* */ comments, to look for entry points
fn strip_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    while let Some(i) = rest.find(['/']) {
        out.push_str(&rest[..i]);
        rest = &rest[i..];
        if rest.starts_with("//") {
            rest = rest.find('\n').map_or("", |j| &rest[j..]);
        } else if rest.starts_with("/*") {
            rest = rest[2..].find("*/").map_or("", |j| &rest[j + 4..]);
        } else {
            out.push('/');
            rest = &rest[1..];
        }
    }
    out.push_str(rest);
    out
}

struct ViewResources<'a> {
    globals: &'a wgpu::Buffer,
    sun: &'a wgpu::Buffer,
    sampler: &'a wgpu::Sampler,
    lights: &'a wgpu::Buffer,
    light_views: &'a wgpu::Buffer,
}

fn view_group(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, r: &ViewResources, view: &wgpu::Buffer, sun_map: &wgpu::TextureView, light_maps: &wgpu::TextureView) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Mesh View Group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: r.globals.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: view.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: r.sun.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(sun_map) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(r.sampler) },
            wgpu::BindGroupEntry { binding: 5, resource: r.lights.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(light_maps) },
            wgpu::BindGroupEntry { binding: 7, resource: r.light_views.as_entire_binding() },
        ],
    })
}

fn depth_array(device: &wgpu::Device, size: u32, layers: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Mesh Light Shadow Maps"),
        size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: layers.max(1) },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

fn depth_texture(device: &wgpu::Device, size: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Mesh Shadow Map"),
        size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

fn uniform_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(16).div_ceil(16) * 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

// full mip chain, averaged in linear light for colour textures
fn mip_chain(img: &MeshImage, srgb: bool) -> (u32, Vec<u8>) {
    let lut: Vec<f32> = (0..256).map(|v| {
        let c = v as f32 / 255.0;
        if !srgb { c } else if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    }).collect();
    let encode = |c: f32| {
        let c = c.clamp(0.0, 1.0);
        let s = if !srgb { c } else if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
        (s * 255.0 + 0.5) as u8
    };
    let (mut w, mut h) = (img.width.max(1), img.height.max(1));
    let mut level = img.pixels.clone();
    level.resize((w * h * 4) as usize, 255);
    let mut out = level.clone();
    let mut levels = 1;
    while w > 1 || h > 1 {
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        let mut next = vec![0u8; (nw * nh * 4) as usize];
        for y in 0..nh {
            for x in 0..nw {
                for c in 0..4 {
                    let mut s = 0.0;
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let (sx, sy) = ((x * 2 + dx).min(w - 1), (y * 2 + dy).min(h - 1));
                        let v = level[((sy * w + sx) * 4 + c) as usize];
                        s += if c == 3 { v as f32 / 255.0 } else { lut[v as usize] };
                    }
                    s *= 0.25;
                    next[((y * nw + x) * 4 + c) as usize] = if c == 3 { (s * 255.0 + 0.5) as u8 } else { encode(s) };
                }
            }
        }
        out.extend_from_slice(&next);
        (level, w, h) = (next, nw, nh);
        levels += 1;
    }
    (levels, out)
}

fn f32_to_f16(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let m = b & 0x7f_ffff;
    if v.is_nan() { return 0x7e00; }
    if e >= 31 { return sign | 0x7c00; }
    if e <= 0 {
        if e < -10 { return sign; }
        let m = (m | 0x80_0000) >> (1 - e);
        return sign | ((m + 0x1000) >> 13) as u16;
    }
    sign | (((e as u32) << 10) + ((m + 0x1000) >> 13)) as u16
}

// linear float image, box-filtered mips, stored as Rgba16Float (filterable everywhere)
fn upload_hdr(core: &Core, w: u32, h: u32, px: &[f32]) -> wgpu::Texture {
    use wgpu::util::DeviceExt;
    let (mut w0, mut h0) = (w.max(1), h.max(1));
    let mut level: Vec<f32> = px.iter().map(|v| v.max(0.0)).collect();
    level.resize((w0 * h0 * 4) as usize, 0.0);
    let mut out: Vec<u16> = level.iter().map(|&v| f32_to_f16(v)).collect();
    let mut levels = 1;
    while w0 > 1 || h0 > 1 {
        let (nw, nh) = ((w0 / 2).max(1), (h0 / 2).max(1));
        let mut next = vec![0.0f32; (nw * nh * 4) as usize];
        for y in 0..nh {
            for x in 0..nw {
                for c in 0..4 {
                    let s: f32 = [(0, 0), (1, 0), (0, 1), (1, 1)].iter()
                        .map(|&(dx, dy)| level[((((y * 2 + dy).min(h0 - 1)) * w0 + (x * 2 + dx).min(w0 - 1)) * 4 + c) as usize])
                        .sum();
                    next[((y * nw + x) * 4 + c) as usize] = s * 0.25;
                }
            }
        }
        out.extend(next.iter().map(|&v| f32_to_f16(v)));
        (level, w0, h0) = (next, nw, nh);
        levels += 1;
    }
    core.device.create_texture_with_data(
        &core.queue,
        &wgpu::TextureDescriptor {
            label: Some("Mesh HDR Texture"),
            size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
            mip_level_count: levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        bytemuck::cast_slice(&out),
    )
}

fn upload_image(core: &Core, img: &MeshImage, srgb: bool) -> wgpu::Texture {
    use wgpu::util::DeviceExt;
    let (levels, pixels) = mip_chain(img, srgb);
    core.device.create_texture_with_data(
        &core.queue,
        &wgpu::TextureDescriptor {
            label: Some("Mesh Texture"),
            size: wgpu::Extent3d { width: img.width.max(1), height: img.height.max(1), depth_or_array_layers: 1 },
            mip_level_count: levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &pixels,
    )
}

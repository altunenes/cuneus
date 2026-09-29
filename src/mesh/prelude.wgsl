// Mesh material prelude: prepended to every material shader (see MeshScene).
// Mesh helper names are inspired by Bevy's mesh functions.

struct Globals { time: f32, delta_time: f32, frame_count: u32, _pad: u32 };
struct View {
    clip_from_world: mat4x4<f32>,
    view_from_world: mat4x4<f32>,
    world_position: vec3<f32>,
    _pad: f32,
    viewport: vec4<f32>,
};
// anim: clip, time, next clip, next time (crowd objects); anim_b.x: fade to next; ids: object + 1, copy
struct MeshTransform {
    world_from_local: mat4x4<f32>,
    normal_from_local: mat4x4<f32>,
    data: vec4<f32>,
    anim: vec4<f32>,
    anim_b: vec4<f32>,
    ids: vec4<u32>,
};
// joints (palette indices), weights, morph info per vertex
struct SkinVertex { joints: vec4<u32>, weights: vec4<f32>, morph: vec4<u32> };
struct GltfMaterial {
    base_color: vec4<f32>,
    emissive: vec4<f32>,
    metallic: f32,
    roughness: f32,
    normal_scale: f32,
    occlusion_strength: f32,
    alpha_cutoff: f32,
    // 0 opaque, 1 mask, 2 blend
    alpha_mode: u32,
    // GLTF_* bits
    flags: u32,
    _pad: u32,
    // texture transform rows (xy scale/rotation, z offset); uv_row0.w = uv set
    uv_row0: vec4<f32>,
    uv_row1: vec4<f32>,
};
const GLTF_BASE_COLOR_TEXTURE: u32 = 1u;
const GLTF_NORMAL_TEXTURE: u32 = 2u;
const GLTF_METALLIC_ROUGHNESS_TEXTURE: u32 = 4u;
const GLTF_EMISSIVE_TEXTURE: u32 = 8u;
const GLTF_OCCLUSION_TEXTURE: u32 = 16u;
const GLTF_DOUBLE_SIDED: u32 = 32u;
const PI: f32 = 3.14159265359;
struct Sun {
    // shadow map projection
    clip_from_world: mat4x4<f32>,
    // towards the light
    direction: vec3<f32>,
    _pad: f32,
    // linear colour x intensity
    color: vec3<f32>,
    shadows: u32,
    // penumbra radius, shadow map uv
    softness: f32,
    // world units
    normal_bias: f32,
    light_count: u32,
    // 2 / light shadow map size
    light_texel: f32,
};
// point (kind 0) or spot (kind 1) light; shadow = first shadow view (6 for a point light) or -1
struct Light {
    position: vec3<f32>,
    range: f32,
    color: vec3<f32>,
    kind: u32,
    direction: vec3<f32>,
    cos_outer: f32,
    cos_inner: f32,
    shadow: i32,
    _p0: f32,
    _p1: f32,
};
struct LightSample { direction: vec3<f32>, radiance: vec3<f32> };

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<uniform> view: View;
@group(0) @binding(2) var<uniform> sun: Sun;
@group(0) @binding(3) var shadow_map: texture_depth_2d;
@group(0) @binding(4) var shadow_sampler: sampler_comparison;
@group(0) @binding(5) var<storage, read> lights: array<Light>;
@group(0) @binding(6) var light_shadow_maps: texture_depth_2d_array;
@group(0) @binding(7) var<storage, read> light_views: array<mat4x4<f32>>;
@group(2) @binding(0) var<uniform> gltf_material: GltfMaterial;
@group(2) @binding(1) var base_color_texture: texture_2d<f32>;
@group(2) @binding(2) var base_color_sampler: sampler;
@group(2) @binding(3) var normal_map_texture: texture_2d<f32>;
@group(2) @binding(4) var metallic_roughness_texture: texture_2d<f32>;
@group(2) @binding(5) var emissive_texture: texture_2d<f32>;
@group(2) @binding(6) var occlusion_texture: texture_2d<f32>;
// your own textures per material (set_texture / set_image / set_image_file), white when unset
@group(1) @binding(1) var material_texture0: texture_2d<f32>;
@group(1) @binding(2) var material_texture1: texture_2d<f32>;
@group(1) @binding(3) var material_texture2: texture_2d<f32>;
@group(1) @binding(4) var material_texture3: texture_2d<f32>;
@group(1) @binding(5) var material_sampler: sampler;
// one entry per instance, indexed by instance_index
@group(3) @binding(0) var<storage, read> mesh_transforms: array<MeshTransform>;
// crowd skinning (MeshScene::set_crowd): baked bone matrices, clip table (first frame, frames,
// duration, bones), the last clip is the rest pose
@group(3) @binding(1) var<storage, read> crowd_skin: array<SkinVertex>;
@group(3) @binding(2) var<storage, read> crowd_palettes: array<mat4x4<f32>>;
@group(3) @binding(3) var<storage, read> crowd_clips: array<vec4<f32>>;

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
    @location(4) color: vec4<f32>,
    @location(5) uv_b: vec2<f32>,
};

struct VertexOutput {
    // invariant: the depth prepass and the main pass must produce identical depth
    @builtin(position) @invariant position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) world_tangent: vec4<f32>,
    @location(4) color: vec4<f32>,
    @location(5) @interpolate(flat) instance_index: u32,
    @location(6) uv_b: vec2<f32>,
};

fn get_world_from_local(instance_index: u32) -> mat4x4<f32> { return mesh_transforms[instance_index].world_from_local; }
fn get_instance_data(instance_index: u32) -> vec4<f32> { return mesh_transforms[instance_index].data; }
fn mesh_position_local_to_world(world_from_local: mat4x4<f32>, p: vec4<f32>) -> vec4<f32> { return world_from_local * p; }
fn mesh_normal_local_to_world(n: vec3<f32>, instance_index: u32) -> vec3<f32> {
    let m = mesh_transforms[instance_index].normal_from_local;
    return normalize(mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz) * n);
}
fn mesh_tangent_local_to_world(world_from_local: mat4x4<f32>, t: vec4<f32>, instance_index: u32) -> vec4<f32> {
    if dot(t.xyz, t.xyz) == 0.0 { return vec4<f32>(0.0); }
    let m = mat3x3<f32>(world_from_local[0].xyz, world_from_local[1].xyz, world_from_local[2].xyz);
    return vec4<f32>(normalize(m * t.xyz), t.w * select(1.0, -1.0, determinant(m) < 0.0));
}
fn position_world_to_clip(p: vec3<f32>) -> vec4<f32> { return view.clip_from_world * vec4<f32>(p, 1.0); }

// direction -> uv of an equirectangular (lat-long) image, e.g. an HDRI in a material texture
fn equirect_uv(d: vec3<f32>) -> vec2<f32> {
    let n = normalize(d);
    return vec2<f32>(atan2(n.z, n.x) / (2.0 * PI) + 0.5, acos(clamp(n.y, -1.0, 1.0)) / PI);
}

// the uv the model's textures read: chosen set + KHR_texture_transform
fn gltf_uv(mesh: VertexOutput) -> vec2<f32> {
    let uv = select(mesh.uv, mesh.uv_b, gltf_material.uv_row0.w > 0.5);
    return vec2<f32>(dot(gltf_material.uv_row0.xy, uv) + gltf_material.uv_row0.z, dot(gltf_material.uv_row1.xy, uv) + gltf_material.uv_row1.z);
}

// glTF inputs (missing textures read as neutral), linear
fn gltf_base_color(mesh: VertexOutput) -> vec4<f32> {
    return gltf_material.base_color * mesh.color * textureSample(base_color_texture, base_color_sampler, gltf_uv(mesh));
}
fn gltf_normal(mesh: VertexOutput, is_front: bool) -> vec3<f32> {
    let s = textureSample(normal_map_texture, base_color_sampler, gltf_uv(mesh)).xyz * 2.0 - 1.0;
    var n = normalize(mesh.world_normal);
    if !is_front { n = -n; }
    let t = mesh.world_tangent;
    if dot(t.xyz, t.xyz) > 0.0 {
        let tt = normalize(t.xyz - n * dot(n, t.xyz));
        let b = cross(n, tt) * t.w;
        n = normalize(tt * s.x * gltf_material.normal_scale + b * s.y * gltf_material.normal_scale + n * s.z);
    }
    return n;
}
// x metallic, y roughness
fn gltf_metallic_roughness(mesh: VertexOutput) -> vec2<f32> {
    let s = textureSample(metallic_roughness_texture, base_color_sampler, gltf_uv(mesh));
    return vec2<f32>(gltf_material.metallic * s.b, gltf_material.roughness * s.g);
}
fn gltf_emissive(mesh: VertexOutput) -> vec3<f32> {
    return gltf_material.emissive.rgb * textureSample(emissive_texture, base_color_sampler, gltf_uv(mesh)).rgb;
}
fn gltf_occlusion(mesh: VertexOutput) -> f32 {
    let o = textureSample(occlusion_texture, base_color_sampler, gltf_uv(mesh)).r;
    return 1.0 + gltf_material.occlusion_strength * (o - 1.0);
}

// Cook-Torrance GGX for one light; l points to the light, radiance = colour x intensity
fn pbr_direct(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, radiance: vec3<f32>, base: vec3<f32>, metallic: f32, roughness: f32) -> vec3<f32> {
    let h = normalize(l + v);
    let nl = max(dot(n, l), 0.0);
    let nv = max(dot(n, v), 1e-4);
    let nh = max(dot(n, h), 0.0);
    let a = max(roughness * roughness, 2e-3);
    let a2 = a * a;
    let dd = nh * nh * (a2 - 1.0) + 1.0;
    let d = a2 / (PI * dd * dd);
    let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    let g = nv / (nv * (1.0 - k) + k) * nl / (nl * (1.0 - k) + k);
    let f0 = mix(vec3<f32>(0.04), base, metallic);
    let f = f0 + (1.0 - f0) * pow(1.0 - max(dot(v, h), 0.0), 5.0);
    let spec = d * g * f / max(4.0 * nv * nl, 1e-4);
    let kd = (1.0 - f) * (1.0 - metallic);
    return (kd * base / PI + spec) * radiance * nl;
}
// sky/ground hemisphere ambient, diffuse + rough specular
fn pbr_ambient(n: vec3<f32>, v: vec3<f32>, base: vec3<f32>, metallic: f32, roughness: f32, sky: vec3<f32>, ground: vec3<f32>) -> vec3<f32> {
    let f0 = mix(vec3<f32>(0.04), base, metallic);
    let nv = max(dot(n, v), 0.0);
    let f = f0 + (max(vec3<f32>(1.0 - roughness), f0) - f0) * pow(1.0 - nv, 5.0);
    let r = reflect(-v, n);
    let diffuse = mix(ground, sky, n.y * 0.5 + 0.5) * base * (1.0 - f) * (1.0 - metallic);
    let spec = mix(mix(ground, sky, r.y * 0.5 + 0.5), mix(ground, sky, 0.5), roughness) * f;
    return diffuse + spec;
}

// 1 lit, 0 in the sun's shadow; world_normal is the surface (not normal-mapped) normal
fn directional_shadow(world_position: vec3<f32>, world_normal: vec3<f32>) -> f32 {
    if sun.shadows == 0u { return 1.0; }
    let n = normalize(world_normal);
    let p = world_position + n * sun.normal_bias * (1.5 - abs(dot(n, sun.direction)));
    let c = sun.clip_from_world * vec4<f32>(p, 1.0);
    let uv = c.xy * vec2<f32>(0.5, -0.5) + 0.5;
    if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) || c.z < 0.0 { return 1.0; }
    let z = min(c.z, 1.0);
    // 16 taps on a golden-angle spiral, turned per surface point to hide banding
    let rot = fract(sin(dot(world_position, vec3<f32>(12.9898, 78.233, 37.719))) * 43758.545) * 6.2832;
    var s = 0.0;
    for (var i = 0; i < 16; i++) {
        let a = f32(i) * 2.39996 + rot;
        let o = vec2<f32>(cos(a), sin(a)) * sqrt((f32(i) + 0.5) / 16.0) * sun.softness;
        s += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + o, z);
    }
    return s / 16.0;
}

fn light_count() -> u32 { return sun.light_count; }

// 1 lit, 0 shadowed, for light i (1 when it has no shadow)
fn light_shadow(i: u32, world_position: vec3<f32>, world_normal: vec3<f32>) -> f32 {
    let l = lights[i];
    if l.shadow < 0 { return 1.0; }
    var v = u32(l.shadow);
    let d = world_position - l.position;
    if l.kind == 0u {
        // cube face by major axis, order +x -x +y -y +z -z
        let a = abs(d);
        if a.x >= a.y && a.x >= a.z { v += select(1u, 0u, d.x > 0.0); }
        else if a.y >= a.z { v += select(3u, 2u, d.y > 0.0); }
        else { v += select(5u, 4u, d.z > 0.0); }
    }
    let p = world_position + normalize(world_normal) * length(d) * sun.light_texel * 1.5;
    let c = light_views[v] * vec4<f32>(p, 1.0);
    if c.w <= 0.0 { return 1.0; }
    let ndc = c.xyz / c.w;
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + 0.5;
    if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) || ndc.z > 1.0 { return 1.0; }
    let o = sun.light_texel * 0.5;
    var s = 0.0;
    for (var k = 0; k < 4; k++) {
        let off = vec2<f32>(select(-0.5, 0.5, (k & 1) == 1), select(-0.5, 0.5, k > 1)) * o;
        s += textureSampleCompareLevel(light_shadow_maps, shadow_sampler, uv + off, i32(v), ndc.z);
    }
    return s * 0.25;
}

// direction to light i and the light arriving there (colour x falloff x cone x shadow)
fn light_sample(i: u32, world_position: vec3<f32>, world_normal: vec3<f32>) -> LightSample {
    let l = lights[i];
    let to = l.position - world_position;
    let d = length(to);
    let dir = to / max(d, 1e-4);
    // inverse square, faded smoothly to zero at range
    let x = clamp(d / max(l.range, 1e-4), 0.0, 1.0);
    let win = 1.0 - x * x * x * x;
    var att = win * win / max(d * d, 1e-4);
    if l.kind == 1u { att *= smoothstep(l.cos_outer, l.cos_inner, dot(-dir, l.direction)); }
    att *= light_shadow(i, world_position, world_normal);
    return LightSample(dir, l.color * att);
}

// crowd: bone j of `clip` at time t, blended between baked frames (clip < 0 = rest pose)
fn crowd_bone(clip: f32, t: f32, j: u32) -> mat4x4<f32> {
    let n = arrayLength(&crowd_clips) - 1u;
    let ci = select(n, u32(clip), clip >= 0.0 && u32(clip) < n);
    let c = crowd_clips[ci];
    var tt = t;
    if c.z > 0.0 && (t < 0.0 || t > c.z) { tt = t - floor(t / c.z) * c.z; }
    let frames = max(u32(c.y), 1u);
    let f = clamp(tt / max(c.z, 1e-6), 0.0, 1.0) * f32(frames - 1u);
    let f0 = u32(f);
    let f1 = min(f0 + 1u, frames - 1u);
    let base = u32(c.x);
    let bones = u32(c.w);
    return crowd_palettes[(base + f0) * bones + j] * (1.0 - fract(f)) + crowd_palettes[(base + f1) * bones + j] * fract(f);
}
fn crowd_matrix(s: SkinVertex, clip: f32, t: f32) -> mat4x4<f32> {
    return crowd_bone(clip, t, s.joints.x) * s.weights.x + crowd_bone(clip, t, s.joints.y) * s.weights.y
         + crowd_bone(clip, t, s.joints.z) * s.weights.z + crowd_bone(clip, t, s.joints.w) * s.weights.w;
}
// pose a rest vertex with its copy's own clip (and crossfade), before the material's vertex code
fn crowd_pose(vertex: Vertex, vertex_index: u32) -> Vertex {
    let tr = mesh_transforms[vertex.instance_index];
    let s = crowd_skin[vertex_index];
    var m = crowd_matrix(s, tr.anim.x, tr.anim.y);
    if tr.anim_b.x > 0.0 { m = m * (1.0 - tr.anim_b.x) + crowd_matrix(s, tr.anim.z, tr.anim.w) * tr.anim_b.x; }
    let r = mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz);
    var out = vertex;
    out.position = (m * vec4<f32>(vertex.position, 1.0)).xyz;
    out.normal = r * vertex.normal;
    out.tangent = vec4<f32>(r * vertex.tangent.xyz, vertex.tangent.w);
    return out;
}

// what the engine runs when a material has no `fn vertex`
fn mesh_vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    let world_from_local = get_world_from_local(vertex.instance_index);
    out.world_position = mesh_position_local_to_world(world_from_local, vec4<f32>(vertex.position, 1.0));
    out.position = position_world_to_clip(out.world_position.xyz);
    out.world_normal = mesh_normal_local_to_world(vertex.normal, vertex.instance_index);
    out.uv = vertex.uv;
    out.world_tangent = mesh_tangent_local_to_world(world_from_local, vertex.tangent, vertex.instance_index);
    out.color = vertex.color;
    out.instance_index = vertex.instance_index;
    out.uv_b = vertex.uv_b;
    return out;
}

// depth prepass: alpha mask + world normal and view depth for post passes
@fragment
fn cuneus_prepass(mesh: VertexOutput, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    let a = gltf_base_color(mesh).a;
    let n = gltf_normal(mesh, is_front);
    if gltf_material.alpha_mode == 1u && a < gltf_material.alpha_cutoff { discard; }
    return vec4<f32>(n, -(view.view_from_world * mesh.world_position).z);
}

// picking: object + 1 and copy here (0 = nothing), and the world point
struct PickOutput { @location(0) id: vec2<u32>, @location(1) position: vec4<f32> };
@fragment
fn cuneus_pick(mesh: VertexOutput) -> PickOutput {
    let a = gltf_base_color(mesh).a;
    if gltf_material.alpha_mode == 1u && a < gltf_material.alpha_cutoff { discard; }
    return PickOutput(mesh_transforms[mesh.instance_index].ids.xy, vec4<f32>(mesh.world_position.xyz, 1.0));
}

// shadow map: depth only, alpha mask cut
@fragment
fn cuneus_shadow(mesh: VertexOutput) {
    let a = gltf_base_color(mesh).a;
    if gltf_material.alpha_mode == 1u && a < gltf_material.alpha_cutoff { discard; }
}

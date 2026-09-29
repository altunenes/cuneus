//! glTF loading into CPU mesh data, plus procedural shapes

use super::anim::{AnimChannel, AnimInterpolation, AnimProperty, MeshAnimation, MeshNode, MeshSkinning, MorphSlot};
use crate::math::{Mat4, Quat, Vec3};
use std::path::Path;

/// Vertex layout shared by the loader and the pipeline (80 bytes = 5 vec4, the skinning shader relies on it)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    /// xyz tangent, w handedness; zero when the mesh has no uvs
    pub tangent: [f32; 4],
    pub color: [f32; 4],
    /// Second uv set (TEXCOORD_1)
    pub uv_b: [f32; 2],
    pub _pad: [f32; 2],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MeshWrap {
    #[default]
    Repeat,
    Clamp,
    Mirror,
}

/// Texture filtering and wrapping from the file (pixel art uses nearest)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MeshSampler {
    pub wrap_u: MeshWrap,
    pub wrap_v: MeshWrap,
    pub mag_nearest: bool,
    pub min_nearest: bool,
    pub mip_nearest: bool,
}

/// glTF alphaMode
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MeshAlpha {
    #[default]
    Opaque,
    Mask,
    Blend,
}

#[derive(Clone, Debug)]
pub struct MeshMaterial {
    pub base_color: [f32; 4],
    /// Emissive factor x KHR_materials_emissive_strength
    pub emissive: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    pub normal_scale: f32,
    pub occlusion_strength: f32,
    pub alpha: MeshAlpha,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
    /// Indices into `MeshData::images`
    pub base_color_image: Option<usize>,
    pub metallic_roughness_image: Option<usize>,
    pub normal_image: Option<usize>,
    pub emissive_image: Option<usize>,
    pub occlusion_image: Option<usize>,
    /// Which uv set the textures read (0 or 1) and its KHR_texture_transform, as two rows of a
    /// 2x3 matrix; taken from the base colour texture (else the first texture) and applied to all
    pub uv_set: u32,
    pub uv_transform: [[f32; 3]; 2],
    pub sampler: MeshSampler,
}

impl Default for MeshMaterial {
    fn default() -> Self {
        Self {
            base_color: [0.8, 0.8, 0.8, 1.0], emissive: [0.0; 3], metallic: 0.0, roughness: 0.6,
            normal_scale: 1.0, occlusion_strength: 1.0, alpha: MeshAlpha::Opaque, alpha_cutoff: 0.5, double_sided: false,
            base_color_image: None, metallic_roughness_image: None, normal_image: None, emissive_image: None, occlusion_image: None,
            uv_set: 0, uv_transform: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], sampler: MeshSampler::default(),
        }
    }
}

/// A run of indices drawn with one material
#[derive(Clone, Debug)]
pub struct MeshPrimitive {
    pub first_index: u32,
    pub index_count: u32,
    pub material: usize,
}

/// RGBA8 image
#[derive(Clone)]
pub struct MeshImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

/// A whole glTF scene flattened into one vertex and index buffer (node transforms applied)
pub struct MeshData {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
    pub primitives: Vec<MeshPrimitive>,
    pub materials: Vec<MeshMaterial>,
    pub images: Vec<MeshImage>,
    /// Bounding box centre and half diagonal (rest pose)
    pub center: Vec3,
    pub radius: f32,
    /// Node hierarchy, skins and animations; `None` for static models (baked to world space)
    pub skinning: Option<MeshSkinning>,
}

#[derive(Clone, Copy)]
enum Joints { Rigid(u32), Skinned(u32, u32) }

// weights base + count of the node's blend shapes
type MorphRef = Option<(u32, u32)>;
// one blend shape's position and normal deltas
type MorphTarget = (Vec<[f32; 3]>, Vec<[f32; 3]>);

impl MeshData {
    /// Load a .glb or .gltf file
    pub fn from_gltf<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let (doc, buffers, images) = gltf::import(path.as_ref())?;

        let mut data = MeshData {
            vertices: Vec::new(), indices: Vec::new(), primitives: Vec::new(),
            materials: Vec::new(), images: images.iter().map(to_rgba8).collect(), center: Vec3::ZERO, radius: 1.0, skinning: None,
        };

        let src = |t: gltf::texture::Texture| t.source().index();
        for m in doc.materials() {
            let pbr = m.pbr_metallic_roughness();
            let (e, k) = (m.emissive_factor(), m.emissive_strength().unwrap_or(1.0));
            // uv set, transform and sampler come from the main texture
            let info = pbr.base_color_texture().or_else(|| pbr.metallic_roughness_texture()).or_else(|| m.emissive_texture());
            let (tex, mut uv_set) = match &info {
                Some(i) => (Some(i.texture()), i.tex_coord()),
                None => match (m.normal_texture(), m.occlusion_texture()) {
                    (Some(n), _) => (Some(n.texture()), n.tex_coord()),
                    (None, Some(o)) => (Some(o.texture()), o.tex_coord()),
                    _ => (None, 0),
                },
            };
            let mut uv_transform = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
            if let Some(tt) = info.as_ref().and_then(|i| i.texture_transform()) {
                let ([ox, oy], r, [sx, sy]) = (tt.offset(), tt.rotation(), tt.scale());
                let (sn, cs) = r.sin_cos();
                uv_transform = [[cs * sx, sn * sy, ox], [-sn * sx, cs * sy, oy]];
                if let Some(t) = tt.tex_coord() { uv_set = t; }
            }
            let sampler = tex.map(|t| sampler_of(&t.sampler())).unwrap_or_default();
            data.materials.push(MeshMaterial {
                base_color: pbr.base_color_factor(),
                emissive: [e[0] * k, e[1] * k, e[2] * k],
                metallic: pbr.metallic_factor(),
                roughness: pbr.roughness_factor(),
                normal_scale: m.normal_texture().map(|t| t.scale()).unwrap_or(1.0),
                occlusion_strength: m.occlusion_texture().map(|t| t.strength()).unwrap_or(1.0),
                alpha: match m.alpha_mode() {
                    gltf::material::AlphaMode::Mask => MeshAlpha::Mask,
                    gltf::material::AlphaMode::Blend => MeshAlpha::Blend,
                    gltf::material::AlphaMode::Opaque => MeshAlpha::Opaque,
                },
                alpha_cutoff: m.alpha_cutoff().unwrap_or(0.5),
                double_sided: m.double_sided(),
                base_color_image: pbr.base_color_texture().map(|t| src(t.texture())),
                metallic_roughness_image: pbr.metallic_roughness_texture().map(|t| src(t.texture())),
                normal_image: m.normal_texture().map(|t| src(t.texture())),
                emissive_image: m.emissive_texture().map(|t| src(t.texture())),
                occlusion_image: m.occlusion_texture().map(|t| src(t.texture())),
                uv_set: uv_set.min(1),
                uv_transform,
                sampler,
            });
        }
        // primitives without a material use this one
        let default_material = data.materials.len();
        data.materials.push(MeshMaterial::default());

        let scene = doc.default_scene().or_else(|| doc.scenes().next());
        let roots: Vec<gltf::Node> = match scene {
            Some(s) => s.nodes().collect(),
            None => doc.nodes().collect(),
        };
        let morphs = doc.meshes().any(|m| m.primitives().any(|p| p.morph_targets().len() > 0));
        if doc.skins().len() > 0 || doc.animations().len() > 0 || morphs {
            data.load_rigged(&doc, &roots, &buffers, default_material);
        } else {
            for node in roots {
                data.add_node(&node, Mat4::IDENTITY, &buffers, default_material);
            }
        }
        if data.vertices.is_empty() {
            anyhow::bail!("no triangle meshes found");
        }
        data.fit_bounds();
        Ok(data)
    }

    fn add_node(&mut self, node: &gltf::Node, parent: Mat4, buffers: &[gltf::buffer::Data], default_material: usize) {
        let world = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
        if let Some(mesh) = node.mesh() {
            for prim in mesh.primitives() {
                self.push_primitive(&prim, world, buffers, default_material, None);
            }
        }
        for child in node.children() {
            self.add_node(&child, world, buffers, default_material);
        }
    }

    // animated / skinned: vertices stay local, every vertex gets palette joints + weights
    fn load_rigged(&mut self, doc: &gltf::Document, roots: &[gltf::Node], buffers: &[gltf::buffer::Data], default_material: usize) {
        let count = doc.nodes().len();
        let mut parent = vec![None; count];
        for n in doc.nodes() {
            for c in n.children() { parent[c.index()] = Some(n.index()); }
        }
        let nodes = doc.nodes().map(|n| {
            let (t, r, sc) = n.transform().decomposed();
            MeshNode { name: n.name().map(String::from), parent: parent[n.index()], translation: Vec3::from(t), rotation: Quat::from_array(r), scale: Vec3::from(sc) }
        }).collect();
        let mut order = Vec::with_capacity(count);
        let mut stack: Vec<usize> = (0..count).filter(|&i| parent[i].is_none()).collect();
        let mut seen = vec![false; count];
        let all: Vec<gltf::Node> = doc.nodes().collect();
        while let Some(i) = stack.pop() {
            if std::mem::replace(&mut seen[i], true) { continue; }
            order.push(i);
            stack.extend(all[i].children().map(|c| c.index()));
        }
        self.skinning = Some(MeshSkinning {
            joints: Vec::new(), weights: Vec::new(), morph: Vec::new(), morph_deltas: Vec::new(), morph_slots: Vec::new(),
            nodes, order, palette: Vec::new(), animations: Vec::new(),
        });

        let mut skin_base = std::collections::HashMap::new();
        let mut node_slot = std::collections::HashMap::new();
        let mut stack: Vec<gltf::Node> = roots.to_vec();
        while let Some(node) = stack.pop() {
            stack.extend(node.children());
            let Some(mesh) = node.mesh() else { continue };
            let sk = self.skinning.as_mut().expect("rigged");
            let joints = match node.skin().filter(|s| s.joints().len() > 0) {
                Some(skin) => {
                    let (base, n) = *skin_base.entry(skin.index()).or_insert_with(|| {
                        let base = sk.palette.len() as u32;
                        let ibs: Vec<[[f32; 4]; 4]> = skin.reader(|b| buffers.get(b.index()).map(|d| &d.0[..]))
                            .read_inverse_bind_matrices().map(|i| i.collect()).unwrap_or_default();
                        for (j, jn) in skin.joints().enumerate() {
                            sk.palette.push((jn.index(), ibs.get(j).map(Mat4::from_cols_array_2d).unwrap_or(Mat4::IDENTITY)));
                        }
                        (base, skin.joints().len() as u32)
                    });
                    Joints::Skinned(base, n)
                }
                None => Joints::Rigid(*node_slot.entry(node.index()).or_insert_with(|| {
                    sk.palette.push((node.index(), Mat4::IDENTITY));
                    sk.palette.len() as u32 - 1
                })),
            };
            // blend shapes: one weight slot per mesh node (node weights > mesh weights > zeros)
            let targets = mesh.primitives().map(|p| p.morph_targets().len()).max().unwrap_or(0) as u32;
            let morph = (targets > 0).then(|| {
                let base = sk.morph_slots.iter().map(|m| m.base + m.count).max().unwrap_or(0);
                let names = mesh.extras().as_ref()
                    .and_then(|e| serde_json::from_str::<serde_json::Value>(e.get()).ok())
                    .and_then(|v| v.get("targetNames").cloned())
                    .and_then(|v| serde_json::from_value::<Vec<String>>(v).ok())
                    .unwrap_or_default();
                let default = node.weights().or(mesh.weights()).map(|w| w.to_vec()).unwrap_or_else(|| vec![0.0; targets as usize]);
                sk.morph_slots.push(MorphSlot { node: node.index(), base, count: targets, names, default });
                (base, targets)
            });
            for prim in mesh.primitives() {
                self.push_primitive(&prim, Mat4::IDENTITY, buffers, default_material, Some((joints, morph)));
            }
        }

        use gltf::animation::util::ReadOutputs;
        let mut animations = Vec::new();
        for (ai, a) in doc.animations().enumerate() {
            let mut channels = Vec::new();
            let mut duration: f32 = 0.0;
            for ch in a.channels() {
                let r = ch.reader(|b| buffers.get(b.index()).map(|d| &d.0[..]));
                let (Some(times), Some(out)) = (r.read_inputs(), r.read_outputs()) else { continue };
                let times: Vec<f32> = times.collect();
                let (property, values): (AnimProperty, Vec<f32>) = match out {
                    ReadOutputs::Translations(v) => (AnimProperty::Translation, v.flatten().collect()),
                    ReadOutputs::Rotations(v) => (AnimProperty::Rotation, v.into_f32().flatten().collect()),
                    ReadOutputs::Scales(v) => (AnimProperty::Scale, v.flatten().collect()),
                    ReadOutputs::MorphTargetWeights(v) => (AnimProperty::MorphWeights, v.into_f32().collect()),
                };
                let interpolation = match ch.sampler().interpolation() {
                    gltf::animation::Interpolation::Linear => AnimInterpolation::Linear,
                    gltf::animation::Interpolation::Step => AnimInterpolation::Step,
                    gltf::animation::Interpolation::CubicSpline => AnimInterpolation::CubicSpline,
                };
                let keys = if interpolation == AnimInterpolation::CubicSpline { times.len() * 3 } else { times.len() };
                let width = match property {
                    AnimProperty::Translation | AnimProperty::Scale => 3,
                    AnimProperty::Rotation => 4,
                    AnimProperty::MorphWeights => values.len() / keys.max(1),
                };
                if times.is_empty() || width == 0 || values.len() < keys * width { continue; }
                duration = duration.max(*times.last().unwrap_or(&0.0));
                channels.push(AnimChannel { node: ch.target().node().index(), property, interpolation, times, values, width });
            }
            animations.push(MeshAnimation { name: a.name().map(String::from).unwrap_or_else(|| format!("animation {ai}")), duration, channels });
        }
        if let Some(sk) = self.skinning.as_mut() { sk.animations = animations; }
    }

    fn push_primitive(&mut self, prim: &gltf::Primitive, world: Mat4, buffers: &[gltf::buffer::Data], default_material: usize, rig: Option<(Joints, MorphRef)>) {
        if prim.mode() != gltf::mesh::Mode::Triangles {
            return;
        }
        let normal_m = world.normal_matrix();
        let mirrored = world.det3() < 0.0;
        let reader = prim.reader(|b| buffers.get(b.index()).map(|d| &d.0[..]));
        let Some(positions) = reader.read_positions() else { return };
        let positions: Vec<[f32; 3]> = positions.collect();
        let normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(|n| n.collect());
        let uvs: Option<Vec<[f32; 2]>> = reader.read_tex_coords(0).map(|t| t.into_f32().collect());
        let uvs_b: Option<Vec<[f32; 2]>> = reader.read_tex_coords(1).map(|t| t.into_f32().collect());
        let tangents: Option<Vec<[f32; 4]>> = reader.read_tangents().map(|t| t.collect());
        let colors: Option<Vec<[f32; 4]>> = reader.read_colors(0).map(|c| c.into_rgba_f32().collect());
        let n = positions.len() as u32;
        let raw: Vec<u32> = match reader.read_indices() {
            Some(i) => i.into_u32().collect(),
            None => (0..n).collect(),
        };
        // drop broken triangles; keep winding front-facing under mirrored transforms
        let mut local: Vec<u32> = raw.chunks_exact(3).filter(|t| t.iter().all(|&i| i < n)).flatten().copied().collect();
        if mirrored {
            for t in local.chunks_exact_mut(3) { t.swap(1, 2); }
        }

        let base = self.vertices.len();
        let sign = if mirrored { -1.0 } else { 1.0 };
        for (k, p) in positions.iter().enumerate() {
            let normal = normals.as_ref().map(|v| normal_m.transform_vector3(Vec3::from(v[k])).normalize_or_zero()).unwrap_or(Vec3::ZERO);
            let tangent = tangents.as_ref().map(|v| {
                let t = world.transform_vector3(Vec3::new(v[k][0], v[k][1], v[k][2])).normalize_or_zero();
                [t.x, t.y, t.z, v[k][3] * sign]
            }).unwrap_or([0.0; 4]);
            self.vertices.push(MeshVertex {
                position: world.transform_point3(Vec3::from(*p)).to_array(),
                normal: normal.to_array(),
                uv: uvs.as_ref().map(|u| u[k]).unwrap_or([0.0, 0.0]),
                tangent,
                color: colors.as_ref().map(|c| c[k]).unwrap_or([1.0; 4]),
                uv_b: uvs_b.as_ref().map(|u| u[k]).unwrap_or([0.0, 0.0]),
                _pad: [0.0; 2],
            });
        }
        let verts = &mut self.vertices[base..];
        if normals.is_none() { smooth_normals(verts, &local); }
        if tangents.is_none() && uvs.is_some() { generate_tangents(verts, &local); }

        if let (Some((j, morph)), Some(sk)) = (rig, self.skinning.as_mut()) {
            // blend shape deltas, interleaved per vertex: [target0 pos, target0 normal, target1 pos, ...]
            let targets: Vec<MorphTarget> = match morph {
                Some(_) => reader.read_morph_targets()
                    .map(|(p, nm, _)| (p.map(|i| i.collect()).unwrap_or_default(), nm.map(|i| i.collect()).unwrap_or_default()))
                    .collect(),
                None => Vec::new(),
            };
            let (wbase, wcount) = morph.unwrap_or((0, 0));
            let count = (targets.len() as u32).min(wcount);
            for k in 0..n as usize {
                if count == 0 { sk.morph.push([0; 4]); continue; }
                sk.morph.push([(sk.morph_deltas.len() / 2) as u32, count, wbase, 0]);
                for (tp, tn) in targets.iter().take(count as usize) {
                    let (p, nm) = (tp.get(k).copied().unwrap_or([0.0; 3]), tn.get(k).copied().unwrap_or([0.0; 3]));
                    sk.morph_deltas.push([p[0], p[1], p[2], 0.0]);
                    sk.morph_deltas.push([nm[0], nm[1], nm[2], 0.0]);
                }
            }
            match j {
                Joints::Rigid(slot) => for _ in 0..n {
                    sk.joints.push([slot, 0, 0, 0]);
                    sk.weights.push([1.0, 0.0, 0.0, 0.0]);
                },
                Joints::Skinned(first, count) => {
                    let js: Vec<[u16; 4]> = reader.read_joints(0).map(|j| j.into_u16().collect()).unwrap_or_default();
                    let ws: Vec<[f32; 4]> = reader.read_weights(0).map(|w| w.into_f32().collect()).unwrap_or_default();
                    for k in 0..n as usize {
                        let w = ws.get(k).copied().unwrap_or([1.0, 0.0, 0.0, 0.0]);
                        let sum: f32 = w.iter().sum();
                        sk.weights.push(if sum > 1e-6 { w.map(|x| x / sum) } else { [1.0, 0.0, 0.0, 0.0] });
                        sk.joints.push(js.get(k).copied().unwrap_or([0; 4]).map(|i| first + (i as u32).min(count - 1)));
                    }
                }
            }
        }

        let first_index = self.indices.len() as u32;
        self.indices.extend(local.iter().map(|i| base as u32 + i));
        self.primitives.push(MeshPrimitive {
            first_index,
            index_count: local.len() as u32,
            material: prim.material().index().unwrap_or(default_material),
        });
    }

    /// A (p, q) torus knot, handy before any model is loaded
    pub fn torus_knot(p: f32, q: f32, segments: u32, sides: u32) -> Self {
        let curve = |t: f32| {
            let r = 2.0 + (q * t).cos();
            Vec3::new(r * (p * t).cos(), -(q * t).sin(), r * (p * t).sin())
        };
        let (tube, e) = (0.45, 1e-3);
        let mut vertices = Vec::with_capacity(((segments + 1) * (sides + 1)) as usize);
        for i in 0..=segments {
            let u = i as f32 / segments as f32;
            let t = u * std::f32::consts::TAU;
            let (c, a, b) = (curve(t), curve(t - e), curve(t + e));
            let tan = (b - a).normalize();
            let bin = tan.cross(a + b - 2.0 * c).normalize();
            let nrm = bin.cross(tan);
            for j in 0..=sides {
                let v = j as f32 / sides as f32;
                let (s, co) = (v * std::f32::consts::TAU).sin_cos();
                let n = nrm * co + bin * s;
                vertices.push(MeshVertex { position: (c + n * tube).to_array(), normal: n.to_array(), uv: [u * 8.0, v], tangent: [0.0; 4], color: [1.0; 4], uv_b: [u, v], _pad: [0.0; 2] });
            }
        }
        let mut indices = Vec::with_capacity((segments * sides * 6) as usize);
        for i in 0..segments {
            for j in 0..sides {
                let (a, b) = (i * (sides + 1) + j, (i + 1) * (sides + 1) + j);
                indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
            }
        }
        generate_tangents(&mut vertices, &indices);
        let mut data = MeshData {
            primitives: vec![MeshPrimitive { first_index: 0, index_count: indices.len() as u32, material: 0 }],
            vertices, indices, materials: vec![MeshMaterial { double_sided: true, ..Default::default() }], images: Vec::new(),
            center: Vec3::ZERO, radius: 1.0, skinning: None,
        };
        data.fit_bounds();
        data
    }

    /// A flat square on the xz plane facing +y, `size` wide, uv repeating `tiles` times
    pub fn plane(size: f32, tiles: f32) -> Self {
        let h = size * 0.5;
        let mut vertices: Vec<MeshVertex> = [(-h, -h, 0.0, 0.0), (h, -h, tiles, 0.0), (h, h, tiles, tiles), (-h, h, 0.0, tiles)].iter()
            .map(|&(x, z, u, v)| MeshVertex { position: [x, 0.0, z], normal: [0.0, 1.0, 0.0], uv: [u, v], tangent: [0.0; 4], color: [1.0; 4], uv_b: [u / tiles.max(1e-6), v / tiles.max(1e-6)], _pad: [0.0; 2] })
            .collect();
        let indices = vec![0, 2, 1, 0, 3, 2];
        generate_tangents(&mut vertices, &indices);
        let mut data = MeshData {
            primitives: vec![MeshPrimitive { first_index: 0, index_count: 6, material: 0 }],
            vertices, indices, materials: vec![MeshMaterial::default()], images: Vec::new(),
            center: Vec3::ZERO, radius: 1.0, skinning: None,
        };
        data.fit_bounds();
        data
    }

    /// Lowest y of the model after `normalize_transform` (to stand it on a floor)
    pub fn normalized_bottom(&self) -> f32 {
        let lo = self.rest_positions().iter().map(|p| p.y).fold(f32::MAX, f32::min);
        (lo.min(self.center.y) - self.center.y) / self.radius
    }

    /// Vertex positions as displayed at rest (skinned models posed by their rest palette)
    pub fn rest_positions(&self) -> Vec<Vec3> {
        let Some(sk) = &self.skinning else { return self.vertices.iter().map(|v| Vec3::from(v.position)).collect() };
        let pal = sk.rest().palette;
        self.vertices.iter().enumerate().map(|(i, v)| {
            let p = Vec3::from(v.position);
            let (j, w) = (sk.joints.get(i).copied().unwrap_or([0; 4]), sk.weights.get(i).copied().unwrap_or([1.0, 0.0, 0.0, 0.0]));
            (0..4).fold(Vec3::ZERO, |acc, k| acc + pal.get(j[k] as usize).map(|m| m.transform_point3(p)).unwrap_or(p) * w[k])
        }).collect()
    }

    /// Recompute `center` and `radius` from the rest pose
    pub fn fit_bounds(&mut self) {
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for p in self.rest_positions() {
            lo = lo.min(p);
            hi = hi.max(p);
        }
        self.center = (lo + hi) * 0.5;
        self.radius = ((hi - lo) * 0.5).length().max(1e-6);
    }

    /// Model matrix that centres the model and scales it to unit radius
    pub fn normalize_transform(&self) -> Mat4 {
        Mat4::from_scale(Vec3::splat(1.0 / self.radius)) * Mat4::from_translation(-self.center)
    }
}

fn smooth_normals(verts: &mut [MeshVertex], tris: &[u32]) {
    for t in tris.chunks_exact(3) {
        let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
        let (pa, pb, pc) = (Vec3::from(verts[a].position), Vec3::from(verts[b].position), Vec3::from(verts[c].position));
        let fnrm = (pb - pa).cross(pc - pa);
        for i in [a, b, c] {
            verts[i].normal = (Vec3::from(verts[i].normal) + fnrm).to_array();
        }
    }
    for v in verts.iter_mut() {
        v.normal = Vec3::from(v.normal).normalize_or_zero().to_array();
    }
}

// per-vertex tangents from uv gradients (same handedness convention as MikkTSpace)
fn generate_tangents(verts: &mut [MeshVertex], tris: &[u32]) {
    let mut tan = vec![Vec3::ZERO; verts.len()];
    let mut bit = vec![Vec3::ZERO; verts.len()];
    for t in tris.chunks_exact(3) {
        let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
        let (p0, p1, p2) = (Vec3::from(verts[a].position), Vec3::from(verts[b].position), Vec3::from(verts[c].position));
        let (w0, w1, w2) = (verts[a].uv, verts[b].uv, verts[c].uv);
        let (e1, e2) = (p1 - p0, p2 - p0);
        let (s1, t1, s2, t2) = (w1[0] - w0[0], w1[1] - w0[1], w2[0] - w0[0], w2[1] - w0[1]);
        let r = s1 * t2 - s2 * t1;
        if r.abs() < 1e-12 { continue; }
        let sdir = (e1 * t2 - e2 * t1) * (1.0 / r);
        let tdir = (e2 * s1 - e1 * s2) * (1.0 / r);
        for i in [a, b, c] {
            tan[i] += sdir;
            bit[i] += tdir;
        }
    }
    for (i, v) in verts.iter_mut().enumerate() {
        let n = Vec3::from(v.normal);
        let t = (tan[i] - n * n.dot(tan[i])).normalize_or_zero();
        if t.length() == 0.0 { continue; }
        let w = if n.cross(t).dot(bit[i]) < 0.0 { -1.0 } else { 1.0 };
        v.tangent = [t.x, t.y, t.z, w];
    }
}

fn sampler_of(s: &gltf::texture::Sampler) -> MeshSampler {
    use gltf::texture::{MagFilter, MinFilter, WrappingMode};
    let wrap = |w| match w { WrappingMode::ClampToEdge => MeshWrap::Clamp, WrappingMode::MirroredRepeat => MeshWrap::Mirror, WrappingMode::Repeat => MeshWrap::Repeat };
    let min = s.min_filter();
    MeshSampler {
        wrap_u: wrap(s.wrap_s()),
        wrap_v: wrap(s.wrap_t()),
        mag_nearest: s.mag_filter() == Some(MagFilter::Nearest),
        min_nearest: matches!(min, Some(MinFilter::Nearest | MinFilter::NearestMipmapNearest | MinFilter::NearestMipmapLinear)),
        mip_nearest: matches!(min, Some(MinFilter::Nearest | MinFilter::Linear | MinFilter::NearestMipmapNearest | MinFilter::LinearMipmapNearest)),
    }
}

fn to_rgba8(img: &gltf::image::Data) -> MeshImage {
    use gltf::image::Format;
    let (ch, bpc) = match img.format {
        Format::R8 => (1, 1),
        Format::R8G8 => (2, 1),
        Format::R8G8B8 => (3, 1),
        Format::R8G8B8A8 => (4, 1),
        Format::R16 => (1, 2),
        Format::R16G16 => (2, 2),
        Format::R16G16B16 => (3, 2),
        Format::R16G16B16A16 => (4, 2),
        Format::R32G32B32FLOAT => (3, 4),
        Format::R32G32B32A32FLOAT => (4, 4),
    };
    let mut out = Vec::with_capacity((img.width * img.height * 4) as usize);
    for px in img.pixels.chunks_exact(ch * bpc) {
        let c = |i: usize| -> u8 {
            let b = &px[i * bpc..(i + 1) * bpc];
            match bpc {
                1 => b[0],
                2 => (u16::from_le_bytes([b[0], b[1]]) >> 8) as u8,
                _ => (f32::from_le_bytes([b[0], b[1], b[2], b[3]]).clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
            }
        };
        out.extend_from_slice(&match ch {
            1 => [c(0), c(0), c(0), 255],
            2 => [c(0), c(0), c(0), c(1)],
            3 => [c(0), c(1), c(2), 255],
            _ => [c(0), c(1), c(2), c(3)],
        });
    }
    MeshImage { width: img.width, height: img.height, pixels: out }
}


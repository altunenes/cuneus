//! Node hierarchy, clips, blend shapes and poses of animated glTF models

use crate::math::{Mat4, Quat, Vec3};

/// glTF node in its rest pose
#[derive(Clone, Debug)]
pub struct MeshNode {
    pub name: Option<String>,
    pub parent: Option<usize>,
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnimProperty { Translation, Rotation, Scale, MorphWeights }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnimInterpolation { Linear, Step, CubicSpline }

/// Keyframes of one node property, `width` floats per key; cubic keys store in tangent, value, out tangent
#[derive(Clone, Debug)]
pub struct AnimChannel {
    pub node: usize,
    pub property: AnimProperty,
    pub interpolation: AnimInterpolation,
    pub times: Vec<f32>,
    pub values: Vec<f32>,
    pub width: usize,
}

#[derive(Clone, Debug)]
pub struct MeshAnimation {
    pub name: String,
    /// Seconds
    pub duration: f32,
    pub channels: Vec<AnimChannel>,
}

impl MeshAnimation {
    /// Only drives blend shapes (blink, breath...): meant as a layer over a body clip
    pub fn morph_only(&self) -> bool {
        !self.channels.is_empty() && self.channels.iter().all(|c| c.property == AnimProperty::MorphWeights)
    }
}

/// A node's blend shapes: weights `base..base+count` of `Pose::morph_weights`
#[derive(Clone, Debug)]
pub struct MorphSlot {
    pub node: usize,
    pub base: u32,
    pub count: u32,
    pub names: Vec<String>,
    pub default: Vec<f32>,
}

/// Bone palette + blend shape weights for one object
#[derive(Clone, Debug, Default)]
pub struct Pose {
    pub palette: Vec<Mat4>,
    pub morph_weights: Vec<f32>,
}

/// Nodes, skins, blend shapes and clips; a palette entry is a node's world matrix times its inverse bind
#[derive(Clone, Debug)]
pub struct MeshSkinning {
    pub joints: Vec<[u32; 4]>,
    pub weights: Vec<[f32; 4]>,
    /// Per vertex: first delta pair, target count, first weight, unused
    pub morph: Vec<[u32; 4]>,
    /// Per vertex and target: position delta, normal delta
    pub morph_deltas: Vec<[f32; 4]>,
    pub morph_slots: Vec<MorphSlot>,
    pub nodes: Vec<MeshNode>,
    /// Parents before children
    pub order: Vec<usize>,
    pub palette: Vec<(usize, Mat4)>,
    pub animations: Vec<MeshAnimation>,
}

impl AnimChannel {
    fn sample(&self, t: f32) -> Vec<f32> {
        let (n, w) = (self.times.len(), self.width);
        let cubic = self.interpolation == AnimInterpolation::CubicSpline;
        let at = |k: usize, part: usize| -> &[f32] { let i = if cubic { k * 3 + part } else { k }; &self.values[i * w..(i + 1) * w] };
        let val = |k: usize| at(k, 1).to_vec();
        if n == 1 || t <= self.times[0] { return val(0); }
        if t >= self.times[n - 1] { return val(n - 1); }
        let k = self.times.partition_point(|&x| x <= t).saturating_sub(1).min(n - 2);
        let dt = self.times[k + 1] - self.times[k];
        let s = if dt > 0.0 { (t - self.times[k]) / dt } else { 0.0 };
        let rot = self.property == AnimProperty::Rotation;
        let v: Vec<f32> = match self.interpolation {
            AnimInterpolation::Step => return val(k),
            AnimInterpolation::Linear if rot => {
                let q = |k| { let a = at(k, 1); Quat::from_array([a[0], a[1], a[2], a[3]]) };
                return q(k).slerp(q(k + 1), s).to_array().to_vec();
            }
            AnimInterpolation::Linear => at(k, 1).iter().zip(at(k + 1, 1)).map(|(a, b)| a + (b - a) * s).collect(),
            AnimInterpolation::CubicSpline => {
                let (v0, b0, a1, v1) = (at(k, 1), at(k, 2), at(k + 1, 0), at(k + 1, 1));
                let (s2, s3) = (s * s, s * s * s);
                (0..w).map(|i| (2.0 * s3 - 3.0 * s2 + 1.0) * v0[i] + (s3 - 2.0 * s2 + s) * dt * b0[i] + (-2.0 * s3 + 3.0 * s2) * v1[i] + (s3 - s2) * dt * a1[i]).collect()
            }
        };
        if rot { Quat::from_array([v[0], v[1], v[2], v[3]]).normalize().to_array().to_vec() } else { v }
    }
}

fn lerp3(a: Vec3, v: &[f32], w: f32) -> Vec3 { a + (Vec3::new(v[0], v[1], v[2]) - a) * w }

impl MeshSkinning {
    /// Rest pose: node rest transforms and default blend shape weights
    pub fn rest(&self) -> Pose { self.pose_layers(&[]) }

    /// One clip at `time` (loops), `None` = rest pose; `time.min(duration)` holds the last frame
    pub fn pose(&self, animation: Option<usize>, time: f32) -> Pose {
        match animation {
            Some(a) => self.pose_layers(&[(a, time, 1.0)]),
            None => self.rest(),
        }
    }

    /// (clip, time, weight) layers in order, each blending what it animates by `weight` (1 = replace)
    pub fn pose_layers(&self, layers: &[(usize, f32, f32)]) -> Pose {
        let mut trs: Vec<(Vec3, Quat, Vec3)> = self.nodes.iter().map(|n| (n.translation, n.rotation, n.scale)).collect();
        let mut morph = vec![0.0; self.morph_slots.iter().map(|m| (m.base + m.count) as usize).max().unwrap_or(0)];
        for m in &self.morph_slots {
            for (i, d) in m.default.iter().enumerate().take(m.count as usize) { morph[m.base as usize + i] = *d; }
        }
        for &(clip, time, weight) in layers {
            let Some(a) = self.animations.get(clip) else { continue };
            // wrap only outside the clip, so time == duration holds the last frame
            let t = if a.duration <= 0.0 { 0.0 } else if (0.0..=a.duration).contains(&time) { time } else { time.rem_euclid(a.duration) };
            let w = weight.clamp(0.0, 1.0);
            for ch in &a.channels {
                let v = ch.sample(t);
                match ch.property {
                    AnimProperty::MorphWeights => {
                        let Some(m) = self.morph_slots.iter().find(|m| m.node == ch.node) else { continue };
                        for (i, x) in v.iter().enumerate().take(m.count as usize) {
                            let d = &mut morph[m.base as usize + i];
                            *d += (x - *d) * w;
                        }
                    }
                    _ => {
                        let Some(n) = trs.get_mut(ch.node) else { continue };
                        match ch.property {
                            AnimProperty::Translation => n.0 = lerp3(n.0, &v, w),
                            AnimProperty::Rotation => n.1 = n.1.slerp(Quat::from_array([v[0], v[1], v[2], v[3]]), w),
                            _ => n.2 = lerp3(n.2, &v, w),
                        }
                    }
                }
            }
        }
        Pose { palette: self.palette_from_local(&trs), morph_weights: morph }
    }

    /// Palette from your own local node transforms (procedural bones, IK, ragdolls...)
    pub fn palette_from_local(&self, trs: &[(Vec3, Quat, Vec3)]) -> Vec<Mat4> {
        let mut world = vec![Mat4::IDENTITY; self.nodes.len()];
        for &i in &self.order {
            let (t, r, s) = trs.get(i).copied().unwrap_or((Vec3::ZERO, Quat::IDENTITY, Vec3::splat(1.0)));
            let local = Mat4::from_trs(t, r, s);
            world[i] = match self.nodes[i].parent { Some(p) => world[p] * local, None => local };
        }
        self.palette.iter().map(|&(node, ib)| world[node] * ib).collect()
    }
}


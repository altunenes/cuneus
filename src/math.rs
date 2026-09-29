//! Poor man's minimal 3D math for meshes and cameras

use std::ops::{Add, AddAssign, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self::splat(0.0);
    pub const Y: Self = Self::new(0.0, 1.0, 0.0);
    pub const fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    pub const fn splat(v: f32) -> Self { Self::new(v, v, v) }
    pub fn to_array(self) -> [f32; 3] { [self.x, self.y, self.z] }
    pub fn dot(self, o: Self) -> f32 { self.x * o.x + self.y * o.y + self.z * o.z }
    pub fn cross(self, o: Self) -> Self { Self::new(self.y * o.z - self.z * o.y, self.z * o.x - self.x * o.z, self.x * o.y - self.y * o.x) }
    pub fn length(self) -> f32 { self.dot(self).sqrt() }
    pub fn normalize(self) -> Self { self * (1.0 / self.length()) }
    pub fn normalize_or_zero(self) -> Self {
        let l = self.length();
        if l > 1e-12 { self * (1.0 / l) } else { Self::ZERO }
    }
    pub fn min(self, o: Self) -> Self { Self::new(self.x.min(o.x), self.y.min(o.y), self.z.min(o.z)) }
    pub fn max(self, o: Self) -> Self { Self::new(self.x.max(o.x), self.y.max(o.y), self.z.max(o.z)) }
}

impl From<[f32; 3]> for Vec3 { fn from(a: [f32; 3]) -> Self { Self::new(a[0], a[1], a[2]) } }
impl Add for Vec3 { type Output = Self; fn add(self, o: Self) -> Self { Self::new(self.x + o.x, self.y + o.y, self.z + o.z) } }
impl Sub for Vec3 { type Output = Self; fn sub(self, o: Self) -> Self { Self::new(self.x - o.x, self.y - o.y, self.z - o.z) } }
impl Mul for Vec3 { type Output = Self; fn mul(self, o: Self) -> Self { Self::new(self.x * o.x, self.y * o.y, self.z * o.z) } }
impl Mul<f32> for Vec3 { type Output = Self; fn mul(self, s: f32) -> Self { Self::new(self.x * s, self.y * s, self.z * s) } }
impl Mul<Vec3> for f32 { type Output = Vec3; fn mul(self, v: Vec3) -> Vec3 { v * self } }
impl Neg for Vec3 { type Output = Self; fn neg(self) -> Self { self * -1.0 } }
impl AddAssign for Vec3 { fn add_assign(&mut self, o: Self) { *self = *self + o; } }

/// 4x4 matrix, stored as columns
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat4 {
    pub cols: [[f32; 4]; 4],
}

impl Mat4 {
    pub const IDENTITY: Self = Self { cols: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]] };
    pub fn from_cols_array_2d(cols: &[[f32; 4]; 4]) -> Self { Self { cols: *cols } }
    pub fn to_cols_array(&self) -> [f32; 16] { bytemuck::cast(self.cols) }
    pub fn from_scale(s: Vec3) -> Self {
        let mut m = Self::IDENTITY;
        (m.cols[0][0], m.cols[1][1], m.cols[2][2]) = (s.x, s.y, s.z);
        m
    }
    pub fn from_translation(t: Vec3) -> Self {
        let mut m = Self::IDENTITY;
        m.cols[3] = [t.x, t.y, t.z, 1.0];
        m
    }
    fn col3(&self, i: usize) -> Vec3 { Vec3::new(self.cols[i][0], self.cols[i][1], self.cols[i][2]) }
    pub fn transform_point3(&self, p: Vec3) -> Vec3 { self.transform_vector3(p) + self.col3(3) }
    pub fn transform_vector3(&self, v: Vec3) -> Vec3 { self.col3(0) * v.x + self.col3(1) * v.y + self.col3(2) * v.z }

    /// Determinant of the upper 3x3; negative when the transform mirrors
    pub fn det3(&self) -> f32 { self.col3(0).dot(self.col3(1).cross(self.col3(2))) }

    /// Inverse transpose of the upper 3x3 (cofactors / det), for normals
    pub fn normal_matrix(&self) -> Self {
        let (a, b, c) = (self.col3(0), self.col3(1), self.col3(2));
        let det = a.dot(b.cross(c));
        let k = if det.abs() > 1e-20 { 1.0 / det } else { 0.0 };
        let (x, y, z) = (b.cross(c) * k, c.cross(a) * k, a.cross(b) * k);
        Self { cols: [[x.x, x.y, x.z, 0.0], [y.x, y.y, y.z, 0.0], [z.x, z.y, z.z, 0.0], [0.0, 0.0, 0.0, 1.0]] }
    }

    pub fn look_at_rh(eye: Vec3, center: Vec3, up: Vec3) -> Self {
        let f = (center - eye).normalize();
        let s = f.cross(up).normalize();
        let u = s.cross(f);
        Self { cols: [[s.x, u.x, -f.x, 0.0], [s.y, u.y, -f.y, 0.0], [s.z, u.z, -f.z, 0.0], [-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0]] }
    }

    /// Right-handed orthographic box, depth 0..1
    pub fn orthographic_rh(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        let (w, h, d) = (1.0 / (right - left), 1.0 / (top - bottom), 1.0 / (near - far));
        Self { cols: [[2.0 * w, 0.0, 0.0, 0.0], [0.0, 2.0 * h, 0.0, 0.0], [0.0, 0.0, d, 0.0], [-(left + right) * w, -(top + bottom) * h, near * d, 1.0]] }
    }

    /// Largest axis scale of the upper 3x3
    pub fn max_scale(&self) -> f32 { self.col3(0).length().max(self.col3(1).length()).max(self.col3(2).length()) }

    /// Right-handed perspective, depth 0..1
    pub fn perspective_rh(fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
        let h = 1.0 / (fov_y * 0.5).tan();
        let r = far / (near - far);
        Self { cols: [[h / aspect, 0.0, 0.0, 0.0], [0.0, h, 0.0, 0.0], [0.0, 0.0, r, -1.0], [0.0, 0.0, r * near, 0.0]] }
    }
}

impl Mul for Mat4 {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        let mut m = [[0.0; 4]; 4];
        for (c, col) in m.iter_mut().enumerate() {
            for (r, v) in col.iter_mut().enumerate() {
                *v = (0..4).map(|k| self.cols[k][r] * o.cols[c][k]).sum();
            }
        }
        Self { cols: m }
    }
}

/// Unit quaternion (x, y, z, w), glTF order
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Quat {
    pub const IDENTITY: Self = Self { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };
    pub fn from_array(a: [f32; 4]) -> Self { Self { x: a[0], y: a[1], z: a[2], w: a[3] } }
    pub fn to_array(self) -> [f32; 4] { [self.x, self.y, self.z, self.w] }
    pub fn dot(self, o: Self) -> f32 { self.x * o.x + self.y * o.y + self.z * o.z + self.w * o.w }
    pub fn normalize(self) -> Self {
        let l = self.dot(self).sqrt();
        if l > 1e-12 { Self { x: self.x / l, y: self.y / l, z: self.z / l, w: self.w / l } } else { Self::IDENTITY }
    }
    /// Shortest-path spherical interpolation
    pub fn slerp(self, o: Self, t: f32) -> Self {
        let (mut o, mut d) = (o, self.dot(o));
        if d < 0.0 { (o, d) = (Self { x: -o.x, y: -o.y, z: -o.z, w: -o.w }, -d); }
        let (a, b) = if d > 0.9995 { (1.0 - t, t) } else {
            let th = d.acos();
            let s = th.sin();
            (((1.0 - t) * th).sin() / s, (t * th).sin() / s)
        };
        Self { x: self.x * a + o.x * b, y: self.y * a + o.y * b, z: self.z * a + o.z * b, w: self.w * a + o.w * b }.normalize()
    }
}

impl Mat4 {
    /// translation * rotation * scale
    pub fn from_trs(t: Vec3, r: Quat, s: Vec3) -> Self {
        let (x2, y2, z2) = (r.x + r.x, r.y + r.y, r.z + r.z);
        let (xx, xy, xz, yy, yz, zz) = (r.x * x2, r.x * y2, r.x * z2, r.y * y2, r.y * z2, r.z * z2);
        let (wx, wy, wz) = (r.w * x2, r.w * y2, r.w * z2);
        Self { cols: [
            [(1.0 - (yy + zz)) * s.x, (xy + wz) * s.x, (xz - wy) * s.x, 0.0],
            [(xy - wz) * s.y, (1.0 - (xx + zz)) * s.y, (yz + wx) * s.y, 0.0],
            [(xz + wy) * s.z, (yz - wx) * s.z, (1.0 - (xx + yy)) * s.z, 0.0],
            [t.x, t.y, t.z, 1.0],
        ] }
    }
}

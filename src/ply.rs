use std::collections::HashMap;
use std::io::BufReader;
use std::path::Path;

/// GPU-ready packed Gaussian data (64 bytes, aligned for optimal GPU access)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PackedGaussian3D {
    /// Position in 3D space (x, y, z)
    pub position: [f32; 3],
    pub _pad0: f32,
    /// Upper triangular 3x3 covariance matrix: [cov_xx, cov_xy, cov_xz, cov_yy, cov_yz, cov_zz]
    pub cov: [f32; 6],
    pub _pad1: [f32; 2],
    /// Base (degree 0) colour, unclamped, and opacity (r, g, b, opacity)
    pub color: [f32; 4],
}

/// Spherical harmonics up to degree 2 as packed f16: coeff j (1..8) x rgb -> half index j*3+c
pub type PackedSH = [u32; 12];

/// Metadata from PLY file
#[derive(Clone, Debug)]
pub struct PlyMetadata {
    pub num_gaussians: u32,
    pub image_size: [u32; 2],
    pub focal_length: f32,
}

/// Result of loading a PLY file
pub struct GaussianCloud {
    pub gaussians: Vec<PackedGaussian3D>,
    /// View-dependent colour, one entry per gaussian (zeros when the file has none)
    pub sh: Vec<PackedSH>,
    /// Spherical harmonics degree kept (0..2)
    pub sh_degree: u32,
    pub metadata: PlyMetadata,
    /// Robust bounds (2nd-98th percentile), ignoring stray floaters
    pub bounds_center: [f32; 3],
    pub bounds_radius: f32,
}

#[derive(Debug)]
pub enum PlyError {
    Io(std::io::Error),
    Parse(String),
    MissingProperty(String),
}

impl std::fmt::Display for PlyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlyError::Io(e) => write!(f, "IO error: {}", e),
            PlyError::Parse(s) => write!(f, "Parse error: {}", s),
            PlyError::MissingProperty(s) => write!(f, "Missing property: {}", s),
        }
    }
}

impl std::error::Error for PlyError {}

impl From<std::io::Error> for PlyError {
    fn from(e: std::io::Error) -> Self {
        PlyError::Io(e)
    }
}

const SH_C0: f32 = 0.28209479177387814;

fn f32_to_f16(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7f_ffff;
    if exp <= 0 {
        if exp < -10 { return sign; }
        let m = (man | 0x80_0000) >> (1 - exp);
        return sign | ((m + 0x1000) >> 13) as u16;
    }
    if exp >= 31 { return sign | 0x7c00; }
    let r = ((exp as u32) << 10) | (man >> 13);
    sign | (r + ((man >> 12) & 1)) as u16
}

fn build_gaussian(get: &dyn Fn(&str) -> Option<f32>, rest_per_channel: usize) -> Result<(PackedGaussian3D, PackedSH), PlyError> {
    let need = |n: &str| get(n).ok_or_else(|| PlyError::MissingProperty(n.to_string()));
    let (x, y, z) = (need("x")?, need("y")?, need("z")?);

    let s = [
        get("scale_0").unwrap_or(0.0).exp(),
        get("scale_1").unwrap_or(0.0).exp(),
        get("scale_2").unwrap_or(0.0).exp(),
    ];
    // rotation quaternion (w, x, y, z)
    let q = [get("rot_0").unwrap_or(1.0), get("rot_1").unwrap_or(0.0), get("rot_2").unwrap_or(0.0), get("rot_3").unwrap_or(0.0)];
    let ql = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let quat = if ql > 0.0 { [q[0] / ql, q[1] / ql, q[2] / ql, q[3] / ql] } else { [1.0, 0.0, 0.0, 0.0] };
    let cov = compute_covariance(s, quat);

    let dc = [get("f_dc_0").unwrap_or(0.0), get("f_dc_1").unwrap_or(0.0), get("f_dc_2").unwrap_or(0.0)];
    let opacity = 1.0 / (1.0 + (-get("opacity").unwrap_or(0.0)).exp());

    // f_rest is channel-major: all coeffs of r, then g, then b
    let mut halves = [0u16; 24];
    for j in 0..8usize.min(rest_per_channel) {
        for c in 0..3 {
            let v = get(&format!("f_rest_{}", c * rest_per_channel + j)).unwrap_or(0.0);
            halves[j * 3 + c] = f32_to_f16(v);
        }
    }
    let mut sh = [0u32; 12];
    for k in 0..12 {
        sh[k] = halves[k * 2] as u32 | ((halves[k * 2 + 1] as u32) << 16);
    }

    Ok((
        PackedGaussian3D {
            position: [x, y, z],
            _pad0: 0.0,
            cov,
            _pad1: [0.0, 0.0],
            color: [0.5 + SH_C0 * dc[0], 0.5 + SH_C0 * dc[1], 0.5 + SH_C0 * dc[2], opacity],
        },
        sh,
    ))
}

fn sh_degree_for(rest_per_channel: usize) -> u32 {
    if rest_per_channel >= 8 { 2 } else if rest_per_channel >= 3 { 1 } else { 0 }
}

/// Fast path: direct reader for binary little-endian files (the usual 3DGS export)
fn read_binary_le(bytes: &[u8]) -> Option<Result<(Vec<PackedGaussian3D>, Vec<PackedSH>, u32), PlyError>> {
    let marker = b"end_header\n";
    let hend = bytes.windows(marker.len()).position(|w| w == marker)? + marker.len();
    let header = std::str::from_utf8(&bytes[..hend]).ok()?;

    let mut binary_le = false;
    let mut in_vertex = false;
    let mut first_element = true;
    let mut count = 0usize;
    let mut props: Vec<(String, &str)> = Vec::new();
    for line in header.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        match t.as_slice() {
            ["format", "binary_little_endian", ..] => binary_le = true,
            ["element", name, n] => {
                // vertex must be the first element for the offset maths below
                if *name == "vertex" && first_element { in_vertex = true; count = n.parse().ok()?; }
                else if in_vertex { in_vertex = false; } else if first_element { return None; }
                first_element = false;
            }
            ["property", "list", ..] if in_vertex => return None,
            ["property", ty, name] if in_vertex => props.push((name.to_string(), ty)),
            _ => {}
        }
    }
    if !binary_le || count == 0 { return None; }

    let size_of = |ty: &str| match ty {
        "char" | "uchar" | "int8" | "uint8" => Some(1usize),
        "short" | "ushort" | "int16" | "uint16" => Some(2),
        "int" | "uint" | "float" | "int32" | "uint32" | "float32" => Some(4),
        "double" | "float64" => Some(8),
        _ => None,
    };
    let mut offsets: HashMap<String, (usize, String)> = HashMap::new();
    let mut stride = 0usize;
    for (name, ty) in &props {
        offsets.insert(name.clone(), (stride, ty.to_string()));
        stride += size_of(ty)?;
    }
    if bytes.len() < hend + stride * count { return Some(Err(PlyError::Parse("PLY data truncated".into()))); }

    let rest = props.iter().filter(|(n, _)| n.starts_with("f_rest_")).count() / 3;
    let mut gs = Vec::with_capacity(count);
    let mut shs = Vec::with_capacity(count);
    for i in 0..count {
        let base = hend + i * stride;
        let get = |name: &str| -> Option<f32> {
            let (o, ty) = offsets.get(name)?;
            let b = &bytes[base + o..];
            Some(match ty.as_str() {
                "float" | "float32" => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                "double" | "float64" => f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32,
                "uchar" | "uint8" => b[0] as f32,
                "char" | "int8" => b[0] as i8 as f32,
                "ushort" | "uint16" => u16::from_le_bytes([b[0], b[1]]) as f32,
                "short" | "int16" => i16::from_le_bytes([b[0], b[1]]) as f32,
                "uint" | "uint32" => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
                _ => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
            })
        };
        match build_gaussian(&get, rest) {
            Ok((g, s)) => { gs.push(g); shs.push(s); }
            Err(e) => return Some(Err(e)),
        }
    }
    Some(Ok((gs, shs, sh_degree_for(rest))))
}

/// Centre and radius from the 2nd-98th percentile of a position sample
fn robust_bounds(gs: &[PackedGaussian3D]) -> ([f32; 3], f32) {
    if gs.is_empty() { return ([0.0; 3], 1.0); }
    let step = (gs.len() / 100_000).max(1);
    let mut ax: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for g in gs.iter().step_by(step) {
        for k in 0..3 { ax[k].push(g.position[k]); }
    }
    let mut c = [0.0f32; 3];
    let mut r = 0.0f32;
    for k in 0..3 {
        ax[k].sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = ax[k].len();
        let lo = ax[k][n * 2 / 100];
        let hi = ax[k][(n * 98 / 100).min(n - 1)];
        c[k] = 0.5 * (lo + hi);
        r = r.max(0.5 * (hi - lo));
    }
    (c, r.max(1e-3))
}

impl GaussianCloud {
    pub fn from_ply<P: AsRef<Path>>(path: P) -> Result<Self, PlyError> {
        let bytes = std::fs::read(path.as_ref())?;
        let (gaussians, sh, sh_degree) = match read_binary_le(&bytes) {
            Some(r) => r?,
            None => Self::read_generic(&bytes)?,
        };
        let num_gaussians = gaussians.len() as u32;
        let (bounds_center, bounds_radius) = robust_bounds(&gaussians);
        Ok(GaussianCloud {
            gaussians,
            sh,
            sh_degree,
            metadata: PlyMetadata { num_gaussians, image_size: [640, 480], focal_length: 512.0 },
            bounds_center,
            bounds_radius,
        })
    }

    /// Slow path for ascii / big-endian files
    fn read_generic(bytes: &[u8]) -> Result<(Vec<PackedGaussian3D>, Vec<PackedSH>, u32), PlyError> {
        use ply_rs_bw::parser::Parser;
        use ply_rs_bw::ply::DefaultElement;

        let mut reader = BufReader::new(bytes);
        let parser = Parser::<DefaultElement>::new();
        let ply = parser.read_ply(&mut reader).map_err(|e| PlyError::Parse(e.to_string()))?;
        let vertices = ply.payload.get("vertex").ok_or_else(|| PlyError::Parse("Missing vertex element".to_string()))?;

        let rest = vertices.first().map(|v| v.keys().filter(|k| k.starts_with("f_rest_")).count() / 3).unwrap_or(0);
        let mut gs = Vec::with_capacity(vertices.len());
        let mut shs = Vec::with_capacity(vertices.len());
        for vertex in vertices {
            let get = |n: &str| get_float_property(vertex, n).ok();
            let (g, s) = build_gaussian(&get, rest)?;
            gs.push(g);
            shs.push(s);
        }
        Ok((gs, shs, sh_degree_for(rest)))
    }

    /// Get the raw byte data for GPU upload
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.gaussians)
    }

    /// Raw spherical harmonics bytes for GPU upload
    pub fn sh_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.sh)
    }

    /// Get the size in bytes of the Gaussian data
    pub fn size_bytes(&self) -> u64 {
        (self.gaussians.len() * std::mem::size_of::<PackedGaussian3D>()) as u64
    }

    /// Compute the centroid (average position) of all Gaussians
    pub fn centroid(&self) -> [f32; 3] {
        if self.gaussians.is_empty() {
            return [0.0, 0.0, 0.0];
        }
        let mut sum = [0.0f64, 0.0f64, 0.0f64];
        for g in &self.gaussians {
            sum[0] += g.position[0] as f64;
            sum[1] += g.position[1] as f64;
            sum[2] += g.position[2] as f64;
        }
        let n = self.gaussians.len() as f64;
        [(sum[0] / n) as f32, (sum[1] / n) as f32, (sum[2] / n) as f32]
    }

    /// Compute the bounding box extent (max dimension)
    pub fn extent(&self) -> f32 {
        if self.gaussians.is_empty() {
            return 1.0;
        }
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        for g in &self.gaussians {
            for i in 0..3 {
                min[i] = min[i].min(g.position[i]);
                max[i] = max[i].max(g.position[i]);
            }
        }
        let size = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
        size[0].max(size[1]).max(size[2]).max(0.001)
    }
}

/// Compute 3D covariance matrix from quaternion rotation and scale
fn compute_covariance(scale: [f32; 3], quat: [f32; 4]) -> [f32; 6] {
    let [w, x, y, z] = quat;

    // Rotation matrix from quaternion
    let r00 = 1.0 - 2.0 * (y * y + z * z);
    let r01 = 2.0 * (x * y - w * z);
    let r02 = 2.0 * (x * z + w * y);
    let r10 = 2.0 * (x * y + w * z);
    let r11 = 1.0 - 2.0 * (x * x + z * z);
    let r12 = 2.0 * (y * z - w * x);
    let r20 = 2.0 * (x * z - w * y);
    let r21 = 2.0 * (y * z + w * x);
    let r22 = 1.0 - 2.0 * (x * x + y * y);

    // Scale squared
    let [sx, sy, sz] = scale;
    let s2 = [sx * sx, sy * sy, sz * sz];

    // Covariance = R * S^2 * R^T (upper triangular)
    let cov_xx = r00 * r00 * s2[0] + r01 * r01 * s2[1] + r02 * r02 * s2[2];
    let cov_xy = r00 * r10 * s2[0] + r01 * r11 * s2[1] + r02 * r12 * s2[2];
    let cov_xz = r00 * r20 * s2[0] + r01 * r21 * s2[1] + r02 * r22 * s2[2];
    let cov_yy = r10 * r10 * s2[0] + r11 * r11 * s2[1] + r12 * r12 * s2[2];
    let cov_yz = r10 * r20 * s2[0] + r11 * r21 * s2[1] + r12 * r22 * s2[2];
    let cov_zz = r20 * r20 * s2[0] + r21 * r21 * s2[1] + r22 * r22 * s2[2];

    [cov_xx, cov_xy, cov_xz, cov_yy, cov_yz, cov_zz]
}

/// Helper to extract float property from PLY element
fn get_float_property(element: &ply_rs_bw::ply::DefaultElement, name: &str) -> Result<f32, PlyError> {
    use ply_rs_bw::ply::Property;

    element.get(name)
        .ok_or_else(|| PlyError::MissingProperty(name.to_string()))
        .and_then(|prop| match prop {
            Property::Float(v) => Ok(*v),
            Property::Double(v) => Ok(*v as f32),
            Property::Int(v) => Ok(*v as f32),
            Property::UInt(v) => Ok(*v as f32),
            Property::Short(v) => Ok(*v as f32),
            Property::UShort(v) => Ok(*v as f32),
            Property::Char(v) => Ok(*v as f32),
            Property::UChar(v) => Ok(*v as f32),
            _ => Err(PlyError::Parse(format!("Property {} is not a number", name))),
        })
}

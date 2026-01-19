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
    /// Color and opacity (r, g, b, opacity)
    pub color: [f32; 4],
}

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
    pub metadata: PlyMetadata,
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

impl GaussianCloud {
    pub fn from_ply<P: AsRef<Path>>(path: P) -> Result<Self, PlyError> {
        use ply_rs_bw::parser::Parser;
        use ply_rs_bw::ply::DefaultElement;

        let file = std::fs::File::open(path.as_ref())?;
        let mut reader = BufReader::new(file);

        let parser = Parser::<DefaultElement>::new();
        let ply = parser.read_ply(&mut reader).map_err(|e| PlyError::Parse(e.to_string()))?;

        let vertices = ply.payload.get("vertex")
            .ok_or_else(|| PlyError::Parse("Missing vertex element".to_string()))?;

        let num_gaussians = vertices.len() as u32;
        let mut gaussians = Vec::with_capacity(num_gaussians as usize);

        let sh_c0: f32 = 0.28209479177387814;

        for vertex in vertices {
            // Extract position
            let x = get_float_property(vertex, "x")?;
            let y = get_float_property(vertex, "y")?;
            let z = get_float_property(vertex, "z")?;

            // Extract scale (log-space, need exp)
            let scale_0 = get_float_property(vertex, "scale_0").unwrap_or(0.0).exp();
            let scale_1 = get_float_property(vertex, "scale_1").unwrap_or(0.0).exp();
            let scale_2 = get_float_property(vertex, "scale_2").unwrap_or(0.0).exp();

            // Extract rotation quaternion (w, x, y, z)
            let rot_0 = get_float_property(vertex, "rot_0").unwrap_or(1.0);
            let rot_1 = get_float_property(vertex, "rot_1").unwrap_or(0.0);
            let rot_2 = get_float_property(vertex, "rot_2").unwrap_or(0.0);
            let rot_3 = get_float_property(vertex, "rot_3").unwrap_or(0.0);

            // Normalize quaternion
            let quat_len = (rot_0 * rot_0 + rot_1 * rot_1 + rot_2 * rot_2 + rot_3 * rot_3).sqrt();
            let quat = if quat_len > 0.0 {
                [rot_0 / quat_len, rot_1 / quat_len, rot_2 / quat_len, rot_3 / quat_len]
            } else {
                [1.0, 0.0, 0.0, 0.0]
            };

            // Compute covariance matrix from quaternion and scale
            let cov = compute_covariance([scale_0, scale_1, scale_2], quat);

            // Extract color
            let f_dc_0 = get_float_property(vertex, "f_dc_0").unwrap_or(0.0);
            let f_dc_1 = get_float_property(vertex, "f_dc_1").unwrap_or(0.0);
            let f_dc_2 = get_float_property(vertex, "f_dc_2").unwrap_or(0.0);

            let r = (0.5 + sh_c0 * f_dc_0).clamp(0.0, 1.0);
            let g = (0.5 + sh_c0 * f_dc_1).clamp(0.0, 1.0);
            let b = (0.5 + sh_c0 * f_dc_2).clamp(0.0, 1.0);

            let opacity_raw = get_float_property(vertex, "opacity").unwrap_or(0.0);
            let opacity = 1.0 / (1.0 + (-opacity_raw).exp());

            gaussians.push(PackedGaussian3D {
                position: [x, y, z],
                _pad0: 0.0,
                cov,
                _pad1: [0.0, 0.0],
                color: [r, g, b, opacity],
            });
        }

        let mut image_size = [640u32, 480u32];
        let mut focal_length = 512.0f32;

        if let Some(image_size_element) = ply.payload.get("image_size") {
            if image_size_element.len() >= 2 {
                if let (Some(w), Some(h)) = (
                    get_uint_from_element(&image_size_element[0], "image_size"),
                    get_uint_from_element(&image_size_element[1], "image_size"),
                ) {
                    image_size = [w, h];
                }
            }
        }


        if let Some(intrinsic_element) = ply.payload.get("intrinsic") {
            if !intrinsic_element.is_empty() {
                if let Some(f) = get_float_from_element(&intrinsic_element[0], "intrinsic") {
                    focal_length = f;
                }
            }
        }

        Ok(GaussianCloud {
            gaussians,
            metadata: PlyMetadata {
                num_gaussians,
                image_size,
                focal_length,
            },
        })
    }

    /// Get the raw byte data for GPU upload
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.gaussians)
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

fn get_float_from_element(element: &ply_rs_bw::ply::DefaultElement, name: &str) -> Option<f32> {
    use ply_rs_bw::ply::Property;

    element.get(name).and_then(|prop| match prop {
        Property::Float(v) => Some(*v),
        Property::Double(v) => Some(*v as f32),
        _ => None,
    })
}

fn get_uint_from_element(element: &ply_rs_bw::ply::DefaultElement, name: &str) -> Option<u32> {
    use ply_rs_bw::ply::Property;

    element.get(name).and_then(|prop| match prop {
        Property::UInt(v) => Some(*v),
        Property::Int(v) => Some(*v as u32),
        Property::UShort(v) => Some(*v as u32),
        Property::Short(v) => Some(*v as u32),
        _ => None,
    })
}
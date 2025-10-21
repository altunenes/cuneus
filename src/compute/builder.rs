use crate::UniformProvider;
use wgpu;

/// Pass description for multi-pass shaders
#[derive(Debug, Clone)]
pub struct PassDescription {
    pub name: String,
    pub inputs: Vec<String>,
    pub workgroup_size: Option<[u32; 3]>,
}

impl PassDescription {
    pub fn new(name: &str, inputs: &[&str]) -> Self {
        Self {
            name: name.to_string(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            workgroup_size: None,
        }
    }

    pub fn with_workgroup_size(mut self, size: [u32; 3]) -> Self {
        self.workgroup_size = Some(size);
        self
    }
}

/// User-defined storage buffer specification
#[derive(Debug, Clone)]
pub struct StorageBufferSpec {
    pub name: String,
    pub size_bytes: u64,
    pub read_only: bool,
}

impl StorageBufferSpec {
    pub fn new(name: &str, size_bytes: u64) -> Self {
        Self {
            name: name.to_string(),
            size_bytes,
            read_only: false,
        }
    }

    pub fn read_only(mut self) -> Self {
        self.read_only = true;
        self
    }
}

/// Configuration built by the builder
#[derive(Debug)]
pub struct ComputeConfiguration {
    pub entry_points: Vec<String>,
    pub passes: Option<Vec<PassDescription>>,
    pub custom_uniform_size: Option<u64>,
    pub has_input_texture: bool,
    pub has_mouse: bool,
    pub has_fonts: bool,
    pub has_audio: bool,
    pub has_atomic_buffer: bool,
    pub audio_buffer_size: usize,
    pub has_audio_spectrum: bool,
    pub audio_spectrum_size: usize,
    pub storage_buffers: Vec<StorageBufferSpec>,
    pub workgroup_size: [u32; 3],
    pub dispatch_once: bool,
    pub texture_format: wgpu::TextureFormat,
    pub label: String,
    pub num_channels: Option<u32>,
}

/// Builder for compute shader configurations
/// @group(0): Per-Frame Resources (TimeUniform)
/// @group(1): Primary Pass I/O & Parameters (output texture, shader params, input textures)
/// @group(2): Global Engine Resources (fonts, audio, atomics, mouse)
/// @group(3): User-Defined Data Buffers (custom storage buffers)
pub struct ComputeShaderBuilder {
    config: ComputeConfiguration,
}

impl ComputeShaderBuilder {
    pub fn new() -> Self {
        Self {
            config: ComputeConfiguration {
                entry_points: vec!["main".to_string()],
                passes: None,
                custom_uniform_size: None,
                has_input_texture: false,
                has_mouse: false,
                has_fonts: false,
                has_audio: false,
                has_atomic_buffer: false,
                audio_buffer_size: 1024,
                has_audio_spectrum: false,
                audio_spectrum_size: 128,
                storage_buffers: Vec::new(),
                workgroup_size: [16, 16, 1],
                dispatch_once: false,
                texture_format: wgpu::TextureFormat::Rgba16Float,
                label: "Compute Shader".to_string(),
                num_channels: None,
            },
        }
    }

    /// Set the entry point for single-pass shaders
    pub fn with_entry_point(mut self, entry_point: &str) -> Self {
        self.config.entry_points = vec![entry_point.to_string()];
        self
    }

    /// Configure multi-pass execution with ping-pong buffers
    pub fn with_multi_pass(mut self, passes: &[PassDescription]) -> Self {
        self.config.passes = Some(passes.to_vec());
        self.config.entry_points = passes.iter().map(|p| p.name.clone()).collect();
        self
    }

    /// Add custom uniform parameters (goes to @group(1))
    pub fn with_custom_uniforms<T: UniformProvider>(mut self) -> Self {
        self.config.custom_uniform_size = Some(std::mem::size_of::<T>() as u64);
        self
    }

    /// Enable input texture support (goes to @group(1))
    pub fn with_input_texture(mut self) -> Self {
        self.config.has_input_texture = true;
        self
    }

    /// Enable channel textures for external media (goes to @group(2))
    pub fn with_channels(mut self, num_channels: u32) -> Self {
        self.config.num_channels = Some(num_channels);
        self
    }

    /// Enable mouse input (goes to @group(2))
    pub fn with_mouse(mut self) -> Self {
        self.config.has_mouse = true;
        self
    }

    /// Enable font rendering (goes to @group(2))
    pub fn with_fonts(mut self) -> Self {
        self.config.has_fonts = true;
        self
    }

    /// Enable audio buffer (goes to @group(2))
    pub fn with_audio(mut self, buffer_size: usize) -> Self {
        self.config.has_audio = true;
        self.config.audio_buffer_size = buffer_size;
        self
    }

    /// Enable audio spectrum data buffer for visualizers (goes to @group(2))
    pub fn with_audio_spectrum(mut self, spectrum_size: usize) -> Self {
        self.config.has_audio_spectrum = true;
        self.config.audio_spectrum_size = spectrum_size;
        self
    }

    /// Enable atomic buffer for particle systems (goes to @group(2))
    pub fn with_atomic_buffer(mut self) -> Self {
        self.config.has_atomic_buffer = true;
        self
    }

    /// Add user-defined storage buffers (goes to @group(3))
    pub fn with_storage_buffer(mut self, buffer: StorageBufferSpec) -> Self {
        self.config.storage_buffers.push(buffer);
        self
    }

    /// Add multiple storage buffers
    pub fn with_storage_buffers(mut self, buffers: &[StorageBufferSpec]) -> Self {
        self.config.storage_buffers.extend_from_slice(buffers);
        self
    }

    /// Set workgroup size
    pub fn with_workgroup_size(mut self, size: [u32; 3]) -> Self {
        self.config.workgroup_size = size;
        self
    }

    /// Run only once (for initialization shaders)
    pub fn dispatch_once(mut self) -> Self {
        self.config.dispatch_once = true;
        self
    }

    /// Set output texture format
    pub fn with_texture_format(mut self, format: wgpu::TextureFormat) -> Self {
        self.config.texture_format = format;
        self
    }

    /// Set debug label
    pub fn with_label(mut self, label: &str) -> Self {
        self.config.label = label.to_string();
        self
    }

    /// Build the configuration (will be used by ComputeShader::from_builder)
    pub fn build(self) -> ComputeConfiguration {
        self.config
    }
}

impl Default for ComputeShaderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

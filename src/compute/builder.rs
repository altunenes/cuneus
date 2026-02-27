use crate::UniformProvider;
use std::path::PathBuf;
use wgpu;

/// A single pass in a multi-pass compute pipeline.
///
/// Each pass corresponds to a WGSL entry point (`@compute @workgroup_size(...) fn name(...)`)
/// and declares which other passes it reads from via `inputs`.
///
/// # Dependency mapping
///
/// The `inputs` slice maps **sequentially** to Group 3 input textures in your WGSL shader:
///
/// - `inputs[0]` → `@group(3) @binding(0) var input_texture0`
/// - `inputs[1]` → `@group(3) @binding(2) var input_texture1`
/// - `inputs[2]` → `@group(3) @binding(4) var input_texture2`
///
/// If fewer than 3 dependencies are listed, the remaining slots repeat `inputs[0]`.
///
/// # Cross-frame feedback
///
/// Including a pass's own name in its inputs enables automatic temporal feedback.
/// The engine's per-buffer ping-pong tracking ensures you read the **previous frame's**
/// output while writing to the current frame:
///
/// ```rust,ignore
/// // buffer_a reads its own previous frame (temporal accumulation)
/// PassDescription::new("buffer_a", &["buffer_a"])
///
/// // buffer_c reads buffer_b (current frame) + its own previous frame
/// PassDescription::new("buffer_c", &["buffer_b", "buffer_c"])
/// ```
///
/// # Non-adjacent reads
///
/// Any pass can read from any other pass regardless of ordering distance:
///
/// ```rust,ignore
/// PassDescription::new("lic_edges", &["tensor_field", "kuwahara_filter"])
/// // Works even though tensor_field is 2 passes earlier
/// ```
#[derive(Debug, Clone)]
pub struct PassDescription {
    /// The WGSL entry point name for this pass (e.g., `"buffer_a"`, `"main_image"`).
    pub name: String,
    /// Buffer names this pass reads from, mapped by position to `input_texture0..2` in WGSL.
    pub inputs: Vec<String>,
    /// Optional per-pass dispatch count override. If `None`, the engine computes
    /// dispatch dimensions from `screen_size / builder_workgroup_size`.
    /// If `Some`, the value is passed directly to `dispatch_workgroups(x, y, z)`.
    pub workgroup_size: Option<[u32; 3]>,
}

impl PassDescription {
    /// Create a new pass description.
    ///
    /// - `name`: the WGSL `@compute` entry point name.
    /// - `inputs`: buffer names this pass reads from (max 3). Order determines WGSL binding:
    ///   `inputs[0]` → `input_texture0`, `inputs[1]` → `input_texture1`, etc.
    ///   Use `&[]` for passes with no dependencies (e.g., the first pass reading external input).
    pub fn new(name: &str, inputs: &[&str]) -> Self {
        Self {
            name: name.to_string(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            workgroup_size: None,
        }
    }

    /// Override the dispatch dimensions for this specific pass.
    ///
    /// When set, the value is passed directly to `dispatch_workgroups(x, y, z)`,
    /// bypassing the default `screen_size / workgroup_size` calculation.
    /// Useful for passes that don't operate on screen-sized data (e.g., CNN layers, FFT butterflies).
    pub fn with_workgroup_size(mut self, size: [u32; 3]) -> Self {
        self.workgroup_size = Some(size);
        self
    }
}

/// Specification for a user-defined storage buffer bound to Group 3.
///
/// Use this instead of (or alongside) multi-pass texture ping-pong when your algorithm
/// needs shared read-write memory across passes (e.g., FFT twiddle factors, CNN weights).
#[derive(Debug, Clone)]
pub struct StorageBufferSpec {
    pub name: String,
    pub size_bytes: u64,
}

impl StorageBufferSpec {
    pub fn new(name: &str, size_bytes: u64) -> Self {
        Self {
            name: name.to_string(),
            size_bytes,
        }
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
    pub hot_reload_path: Option<PathBuf>,
}

/// Declarative builder for compute shader pipelines.
///
/// Configures everything your shader needs — the engine handles all bind group layouts,
/// pipeline creation, ping-pong buffers, and hot reload wiring.
///
/// # Bind group convention
///
/// | Group | Contents | Builder methods |
/// |-------|----------|-----------------|
/// | 0 | Time / frame data | Always present |
/// | 1 | Output texture, custom uniforms, input texture | [`with_custom_uniforms`], [`with_input_texture`] |
/// | 2 | Mouse, fonts, audio, atomics, channels | [`with_mouse`], [`with_fonts`], [`with_audio`], [`with_channels`], etc. |
/// | 3 | Multi-pass input textures **or** storage buffers | [`with_multi_pass`], [`with_storage_buffer`] |
///
/// Group 2 bindings are **dynamic** — resources are assigned in a fixed order
/// (mouse → fonts → audio → audio_spectrum → atomics → channels) and only the
/// ones you enable get binding slots.
///
/// # Example
///
/// ```rust,ignore
/// let config = ComputeShader::builder()
///     .with_multi_pass(&passes)
///     .with_custom_uniforms::<MyParams>()
///     .with_mouse()
///     .with_channels(1)
///     .build();
/// ```
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
                hot_reload_path: None,
            },
        }
    }

    /// Set the WGSL entry point for single-pass shaders.
    /// For multi-pass, use [`with_multi_pass`] instead (it overrides the entry points set here).
    pub fn with_entry_point(mut self, entry_point: &str) -> Self {
        self.config.entry_points = vec![entry_point.to_string()];
        self
    }

    /// Configure a multi-pass pipeline with automatic ping-pong double-buffering.
    ///
    /// Each [`PassDescription`] declares a WGSL entry point and its input dependencies.
    /// The engine creates per-buffer texture pairs, tracks write sides independently,
    /// and rebuilds Group 3 bind groups every frame so each pass sees the correct inputs.
    ///
    /// A single `.dispatch()` call runs all passes in order.
    /// Overrides any entry point set by [`with_entry_point`].
    pub fn with_multi_pass(mut self, passes: &[PassDescription]) -> Self {
        self.config.passes = Some(passes.to_vec());
        self.config.entry_points = passes.iter().map(|p| p.name.clone()).collect();
        self
    }

    /// Register a custom uniform struct at `@group(1) @binding(1)`.
    ///
    /// The struct must implement [`UniformProvider`] (use `cuneus::uniform_params!` for this).
    /// Update values at runtime with `compute_shader.set_custom_params(params, &queue)`.
    pub fn with_custom_uniforms<T: UniformProvider>(mut self) -> Self {
        self.config.custom_uniform_size = Some(std::mem::size_of::<T>() as u64);
        self
    }

    /// Enable a single input texture in Group 1 (texture + sampler, 2 bindings after output and
    /// optional custom uniform).
    ///
    /// **Multi-pass note:** with `.dispatch()`, this is only available in the `main_image` pass.
    /// Use `.dispatch_stage()` or `.with_channels()` if you need it in intermediate passes.
    pub fn with_input_texture(mut self) -> Self {
        self.config.has_input_texture = true;
        self
    }

    /// Enable `N` external texture channels in Group 2 (video, webcam, HDRI).
    ///
    /// Each channel occupies 2 bindings (texture + sampler). Unlike `with_input_texture`,
    /// channels are accessible from **all** passes in both `.dispatch()` and `.dispatch_stage()`.
    pub fn with_channels(mut self, num_channels: u32) -> Self {
        self.config.num_channels = Some(num_channels);
        self
    }

    /// Enable mouse uniform in Group 2. Access as `var<uniform> mouse: MouseUniform` in WGSL.
    pub fn with_mouse(mut self) -> Self {
        self.config.has_mouse = true;
        self
    }

    /// Enable font texture + uniform in Group 2 (2 bindings).
    pub fn with_fonts(mut self) -> Self {
        self.config.has_fonts = true;
        self
    }

    /// Enable a read-write audio buffer in Group 2 for GPU audio synthesis.
    ///
    /// The buffer is `storage, read_write` — your shader writes synthesis parameters,
    /// the CPU reads them back for playback. Also usable as generic persistent storage.
    pub fn with_audio(mut self, buffer_size: usize) -> Self {
        self.config.has_audio = true;
        self.config.audio_buffer_size = buffer_size;
        self
    }

    /// Enable a read-only audio spectrum buffer in Group 2 for visualization.
    ///
    /// Fed by GStreamer's spectrum analyzer from loaded media files.
    /// Indices 0..63 are frequency bands; 64=BPM, 65=bass, 66=mid, 67=high, 68=total energy.
    pub fn with_audio_spectrum(mut self, spectrum_size: usize) -> Self {
        self.config.has_audio_spectrum = true;
        self.config.audio_spectrum_size = spectrum_size;
        self
    }

    /// Enable an atomic `u32` buffer in Group 2 for lock-free GPU algorithms (histograms, particle systems).
    pub fn with_atomic_buffer(mut self) -> Self {
        self.config.has_atomic_buffer = true;
        self
    }

    /// Add a user-defined storage buffer to Group 3.
    ///
    /// When used with `.with_multi_pass()`, the passes share these buffers
    /// instead of using texture ping-pong for Group 3.
    pub fn with_storage_buffer(mut self, buffer: StorageBufferSpec) -> Self {
        self.config.storage_buffers.push(buffer);
        self
    }

    /// Add multiple storage buffers to Group 3 at once.
    pub fn with_storage_buffers(mut self, buffers: &[StorageBufferSpec]) -> Self {
        self.config.storage_buffers.extend_from_slice(buffers);
        self
    }

    /// Set the workgroup size `[x, y, z]` used to calculate dispatch dimensions.
    ///
    /// The engine dispatches `ceil(screen_width / x)` by `ceil(screen_height / y)` workgroups.
    /// This value should match the `@workgroup_size()` in your WGSL shader.
    /// For multi-pass, use [`PassDescription::with_workgroup_size`] for per-pass overrides.
    pub fn with_workgroup_size(mut self, size: [u32; 3]) -> Self {
        self.config.workgroup_size = size;
        self
    }

    /// Run the pipeline only once (useful for initialization or precomputation shaders).
    pub fn dispatch_once(mut self) -> Self {
        self.config.dispatch_once = true;
        self
    }

    /// Set the output texture format. Default is `Rgba16Float`.
    pub fn with_texture_format(mut self, format: wgpu::TextureFormat) -> Self {
        self.config.texture_format = format;
        self
    }

    /// Set a debug label (visible in GPU debuggers like RenderDoc).
    pub fn with_label(mut self, label: &str) -> Self {
        self.config.label = label.to_string();
        self
    }

    /// Enable hot reload by watching a shader file for changes.
    /// Note: the `compute_shader!` macro sets this automatically.
    pub fn with_hot_reload(mut self, shader_path: &str) -> Self {
        self.config.hot_reload_path = Some(PathBuf::from(shader_path));
        self
    }

    /// Consume the builder and return the final [`ComputeConfiguration`].
    pub fn build(self) -> ComputeConfiguration {
        self.config
    }
}

impl Default for ComputeShaderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

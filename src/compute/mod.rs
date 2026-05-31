// @group(0): Per-Frame Resources (TimeUniform)
// @group(1): Primary Pass I/O & Parameters (output texture, shader params, input textures)
// @group(2): Global Engine Resources (fonts, audio, atomics, mouse)
// @group(3): User-Defined Data Buffers (custom storage buffers)

pub mod builder;
pub mod core;
pub mod multipass;
pub mod resource;

pub use builder::*;
pub use core::*;
pub use multipass::*;
pub use resource::*;

// Texture format constants
pub const COMPUTE_TEXTURE_FORMAT_RGBA16: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const COMPUTE_TEXTURE_FORMAT_RGBA8: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

use crate::Core;

/// Main entry point for creating compute shaders
impl ComputeShader {
    /// Create a compute shader using the builder pattern.
    /// This is the primary API for all compute shader creation.
    ///
    /// # Example
    ///
    /// Single-pass configuration:
    ///
    /// ```no_run
    /// # use cuneus::compute::ComputeShader;
    /// # use cuneus::UniformProvider;
    /// # cuneus::uniform_params! { struct MyParams { x: f32, _pad: [f32; 3] } }
    /// let config = ComputeShader::builder()
    ///     .with_entry_point("main")
    ///     .with_custom_uniforms::<MyParams>()
    ///     .build();
    /// ```
    ///
    /// Multi-pass configuration. A pass that names itself in its `inputs`
    /// reads its own previous-frame output:
    ///
    /// ```no_run
    /// # use cuneus::compute::{ComputeShader, PassDescription};
    /// # use cuneus::UniformProvider;
    /// # cuneus::uniform_params! { struct MyParams { x: f32, _pad: [f32; 3] } }
    /// let passes = vec![
    ///     PassDescription::new("simulate", &["simulate"]),
    ///     PassDescription::new("main_image", &["simulate"]),
    /// ];
    /// let config = ComputeShader::builder()
    ///     .with_multi_pass(&passes)
    ///     .with_custom_uniforms::<MyParams>()
    ///     .build();
    /// ```
    ///
    /// Construct the shader from the resulting config with the
    /// [`compute_shader!`](crate::compute_shader) macro.
    ///
    /// # Binding layout
    ///
    /// The builder produces the following 4-group layout (referenced from your WGSL):
    /// - `@group(0)` — Per-frame data (`TimeUniform`)
    /// - `@group(1)` — Output texture (and custom uniform if `with_custom_uniforms` is set)
    /// - `@group(2)` — Engine resources enabled via `.with_*` methods (mouse, fonts, audio, atomics, channels)
    /// - `@group(3)` — Multi-pass input textures or user storage buffers
    ///
    /// See [`ComputeShaderBuilder`] for the full list of configuration methods,
    /// and `usage.md` in the repo root for additional patterns.
    pub fn builder() -> ComputeShaderBuilder {
        ComputeShaderBuilder::new()
    }

    /// Create a simple compute shader with basic configuration
    pub fn new(core: &Core, shader_source: &str) -> Self {
        let config = ComputeShaderBuilder::new()
            .with_label("Simple Compute Shader")
            .build();

        Self::from_builder(core, shader_source, config)
    }

    /// Create a compute shader with custom uniform parameters
    pub fn with_uniforms<T: crate::UniformProvider>(
        core: &Core,
        shader_source: &str,
        label: &str,
    ) -> Self {
        let config = ComputeShaderBuilder::new()
            .with_custom_uniforms::<T>()
            .with_label(label)
            .build();

        Self::from_builder(core, shader_source, config)
    }

    /// Create a multi-pass compute shader
    pub fn with_multi_pass(
        core: &Core,
        shader_source: &str,
        passes: &[PassDescription],
        label: &str,
    ) -> Self {
        let config = ComputeShaderBuilder::new()
            .with_multi_pass(passes)
            .with_label(label)
            .build();

        Self::from_builder(core, shader_source, config)
    }
}

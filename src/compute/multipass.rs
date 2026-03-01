use crate::Core;
use std::collections::HashMap;
use wgpu;

/// Manages ping-pong buffers for multi-pass compute shaders.
///
/// Each buffer independently tracks which side (.0 or .1) was last written.
/// This means any pass can read from any previous pass's output, regardless of
/// how many passes have elapsed. The old global-flip approach only allowed
/// reading from the immediately preceding pass.
pub struct MultiPassManager {
    buffers: HashMap<String, (wgpu::Texture, wgpu::Texture)>,
    bind_groups: HashMap<String, (wgpu::BindGroup, wgpu::BindGroup)>,
    /// Per-buffer write-side tracking. `true` means the last write went to `.0`,
    /// so the next write goes to `.1` and reads return `.0`.
    write_side: HashMap<String, bool>,
    output_texture: wgpu::Texture,
    output_bind_group: wgpu::BindGroup,
    storage_layout: wgpu::BindGroupLayout,
    input_layout: wgpu::BindGroupLayout,
    width: u32,
    height: u32,
    texture_format: wgpu::TextureFormat,
}

/// Note: storage layout currently un-used. I try to create our own storage-only layout
impl MultiPassManager {
    pub fn new(
        core: &Core,
        buffer_names: &[String],
        texture_format: wgpu::TextureFormat,
        _storage_layout: wgpu::BindGroupLayout,
    ) -> Self {
        let width = core.size.width;
        let height = core.size.height;

        // Create dedicated storage layout (only storage texture, no custom uniform)
        let storage_layout =
            core.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Multi-Pass Storage Layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: texture_format,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    }],
                });

        // Create input texture layout for multi-buffer reading
        let input_layout = Self::create_input_layout(&core.device);

        let mut buffers = HashMap::new();
        let mut bind_groups = HashMap::new();

        // Create ping-pong texture pairs for each buffer
        for name in buffer_names {
            let texture0 = Self::create_storage_texture(
                &core.device,
                width,
                height,
                texture_format,
                &format!("{name}_0"),
            );
            let texture1 = Self::create_storage_texture(
                &core.device,
                width,
                height,
                texture_format,
                &format!("{name}_1"),
            );

            let bind_group0 = Self::create_storage_bind_group(
                &core.device,
                &storage_layout,
                &texture0,
                &format!("{name}_0_bind"),
            );
            let bind_group1 = Self::create_storage_bind_group(
                &core.device,
                &storage_layout,
                &texture1,
                &format!("{name}_1_bind"),
            );

            buffers.insert(name.clone(), (texture0, texture1));
            bind_groups.insert(name.clone(), (bind_group0, bind_group1));
        }

        // Create output texture
        let output_texture = Self::create_storage_texture(
            &core.device,
            width,
            height,
            texture_format,
            "multipass_output",
        );
        let output_bind_group = Self::create_storage_bind_group(
            &core.device,
            &storage_layout,
            &output_texture,
            "output_bind",
        );

        let mut write_side = HashMap::new();
        for name in buffer_names {
            write_side.insert(name.clone(), false);
        }

        Self {
            buffers,
            bind_groups,
            write_side,
            output_texture,
            output_bind_group,
            storage_layout,
            input_layout,
            width,
            height,
            texture_format,
        }
    }

    fn create_storage_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        label: &str,
    ) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    fn create_storage_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        texture: &wgpu::Texture,
        label: &str,
    ) -> wgpu::BindGroup {
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
            label: Some(label),
        })
    }

    fn create_input_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: Some("Multi-Pass Input Layout"),
        })
    }

    /// Get the write bind group for a buffer (writes to the side not last written)
    pub fn get_write_bind_group(&self, buffer_name: &str) -> &wgpu::BindGroup {
        let bind_groups = self.bind_groups.get(buffer_name).expect("Buffer not found");
        let last_wrote_0 = self.write_side.get(buffer_name).copied().unwrap_or(false);
        if last_wrote_0 {
            &bind_groups.1 // Last write was .0, next write goes to .1
        } else {
            &bind_groups.0 // Last write was .1 (or never), next write goes to .0
        }
    }

    /// Get the write texture for a buffer (writes to the side not last written)
    pub fn get_write_texture(&self, buffer_name: &str) -> &wgpu::Texture {
        let textures = self.buffers.get(buffer_name).expect("Buffer not found");
        let last_wrote_0 = self.write_side.get(buffer_name).copied().unwrap_or(false);
        if last_wrote_0 {
            &textures.1
        } else {
            &textures.0
        }
    }

    /// Get the read texture for a buffer (returns the side that was last written)
    pub fn get_read_texture(&self, buffer_name: &str) -> &wgpu::Texture {
        let textures = self.buffers.get(buffer_name).expect("Buffer not found");
        let last_wrote_0 = self.write_side.get(buffer_name).copied().unwrap_or(false);
        if last_wrote_0 {
            &textures.0 // Last write was to .0, read from .0
        } else {
            &textures.1 // Last write was to .1, read from .1
        }
    }


    /// Get output bind group
    pub fn get_output_bind_group(&self) -> &wgpu::BindGroup {
        &self.output_bind_group
    }

    /// Get output texture
    pub fn get_output_texture(&self) -> &wgpu::Texture {
        &self.output_texture
    }

    /// Mark a specific buffer as having been written to.
    /// Flips that buffer's write side so the next read returns what was just written,
    /// and the next write goes to the other side.
    pub fn mark_written(&mut self, buffer_name: &str) {
        if let Some(side) = self.write_side.get_mut(buffer_name) {
            *side = !*side;
        }
    }

    /// Flip all buffers (for cross-frame feedback in temporal effects).
    /// Call this after frame presentation to preserve state for the next frame.
    pub fn flip_buffers(&mut self) {
        for side in self.write_side.values_mut() {
            *side = !*side;
        }
    }

    /// Clear all buffers
    pub fn clear_all(&mut self, core: &Core) {
        // Recreate all buffer textures
        for (name, textures) in &mut self.buffers {
            textures.0 = Self::create_storage_texture(
                &core.device,
                self.width,
                self.height,
                self.texture_format,
                &format!("{name}_0"),
            );
            textures.1 = Self::create_storage_texture(
                &core.device,
                self.width,
                self.height,
                self.texture_format,
                &format!("{name}_1"),
            );
        }

        // Recreate all bind groups
        for (name, bind_groups) in &mut self.bind_groups {
            let textures = self.buffers.get(name).unwrap();
            bind_groups.0 = Self::create_storage_bind_group(
                &core.device,
                &self.storage_layout,
                &textures.0,
                &format!("{name}_0_bind"),
            );
            bind_groups.1 = Self::create_storage_bind_group(
                &core.device,
                &self.storage_layout,
                &textures.1,
                &format!("{name}_1_bind"),
            );
        }

        // Recreate output texture and bind group
        self.output_texture = Self::create_storage_texture(
            &core.device,
            self.width,
            self.height,
            self.texture_format,
            "multipass_output",
        );
        self.output_bind_group = Self::create_storage_bind_group(
            &core.device,
            &self.storage_layout,
            &self.output_texture,
            "output_bind",
        );

        for side in self.write_side.values_mut() {
            *side = false;
        }
    }

    /// Resize all buffers
    pub fn resize(&mut self, core: &Core, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.clear_all(core);
    }

    /// Get the input layout for pipeline creation
    pub fn get_input_layout(&self) -> &wgpu::BindGroupLayout {
        &self.input_layout
    }

    /// Get the storage layout for pipeline creation
    pub fn get_storage_layout(&self) -> &wgpu::BindGroupLayout {
        &self.storage_layout
    }

    /// Get the write_side state for a buffer
    pub fn get_write_side(&self, buffer_name: &str) -> bool {
        self.write_side.get(buffer_name).copied().unwrap_or(false)
    }

    /// Get both ping-pong textures for a buffer
    pub fn get_buffer_pair(&self, buffer_name: &str) -> Option<&(wgpu::Texture, wgpu::Texture)> {
        self.buffers.get(buffer_name)
    }

    /// Get the first buffer name (for passes with no dependencies)
    pub fn first_buffer_name(&self) -> Option<&String> {
        self.buffers.keys().next()
    }
}

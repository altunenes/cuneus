use crate::Core;
use std::collections::HashMap;
use wgpu;

/// Manages ping-pong buffers for multi-pass compute shaders
pub struct MultiPassManager {
    buffers: HashMap<String, (wgpu::Texture, wgpu::Texture)>,
    bind_groups: HashMap<String, (wgpu::BindGroup, wgpu::BindGroup)>,
    output_texture: wgpu::Texture,
    output_bind_group: wgpu::BindGroup,
    storage_layout: wgpu::BindGroupLayout,
    input_layout: wgpu::BindGroupLayout,
    frame_flip: bool,
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
                &format!("{}_0", name),
            );
            let texture1 = Self::create_storage_texture(
                &core.device,
                width,
                height,
                texture_format,
                &format!("{}_1", name),
            );

            let bind_group0 = Self::create_storage_bind_group(
                &core.device,
                &storage_layout,
                &texture0,
                &format!("{}_0_bind", name),
            );
            let bind_group1 = Self::create_storage_bind_group(
                &core.device,
                &storage_layout,
                &texture1,
                &format!("{}_1_bind", name),
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

        Self {
            buffers,
            bind_groups,
            output_texture,
            output_bind_group,
            storage_layout,
            input_layout,
            frame_flip: false,
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

    /// Get the write bind group for current frame
    pub fn get_write_bind_group(&self, buffer_name: &str) -> &wgpu::BindGroup {
        let bind_groups = self.bind_groups.get(buffer_name).expect("Buffer not found");
        if self.frame_flip {
            &bind_groups.1
        } else {
            &bind_groups.0
        }
    }

    /// Get the write texture for current frame
    pub fn get_write_texture(&self, buffer_name: &str) -> &wgpu::Texture {
        let textures = self.buffers.get(buffer_name).expect("Buffer not found");
        if self.frame_flip {
            &textures.1
        } else {
            &textures.0
        }
    }

    /// Get the read texture for previous frame
    pub fn get_read_texture(&self, buffer_name: &str) -> &wgpu::Texture {
        let textures = self.buffers.get(buffer_name).expect("Buffer not found");
        if self.frame_flip {
            &textures.0
        } else {
            &textures.1
        }
    }

    /// Create input bind group for a pass with its dependencies
    pub fn create_input_bind_group(
        &self,
        device: &wgpu::Device,
        sampler: &wgpu::Sampler,
        input_buffers: &[String],
    ) -> wgpu::BindGroup {
        let mut views = Vec::new();

        // Create views for up to 3 input textures
        for i in 0..3 {
            let buffer_name = if input_buffers.is_empty() {
                // For first pass with no dependencies, use the first buffer or create a dummy
                self.buffers
                    .keys()
                    .next()
                    .cloned()
                    .unwrap_or_else(|| "buffer_a".to_string())
            } else {
                input_buffers.get(i).unwrap_or(&input_buffers[0]).clone()
            };
            let texture = self.get_read_texture(&buffer_name);
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            views.push(view);
        }

        let entries = [
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&views[0]),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&views[1]),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&views[2]),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ];

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.input_layout,
            entries: &entries,
            label: Some("Multi-Pass Input"),
        })
    }

    /// Get output bind group
    pub fn get_output_bind_group(&self) -> &wgpu::BindGroup {
        &self.output_bind_group
    }

    /// Get output texture
    pub fn get_output_texture(&self) -> &wgpu::Texture {
        &self.output_texture
    }

    /// Flip ping-pong buffers
    pub fn flip_buffers(&mut self) {
        self.frame_flip = !self.frame_flip;
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
                &format!("{}_0", name),
            );
            textures.1 = Self::create_storage_texture(
                &core.device,
                self.width,
                self.height,
                self.texture_format,
                &format!("{}_1", name),
            );
        }

        // Recreate all bind groups
        for (name, bind_groups) in &mut self.bind_groups {
            let textures = self.buffers.get(name).unwrap();
            bind_groups.0 = Self::create_storage_bind_group(
                &core.device,
                &self.storage_layout,
                &textures.0,
                &format!("{}_0_bind", name),
            );
            bind_groups.1 = Self::create_storage_bind_group(
                &core.device,
                &self.storage_layout,
                &textures.1,
                &format!("{}_1_bind", name),
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

        self.frame_flip = false;
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
}

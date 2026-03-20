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
    /// Default screen dimensions
    width: u32,
    height: u32,
    /// Per-buffer dimensions (may differ from screen size)
    buffer_dimensions: HashMap<String, (u32, u32)>,
    /// Per-buffer resolution cfgs: Some(absolute) or None (use scale or screen size)
    buffer_resolution: HashMap<String, Option<[u32; 2]>>,
    /// Per-buffer scale factors relative to screen size
    buffer_scale: HashMap<String, Option<f32>>,
    texture_format: wgpu::TextureFormat,
    max_input_deps: usize,
}

/// Note: storage layout currently un-used. I try to create our own storage-only layout
impl MultiPassManager {
    pub fn new(
        core: &Core,
        buffer_names: &[String],
        texture_format: wgpu::TextureFormat,
        _storage_layout: wgpu::BindGroupLayout,
        max_input_deps: usize,
        passes: &[crate::compute::PassDescription],
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
        let input_layout = Self::create_input_layout(&core.device, max_input_deps);

        // Build per-buffer resolution config from pass descriptions
        let mut buffer_resolution: HashMap<String, Option<[u32; 2]>> = HashMap::new();
        let mut buffer_scale: HashMap<String, Option<f32>> = HashMap::new();
        let mut buffer_dimensions: HashMap<String, (u32, u32)> = HashMap::new();

        for pass in passes {
            buffer_resolution.insert(pass.name.clone(), pass.resolution);
            buffer_scale.insert(pass.name.clone(), pass.resolution_scale);

            let (bw, bh) = Self::compute_buffer_dims(
                width, height, pass.resolution, pass.resolution_scale,
            );
            buffer_dimensions.insert(pass.name.clone(), (bw, bh));
        }

        let mut buffers = HashMap::new();
        let mut bind_groups = HashMap::new();

        // Create ping-pong texture pairs for each buffer at its own resolution
        for name in buffer_names {
            let (bw, bh) = buffer_dimensions.get(name).copied().unwrap_or((width, height));

            let texture0 = Self::create_storage_texture(
                &core.device,
                bw,
                bh,
                texture_format,
                &format!("{name}_0"),
            );
            let texture1 = Self::create_storage_texture(
                &core.device,
                bw,
                bh,
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
            buffer_dimensions,
            buffer_resolution,
            buffer_scale,
            texture_format,
            max_input_deps,
        }
    }

    /// Compute buffer dimensions from resolution config
    fn compute_buffer_dims(
        screen_w: u32,
        screen_h: u32,
        resolution: Option<[u32; 2]>,
        scale: Option<f32>,
    ) -> (u32, u32) {
        if let Some([w, h]) = resolution {
            // Absolute resolution takes precedence
            (w.max(1), h.max(1))
        } else if let Some(s) = scale {
            // Scale relative to screen
            ((screen_w as f32 * s).round().max(1.0) as u32,
             (screen_h as f32 * s).round().max(1.0) as u32)
        } else {
            // Default: match screen
            (screen_w, screen_h)
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

    fn create_input_layout(device: &wgpu::Device, max_input_deps: usize) -> wgpu::BindGroupLayout {
        let mut entries = Vec::with_capacity(max_input_deps * 2);
        for i in 0..max_input_deps {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: (i * 2) as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            });
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: (i * 2 + 1) as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });
        }
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &entries,
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
        let names: Vec<String> = self.buffers.keys().cloned().collect();

        // Recreate all buffer textures at their respective dimensions
        for name in &names {
            let (bw, bh) = self.buffer_dimensions.get(name).copied()
                .unwrap_or((self.width, self.height));

            let texture0 = Self::create_storage_texture(
                &core.device, bw, bh, self.texture_format, &format!("{name}_0"),
            );
            let texture1 = Self::create_storage_texture(
                &core.device, bw, bh, self.texture_format, &format!("{name}_1"),
            );

            let bind_group0 = Self::create_storage_bind_group(
                &core.device, &self.storage_layout, &texture0, &format!("{name}_0_bind"),
            );
            let bind_group1 = Self::create_storage_bind_group(
                &core.device, &self.storage_layout, &texture1, &format!("{name}_1_bind"),
            );

            self.buffers.insert(name.clone(), (texture0, texture1));
            self.bind_groups.insert(name.clone(), (bind_group0, bind_group1));
        }

        // Recreate output texture and bind group (always at screen resolution)
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

    /// Resize all buffers (recomputes scaled dimensions from new screen size)
    pub fn resize(&mut self, core: &Core, width: u32, height: u32) {
        self.width = width;
        self.height = height;

        // Recompute per-buffer dimensions based on new screen size
        let names: Vec<String> = self.buffer_dimensions.keys().cloned().collect();
        for name in names {
            let resolution = self.buffer_resolution.get(&name).copied().flatten();
            let scale = self.buffer_scale.get(&name).copied().flatten();
            let (bw, bh) = Self::compute_buffer_dims(width, height, resolution, scale);
            self.buffer_dimensions.insert(name, (bw, bh));
        }

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

    /// Get the maximum number of input dependencies per pass
    pub fn max_input_deps(&self) -> usize {
        self.max_input_deps
    }

    /// Get the dimensions of a specific buffer (may differ from screen size)
    pub fn get_buffer_dimensions(&self, buffer_name: &str) -> (u32, u32) {
        self.buffer_dimensions.get(buffer_name).copied()
            .unwrap_or((self.width, self.height))
    }
}

use crate::compute::ComputeShader;
use crate::radix_sort::RadixSorter;
use crate::{Core, ExportSettings};

/// GPU Sorter for Gaussian depth ordering
pub struct GaussianSorter {
    radix_sorter: RadixSorter,
    bind_group: Option<wgpu::BindGroup>,
    aux_keys: Option<wgpu::Buffer>,
    aux_payload: Option<wgpu::Buffer>,
    internal_buffer: Option<wgpu::Buffer>,
    state_buffer: Option<wgpu::Buffer>,
    current_count: u32,
    last_camera_forward: Option<[f32; 3]>,
}

impl GaussianSorter {
    /// Create a new GaussianSorter (32-bit radix sort, 4 passes)
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            radix_sorter: RadixSorter::new(device),
            bind_group: None,
            aux_keys: None,
            aux_payload: None,
            internal_buffer: None,
            state_buffer: None,
            current_count: 0,
            last_camera_forward: None,
        }
    }

    /// 16-bit 2 passes (note fn new is 4 passes and 32)
    pub fn new_16bit(device: &wgpu::Device) -> Self {
        Self {
            radix_sorter: RadixSorter::new_16bit(device),
            bind_group: None,
            aux_keys: None,
            aux_payload: None,
            internal_buffer: None,
            state_buffer: None,
            current_count: 0,
            last_camera_forward: None,
        }
    }

    /// Prepare sorter for specific buffers
    /// This binds directly to the depth_keys and sorted_indices buffers
    pub fn prepare_with_buffers(
        &mut self,
        device: &wgpu::Device,
        depth_keys_buffer: &wgpu::Buffer,
        sorted_indices_buffer: &wgpu::Buffer,
        count: u32,
    ) {
        if self.current_count != count {
            let (state_buffer, aux_keys, aux_payload, internal_buffer, bind_group) =
                self.radix_sorter.create_direct_bind_group(
                    device,
                    depth_keys_buffer,
                    sorted_indices_buffer,
                    count,
                );
            self.bind_group = Some(bind_group);
            self.aux_keys = Some(aux_keys);
            self.aux_payload = Some(aux_payload);
            self.internal_buffer = Some(internal_buffer);
            self.state_buffer = Some(state_buffer);
            self.current_count = count;
        }
    }

    pub fn sort(&self, encoder: &mut wgpu::CommandEncoder, count: u32) {
        let Some(ref bind_group) = self.bind_group else {
            return;
        };

        self.radix_sorter.sort_with_bind_group(encoder, bind_group, count);
    }

    /// Check if sorting is needed based on camera forward vector change.
    /// Returns true if the camera has moved enough to warrant re-sorting.
    /// Updates internal state when returning true.
    pub fn needs_sort(&mut self, camera_forward: [f32; 3]) -> bool {
        if let Some(last) = self.last_camera_forward {
            let dot = last[0] * camera_forward[0]
                + last[1] * camera_forward[1]
                + last[2] * camera_forward[2];
            if dot > 0.9999 {
                return false;
            }
        }
        self.last_camera_forward = Some(camera_forward);
        true
    }

    /// Force a sort on the next frame (e.g. after loading new data)
    pub fn force_sort(&mut self) {
        self.last_camera_forward = None;
    }

    /// Get the current gaussian count this sorter is prepared for
    pub fn count(&self) -> u32 {
        self.current_count
    }
}


pub struct GaussianRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GaussianRenderer {
    /// Create a new GaussianRenderer
    ///
    /// The shader_source should contain `vs_main` and `fs_main` entry points.
    pub fn new(device: &wgpu::Device, texture_format: wgpu::TextureFormat, shader_source: &str) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Gaussian Render Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Gaussian Render Bind Group Layout"),
            entries: &[
                // Params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // cam uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Gaussian 2D data
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Sorted indices
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Gaussian Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Gaussian Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: texture_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    /// for rendering
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        params_buffer: &wgpu::Buffer,
        camera_buffer: &wgpu::Buffer,
        gaussian_2d_buffer: &wgpu::Buffer,
        sorted_indices_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gaussian Render Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: gaussian_2d_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: sorted_indices_buffer.as_entire_binding() },
            ],
        })
    }

    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// - `pass`: Active render pass
    /// - `bind_group`: Bind group created with `create_bind_group`
    /// - `count`: Number of gaussians to render
    pub fn render<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        bind_group: &'a wgpu::BindGroup,
        count: u32,
    ) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..6, 0..count);
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GaussianCamera {
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub viewport: [f32; 2],
    pub focal: [f32; 2],
}

impl GaussianCamera {
    pub fn from_orbit(
        yaw: f32,
        pitch: f32,
        distance: f32,
        target: [f32; 3],
        fov: f32,
        viewport: [f32; 2],
    ) -> Self {
        let (sy, cy) = (yaw.sin(), yaw.cos());
        let (sp, cp) = (pitch.sin(), pitch.cos());

        let pos = [
            target[0] + distance * cp * sy,
            target[1] + distance * sp,
            target[2] + distance * cp * cy,
        ];

        let f = [target[0] - pos[0], target[1] - pos[1], target[2] - pos[2]];
        let fl = (f[0]*f[0] + f[1]*f[1] + f[2]*f[2]).sqrt();
        let f = [f[0]/fl, f[1]/fl, f[2]/fl];

        let up = [0.0, 1.0, 0.0];
        let r = [f[1]*up[2] - f[2]*up[1], f[2]*up[0] - f[0]*up[2], f[0]*up[1] - f[1]*up[0]];
        let rl = (r[0]*r[0] + r[1]*r[1] + r[2]*r[2]).sqrt().max(0.0001);
        let r = [r[0]/rl, r[1]/rl, r[2]/rl];

        let u = [r[1]*f[2] - r[2]*f[1], r[2]*f[0] - r[0]*f[2], r[0]*f[1] - r[1]*f[0]];

        let tx = -(r[0]*pos[0] + r[1]*pos[1] + r[2]*pos[2]);
        let ty = -(u[0]*pos[0] + u[1]*pos[1] + u[2]*pos[2]);
        let tz = f[0]*pos[0] + f[1]*pos[1] + f[2]*pos[2];

        let view = [
            [r[0], u[0], -f[0], 0.0],
            [r[1], u[1], -f[1], 0.0],
            [r[2], u[2], -f[2], 0.0],
            [tx, ty, tz, 1.0],
        ];

        let aspect = viewport[0] / viewport[1];
        let focal_len = 1.0 / (fov / 2.0).tan();
        let (near, far) = (0.01, 1000.0); 
        let proj = [
            [focal_len / aspect, 0.0, 0.0, 0.0],
            [0.0, focal_len, 0.0, 0.0],
            [0.0, 0.0, (far + near) / (near - far), -1.0],
            [0.0, 0.0, (2.0 * far * near) / (near - far), 0.0],
        ];

        let focal = [
            focal_len * viewport[0] * 0.5,
            focal_len * viewport[1] * 0.5,
        ];

        Self { view, proj, viewport, focal }
    }
}

pub struct GaussianExporter;

impl GaussianExporter {
    /// Capture a single frame of gaussian rendering to CPU memory.
    ///
    /// preprocess → sort → render
    pub fn capture_frame(
        core: &Core,
        preprocess: &mut ComputeShader,
        sorter: &GaussianSorter,
        renderer: &GaussianRenderer,
        render_bind_group: &wgpu::BindGroup,
        count: u32,
        settings: &ExportSettings,
        texture_format: wgpu::TextureFormat,
    ) -> Result<Vec<u8>, wgpu::SurfaceError> {
        let capture_texture = core.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Gaussian Export Capture"),
            size: wgpu::Extent3d {
                width: settings.width,
                height: settings.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: texture_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let align = 256u32;
        let unpadded_bytes_per_row = settings.width * 4;
        let padding = (align - unpadded_bytes_per_row % align) % align;
        let padded_bytes_per_row = unpadded_bytes_per_row + padding;

        let output_buffer = core.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gaussian Export Buffer"),
            size: (padded_bytes_per_row * settings.height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let capture_view = capture_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Gaussian Export Encoder"),
        });

        if count > 0 {
            let workgroups = (count + 255) / 256;
            preprocess.dispatch_stage_with_workgroups(&mut encoder, 0, [workgroups, 1, 1]);
            sorter.sort(&mut encoder, count);
            encoder = core.flush_encoder(encoder);

            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Gaussian Export Render"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &capture_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    ..Default::default()
                });
                renderer.render(&mut pass, render_bind_group, count);
            }
        }

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &capture_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(settings.height),
                },
            },
            wgpu::Extent3d {
                width: settings.width,
                height: settings.height,
                depth_or_array_layers: 1,
            },
        );

        core.queue.submit(Some(encoder.finish()));

        let buffer_slice = output_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        let _ = core.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();

        let padded_data = buffer_slice.get_mapped_range().to_vec();
        let mut data = Vec::with_capacity((settings.width * settings.height * 4) as usize);
        for chunk in padded_data.chunks(padded_bytes_per_row as usize) {
            data.extend_from_slice(&chunk[..unpadded_bytes_per_row as usize]);
        }

        Ok(data)
    }

    /// Capture and save a single export frame.
    ///
    /// Convenience wrapper that calls `capture_frame` and then `save_frame`.
    /// The caller should update camera and time uniforms before calling this.
    pub fn export_frame(
        core: &Core,
        preprocess: &mut ComputeShader,
        sorter: &GaussianSorter,
        renderer: &GaussianRenderer,
        render_bind_group: &wgpu::BindGroup,
        count: u32,
        frame: u32,
        settings: &ExportSettings,
        texture_format: wgpu::TextureFormat,
    ) {
        match Self::capture_frame(
            core, preprocess, sorter, renderer,
            render_bind_group, count, settings, texture_format,
        ) {
            Ok(data) => {
                if let Err(e) = crate::save_frame(data, frame, settings) {
                    eprintln!("Error saving gaussian export frame {frame}: {e:?}");
                }
            }
            Err(e) => eprintln!("Error capturing gaussian export frame {frame}: {e}"),
        }
    }
}

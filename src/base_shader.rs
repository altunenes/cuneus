use std::time::Instant;
use egui_wgpu::ScreenDescriptor;
use egui::ViewportId;
use std::path::PathBuf;
use crate::{Core, Renderer, TextureManager, UniformProvider, UniformBinding,KeyInputHandler,ExportManager,ShaderControls,ControlsRequest,ResolutionUniform,VideoManager};
#[cfg(target_os = "macos")]
pub const CAPTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;
#[cfg(not(target_os = "macos"))]
pub const CAPTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TimeUniform {
    pub time: f32,
    pub frame: u32,
}
impl UniformProvider for TimeUniform {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}
pub struct BaseShader {
    pub renderer: Renderer,
    pub texture_manager: Option<TextureManager>,
    pub video_manager: Option<VideoManager>,
    pub last_media_path: Option<PathBuf>,
    pub egui_renderer: egui_wgpu::Renderer,
    pub egui_state: egui_winit::State,
    pub context: egui::Context,
    pub texture_bind_group_layout: wgpu::BindGroupLayout,
    pub start_time: Instant,
    pub time_uniform: UniformBinding<TimeUniform>,
    pub resolution_uniform: UniformBinding<ResolutionUniform>,
    pub key_handler: KeyInputHandler,
    pub export_manager: ExportManager,
    pub controls: ShaderControls,
}
impl BaseShader {
    pub fn new(
        core: &Core,
        vs_source: &str,
        fs_source: &str,
        bind_group_layouts: &[&wgpu::BindGroupLayout],
        fragment_entry: Option<&str>,
    ) -> Self {
        let time_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("time_bind_group_layout"),
        });
        let resolution_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("resolution_bind_group_layout"),
        });
        let time_uniform = UniformBinding::new(
            &core.device,
            "Time Uniform",
            TimeUniform { 
                time: 0.0,
                frame: 0,
            },
            &time_bind_group_layout,
            0,
        );
        let resolution_uniform = UniformBinding::new(
            &core.device,
            "Resolution Uniform",
            ResolutionUniform {
                dimensions: [core.size.width as f32, core.size.height as f32],
                _padding: [0.0; 2],
            },
            &resolution_bind_group_layout,
            0,
        );
        let vs_shader = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vertex Shader"),
            source: wgpu::ShaderSource::Wgsl(vs_source.into()),
        });
        let fs_shader = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fragment Shader"),
            source: wgpu::ShaderSource::Wgsl(fs_source.into()),
        });
        let texture_bind_group_layout = core.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            },
        );
        let pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts,
            push_constant_ranges: &[],
        });
        let renderer = Renderer::new(
            &core.device,
            &vs_shader,
            &fs_shader,
            core.config.format,
            &pipeline_layout,
            fragment_entry, 
        );
        let context = egui::Context::default();
        let egui_state = egui_winit::State::new(
            context.clone(),
            ViewportId::default(),
            core.window(),
            None,
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            &core.device,
            core.config.format,
            None,
            1,
            false,
        );

        //  default texture manager
        let texture_manager = Self::create_default_texture_manager(core, &texture_bind_group_layout);

        Self {
            renderer,
            texture_manager: Some(texture_manager),
            video_manager: None,
            last_media_path: None, 
            egui_renderer,
            egui_state,
            context,
            texture_bind_group_layout,
            start_time: Instant::now(),
            time_uniform,
            resolution_uniform,
            key_handler: KeyInputHandler::new(),
            export_manager: ExportManager::new(),
            controls: ShaderControls::new(),
        }
    }

    pub fn update_time(&mut self, queue: &wgpu::Queue) {
        self.time_uniform.data.time = self.start_time.elapsed().as_secs_f32();
        self.time_uniform.update(queue);
    }
    pub fn update_resolution(&mut self, queue: &wgpu::Queue, new_size: winit::dpi::PhysicalSize<u32>) {
        self.resolution_uniform.data.dimensions = [new_size.width as f32, new_size.height as f32];
        self.resolution_uniform.update(queue);
    }
    fn create_default_texture_manager(
        core: &Core,
        texture_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> TextureManager {
        let default_texture = core.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Default Texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let default_view = default_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = core.device.create_sampler(&wgpu::SamplerDescriptor::default());

        let bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&default_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("Default Texture Bind Group"),
        });

        TextureManager {
            texture: default_texture,
            view: default_view,
            sampler,
            bind_group,
        }
    }

    pub fn render_ui<F>(&mut self, core: &Core, mut ui_builder: F) -> egui::FullOutput 
    where
        F: FnMut(&egui::Context),
    {
        let raw_input = self.egui_state.take_egui_input(core.window());
        self.context.run(raw_input, |ctx| ui_builder(ctx))
    }

    pub fn handle_render_output(
        &mut self,
        core: &Core,
        view: &wgpu::TextureView,
        full_output: egui::FullOutput,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [core.config.width, core.config.height],
            pixels_per_point: core.window().scale_factor() as f32,
        };

        let clipped_primitives = self.context.tessellate(
            full_output.shapes,
            screen_descriptor.pixels_per_point,
        );
        // Update egui textures
        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer.update_texture(
                &core.device,
                &core.queue,
                *id,
                image_delta,
            );
        }

        self.egui_renderer.update_buffers(
            &core.device,
            &core.queue,
            encoder,
            &clipped_primitives,
            &screen_descriptor,
        );

        {
            let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Egui Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            let mut render_pass = render_pass.forget_lifetime();
            self.egui_renderer.render(
                &mut render_pass,
                &clipped_primitives,
                &screen_descriptor,
            );
        }
        // Cleanup egui textures
        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
    }
    pub fn load_media(&mut self, core: &Core, path: std::path::PathBuf) {
        self.last_media_path = Some(path.clone());
        
        let extension = path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();
            
        match extension.as_str() {
            // Video formats
            "mp4" | "webm" | "mov" | "avi" | "mkv" => {
                match VideoManager::new(
                    &core.device,
                    &core.queue,
                    &path,
                    &self.texture_bind_group_layout,
                    30,
                ) {
                    Ok(video_mgr) => {
                        let frame_count = video_mgr.frame_count;
                        let fps = video_mgr.fps;
                        
                        self.video_manager = Some(video_mgr);
                        self.texture_manager = None;
                        
                        println!("Loaded video: {} frames @ {:.2} fps", 
                                frame_count, fps);
                    },
                    Err(e) => {
                        eprintln!("Failed to load video: {:?}", e);
                        self.video_manager = None;
                        self.texture_manager = Some(Self::create_default_texture_manager(
                            core, &self.texture_bind_group_layout
                        ));
                    }
                }
            },
            "png" | "jpg" | "jpeg" | "bmp" | "tga" => {
                if let Ok(img) = image::open(&path) {
                    let rgba_image = img.into_rgba8();
                    self.texture_manager = Some(TextureManager::new(
                        &core.device,
                        &core.queue,
                        &rgba_image,
                        &self.texture_bind_group_layout,
                    ));
                    self.video_manager = None;
                }
            },
            _ => {
                eprintln!("Unsupported file format: {}", extension);
            }
        }
    }
    
    pub fn update_video(&mut self, queue: &wgpu::Queue) -> bool {
        if let (Some(video_mgr), Some(path)) = (&mut self.video_manager, &self.last_media_path) {
            match video_mgr.update(queue, path) {
                Ok(updated) => return updated,
                Err(e) => {
                    eprintln!("Error updating video: {:?}", e);
                    return false;
                }
            }
        }
        false
    }
    pub fn load_image(&mut self, core: &Core, path: std::path::PathBuf) {
        self.load_media(core, path);
    }
    pub fn create_capture_texture(
        &self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::Buffer) {
        let capture_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Capture Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: CAPTURE_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let align = 256;
        let unpadded_bytes_per_row = width * 4;
        let padding = (align - unpadded_bytes_per_row % align) % align;
        let padded_bytes_per_row = unpadded_bytes_per_row + padding;
        let buffer_size = padded_bytes_per_row * height;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Capture Buffer"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        (capture_texture, output_buffer)
    }
    pub fn apply_control_request(&mut self, request: ControlsRequest) {
        if request.should_reset {
            self.start_time = Instant::now();
        }
        self.controls.apply_ui_request(request);
    }
}
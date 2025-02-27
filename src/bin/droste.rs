use cuneus::{Core,Renderer,ShaderApp, ShaderManager, UniformProvider, UniformBinding, BaseShader,ExportSettings, ExportError, ExportManager,ShaderHotReload,ShaderControls};
use winit::event::*;
use image::ImageError;
use std::path::PathBuf;
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ShaderParams {
    branches: f32,
    scale: f32,
    time_scale: f32,
    rotation: f32,
    zoom: f32,
    offset_x: f32,
    offset_y: f32,
    iterations: f32,
    smoothing: f32,
    use_animation: f32,
}
impl UniformProvider for ShaderParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    ffmpeg_sidecar::download::auto_download()?;
    let (app, event_loop) = ShaderApp::new("Droste", 800, 600);
    let shader = SpiralShader::init(app.core());
    app.run(event_loop, shader)
}
struct SpiralShader {
    base: BaseShader,
    params_uniform: UniformBinding<ShaderParams>,
    hot_reload: ShaderHotReload,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    time_bind_group_layout: wgpu::BindGroupLayout,
    resolution_bind_group_layout: wgpu::BindGroupLayout,
    params_bind_group_layout: wgpu::BindGroupLayout,
}
impl SpiralShader {
    fn capture_frame(&mut self, core: &Core, time: f32) -> Result<Vec<u8>, wgpu::SurfaceError> {
        let settings = self.base.export_manager.settings();
        let (capture_texture, output_buffer) = self.base.create_capture_texture(
            &core.device,
            settings.width,
            settings.height
        );
        let align = 256;
        let unpadded_bytes_per_row = settings.width * 4;
        let padding = (align - unpadded_bytes_per_row % align) % align;
        let padded_bytes_per_row = unpadded_bytes_per_row + padding;
        let capture_view = capture_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Capture Encoder"),
        });
        self.base.time_uniform.data.time = time;
        self.base.time_uniform.update(&core.queue);
        self.base.resolution_uniform.data.dimensions = [settings.width as f32, settings.height as f32];
        self.base.resolution_uniform.update(&core.queue);
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Capture Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &capture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            render_pass.set_pipeline(&self.base.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
            if let Some(video_mgr) = &self.base.video_manager {
                render_pass.set_bind_group(0, &video_mgr.texture_manager.bind_group, &[]);
            } else if let Some(texture_manager) = &self.base.texture_manager {
                render_pass.set_bind_group(0, &texture_manager.bind_group, &[]);
            }
            // Time (group 1)
            render_pass.set_bind_group(1, &self.base.time_uniform.bind_group, &[]);
            // Params (group 2)
            render_pass.set_bind_group(2, &self.params_uniform.bind_group, &[]);
            // Resolution (group 3)
            render_pass.set_bind_group(3, &self.base.resolution_uniform.bind_group, &[]);
            render_pass.draw(0..4, 0..1);
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
        core.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let padded_data = buffer_slice.get_mapped_range().to_vec();
        let mut unpadded_data = Vec::with_capacity((settings.width * settings.height * 4) as usize);
        for chunk in padded_data.chunks(padded_bytes_per_row as usize) {
            unpadded_data.extend_from_slice(&chunk[..unpadded_bytes_per_row as usize]);
        }
        Ok(unpadded_data)
    }

    #[allow(unused_mut)]
    fn save_frame(&self, mut data: Vec<u8>, frame: u32, settings: &ExportSettings) -> Result<(), ExportError> {
        let frame_path = settings.export_path
            .join(format!("frame_{:05}.png", frame));
        
        if let Some(parent) = frame_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        #[cfg(target_os = "macos")]
        {
            for chunk in data.chunks_mut(4) {
                chunk.swap(0, 2);
            }
        }
        let image = image::ImageBuffer::<image::Rgba<u8>, Vec<u8>>::from_raw(
            settings.width,
            settings.height,
            data
        ).ok_or_else(|| ImageError::Parameter(
            image::error::ParameterError::from_kind(
                image::error::ParameterErrorKind::Generic(
                    "Failed to create image buffer".to_string()
                )
            )
        ))?;
        
        image.save(&frame_path)?;
        Ok(())
    }

    fn handle_export(&mut self, core: &Core) {
        if let Some((frame, time)) = self.base.export_manager.try_get_next_frame() {
            if let Ok(data) = self.capture_frame(core, time) {
                let settings = self.base.export_manager.settings();
                if let Err(e) = self.save_frame(data, frame, settings) {
                    eprintln!("Error saving frame: {:?}", e);
                }
            }
        } else {
            self.base.export_manager.complete_export();
        }
    }
}
impl ShaderManager for SpiralShader {
    fn init(core: &cuneus::Core) -> Self {
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
        let params_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            label: Some("params_bind_group_layout"),
        });
        let params_uniform = UniformBinding::new(
            &core.device,
            "Params Uniform",
            ShaderParams {
                branches: 1.0,
                scale: 0.5,
                time_scale: 1.0,
                rotation: 0.0,
                zoom: 1.0,
                offset_x: 0.0,
                offset_y: 0.0,
                iterations: 1.0,
                smoothing: 0.5,
                use_animation: 1.0,
            },
            &params_bind_group_layout,
            0,
        );
        let texture_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        });
        let bind_group_layouts = vec![
            &texture_bind_group_layout,    // group 0
            &time_bind_group_layout,       // group 1 
            &params_bind_group_layout,     // group 2
            &resolution_bind_group_layout, // group 3
        ];
        let vs_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vertex Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/vertex.wgsl").into()),
        });

        let fs_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fragment Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/droste.wgsl").into()),
        });

        let shader_paths = vec![
            PathBuf::from("shaders/vertex.wgsl"),
            PathBuf::from("shaders/droste.wgsl"),
        ];

        let hot_reload = ShaderHotReload::new(
            core.device.clone(),
            shader_paths,
            vs_module,
            fs_module,
        ).expect("Failed to initialize hot reload");
        let base = BaseShader::new(
            core,
            include_str!("../../shaders/vertex.wgsl"),
            include_str!("../../shaders/droste.wgsl"),
            &bind_group_layouts,
            None,
        );
        Self {
            base,
            params_uniform,
            hot_reload,
            texture_bind_group_layout,
            time_bind_group_layout,
            resolution_bind_group_layout,
            params_bind_group_layout,
        }
    }

    fn update(&mut self, core: &Core) {
        if let Some((new_vs, new_fs)) = self.hot_reload.check_and_reload() {
            println!("Reloading shaders at time: {:.2}s", self.base.start_time.elapsed().as_secs_f32());
            let pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &self.texture_bind_group_layout,    // group 0
                    &self.time_bind_group_layout,       // group 1
                    &self.resolution_bind_group_layout, // group 2
                    &self.params_bind_group_layout,     // group 3
                ],
                push_constant_ranges: &[],
            });
            self.base.renderer = Renderer::new(
                &core.device,
                new_vs,
                new_fs,
                core.config.format,
                &pipeline_layout,
                None,
            );
        }
        self.base.update_video(&core.queue);
        
        if self.base.export_manager.is_exporting() {
            self.handle_export(core);
        }
    }
    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut reload_media = false;
        let mut selected_path = None;
        
        let mut params = self.params_uniform.data;
        let mut changed = false;
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self.base.controls.get_ui_request(
            &self.base.start_time,
            &core.size
        );
        let has_video = self.base.video_manager.is_some();
        let mut video_is_playing = false;
        let mut video_loop = false;
        let mut video_current_frame = 0;
        let mut video_frame_count = 0;
        let mut video_width = 0;
        let mut video_height = 0;
        let mut video_fps = 0.0;
        
        if let Some(video_mgr) = &self.base.video_manager {
            video_is_playing = video_mgr.is_playing;
            video_loop = video_mgr.loop_video;
            video_current_frame = video_mgr.current_frame;
            video_frame_count = video_mgr.frame_count;
            video_width = video_mgr.width;
            video_height = video_mgr.height;
            video_fps = video_mgr.fps;
        }
        
        let mut video_state_changed = false;
        let mut video_seek_to_frame = None;
        
        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                egui::Window::new("Shader Settings").show(ctx, |ui| {
                    ui.group(|ui| {
                        ui.label("Media");
                        
                        if ui.button("Load Media").clicked() {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("Media", &["png", "jpg", "jpeg", "mp4", "webm", "mov", "avi", "mkv"])
                                .pick_file() 
                            {
                                selected_path = Some(path);
                                reload_media = true;
                            }
                        }
                    });
                    if has_video {
                        ui.group(|ui| {
                            ui.label("Video Controls");
                            
                            if ui.button(if video_is_playing { "Pause" } else { "Play" }).clicked() {
                                video_is_playing = !video_is_playing;
                                video_state_changed = true;
                            }
                            
                            if ui.checkbox(&mut video_loop, "Loop Video").changed() {
                                video_state_changed = true;
                            }
                            
                            let mut current_frame = video_current_frame as i32;
                            
                            if ui.add(egui::Slider::new(&mut current_frame, 0..=(video_frame_count as i32 - 1))
                                .text("Frame")
                                .clamp_to_range(true)).changed() {
                                
                                video_seek_to_frame = Some(current_frame as usize);
                            }
                            
                            ui.label(format!("Frame: {}/{}", video_current_frame + 1, video_frame_count));
                            ui.label(format!("Time: {:.2}s / {:.2}s", 
                                            video_current_frame as f32 / video_fps, 
                                            video_frame_count as f32 / video_fps));
                            ui.label(format!("FPS: {:.2}", video_fps));
                            ui.label(format!("Resolution: {}x{}", video_width, video_height));
                        });
                    }
                    
                    ui.group(|ui| {
                        ui.label("Basic Parameters");
                        changed |= ui.add(egui::Slider::new(&mut params.branches, -20.0..=20.0).text("Branches")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.scale, 0.0..=2.0).text("Scale")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.zoom, 0.1..=5.0).text("uv")).changed();
                    });
                    
                    ui.group(|ui| {
                        ui.label("Animation");
                        let mut use_anim = params.use_animation > 0.5;
                        if ui.checkbox(&mut use_anim, "Enable Animation").changed() {
                            changed = true;
                            params.use_animation = if use_anim { 1.0 } else { 0.0 };
                        }
                        if use_anim {
                            changed |= ui.add(egui::Slider::new(&mut params.time_scale, -5.0..=5.0).text("Animation Speed")).changed();
                        }
                        changed |= ui.add(egui::Slider::new(&mut params.rotation, -6.28..=6.28).text("Rotation")).changed();
                    });
    
                    ui.group(|ui| {
                        ui.label("Advanced Parameters");
                        changed |= ui.add(egui::Slider::new(&mut params.iterations, -10.0..=10.0).text("Iterations")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.smoothing, -1.0..=1.0).text("Smoothing")).changed();
                    });
                    
                    ui.group(|ui| {
                        ui.label("Texture Offset");
                        changed |= ui.add(egui::Slider::new(&mut params.offset_x, -1.0..=1.0).text("X Offset")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.offset_y, -1.0..=1.0).text("Y Offset")).changed();
                    });
    
                    ui.separator();
                    ShaderControls::render_controls_widget(ui, &mut controls_request);
                    ui.separator();
                    should_start_export = ExportManager::render_export_ui_widget(ui, &mut export_request);
                });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };
        if video_state_changed {
            if let Some(video_mgr) = &mut self.base.video_manager {
                video_mgr.is_playing = video_is_playing;
                video_mgr.loop_video = video_loop;
                
                if video_is_playing {
                    video_mgr.last_update_time = std::time::Instant::now();
                }
            }
        }
        if let Some(frame) = video_seek_to_frame {
            if let (Some(video_mgr), Some(path)) = (&mut self.base.video_manager, &self.base.last_media_path) {
                if let Err(e) = video_mgr.seek_to_frame(&core.queue, path, frame) {
                    eprintln!("Error seeking: {:?}", e);
                }
            }
        }
        
        self.base.export_manager.apply_ui_request(export_request);
        self.base.apply_control_request(controls_request);
        let current_time = self.base.controls.get_time(&self.base.start_time);
        self.base.time_uniform.data.time = current_time;
        self.base.time_uniform.update(&core.queue);
        if changed {
            self.params_uniform.data = params;
            self.params_uniform.update(&core.queue);
        }

        if should_start_export {
            self.base.export_manager.start_export();
        }
        if reload_media {
            if let Some(path) = selected_path {
                self.base.load_media(core, path);
            }
        }
        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            render_pass.set_pipeline(&self.base.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
            
            // Texture (group 0)
            if let Some(video_mgr) = &self.base.video_manager {
                render_pass.set_bind_group(0, &video_mgr.texture_manager.bind_group, &[]);
            } else if let Some(texture_manager) = &self.base.texture_manager {
                render_pass.set_bind_group(0, &texture_manager.bind_group, &[]);
            }
            // Time (group 1)
            render_pass.set_bind_group(1, &self.base.time_uniform.bind_group, &[]);
            // Params (group 2)
            render_pass.set_bind_group(2, &self.params_uniform.bind_group, &[]);
            // Resolution (group 3)
            render_pass.set_bind_group(3, &self.base.resolution_uniform.bind_group, &[]);
            render_pass.draw(0..4, 0..1);
        }
        self.base.handle_render_output(core, &view, full_output, &mut encoder);
        core.queue.submit(Some(encoder.finish()));
        output.present();
        Ok(())
    }
    fn resize(&mut self, core: &Core) {
        self.base.update_resolution(&core.queue, core.size);
    }
    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        if self.base.egui_state.on_window_event(core.window(), event).consumed {
            return true;
        }
        if let WindowEvent::KeyboardInput { event, .. } = event {
            return self.base.key_handler.handle_keyboard_input(core.window(), event);
        }
    
        false
    }
}
use cuneus::{Core, ShaderManager,UniformProvider, UniformBinding,RenderKit,TextureManager,ShaderHotReload,ShaderControls,AtomicBuffer};
use winit::event::WindowEvent;
use cuneus::ShaderApp;
use cuneus::Renderer;
use cuneus::create_feedback_texture_pair;
use cuneus::ExportManager;
use std::path::PathBuf;
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ShaderParams {
    // Matrix 1
    m1_scale: f32,
    m1_y_scale: f32,
    // Matrix 2
    m2_scale: f32,
    m2_shear: f32,
    m2_shift: f32,
    // Matrix 3
    m3_scale: f32,
    m3_shear: f32,
    m3_shift: f32,
    // Matrix 4
    m4_scale: f32,
    m4_shift: f32,
    // Matrix 5
    m5_scale: f32,
    m5_shift: f32,
    time_scale: f32,
    decay: f32,
    intensity: f32,
}

impl UniformProvider for ShaderParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}
struct Shader {
    base: RenderKit,
    renderer_pass2: Renderer,
    params_uniform: UniformBinding<ShaderParams>,
    texture_a: Option<TextureManager>,
    texture_b: Option<TextureManager>,
    frame_count: u32,
    hot_reload: ShaderHotReload,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    time_bind_group_layout: wgpu::BindGroupLayout,
    params_bind_group_layout: wgpu::BindGroupLayout,
    atomic_buffer: AtomicBuffer,
    atomic_bind_group_layout: wgpu::BindGroupLayout,
}
impl Shader {
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
        {
            self.atomic_buffer.clear(&core.queue);
            let mut render_pass = Renderer::begin_render_pass(
                &mut encoder,
                &capture_view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("Capture Pass"),
            );
            render_pass.set_pipeline(&self.renderer_pass2.render_pipeline);
            render_pass.set_vertex_buffer(0, self.renderer_pass2.vertex_buffer.slice(..));
            if let Some(texture) = &self.texture_a {
                render_pass.set_bind_group(0, &texture.bind_group, &[]);
            }
            render_pass.set_bind_group(1, &self.base.time_uniform.bind_group, &[]);
            render_pass.set_bind_group(2, &self.params_uniform.bind_group, &[]);
            render_pass.set_bind_group(3, &self.atomic_buffer.bind_group, &[]);
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

    fn handle_export(&mut self, core: &Core) {
        if let Some((frame, time)) = self.base.export_manager.try_get_next_frame() {
            if let Ok(data) = self.capture_frame(core, time) {
                let settings = self.base.export_manager.settings();
                if let Err(e) = cuneus::save_frame(data, frame, settings) {
                    eprintln!("Error saving frame: {:?}", e);
                }
            }
        } else {
            self.base.export_manager.complete_export();
        }
    }
}


impl ShaderManager for Shader {
    fn init(core: &Core) -> Self {
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
        let atomic_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("atomic_bind_group_layout"),
        });

        let buffer_size = core.config.width * core.config.height;
        let atomic_buffer = AtomicBuffer::new(
            &core.device,
            buffer_size,
            &atomic_bind_group_layout,
        );
        let params_uniform = UniformBinding::new(
            &core.device,
            "Params Uniform",
            ShaderParams {
                m1_scale: 0.8,
                m1_y_scale: 0.5,
                m2_scale: 0.4,
                m2_shear: 0.2,
                m2_shift: 0.3,
                m3_scale: 0.4,
                m3_shear: 0.2,
                m3_shift: 0.3,
                m4_scale: 0.3,
                m4_shift: 0.2,
                m5_scale: 0.2,
                m5_shift: 0.4,
                time_scale: 0.1,
                decay: 0.0,
                intensity: 0.0,
            },
            &params_bind_group_layout,
            0,
        );
        let (texture_a, texture_b) = create_feedback_texture_pair(
            core,
            core.config.width,
            core.config.height,
            &texture_bind_group_layout,
        );
        let vs_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vertex Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/vertex.wgsl").into()),
        });
        let fs_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fragment Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/rorschach.wgsl").into()),
        });
        let pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[
                &texture_bind_group_layout,
                &time_bind_group_layout,
                &params_bind_group_layout,
                &atomic_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
        let shader_paths = vec![
            PathBuf::from("shaders/vertex.wgsl"),
            PathBuf::from("shaders/rorschach.wgsl"),
        ];
        let hot_reload = ShaderHotReload::new(
            core.device.clone(),
            shader_paths,
            vs_module,
            fs_module,
        ).expect("Failed to initialize hot reload");
        let renderer_pass2 = Renderer::new(
            &core.device,
            &hot_reload.vs_module,
            &hot_reload.fs_module,
            core.config.format,
            &pipeline_layout,
            Some("fs_pass2"),
        );
        let base = RenderKit::new(
            core,
            include_str!("../../shaders/vertex.wgsl"),
            include_str!("../../shaders/rorschach.wgsl"),
            &[
                &texture_bind_group_layout,
                &time_bind_group_layout,
                &params_bind_group_layout,
                &atomic_bind_group_layout,
            ],
            Some("fs_pass1"),
        );
        Self {
            base,
            renderer_pass2,
            params_uniform,
            texture_a: Some(texture_a),
            texture_b: Some(texture_b),
            frame_count: 0,
            hot_reload,
            texture_bind_group_layout,
            time_bind_group_layout,
            params_bind_group_layout,
            atomic_buffer,
            atomic_bind_group_layout,
        }
    }
    fn update(&mut self, core: &Core) {
        if let Some((new_vs, new_fs)) = self.hot_reload.check_and_reload() {
            println!("Reloading shaders at time: {:.2}s", self.base.start_time.elapsed().as_secs_f32());
            let pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &self.texture_bind_group_layout,
                    &self.time_bind_group_layout,
                    &self.params_bind_group_layout,
                    &self.atomic_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
            self.renderer_pass2 = Renderer::new(
                &core.device,
                new_vs,
                new_fs,
                core.config.format,
                &pipeline_layout,
                Some("fs_pass2"),
            );
    
            self.base.renderer = Renderer::new(
                &core.device,
                new_vs,
                new_fs,
                core.config.format,
                &pipeline_layout,
                Some("fs_pass1"),
            );
        }
        if self.base.export_manager.is_exporting() {
            self.handle_export(core);
        }
        self.base.fps_tracker.update();
    }
    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        let mut params = self.params_uniform.data;
        let mut changed = false;
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self.base.controls.get_ui_request(
            &self.base.start_time,
            &core.size
        );
        controls_request.current_fps = Some(self.base.fps_tracker.fps());
        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                ctx.style_mut(|style| {
                    style.visuals.window_fill = egui::Color32::from_rgba_premultiplied(0, 0, 0, 180);
                });                egui::Window::new("Rorschach Settings").show(ctx, |ui| {
                    ui.group(|ui| {
                        ui.heading("General");
                        changed |= ui.add(egui::Slider::new(&mut params.time_scale, 0.01..=1.0).text("Time Scale")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.decay, 0.0..=0.99).text("Decay")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.intensity, 0.0..=1.0).text("Clean/Blend")).changed();
                    });
            
                    ui.add_space(10.0);
            
                    ui.group(|ui| {
                        ui.heading("Matrix 1");
                        changed |= ui.add(egui::Slider::new(&mut params.m1_scale, 0.1..=1.0).text("Scale")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.m1_y_scale, 0.1..=1.0).text("Y Scale")).changed();
                    });
            
                    ui.group(|ui| {
                        ui.heading("Matrix 2");
                        changed |= ui.add(egui::Slider::new(&mut params.m2_scale, 0.1..=1.0).text("Scale")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.m2_shear, -0.5..=0.5).text("Shear")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.m2_shift, -0.5..=0.5).text("Shift")).changed();
                    });
            
                    ui.group(|ui| {
                        ui.heading("Matrix 3");
                        changed |= ui.add(egui::Slider::new(&mut params.m3_scale, 0.1..=1.0).text("Scale")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.m3_shear, -0.5..=0.5).text("Shear")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.m3_shift, -0.5..=0.5).text("Shift")).changed();
                    });
            
                    ui.group(|ui| {
                        ui.heading("Matrix 4 & 5");
                        changed |= ui.add(egui::Slider::new(&mut params.m4_scale, 0.1..=1.0).text("M4 Scale")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.m4_shift, -0.5..=0.5).text("M4 Shift")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.m5_scale, 0.1..=1.0).text("M5 Scale")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.m5_shift, -0.5..=0.5).text("M5 Shift")).changed();
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
        if controls_request.should_clear_buffers {
            let (texture_a, texture_b) = create_feedback_texture_pair(
                core,
                core.config.width,
                core.config.height,
                &self.texture_bind_group_layout,
            );
            self.texture_a = Some(texture_a);
            self.texture_b = Some(texture_b);
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
        if let (Some(ref texture_a), Some(ref texture_b)) = (&self.texture_a, &self.texture_b) {
            let (source_texture, target_texture) = if self.frame_count % 2 == 0 {
                (texture_b, texture_a)
            } else {
                (texture_a, texture_b)
            };
            
            // First render pass
{
                self.atomic_buffer.clear(&core.queue);
                let mut render_pass = Renderer::begin_render_pass(
                    &mut encoder,
                    &target_texture.view,
                    wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    Some("Feedback Pass"),
                );
                render_pass.set_pipeline(&self.base.renderer.render_pipeline);
                render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
                render_pass.set_bind_group(0, &source_texture.bind_group, &[]);
                render_pass.set_bind_group(1, &self.base.time_uniform.bind_group, &[]);
                render_pass.set_bind_group(2, &self.params_uniform.bind_group, &[]);
                render_pass.set_bind_group(3, &self.atomic_buffer.bind_group, &[]);
                render_pass.draw(0..4, 0..1);
            }
    
            // Second render pass
            {   self.atomic_buffer.clear(&core.queue);
                let mut render_pass = Renderer::begin_render_pass(
                    &mut encoder,
                    &view,
                    wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    Some("Display Pass"),
                );
                render_pass.set_pipeline(&self.renderer_pass2.render_pipeline);
                render_pass.set_vertex_buffer(0, self.renderer_pass2.vertex_buffer.slice(..));
                render_pass.set_bind_group(0, &target_texture.bind_group, &[]);
                render_pass.set_bind_group(1, &self.base.time_uniform.bind_group, &[]);
                render_pass.set_bind_group(2, &self.params_uniform.bind_group, &[]);
                render_pass.set_bind_group(3, &self.atomic_buffer.bind_group, &[]);
                render_pass.draw(0..4, 0..1);
            }
            self.frame_count = self.frame_count.wrapping_add(1);
        }
        self.base.handle_render_output(core, &view, full_output, &mut encoder);
        encoder.insert_debug_marker("Transition to Present");
        core.queue.submit(Some(encoder.finish()));
        output.present();
    
        Ok(())
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
fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("inkblot", 800, 600);
    app.run(event_loop, |core| {
        Shader::init(core)
    })
}

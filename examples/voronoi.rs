use cuneus::prelude::ComputeShader;
use cuneus::{Core, RenderKit, ShaderApp, ShaderManager, UniformProvider};
use winit::event::*;
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ShaderParams {
    scale: f32,
    offset_value: f32,
    cell_index: f32,
    edge_width: f32,
    highlight: f32,
}

impl UniformProvider for ShaderParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    cuneus::gst::init()?;
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("voronoi", 800, 600);
    app.run(event_loop, |core| Voronoi::init(core))
}
struct Voronoi {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: ShaderParams,
}
impl ShaderManager for Voronoi {
    fn init(core: &Core) -> Self {
        let texture_bind_group_layout = RenderKit::create_standard_texture_layout(&core.device);
        let base = RenderKit::new(core, &texture_bind_group_layout, None);

        let initial_params = ShaderParams {
            scale: 24.0,
            offset_value: -1.0,
            cell_index: 0.0,
            edge_width: 0.1,
            highlight: 0.15,
        };

        let config = ComputeShader::builder()
            .with_entry_point("main")
            .with_input_texture()
            .with_custom_uniforms::<ShaderParams>()
            .build();

        let mut compute_shader =
            ComputeShader::from_builder(core, include_str!("shaders/voronoi.wgsl"), config);

        // Enable hot reload
        if let Err(e) = compute_shader.enable_hot_reload(
            core.device.clone(),
            std::path::PathBuf::from("examples/shaders/voronoi.wgsl"),
            core.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Voronoi Hot Reload"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("shaders/voronoi.wgsl").into()),
                }),
        ) {
            eprintln!("Failed to enable hot reload for voronoi shader: {}", e);
        }

        compute_shader.set_custom_params(initial_params, &core.queue);

        Self {
            base,
            compute_shader,
            current_params: initial_params,
        }
    }

    fn update(&mut self, core: &Core) {
        // Update time
        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0 / 60.0;
        self.compute_shader
            .set_time(current_time, delta, &core.queue);

        // Update input textures for media processing
        self.base.update_current_texture(core, &core.queue);
        if let Some(texture_manager) = self.base.get_current_texture_manager() {
            self.compute_shader.update_input_texture(
                &texture_manager.view,
                &texture_manager.sampler,
                &core.device,
            );
        }

        self.base.fps_tracker.update();
        // Handle export
        self.compute_shader.handle_export(core, &mut self.base);
        self.compute_shader.check_hot_reload(&core.device);
    }
    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let _video_updated = if self.base.using_video_texture {
            self.base.update_video_texture(core, &core.queue)
        } else {
            false
        };
        let _webcam_updated = if self.base.using_webcam_texture {
            self.base.update_webcam_texture(core, &core.queue)
        } else {
            false
        };

        let mut params = self.current_params;
        let mut changed = false;

        let mut controls_request = self
            .base
            .controls
            .get_ui_request(&self.base.start_time, &core.size);
        controls_request.current_fps = Some(self.base.fps_tracker.fps());

        let using_video_texture = self.base.using_video_texture;
        let using_hdri_texture = self.base.using_hdri_texture;
        let using_webcam_texture = self.base.using_webcam_texture;
        let video_info = self.base.get_video_info();
        let hdri_info = self.base.get_hdri_info();
        let webcam_info = self.base.get_webcam_info();

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                ctx.style_mut(|style| {
                    style.visuals.window_fill =
                        egui::Color32::from_rgba_premultiplied(0, 0, 0, 180);
                });
                egui::Window::new("Voronoi Settings")
                    .collapsible(true)
                    .default_size([300.0, 100.0])
                    .show(ctx, |ui| {
                        ui.collapsing("Media", |ui: &mut egui::Ui| {
                            cuneus::ShaderControls::render_media_panel(
                                ui,
                                &mut controls_request,
                                using_video_texture,
                                video_info,
                                using_hdri_texture,
                                hdri_info,
                                using_webcam_texture,
                                webcam_info,
                            );
                        });

                        ui.separator();

                        // Pattern Settings
                        ui.collapsing("Pattern Settings", |ui| {
                            changed |= ui
                                .add(
                                    egui::Slider::new(&mut params.scale, 1.0..=100.0)
                                        .text("Cell Scale"),
                                )
                                .changed();
                            changed |= ui
                                .add(
                                    egui::Slider::new(&mut params.offset_value, -1.0..=2.0)
                                        .text("Pattern Offset"),
                                )
                                .changed();
                        });

                        // Cell Settings
                        ui.collapsing("Cell Settings", |ui| {
                            changed |= ui
                                .add(
                                    egui::Slider::new(&mut params.cell_index, 0.0..=3.0)
                                        .text("Cell Index"),
                                )
                                .changed();
                        });

                        // Edge Settings
                        ui.collapsing("Edge Settings", |ui| {
                            changed |= ui
                                .add(
                                    egui::Slider::new(&mut params.edge_width, 0.0..=1.0)
                                        .text("Edge Width"),
                                )
                                .changed();
                            changed |= ui
                                .add(
                                    egui::Slider::new(&mut params.highlight, 0.0..=15.0)
                                        .text("Edge Highlight"),
                                )
                                .changed();
                        });

                        ui.separator();
                        cuneus::ShaderControls::render_controls_widget(ui, &mut controls_request);
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        self.base.apply_control_request(controls_request.clone());
        self.base.handle_video_requests(core, &controls_request);
        self.base.handle_webcam_requests(core, &controls_request);
        self.base.handle_hdri_requests(core, &controls_request);

        if changed {
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
        }

        // Create command encoder
        let mut encoder = core
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Update time uniform
        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta_time = 1.0 / 60.0; // Approximate delta time
        self.compute_shader
            .set_time(current_time, delta_time, &core.queue);

        // Dispatch compute shader
        self.compute_shader.dispatch(&mut encoder, core);

        // Render compute output to screen
        {
            let mut render_pass = cuneus::Renderer::begin_render_pass(
                &mut encoder,
                &view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("Main Render Pass"),
            );

            let compute_texture = self.compute_shader.get_output_texture();
            render_pass.set_pipeline(&self.base.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &compute_texture.bind_group, &[]);
            render_pass.draw(0..4, 0..1);
        }

        self.base
            .handle_render_output(core, &view, full_output, &mut encoder);
        core.queue.submit(Some(encoder.finish()));
        output.present();

        Ok(())
    }
    fn resize(&mut self, core: &Core) {
        self.base.update_resolution(&core.queue, core.size);
        self.compute_shader
            .resize(core, core.size.width, core.size.height);
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        if self
            .base
            .egui_state
            .on_window_event(core.window(), event)
            .consumed
        {
            return true;
        }
        if let WindowEvent::KeyboardInput { event, .. } = event {
            return self
                .base
                .key_handler
                .handle_keyboard_input(core.window(), event);
        }

        false
    }
}

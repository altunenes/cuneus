use cuneus::compute::*;
use cuneus::prelude::*;
use winit::event::WindowEvent;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct LichParams {
    cloud_density: f32,
    lightning_intensity: f32,
    branch_count: f32,
    feedback_decay: f32,
    base_color: [f32; 3],
    _pad1: f32,
    color_shift: f32,
    spectrum_mix: f32,
    _pad2: [f32; 2],
}

impl UniformProvider for LichParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct LichShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: LichParams,
}

impl LichShader {
    fn clear_buffers(&mut self, core: &Core) {
        // Clear multipass ping-pong buffers
        self.compute_shader.clear_all_buffers(core);
    }
}

impl ShaderManager for LichShader {
    fn init(core: &Core) -> Self {
        let texture_bind_group_layout = RenderKit::create_standard_texture_layout(&core.device);
        let base = RenderKit::new(core, &texture_bind_group_layout, None);

        let passes = vec![
            PassDescription::new("buffer_a", &[]),
            PassDescription::new("buffer_b", &["buffer_a", "buffer_b"]), // Self-feedback!
            PassDescription::new("main_image", &["buffer_b"]),
        ];

        let config = ComputeShader::builder()
            .with_multi_pass(&passes)
            .with_custom_uniforms::<LichParams>()
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(cuneus::compute::COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Lich Lightning")
            .build();

        let mut compute_shader =
            ComputeShader::from_builder(core, include_str!("shaders/lich.wgsl"), config);

        // Enable hot reload
        if let Err(e) = compute_shader.enable_hot_reload(
            core.device.clone(),
            std::path::PathBuf::from("examples/shaders/lich.wgsl"),
            core.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Lich Hot Reload"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("shaders/lich.wgsl").into()),
                }),
        ) {
            eprintln!("Failed to enable hot reload for Lich shader: {}", e);
        }

        let initial_params = LichParams {
            cloud_density: 3.0,
            lightning_intensity: 1.0,
            branch_count: 1.0,
            feedback_decay: 0.98,
            base_color: [1.0, 1.0, 1.0],
            _pad1: 0.0,
            color_shift: 2.0,
            spectrum_mix: 0.5,
            _pad2: [0.0; 2],
        };

        // Initialize custom uniform with initial parameters
        compute_shader.set_custom_params(initial_params, &core.queue);

        Self {
            base,
            compute_shader,
            current_params: initial_params,
        }
    }

    fn update(&mut self, core: &Core) {
        // Check for hot reload updates
        self.compute_shader.check_hot_reload(&core.device);
        // Handle export
        self.compute_shader.handle_export(core, &mut self.base);

        // Update time
        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0 / 60.0;
        self.compute_shader
            .set_time(current_time, delta, &core.queue);

        self.base.fps_tracker.update();
    }

    fn resize(&mut self, core: &Core) {
        self.compute_shader
            .resize(core, core.size.width, core.size.height);
    }

    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = core
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Lich Render Encoder"),
            });

        let mut params = self.current_params;
        let mut changed = false;
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self
            .base
            .controls
            .get_ui_request(&self.base.start_time, &core.size);
        controls_request.current_fps = Some(self.base.fps_tracker.fps());

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                ctx.style_mut(|style| {
                    style.visuals.window_fill =
                        egui::Color32::from_rgba_premultiplied(0, 0, 0, 180);
                    style
                        .text_styles
                        .get_mut(&egui::TextStyle::Body)
                        .unwrap()
                        .size = 11.0;
                    style
                        .text_styles
                        .get_mut(&egui::TextStyle::Button)
                        .unwrap()
                        .size = 10.0;
                });

                egui::Window::new("Lich Lightning")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(300.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("Lightning Parameters")
                            .default_open(true)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.cloud_density, 0.0..=24.0)
                                            .text("Seed"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(
                                            &mut params.lightning_intensity,
                                            0.1..=6.0,
                                        )
                                        .text("Lightning"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.branch_count, 0.0..=2.0)
                                            .text("Branch"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.feedback_decay, 0.1..=1.5)
                                            .text("Decay"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Color Settings")
                            .default_open(false)
                            .show(ui, |ui| {
                                let mut color = params.base_color;
                                if ui.color_edit_button_rgb(&mut color).changed() {
                                    params.base_color = color;
                                    changed = true;
                                }
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.color_shift, 0.1..=20.0)
                                            .text("Temperature"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.spectrum_mix, 0.0..=1.0)
                                            .text("Spectral"),
                                    )
                                    .changed();
                            });

                        ui.separator();
                        ShaderControls::render_controls_widget(ui, &mut controls_request);

                        ui.separator();
                        should_start_export =
                            ExportManager::render_export_ui_widget(ui, &mut export_request);

                        ui.separator();
                        ui.label("Electric lightning with atomic buffer accumulation");
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        self.compute_shader.dispatch(&mut encoder, core);

        // Render compute output to screen
        {
            let compute_texture = self.compute_shader.get_output_texture();
            let mut render_pass = cuneus::Renderer::begin_render_pass(
                &mut encoder,
                &view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("Lich Display Pass"),
            );

            render_pass.set_pipeline(&self.base.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &compute_texture.bind_group, &[]);
            render_pass.draw(0..4, 0..1);
        }

        // Apply UI changes
        if controls_request.should_clear_buffers {
            self.clear_buffers(core);
        }
        self.base.apply_control_request(controls_request.clone());

        self.base.export_manager.apply_ui_request(export_request);
        if should_start_export {
            self.base.export_manager.start_export();
        }

        if changed {
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
        }

        self.base
            .handle_render_output(core, &view, full_output, &mut encoder);
        core.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        // Flip ping-pong buffers for next frame (required for multi-pass)
        self.compute_shader.flip_buffers();

        Ok(())
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("Lich Lightning", 800, 600);
    app.run(event_loop, |core| LichShader::init(core))
}

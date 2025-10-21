use cuneus::prelude::ComputeShader;
use cuneus::{Core, RenderKit, ShaderApp, ShaderManager, UniformProvider};
use winit::event::*;
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ShaderParams {
    lambda: f32,
    theta: f32,
    alpha: f32,
    sigma: f32,
    gamma: f32,
    blue: f32,
    a: f32,
    b: f32,
    base_color_r: f32,
    base_color_g: f32,
    base_color_b: f32,
    accent_color_r: f32,
    accent_color_g: f32,
    accent_color_b: f32,
    background_r: f32,
    background_g: f32,
    background_b: f32,
    gamma_correction: f32,
    aces_tonemapping: f32,
    _padding: f32,
}

impl UniformProvider for ShaderParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct Shader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: ShaderParams,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("sdvert", 800, 600);
    app.run(event_loop, |core| Shader::init(core))
}
impl ShaderManager for Shader {
    fn init(core: &Core) -> Self {
        // Create texture display layout
        let texture_bind_group_layout = RenderKit::create_standard_texture_layout(&core.device);
        let base = RenderKit::new(core, &texture_bind_group_layout, None);

        let initial_params = ShaderParams {
            sigma: 0.07,
            gamma: 1.5,
            blue: 1.0,
            a: 2.0,
            b: 0.5,
            lambda: 3.0,
            theta: 2.0,
            alpha: 0.3,
            base_color_r: 1.0,
            base_color_g: 1.0,
            base_color_b: 1.0,
            accent_color_r: 1.0,
            accent_color_g: 1.0,
            accent_color_b: 1.0,
            background_r: 0.6,
            background_g: 0.9,
            background_b: 0.9,
            gamma_correction: 0.41,
            aces_tonemapping: 0.4,
            _padding: 0.0,
        };

        let config = ComputeShader::builder()
            .with_entry_point("main")
            .with_custom_uniforms::<ShaderParams>()
            .build();

        let mut compute_shader =
            ComputeShader::from_builder(core, include_str!("shaders/sdvert.wgsl"), config);

        // Enable hot reload
        if let Err(e) = compute_shader.enable_hot_reload(
            core.device.clone(),
            std::path::PathBuf::from("examples/shaders/sdvert.wgsl"),
            core.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("SDVert Hot Reload"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sdvert.wgsl").into()),
                }),
        ) {
            eprintln!("Failed to enable hot reload for sdvert shader: {}", e);
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

                egui::Window::new("SDVert Controls")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(300.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("Geometry")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.lambda, 1.0..=20.0)
                                            .text("Vertices"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.theta, 0.0..=10.0)
                                            .text("Angle Scale"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.gamma, 0.1..=3.0)
                                            .text("Layer Size"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.alpha, 0.001..=0.5)
                                            .text("Layer Min"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.sigma, 0.01..=0.5)
                                            .text("Layer Max"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Shape Parameters")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.a, 0.0..=5.0)
                                            .text("Depth Factor"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.b, 0.0..=5.0)
                                            .text("Fold Pattern"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.blue, 0.0..=5.0)
                                            .text("Hue Shift"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Colors")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Base:");
                                    let mut base_color = [
                                        params.base_color_r,
                                        params.base_color_g,
                                        params.base_color_b,
                                    ];
                                    if ui.color_edit_button_rgb(&mut base_color).changed() {
                                        params.base_color_r = base_color[0];
                                        params.base_color_g = base_color[1];
                                        params.base_color_b = base_color[2];
                                        changed = true;
                                    }
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Accent:");
                                    let mut accent_color = [
                                        params.accent_color_r,
                                        params.accent_color_g,
                                        params.accent_color_b,
                                    ];
                                    if ui.color_edit_button_rgb(&mut accent_color).changed() {
                                        params.accent_color_r = accent_color[0];
                                        params.accent_color_g = accent_color[1];
                                        params.accent_color_b = accent_color[2];
                                        changed = true;
                                    }
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Background:");
                                    let mut bg_color = [
                                        params.background_r,
                                        params.background_g,
                                        params.background_b,
                                    ];
                                    if ui.color_edit_button_rgb(&mut bg_color).changed() {
                                        params.background_r = bg_color[0];
                                        params.background_g = bg_color[1];
                                        params.background_b = bg_color[2];
                                        changed = true;
                                    }
                                });
                            });

                        egui::CollapsingHeader::new("Post-Processing")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.gamma_correction, 0.1..=3.0)
                                            .text("Gamma Correction"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.aces_tonemapping, 0.0..=2.0)
                                            .text("ACES Tonemapping"),
                                    )
                                    .changed();
                            });

                        ui.separator();
                        cuneus::ShaderControls::render_controls_widget(ui, &mut controls_request);

                        ui.separator();
                        should_start_export =
                            cuneus::ExportManager::render_export_ui_widget(ui, &mut export_request);
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        self.base.export_manager.apply_ui_request(export_request);
        self.base.apply_control_request(controls_request);

        if changed {
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
        }

        if should_start_export {
            self.base.export_manager.start_export();
        }

        // Create command encoder
        let mut encoder = core
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Update time uniform
        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta_time = 1.0 / 60.0;
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

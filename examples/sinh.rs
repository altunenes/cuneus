use cuneus::compute::*;
use cuneus::prelude::*;
use winit::event::WindowEvent;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SinhParams {
    aa: i32,
    camera_x: f32,
    camera_y: f32,
    camera_z: f32,
    orbit_speed: f32,
    magic_number: f32,
    cv_min: f32,
    cv_max: f32,
    os_base: f32,
    os_scale: f32,
    base_color_r: f32,
    base_color_g: f32,
    base_color_b: f32,
    light_color_r: f32,
    light_color_g: f32,
    light_color_b: f32,
    ambient_r: f32,
    ambient_g: f32,
    ambient_b: f32,
    gamma: f32,
    iterations: i32,
    bound: f32,
    fractal_scale: f32,
    vignette_offset: f32,
}

impl UniformProvider for SinhParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct SinhShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: SinhParams,
}

impl SinhShader {
    fn clear_buffers(&mut self, core: &Core) {
        self.compute_shader.clear_all_buffers(core);
    }
}

impl ShaderManager for SinhShader {
    fn init(core: &Core) -> Self {
        // Create texture bind group layout for displaying compute shader output
        let texture_bind_group_layout = RenderKit::create_standard_texture_layout(&core.device);
        let base = RenderKit::new(core, &texture_bind_group_layout, None);

        let initial_params = SinhParams {
            aa: 2,
            camera_x: 0.1,
            camera_y: 10.0,
            camera_z: 10.0,
            orbit_speed: 0.3,
            magic_number: 36.0,
            cv_min: 2.197,
            cv_max: 2.99225,
            os_base: 0.00004,
            os_scale: 0.02040101,
            base_color_r: 0.5,
            base_color_g: 0.25,
            base_color_b: 0.05,
            light_color_r: 0.8,
            light_color_g: 1.0,
            light_color_b: 0.3,
            ambient_r: 1.2,
            ambient_g: 1.0,
            ambient_b: 0.8,
            gamma: 0.4,
            iterations: 65,
            bound: 12.25,
            fractal_scale: 0.05,
            vignette_offset: 0.0,
        };

        let config = ComputeShader::builder()
            .with_entry_point("main")
            .with_custom_uniforms::<SinhParams>()
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Sinh Unified")
            .build();

        let mut compute_shader =
            ComputeShader::from_builder(core, include_str!("shaders/sinh.wgsl"), config);

        // Enable hot reload
        if let Err(e) = compute_shader.enable_hot_reload(
            core.device.clone(),
            std::path::PathBuf::from("examples/shaders/sinh.wgsl"),
            core.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Sinh Hot Reload"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sinh.wgsl").into()),
                }),
        ) {
            eprintln!("Failed to enable hot reload for Sinh shader: {}", e);
        }

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

        self.base.fps_tracker.update();
    }
    fn resize(&mut self, core: &Core) {
        self.base.update_resolution(&core.queue, core.size);
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
                label: Some("Render Encoder"),
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

                egui::Window::new("Sinh")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(280.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("Rendering")
                            .default_open(true)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(egui::Slider::new(&mut params.aa, 1..=4).text("AA"))
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.gamma, 0.2..=1.1)
                                            .text("Gamma"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.vignette_offset, 0.0..=1.0)
                                            .text("Vignette"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Camera")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.camera_x, -1.0..=1.0)
                                            .text("X"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.camera_y, 5.0..=20.0)
                                            .text("Y"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.camera_z, 5.0..=20.0)
                                            .text("Z"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.orbit_speed, 0.0..=1.0)
                                            .text("speed"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Fractal")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.iterations, 10..=100)
                                            .text("Iterations"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.bound, 1.0..=25.0)
                                            .text("Bound"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.magic_number, 1.0..=100.0)
                                            .text("Magic Number"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.cv_min, 1.0..=3.0)
                                            .text("CV Min"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.cv_max, 2.0..=4.0)
                                            .text("CV Max"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.os_base, 0.00001..=0.001)
                                            .logarithmic(true)
                                            .text("OS Base"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.os_scale, 0.001..=0.1)
                                            .text("OS Scale"),
                                    )
                                    .changed();
                                ui.separator();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.fractal_scale, 0.01..=1.0)
                                            .text("Fractal Scale"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Colors")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Base Color:");
                                    let mut color = [
                                        params.base_color_r,
                                        params.base_color_g,
                                        params.base_color_b,
                                    ];
                                    if ui.color_edit_button_rgb(&mut color).changed() {
                                        params.base_color_r = color[0];
                                        params.base_color_g = color[1];
                                        params.base_color_b = color[2];
                                        changed = true;
                                    }
                                });

                                ui.horizontal(|ui| {
                                    ui.label("Light Color:");
                                    let mut color = [
                                        params.light_color_r,
                                        params.light_color_g,
                                        params.light_color_b,
                                    ];
                                    if ui.color_edit_button_rgb(&mut color).changed() {
                                        params.light_color_r = color[0];
                                        params.light_color_g = color[1];
                                        params.light_color_b = color[2];
                                        changed = true;
                                    }
                                });

                                ui.horizontal(|ui| {
                                    ui.label("Ambient Color:");
                                    let mut color =
                                        [params.ambient_r, params.ambient_g, params.ambient_b];
                                    if ui.color_edit_button_rgb(&mut color).changed() {
                                        params.ambient_r = color[0];
                                        params.ambient_g = color[1];
                                        params.ambient_b = color[2];
                                        changed = true;
                                    }
                                });
                            });

                        ui.separator();
                        ShaderControls::render_controls_widget(ui, &mut controls_request);
                        ui.separator();
                        should_start_export =
                            ExportManager::render_export_ui_widget(ui, &mut export_request);
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        self.base.export_manager.apply_ui_request(export_request);
        if controls_request.should_clear_buffers {
            self.clear_buffers(core);
        }
        self.base.apply_control_request(controls_request);

        let current_time = self.base.controls.get_time(&self.base.start_time);

        let delta = 1.0 / 60.0;
        self.compute_shader
            .set_time(current_time, delta, &core.queue);

        if changed {
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
        }

        if should_start_export {
            self.base.export_manager.start_export();
        }

        self.compute_shader.dispatch(&mut encoder, core);

        {
            let mut render_pass = cuneus::Renderer::begin_render_pass(
                &mut encoder,
                &view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("Display Pass"),
            );

            render_pass.set_pipeline(&self.base.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &self.compute_shader.output_texture.bind_group, &[]);

            render_pass.draw(0..4, 0..1);
        }

        self.base
            .handle_render_output(core, &view, full_output, &mut encoder);
        core.queue.submit(Some(encoder.finish()));
        output.present();

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
    let (app, event_loop) = cuneus::ShaderApp::new("Sinh 3D", 800, 300);

    app.run(event_loop, |core| SinhShader::init(core))
}

// Photon tracing: currents
// Very complex example demonstrating multi-buffer ping-pong computation
// I hope this example is useful for those who came from the Shadertoy, I tried to use same terminology (bufferA, ichannels etc)
// I used the all buffers (buffera,b,c,d,mainimage) and complex ping-pong logic
use cuneus::compute::{ComputeShader, PassDescription, COMPUTE_TEXTURE_FORMAT_RGBA16};
use cuneus::{Core, RenderKit, ShaderApp, ShaderControls, ShaderManager};
use cuneus::{ExportManager, UniformProvider};
use winit::event::*;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CurrentsParams {
    sphere_radius: f32,
    sphere_pos_x: f32,
    sphere_pos_y: f32,
    critic2_interval: f32,
    critic2_pause: f32,
    critic3_interval: f32,
    metallic_reflection: f32,
    line_intensity: f32,
    pattern_scale: f32,
    noise_strength: f32,
    gradient_r: f32,
    gradient_g: f32,
    gradient_b: f32,
    gradient_w: f32,
    line_color_r: f32,
    line_color_g: f32,
    line_color_b: f32,
    line_color_w: f32,
    gradient_intensity: f32,
    line_intensity_final: f32,
    c2_min: f32,
    c2_max: f32,
    c3_min: f32,
    c3_max: f32,
    fbm_scale: f32,
    fbm_offset: f32,
    gamma: f32,
}

impl Default for CurrentsParams {
    fn default() -> Self {
        Self {
            sphere_radius: 0.2,
            sphere_pos_x: 0.0,
            sphere_pos_y: -0.2,
            critic2_interval: 10.0,
            critic2_pause: 5.0,
            critic3_interval: 10.0,
            metallic_reflection: 1.8,
            line_intensity: 0.8,
            pattern_scale: 150.0,
            noise_strength: 1.0,
            gradient_r: 1.0,
            gradient_g: 2.0,
            gradient_b: 3.0,
            gradient_w: 4.0,
            line_color_r: 1.0,
            line_color_g: 2.0,
            line_color_b: 3.0,
            line_color_w: 4.0,
            gradient_intensity: 1.5,
            line_intensity_final: 1.5,
            c2_min: 333.0,
            c2_max: 1.0,
            c3_min: 1.0,
            c3_max: 3.0,
            fbm_scale: 4.0,
            fbm_offset: 1.0,
            gamma: 2.1,
        }
    }
}

impl UniformProvider for CurrentsParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct CurrentsShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: CurrentsParams,
}

impl CurrentsShader {
    fn clear_buffers(&mut self, core: &Core) {
        self.compute_shader.clear_all_buffers(core);
    }
}

impl ShaderManager for CurrentsShader {
    fn init(core: &Core) -> Self {
        let texture_bind_group_layout = RenderKit::create_standard_texture_layout(&core.device);
        let base = RenderKit::new(core, &texture_bind_group_layout, None);

        // Define the 5 passes
        let passes = vec![
            PassDescription::new("buffer_a", &["buffer_a"]), // self-feedback
            PassDescription::new("buffer_b", &["buffer_b", "buffer_a"]), // reads BufferB + BufferA
            PassDescription::new("buffer_c", &["buffer_c", "buffer_a"]), // reads BufferC + BufferA
            PassDescription::new("buffer_d", &["buffer_d", "buffer_c", "buffer_b"]), // reads BufferD + BufferC + BufferB
            PassDescription::new("main_image", &["buffer_d"]), // reads BufferD for final output
        ];

        let config = ComputeShader::builder()
            .with_entry_point("buffer_a") // Start with buffer_a
            .with_multi_pass(&passes)
            .with_custom_uniforms::<CurrentsParams>()
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Currents Multi-Pass")
            .build();

        let compute_shader = cuneus::compute_shader!(core, "shaders/currents.wgsl", config);

        let initial_params = CurrentsParams::default();
        let shader = Self {
            base,
            compute_shader,
            current_params: initial_params,
        };

        // Initialize custom uniform with default parameters
        shader
            .compute_shader
            .set_custom_params(initial_params, &core.queue);

        shader
    }

    fn update(&mut self, core: &Core) {
        // Update time
        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0 / 60.0;
        self.compute_shader
            .set_time(current_time, delta, &core.queue);

        self.base.fps_tracker.update();

        // Check for hot reload updates
        self.compute_shader.check_hot_reload(&core.device);
        // Handle export
        self.compute_shader.handle_export(core, &mut self.base);
    }

    fn resize(&mut self, core: &Core) {
        self.base.default_resize(core, &mut self.compute_shader);
    }

    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let mut frame = self.base.begin_frame(core)?;

        let mut controls_request = self
            .base
            .controls
            .get_ui_request(&self.base.start_time, &core.size);
        controls_request.current_fps = Some(self.base.fps_tracker.fps());

        // Handle UI and controls
        let mut params = self.current_params;
        let mut changed = false;
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                RenderKit::apply_default_style(ctx);

                egui::Window::new("Multi-Buffer Ping-Pong Example")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(280.0)
                    .show(ctx, |ui| {
                        // CURRENTS MODE UI
                        egui::CollapsingHeader::new("Sphere Settings")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.sphere_radius, 0.05..=0.5)
                                            .text("Sphere Radius"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.sphere_pos_x, -1.0..=1.0)
                                            .text("Sphere X"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.sphere_pos_y, -1.0..=1.0)
                                            .text("Sphere Y"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(
                                            &mut params.metallic_reflection,
                                            0.5..=3.0,
                                        )
                                        .text("Metallic Reflection"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Pattern Control")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.pattern_scale, 50.0..=300.0)
                                            .text("Pattern Scale"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.critic2_interval, 5.0..=20.0)
                                            .text("Flow Interval"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.critic2_pause, 1.0..=10.0)
                                            .text("Flow Pause"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.critic3_interval, 5.0..=20.0)
                                            .text("Scale Interval"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.noise_strength, 0.5..=5.0)
                                            .text("Noise Strength"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Noise")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.label("Oscillator 2 (c2):");
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.c2_min, 1.0..=500.0)
                                            .text("C2 Min"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.c2_max, 0.1..=10.0)
                                            .text("C2 Max"),
                                    )
                                    .changed();

                                ui.separator();
                                ui.label("Oscillator 3 (c3):");
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.c3_min, 0.1..=10.0)
                                            .text("C3 Min"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.c3_max, 0.5..=10.0)
                                            .text("C3 Max"),
                                    )
                                    .changed();

                                ui.separator();
                                ui.label("FBM Noise:");
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.fbm_scale, 1.0..=10.0)
                                            .text("FBM Scale"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.fbm_offset, 0.1..=5.0)
                                            .text("FBM Offset"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Colors & Post-Processing")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Gradient:");
                                    let mut color =
                                        [params.gradient_r, params.gradient_g, params.gradient_b];
                                    if ui.color_edit_button_rgb(&mut color).changed() {
                                        params.gradient_r = color[0];
                                        params.gradient_g = color[1];
                                        params.gradient_b = color[2];
                                        changed = true;
                                    }
                                });

                                ui.horizontal(|ui| {
                                    ui.label("Lines:");
                                    let mut color = [
                                        params.line_color_r,
                                        params.line_color_g,
                                        params.line_color_b,
                                    ];
                                    if ui.color_edit_button_rgb(&mut color).changed() {
                                        params.line_color_r = color[0];
                                        params.line_color_g = color[1];
                                        params.line_color_b = color[2];
                                        changed = true;
                                    }
                                });

                                ui.separator();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(
                                            &mut params.gradient_intensity,
                                            0.1..=2.0,
                                        )
                                        .text("Gradient Intensity"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(
                                            &mut params.line_intensity_final,
                                            0.1..=2.0,
                                        )
                                        .text("Line Final Intensity"),
                                    )
                                    .changed();

                                ui.separator();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.line_intensity, 0.1..=3.0)
                                            .text("Line Intensity"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.gamma, 0.1..=4.0)
                                            .text("Gamma Correction"),
                                    )
                                    .changed();
                            });

                        ui.separator();

                        ShaderControls::render_controls_widget(ui, &mut controls_request);

                        ui.separator();

                        should_start_export =
                            ExportManager::render_export_ui_widget(ui, &mut export_request);

                        ui.separator();
                        ui.label(format!("Frame: {}", self.compute_shader.current_frame));
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

        if changed {
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
            // Reset frame counter for proper photon accumulation restart
            self.compute_shader.current_frame = 0;
        }

        if should_start_export {
            self.base.export_manager.start_export();
        }

        // Create command encoder

        self.compute_shader.dispatch(&mut frame.encoder, core);

        self.base.renderer.render_to_view(&mut frame.encoder, &frame.view, &self.compute_shader);

        self.base.end_frame(core, frame, full_output);

        // Flip ping-pong buffers for next frame
        self.compute_shader.flip_buffers();

        Ok(())
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        self.base.default_handle_input(core, event)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("Photon Tracing", 800, 600);

    app.run(event_loop, CurrentsShader::init)
}

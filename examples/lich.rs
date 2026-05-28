use cuneus::compute::*;
use cuneus::prelude::*;

cuneus::uniform_params! {
    struct LichParams {
        cloud_density: f32,
        lightning_intensity: f32,
        branch_count: f32,
        feedback_decay: f32,
        base_color: [f32; 3],
        glow_intensity: f32,
        specular_strength: f32,
        contrast: f32,
        gamma: f32,
        saturation: f32,
        color_shift: f32,
        spectrum_mix: f32,
        light_intensity: f32,
        _pad: f32
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
        let base = RenderKit::new(core);

        let passes = vec![
            PassDescription::new("lightning", &[]),
            PassDescription::new("feedback", &["lightning", "feedback"]),
            PassDescription::new("main_image", &["feedback"]),
        ];

        let config = ComputeShader::builder()
            .with_multi_pass(&passes)
            .with_custom_uniforms::<LichParams>()
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(cuneus::compute::COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Lich Lightning")
            .build();

        let compute_shader = cuneus::compute_shader!(core, "shaders/lich.wgsl", config);

        let initial_params = LichParams {
            cloud_density: 3.0,
            lightning_intensity: 1.2,
            branch_count: 1.0,
            feedback_decay: 0.94,
            base_color: [0.25, 0.55, 1.0],
            glow_intensity: 1.5,
            specular_strength: 1.2,
            contrast: 1.8,
            gamma: 1.0,
            saturation: 1.4,
            color_shift: 12.0,
            spectrum_mix: 0.4,
            light_intensity: 1.6,
            _pad: 0.0,
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
        // Handle export
        self.compute_shader.handle_export(core, &mut self.base);

        // Update time
        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0 / 60.0;
        self.compute_shader
            .set_time(current_time, delta, &core.queue);
    }

    fn resize(&mut self, core: &Core) {
        self.compute_shader
            .resize(core, core.size.width, core.size.height);
    }

    fn render(&mut self, core: &Core) -> Result<(), cuneus::SurfaceError> {
        let mut frame = self.base.begin_frame(core)?;

        let mut params = self.current_params;
        let mut changed = false;
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self
            .base
            .controls
            .get_ui_request(&self.base.start_time, &core.size, self.base.fps_tracker.fps());

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                RenderKit::apply_default_style(ctx);

                egui::Window::new("Lich")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(320.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("Parameters")
                            .default_open(true)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.cloud_density, 0.0..=24.0).text("Seed")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.lightning_intensity, 0.1..=6.0).text("Intensity")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.branch_count, 0.0..=2.0).text("Branching")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.feedback_decay, 0.1..=1.0).text("Decay Rate")).changed();
                            });

                        egui::CollapsingHeader::new("Impasto")
                            .default_open(true)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.glow_intensity, 0.0..=5.0).text("Plasma Glow")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.specular_strength, 0.0..=4.0).text("Specular Shimmer")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.light_intensity, 0.1..=4.0).text("Light Intensity")).changed();
                            });

                        egui::CollapsingHeader::new("Color Gradients & Post Filters")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.label("Primary Color Conduit:");
                                let mut color = params.base_color;
                                if ui.color_edit_button_rgb(&mut color).changed() {
                                    params.base_color = color;
                                    changed = true;
                                }
                                changed |= ui.add(egui::Slider::new(&mut params.color_shift, 0.1..=20.0).text("Temperature")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.spectrum_mix, 0.0..=1.0).text("Spectral Mix")).changed();
                                ui.separator();
                                changed |= ui.add(egui::Slider::new(&mut params.contrast, 0.0..=5.0).text("Contrast Line")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.saturation, 0.0..=3.0).text("Saturation")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.gamma, 0.1..=3.0).text("Gamma Correction")).changed();
                            });

                        ui.separator();
                        ShaderControls::render_controls_widget(ui, &mut controls_request);

                        ui.separator();
                        should_start_export =
                            ExportManager::render_export_ui_widget(ui, &mut export_request);

                        ui.separator();
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        self.compute_shader.dispatch(&mut frame.encoder, core);

        self.base.renderer.render_to_view(&mut frame.encoder, &frame.view, &self.compute_shader.get_output_texture().bind_group);

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

        self.base.end_frame(core, frame, full_output);

        Ok(())
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        self.base.default_handle_input(core, event)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("Lich Lightning", 800, 600);
    app.run(event_loop, LichShader::init)
}

use cuneus::compute::*;
use cuneus::prelude::*;

cuneus::uniform_params! {
    struct PhysarumParams {
        sensor_angle: f32, sensor_dist: f32, drag: f32, move_speed: f32,
        decay_rate: f32, diffuse_rate: f32, deposit_amt: f32, random_jitter: f32,
        rule_seed: f32, mutation_scale: f32, force_scale: f32, sensor_gain: f32,
        species_attract: f32, species_repel: f32, strafe_power: f32, agent_count_scale: f32,
        glow_intensity: f32, color_shift: f32, specular_strength: f32, gamma: f32,
        attractor_count: f32, attractor_speed: f32, attractor_radius: f32, attractor_strength: f32,
        color_spread: f32, saturation: f32, palette_mix: f32, blur_samples: f32,
        color0_r: f32, color0_g: f32, color0_b: f32,
        color1_r: f32, color1_g: f32, color1_b: f32,
        color2_r: f32, color2_g: f32, color2_b: f32,
        substep: f32, total_samples: f32, wind_strength: f32, turing_scale: f32,
        _pad1: f32, fluid_blend: f32, _pad3: f32,
    }
}

struct PhysarumShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: PhysarumParams,
}

impl ShaderManager for PhysarumShader {
    fn init(core: &Core) -> Self {
        let initial_params = PhysarumParams {
            sensor_angle: 1.50, sensor_dist: 40.0, drag: 0.20, move_speed: 4.95,
            decay_rate: 0.999, diffuse_rate: 0.10, deposit_amt: 30.0, random_jitter: 0.500,
            rule_seed: 1.0, mutation_scale: 0.30, force_scale: 0.80, sensor_gain: 5.0,
            species_attract: 1.00, species_repel: 0.92, strafe_power: 0.15, agent_count_scale: 1.00,
            glow_intensity: 0.00, color_shift: -0.02, specular_strength: 0.25, gamma: 0.50,
            attractor_count: 3.0, attractor_speed: 0.30, attractor_radius: 250.0, attractor_strength: 0.15,
            color_spread: 0.30, saturation: 1.58, palette_mix: 0.57, blur_samples: 0.99,
            color0_r: 1.0, color0_g: 0.2, color0_b: 0.2,
            color1_r: 0.2, color1_g: 0.8, color1_b: 0.3,
            color2_r: 0.2, color2_g: 0.4, color2_b: 1.0,
            substep: 1.0, total_samples: 1.0, wind_strength: 0.8, turing_scale: 0.0,
            _pad1: 0.0, fluid_blend: 0.0, _pad3: 0.0,
        };

        let base = RenderKit::new(core);

        let passes = vec![
            PassDescription::new("agent_update", &["agent_update", "turing_resolve"]).with_resolution(1024, 1024),
            PassDescription::new("process_trails", &["process_trails"]),
            PassDescription::new("diffuse_h", &["process_trails"]),
            PassDescription::new("diffuse_v", &["process_trails", "diffuse_h"]),
            PassDescription::new("inhibitor_down", &["diffuse_v"]).with_resolution_scale(0.125),
            PassDescription::new("turing_resolve", &["process_trails", "diffuse_v", "inhibitor_down"]),
            PassDescription::new("main_image", &["process_trails", "turing_resolve", "inhibitor_down"]),
        ];

        let config = ComputeShader::builder()
            .with_multi_pass(&passes)
            .with_custom_uniforms::<PhysarumParams>()
            .with_atomic_buffer(4)
            .with_label("Physarum Simulation")
            .build();

        let compute_shader = cuneus::compute_shader!(core, "shaders/physarum.wgsl", config);
        compute_shader.set_custom_params(initial_params, &core.queue);

        Self { base, compute_shader, current_params: initial_params }
    }

    fn update(&mut self, core: &Core) {
        let current_time = self.base.controls.get_time(&self.base.start_time);
        self.compute_shader.set_time(current_time, 1.0 / 60.0, &core.queue);
        self.compute_shader.handle_export(core, &mut self.base);
    }

    fn resize(&mut self, core: &Core) {
        self.base.default_resize(core, &mut self.compute_shader);
    }

    fn render(&mut self, core: &Core) -> Result<(), cuneus::SurfaceError> {
        let mut frame = self.base.begin_frame(core)?;
        let mut params = self.current_params;
        let mut changed = false;
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self.base.controls.get_ui_request(
            &self.base.start_time, &core.size, self.base.fps_tracker.fps()
        );

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                RenderKit::apply_default_style(ctx);
                egui::Window::new("Physarum Controls")
                    .collapsible(true).resizable(true).default_width(320.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("Behavior Rule").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.rule_seed, 0.0..=100.0).text("Rule Seed")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.mutation_scale, 0.0..=1.0).text("Species Diversity")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.force_scale, 0.0..=3.0).text("Force Scale")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.sensor_gain, 0.5..=30.0).text("Sensor Gain")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.strafe_power, 0.0..=2.0).text("Strafe Power")).changed();
                        });

                        egui::CollapsingHeader::new("Agent Physics").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.move_speed, 0.1..=5.0).text("Speed")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.drag, 0.0..=0.99).text("Momentum")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.wind_strength, 0.0..=3.0).text("Slipstream Wind")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.fluid_blend, 0.0..=2.0).text("Bubble vs Vein Bias")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.sensor_dist, 1.0..=40.0).text("Sensor Distance")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.sensor_angle, 0.05..=1.5).text("Sensor Angle")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.random_jitter, 0.0..=0.5).text("Random Jitter")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.agent_count_scale, 0.05..=1.0).text("Agent Density")).changed();
                        });

                        egui::CollapsingHeader::new("Trail Environment").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.deposit_amt, 1.0..=30.0).text("Deposit Amount")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.decay_rate, 0.9..=0.999).text("Decay Rate")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.diffuse_rate, 0.0..=1.0).text("Diffusion Rate")).changed();
                        });

                        egui::CollapsingHeader::new("Species Interaction").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.species_attract, 0.0..=1.0).text("Cross-Attraction")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.species_repel, 0.0..=1.0).text("Cross-Repulsion")).changed();
                        });

                        egui::CollapsingHeader::new("Attractors").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.attractor_count, 0.0..=8.0).step_by(1.0).text("Count")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.attractor_speed, 0.0..=2.0).text("Orbit Speed")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.attractor_radius, 50.0..=500.0).text("Radius")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.attractor_strength, 0.0..=1.0).text("Strength")).changed();
                        });

                        egui::CollapsingHeader::new("Rendering").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.blur_samples, 0.0..=1.0).text("Feedback Fade")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.glow_intensity, 0.0..=2.0).text("Glow")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.specular_strength, 0.0..=1.5).text("Specular")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.gamma, 0.1..=2.2).text("Gamma")).changed();
                        });

                        egui::CollapsingHeader::new("Colors").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.palette_mix, 0.0..=1.0).text("Palette Mix")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.color_shift, -0.5..=0.5).text("Color Shift")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.color_spread, 0.0..=1.0).text("Species Hue Spread")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.saturation, 0.0..=2.5).text("Saturation")).changed();
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

        if controls_request.should_clear_buffers { self.compute_shader.current_frame = 0; }
        if !self.base.export_manager.is_exporting() { self.compute_shader.dispatch(&mut frame.encoder, core); }

        self.base.renderer.render_to_view(
            &mut frame.encoder,
            &frame.view,
            &self.compute_shader.get_output_texture().bind_group,
        );
        self.base.apply_control_request(controls_request);
        self.base.export_manager.apply_ui_request(export_request);

        if should_start_export { self.base.export_manager.start_export(); }
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
    let (app, event_loop) = ShaderApp::new("Physarum Engine", 1280, 720);
    app.run(event_loop, PhysarumShader::init)
}
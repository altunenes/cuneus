use cuneus::compute::*;
use cuneus::prelude::*;
cuneus::uniform_params! {
    struct FluidParams {
    viscosity: f32,
    gravity: f32,
    pressure_scale: f32,
    vortex_strength: f32,
    turbulence: f32,
    flow_speed: f32,
    pos_diffusion: f32,
    texture_influence: f32,
    light_intensity: f32,
    spec_power: f32,
    spec_intensity: f32,
    color_vibrancy: f32,
    vortex_radius: f32,
    gamma: f32,
    feedback: f32,
    vortex_speed: f32,
    force_mode: f32,
    force_harmony: f32,
    force_count: f32,
    contrast: f32,
    warp_amount: f32,
    flow_intensity: f32,
    color_advect: f32,
    drift_decay: f32,
    dye_intensity: f32,
    dye_radius: f32,
    bg_boil: f32,
    _padding: f32
}
}
struct FluidShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: FluidParams
}
impl ShaderManager for FluidShader {
    fn init(core: &Core) -> Self {
        let initial_params = FluidParams {
            viscosity: 0.5,
            gravity: 0.002,
            pressure_scale: 1.0,
            vortex_strength: 0.12,
            turbulence: 0.0005,
            flow_speed: 2.0,
            pos_diffusion: 0.3,
            texture_influence: 1.3,
            light_intensity: 1.3,
            spec_power: 36.0,
            spec_intensity: 2.0,
            color_vibrancy: 1.3,
            vortex_radius: 0.001,
            gamma: 1.1,
            feedback: 0.90,
            vortex_speed: 0.04,
            force_mode: 0.0,
            force_harmony: 0.3,
            force_count: 4.0,
            contrast: 0.15,
            warp_amount: 1.0,
            flow_intensity: 1.0,
            color_advect: 1.0,
            drift_decay: 0.0,
            dye_intensity: 0.06,
            dye_radius: 1.0,
            bg_boil: 0.8,
            _padding: 0.0
        };

        let base = RenderKit::new(core);
        let passes = vec![
            PassDescription::new("fluid_sim", &["fluid_sim", "color_map"]),
            PassDescription::new("pressure_refine", &["fluid_sim", "pressure_refine"]),
            PassDescription::new("position_field", &["pressure_refine", "position_field", "color_map"]),
            PassDescription::new("color_map", &["position_field", "color_map"]),
            PassDescription::new("main_image", &["color_map", "pressure_refine"]),
        ];
        let config = ComputeShader::builder()
            .with_entry_point("fluid_sim")
            .with_multi_pass(&passes)
            .with_channels(1)
            .with_custom_uniforms::<FluidParams>()
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Fluid LB")
            .build();
        let compute_shader = cuneus::compute_shader!(core, "shaders/fluid.wgsl", config);
        compute_shader.set_custom_params(initial_params, &core.queue);
        Self {
            base,
            compute_shader,
            current_params: initial_params}
    }
    fn update(&mut self, core: &Core) {
        self.base.update_current_texture(core, &core.queue);
        if let Some(texture_manager) = self.base.get_current_texture_manager() {
            self.compute_shader.update_channel_texture(
                0,
                &texture_manager.view,
                &texture_manager.sampler,
                &core.device,
                &core.queue,
            );
        }
        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0 / 60.0;
        self.compute_shader.set_time(current_time, delta, &core.queue);
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
        let mut controls_request = self.base.controls.get_ui_request(&self.base.start_time, &core.size, self.base.fps_tracker.fps());
        let using_video_texture = self.base.using_video_texture;
        let using_hdri_texture = self.base.using_hdri_texture;
        let using_webcam_texture = self.base.using_webcam_texture;
        let video_info = self.base.get_video_info();
        let hdri_info = self.base.get_hdri_info();
        let webcam_info = self.base.get_webcam_info();
        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                RenderKit::apply_default_style(ctx);
                egui::Window::new("Fluid Simulation")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(300.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("Flow").default_open(true).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.flow_speed, 0.1..=3.0).text("Speed")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.viscosity, 0.0..=3.0).text("Viscosity")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.turbulence, 0.0..=0.5).text("Dissipation")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.feedback, 0.0..=1.01).text("Feedback")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.bg_boil, 0.0..=2.0).text("Global Boil")).changed();
                        });

                        egui::CollapsingHeader::new("Dye & Glow").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.dye_intensity, 0.0..=0.2).text("Paint Brightness")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.dye_radius, 0.1..=3.0).text("Glow Spread")).changed();
                        });

                        egui::CollapsingHeader::new("Vortices").default_open(false).show(ui, |ui| {
                            if ui.add(egui::Slider::new(&mut params.force_count, 0.0..=18.0).step_by(1.0).text("Sources")).changed() {
                                params.force_count = params.force_count.round();
                                changed = true;
                            }
                            changed |= ui.add(egui::Slider::new(&mut params.vortex_strength, 0.0..=0.5).text("Confinement")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.force_harmony, 0.0..=1.0).text("Softness")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.vortex_radius, 0.001..=0.05).text("Radius")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.vortex_speed, 0.005..=0.15).text("Speed")).changed();
                        });
                        egui::CollapsingHeader::new("Distortion").default_open(true).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.warp_amount, 0.5..=5.0).text("Warp")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.flow_intensity, 0.5..=5.0).text("Flow Intensity")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.color_advect, 0.0..=3.0).text("Color Advect")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.drift_decay, 0.0..=0.05).text("Drift Decay")).changed();
                        });
                        egui::CollapsingHeader::new("Physics").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.pressure_scale, 0.0..=2.0).text("Pressure")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.gravity, 0.0..=0.2).text("Gravity")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.texture_influence, 0.0..=1.3).text("Texture Inf")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.pos_diffusion, 0.0..=1.0).text("Pos Diffusion")).changed();
                        });
                        egui::CollapsingHeader::new("Display").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.light_intensity, 0.8..=3.0).text("Light")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.spec_intensity, 0.0..=5.0).text("Specular")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.color_vibrancy, 0.5..=2.5).text("Color Vibrancy")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.contrast, 0.0..=0.8).text("Contrast")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.gamma, 0.5..=2.5).text("Gamma")).changed();
                        });
                        ui.separator();
                        ShaderControls::render_media_panel(ui, &mut controls_request, using_video_texture, video_info, using_hdri_texture, hdri_info, using_webcam_texture, webcam_info);
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
            self.compute_shader.current_frame = 0;
        }
        if !self.base.export_manager.is_exporting() {
            self.compute_shader.dispatch(&mut frame.encoder, core);
        }
        self.base.renderer.render_to_view(&mut frame.encoder, &frame.view, &self.compute_shader.get_output_texture().bind_group);
        self.base.apply_media_requests(core, &controls_request);
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
    cuneus::gst::init()?;
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("Fluid Simulation", 800, 600);
    app.run(event_loop, FluidShader::init)
}
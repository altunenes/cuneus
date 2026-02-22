use cuneus::compute::*;
use cuneus::prelude::*;
use winit::event::WindowEvent;
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
    mixing: f32,
    gamma: f32,
    feedback: f32,
    _pad2: f32}
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
            vortex_strength: 0.08,
            turbulence: 0.003,
            flow_speed: 2.0,
            pos_diffusion: 0.3,
            texture_influence: 1.3,
            light_intensity: 1.3,
            spec_power: 36.0,
            spec_intensity: 2.0,
            color_vibrancy: 1.3,
            mixing: 0.0,
            gamma: 1.1,
            feedback: 0.0,
            _pad2: 0.0};
        let base = RenderKit::new(core);
        let passes = vec![
            PassDescription::new("fluid_sim", &["fluid_sim", "color_map"]),
            PassDescription::new("position_field", &["fluid_sim", "position_field", "color_map"]),
            PassDescription::new("color_map", &["position_field", "color_map"]),
            PassDescription::new("main_image", &["color_map", "fluid_sim"]),
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
    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
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
                            changed |= ui.add(egui::Slider::new(&mut params.flow_speed, 0.1..=5.0).text("Speed")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.vortex_strength, 0.0..=0.5).text("Vorticity")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.turbulence, 0.0..=0.02).text("Turbulence")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.viscosity, 0.0..=5.0).text("Viscosity")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.mixing, 0.0..=1.0).text("Mixing")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.feedback, 0.0..=1.01).text("Feedback")).changed();
                        });
                        egui::CollapsingHeader::new("Physics").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.pressure_scale, 0.0..=2.0).text("Pressure")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.gravity, 0.0..=0.2).text("Gravity")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.texture_influence, 0.0..=1.3).text("texture inf")).changed();
                        });
                        egui::CollapsingHeader::new("Display").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.light_intensity, 0.8..=2.0).text("Light")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.color_vibrancy, 0.5..=2.0).text("Color Vibrancy")).changed();
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
        self.base.renderer.render_to_view(&mut frame.encoder, &frame.view, &self.compute_shader);
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
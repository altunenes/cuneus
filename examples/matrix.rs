use cuneus::prelude::ComputeShader;
use cuneus::{
    Core, ExportManager, RenderKit, ShaderApp, ShaderControls, ShaderManager,
};
use winit::event::*;

cuneus::uniform_params! {
    struct ShaderParams {
        red_power: f32,
        green_power: f32,
        blue_power: f32,
        green_boost: f32,
        contrast: f32,
        gamma: f32,
        glow: f32,
        _pad_m: f32,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    cuneus::gst::init()?;
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("matrix", 800, 600);
    app.run(event_loop, MatrixShader::init)
}

struct MatrixShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: ShaderParams,
}
impl ShaderManager for MatrixShader {
    fn init(core: &Core) -> Self {
        let base = RenderKit::new(core);

        let initial_params = ShaderParams {
            red_power: 0.98,
            green_power: 0.85,
            blue_power: 0.90,
            green_boost: 1.62,
            contrast: 1.0,
            gamma: 1.0,
            glow: 0.05,
            _pad_m: 0.0,
        };

        let config = ComputeShader::builder()
            .with_entry_point("main")
            .with_input_texture()
            .with_custom_uniforms::<ShaderParams>()
            .build();

        let compute_shader = cuneus::compute_shader!(core, "shaders/matrix.wgsl", config);

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
    }

    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let mut frame = self.base.begin_frame(core)?;

        let mut params = self.current_params;
        let mut changed = false;
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self
            .base
            .controls
            .get_ui_request(&self.base.start_time, &core.size, self.base.fps_tracker.fps());

        let using_video_texture = self.base.using_video_texture;
        let using_hdri_texture = self.base.using_hdri_texture;
        let using_webcam_texture = self.base.using_webcam_texture;
        let video_info = self.base.get_video_info();
        let hdri_info = self.base.get_hdri_info();
        let webcam_info = self.base.get_webcam_info();

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                RenderKit::apply_default_style(ctx);

                egui::Window::new("Matrix Effect")
                    .collapsible(true)
                    .resizable(true)
                    .default_size([300.0, 100.0])
                    .show(ctx, |ui| {
                        ui.collapsing("Media", |ui: &mut egui::Ui| {
                            ShaderControls::render_media_panel(
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

                        ui.collapsing("Matrix Color Settings", |ui| {
                            changed |= ui
                                .add(
                                    egui::Slider::new(&mut params.red_power, 0.5..=3.0)
                                        .text("Red Power"),
                                )
                                .changed();

                            changed |= ui
                                .add(
                                    egui::Slider::new(&mut params.green_power, 0.5..=3.0)
                                        .text("Green Power"),
                                )
                                .changed();

                            changed |= ui
                                .add(
                                    egui::Slider::new(&mut params.blue_power, 0.5..=3.0)
                                        .text("Blue Power"),
                                )
                                .changed();

                            changed |= ui
                                .add(
                                    egui::Slider::new(&mut params.green_boost, 0.5..=2.0)
                                        .text("Green Boost"),
                                )
                                .changed();

                            changed |= ui
                                .add(
                                    egui::Slider::new(&mut params.contrast, 0.5..=2.0)
                                        .text("Contrast"),
                                )
                                .changed();

                            changed |= ui
                                .add(egui::Slider::new(&mut params.gamma, 0.2..=2.0).text("Gamma"))
                                .changed();

                            changed |= ui
                                .add(egui::Slider::new(&mut params.glow, -1.0..=1.0).text("Glow"))
                                .changed();
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
        self.base.apply_media_requests(core, &controls_request);

        if changed {
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
        }

        if should_start_export {
            self.base.export_manager.start_export();
        }


        // Run compute shader
        self.compute_shader.dispatch(&mut frame.encoder, core);

        self.base.renderer.render_to_view(&mut frame.encoder, &frame.view, &self.compute_shader);

        self.base.end_frame(core, frame, full_output);
        Ok(())
    }

    fn resize(&mut self, core: &Core) {
        self.base.default_resize(core, &mut self.compute_shader);
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        self.base.default_handle_input(core, event)
    }
}

use cuneus::compute::*;
use cuneus::prelude::*;
use winit::event::WindowEvent;

cuneus::uniform_params! {
    struct LegoParams {
    brick_scale: f32,
    lightdir_x: f32,
    lightdir_y: f32,
    grain: f32,
    gamma: f32,
    shadow_str: f32,
    shadow_dist: f32,
    ao_str: f32,
    spec_pow: f32,
    spec_str: f32,
    edge_enh: f32,
    stud_h: f32,
    base_h: f32,
    rim_str: f32,
    res_scale_mult: f32,
    stud_h_mult: f32,
    light_r: f32,
    light_g: f32,
    light_b: f32,
    depth_scale: f32,
    edge_blend: f32,
    _pad: f32,
    _pad2: f32,
    _pad3: f32}
}

struct LegoShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: LegoParams}

impl ShaderManager for LegoShader {
    fn init(core: &Core) -> Self {
        let initial_params = LegoParams {
            brick_scale: 0.01,
            lightdir_x: 0.8,
            lightdir_y: 0.6,
            grain: 0.04,
            gamma: 0.4,
            shadow_str: 0.5,
            shadow_dist: 1.25,
            ao_str: 0.85,
            spec_pow: 12.0,
            spec_str: 0.3,
            edge_enh: 0.15,
            stud_h: 0.045,
            base_h: 0.2,
            rim_str: 0.5,
            res_scale_mult: 0.2,
            stud_h_mult: 1.0,
            light_r: 0.8,
            light_g: 0.75,
            light_b: 0.7,
            depth_scale: 0.85,
            edge_blend: 0.3,
            _pad: 0.0,
            _pad2: 0.0,
            _pad3: 0.0};
        let base = RenderKit::new(core);

        let config = ComputeShader::builder()
            .with_entry_point("main_image")
            .with_channels(1)
            .with_custom_uniforms::<LegoParams>()
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("LEGO Effect")
            .build();

        let compute_shader = cuneus::compute_shader!(core, "shaders/lego.wgsl", config);

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
        self.compute_shader
            .set_time(current_time, delta, &core.queue);
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

        let current_fps = self.base.fps_tracker.fps();

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                RenderKit::apply_default_style(ctx);

                egui::Window::new("LEGO Effect")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(320.0)
                    .show(ctx, |ui| {
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

                        ui.separator();
                        egui::CollapsingHeader::new("Brick Geom")
                            .default_open(true)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.brick_scale, 0.005..=0.05)
                                            .text("Scale"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.stud_h, 0.01..=0.1)
                                            .text("Stud H"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.base_h, 0.05..=0.3)
                                            .text("Base H"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.stud_h_mult, 1.0..=12.0)
                                            .text("Stud Mult"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Lighting")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.lightdir_x, -1.0..=1.0)
                                            .text("Dir X"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.lightdir_y, -1.0..=1.0)
                                            .text("Dir Y"),
                                    )
                                    .changed();

                                ui.label("Light Color");
                                let mut color = [params.light_r, params.light_g, params.light_b];
                                if ui.color_edit_button_rgb(&mut color).changed() {
                                    params.light_r = color[0];
                                    params.light_g = color[1];
                                    params.light_b = color[2];
                                    changed = true;
                                }

                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.spec_pow, 2.0..=50.0)
                                            .text("Spec Pow"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.spec_str, 0.0..=1.0)
                                            .text("Spec Str"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.rim_str, 0.0..=1.0)
                                            .text("Rim Str"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Shadows & AO")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.shadow_str, 0.0..=1.0)
                                            .text("Shd Str"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.shadow_dist, 0.1..=3.0)
                                            .text("Shd Dist"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.ao_str, 0.5..=1.5)
                                            .text("AO Cntrst"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Post-FX")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.edge_enh, 0.0..=0.5)
                                            .text("Edge Enh"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.edge_blend, 0.01..=0.3)
                                            .text("Edge Blend"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.grain, 0.0..=0.1)
                                            .text("Grain"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.gamma, 0.1..=1.4)
                                            .text("Gamma"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Advanced")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.res_scale_mult, 0.01..=2.0)
                                            .text("Res Scale"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.depth_scale, 0.5..=1.0)
                                            .text("Depth Scl"),
                                    )
                                    .changed();
                            });

                        ui.separator();
                        ShaderControls::render_controls_widget(ui, &mut controls_request);
                        ui.separator();
                        should_start_export =
                            ExportManager::render_export_ui_widget(ui, &mut export_request);
                        ui.separator();
                        ui.label(format!(
                            "Resolution: {}x{}",
                            core.size.width, core.size.height
                        ));
                        ui.label(format!("FPS: {current_fps:.1}"));
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        self.base.apply_media_requests(core, &controls_request);

        self.base.export_manager.apply_ui_request(export_request);
        if should_start_export {
            self.base.export_manager.start_export();
        }

        if changed {
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
        }

        self.compute_shader.dispatch_stage(&mut frame.encoder, core, 0);

        self.base.renderer.render_to_view(&mut frame.encoder, &frame.view, &self.compute_shader);

        self.base.end_frame(core, frame, full_output);

        Ok(())
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        if self.base.default_handle_input(core, event) {
            return true;
        }
        if let WindowEvent::DroppedFile(path) = event {
            if let Err(e) = self.base.load_media(core, path) {
                eprintln!("Failed to load dropped file: {e:?}");
            }
            return true;
        }
        false
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    cuneus::gst::init()?;
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("LEGO Effect", 1280, 720);

    app.run(event_loop, LegoShader::init)
}

use cuneus::compute::*;
use cuneus::prelude::*;

cuneus::uniform_params! {
    struct KuwaharaParams {
    radius: f32,
    q: f32,
    alpha: f32,
    tensor_sigma: f32,
    strength: f32,
    mode: i32,
    lic_length: f32,
    lic_strength: f32,
    lic_step: f32,
    saturation: f32,
    view: i32,
    _pad: f32}
}

impl Default for KuwaharaParams {
    fn default() -> Self {
        Self {
            radius: 6.0,
            q: 8.0,
            alpha: 1.0,
            tensor_sigma: 2.0,
            strength: 1.0,
            mode: 1,
            lic_length: 8.0,
            lic_strength: 0.4,
            lic_step: 1.0,
            saturation: 1.0,
            view: 0,
            _pad: 0.0}
    }
}

fn slider(ui: &mut egui::Ui, v: &mut f32, range: std::ops::RangeInclusive<f32>, label: &str) -> bool {
    ui.add(egui::Slider::new(v, range).text(label)).changed()
}

struct KuwaharaShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: KuwaharaParams,
    // frames left to render; a still image only renders after a change
    dirty: u32,
    was_exporting: bool}

impl ShaderManager for KuwaharaShader {
    fn init(core: &Core) -> Self {
        let initial_params = KuwaharaParams::default();
        let base = RenderKit::new(core);

        let passes = vec![
            PassDescription::new("source", &[]),
            PassDescription::new("structure_tensor", &["source"]),
            PassDescription::new("tensor_blur_h", &["structure_tensor"]),
            PassDescription::new("tensor_field", &["tensor_blur_h"]),
            PassDescription::new("kuwahara_filter", &["tensor_field", "source"]),
            PassDescription::new("lic_edges", &["tensor_field", "kuwahara_filter"]),
            PassDescription::new("main_image", &["lic_edges", "tensor_field", "source"]),
        ];

        let config = ComputeShader::builder()
            .with_entry_point("source")
            .with_multi_pass(&passes)
            .with_custom_uniforms::<KuwaharaParams>()
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_channels(1)
            .with_label("Kuwahara Multi-Pass")
            .build();

        let compute_shader = cuneus::compute_shader!(core, "shaders/kuwahara.wgsl", config);

        compute_shader.set_custom_params(initial_params, &core.queue);

        Self {
            base,
            compute_shader,
            current_params: initial_params,
            dirty: 2,
            was_exporting: false}
    }

    fn update(&mut self, core: &Core) {
        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0 / 60.0;
        self.compute_shader
            .set_time(current_time, delta, &core.queue);

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

        self.compute_shader.handle_export(core, &mut self.base);
    }

    fn resize(&mut self, core: &Core) {
        self.base.default_resize(core, &mut self.compute_shader);
        self.dirty = 2;
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

        let using_video_texture = self.base.using_video_texture;
        let using_hdri_texture = self.base.using_hdri_texture;
        let using_webcam_texture = self.base.using_webcam_texture;
        let video_info = self.base.get_video_info();
        let hdri_info = self.base.get_hdri_info();
        let webcam_info = self.base.get_webcam_info();

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                RenderKit::apply_default_style(ctx);

                egui::Window::new("Filter")
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
                        ui.horizontal(|ui| {
                            for (i, name) in ["Result", "Original", "Flow"].iter().enumerate() {
                                changed |= ui.selectable_value(&mut params.view, i as i32, *name).changed();
                            }
                        });

                        egui::CollapsingHeader::new("Kuwahara").default_open(true).show(ui, |ui| {
                            let mut aniso = params.mode == 1;
                            if ui.checkbox(&mut aniso, "Anisotropic (follow the flow)").changed() {
                                params.mode = aniso as i32;
                                changed = true;
                            }
                            changed |= slider(ui, &mut params.radius, 2.0..=12.0, "Radius");
                            changed |= slider(ui, &mut params.q, 1.0..=18.0, "Sharpness");
                            if params.mode == 1 {
                                changed |= ui.add(egui::Slider::new(&mut params.alpha, 0.2..=10.0).logarithmic(true).text("Stretch (lower = more)")).changed();
                                changed |= slider(ui, &mut params.tensor_sigma, 0.5..=6.0, "Flow smoothing");
                            }
                            changed |= slider(ui, &mut params.strength, 0.0..=1.0, "Strength");
                        });

                        egui::CollapsingHeader::new("Brush strokes").default_open(true).show(ui, |ui| {
                            changed |= slider(ui, &mut params.lic_strength, 0.0..=1.0, "Strength");
                            changed |= slider(ui, &mut params.lic_length, 0.0..=30.0, "Length");
                            changed |= slider(ui, &mut params.lic_step, 0.5..=3.0, "Step");
                        });

                        egui::CollapsingHeader::new("Colour").show(ui, |ui| {
                            changed |= slider(ui, &mut params.saturation, 0.0..=2.0, "Saturation");
                            if ui.button("Reset to defaults").clicked() {
                                params = KuwaharaParams::default();
                                changed = true;
                            }
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

        self.base.apply_media_requests(core, &controls_request);

        self.base.export_manager.apply_ui_request(export_request);
        if should_start_export {
            self.base.export_manager.start_export();
        }

        if changed || controls_request.load_media_path.is_some() || controls_request.start_webcam {
            self.dirty = 2;
        }
        if changed {
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
        }

        // video, webcam, shader edits and exports render every frame; a still image only after a change
        let exporting = self.base.export_manager.is_exporting();
        if self.compute_shader.check_hot_reload(&core.device) || exporting || self.was_exporting {
            self.dirty = 2;
        }
        self.was_exporting = exporting;
        if self.dirty > 0 || self.base.using_video_texture || self.base.using_webcam_texture {
            self.compute_shader.dispatch(&mut frame.encoder, core);
            self.dirty = self.dirty.saturating_sub(1);
        }

        self.base.renderer.render_to_view(&mut frame.encoder, &frame.view, &self.compute_shader.get_output_texture().bind_group);

        self.base.end_frame(core, frame, full_output);

        Ok(())
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        if let WindowEvent::DroppedFile(_) = event {
            self.dirty = 2;
        }
        self.base.default_handle_input(core, event)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    cuneus::gst::init()?;
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("Kuwahara Filter", 800, 600);

    app.run(event_loop, KuwaharaShader::init)
}

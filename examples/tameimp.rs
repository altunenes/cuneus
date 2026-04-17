use cuneus::compute::{ComputeShader, PassDescription};
use cuneus::prelude::*;
use cuneus::{Core, ExportManager, RenderKit, ShaderControls, ShaderManager};
use log::error;

cuneus::uniform_params! {
    struct ExperimentParams {
        col_bg: [f32; 4],
        col_line: [f32; 4],
        col_core: [f32; 4],
        col_amber: [f32; 4],

        ball_offset_x: f32,
        ball_offset_y: f32,
        ball_sink: f32,
        distortion_amt: f32,
        
        noise_amt: f32,
        stream_width: f32,
        scale: f32,
        angle: f32,
        
        line_freq: f32,
        cam_height: f32,
        cam_distance: f32,
        cam_fov: f32,
        
        ball_roughness: f32,
        ball_metalness: f32,
        gamma: f32,
        saturation: f32,
        
        exposure: f32,
        contrast: f32,
        max_bounces: u32,
        samples_per_pixel: u32,
        
        accumulate: u32,
        time_offset: f32,
        dof_strength: f32,
        focal_distance: f32,
        
        rotation_x: f32,
        rotation_y: f32,
        use_hdri: u32,
        animate_flow: u32,
    }
}

impl Default for ExperimentParams {
    fn default() -> Self {
        Self {
            col_bg: [0.05, 0.02, 0.10, 1.0],
            col_line: [0.55, 0.40, 0.85, 1.0],
            col_core: [1.0, 0.1, 0.2, 1.0],
            col_amber: [1.0, 0.6, 0.1, 1.0],

            ball_offset_x: 0.0,
            ball_offset_y: 0.0,
            ball_sink: 1.0,
            distortion_amt: 50.0,
            
            noise_amt: 300.0,
            stream_width: 0.08,
            scale: 1.35,
            angle: -1.785398,
            
            line_freq: 90.0,
            cam_height: 3.58,
            cam_distance: 6.0,
            cam_fov: 1.7,
            
            ball_roughness: 0.0,
            ball_metalness: 0.0,
            gamma: 0.4, 
            saturation: 1.0,
            
            exposure: 1.2,
            contrast: 1.1,
            max_bounces: 4,
            samples_per_pixel: 2,
            
            accumulate: 1,
            time_offset: 10.0,
            dof_strength: 0.02,
            focal_distance: 6.0,
            
            rotation_x: 0.0,
            rotation_y: 0.0,
            use_hdri: 0,
            animate_flow: 1,
        }
    }
}

struct ExperimentShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: ExperimentParams,
    should_reset_accumulation: bool,
}

impl ExperimentShader {
    fn reset_accumulation(&mut self) {
        self.compute_shader.current_frame = 0;
        self.should_reset_accumulation = false;
    }
}

impl ShaderManager for ExperimentShader {
    fn init(core: &Core) -> Self {
        let base = RenderKit::new(core);
        let initial_params = ExperimentParams::default();

        let passes = vec![
            PassDescription::new("accumulate", &["accumulate"]),
            PassDescription::new("main_image", &["accumulate"]),
        ];

        let config = ComputeShader::builder()
            .with_multi_pass(&passes)
            .with_custom_uniforms::<ExperimentParams>()
            .with_channels(1) 
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(cuneus::compute::COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Currents Path Tracer")
            .build();

        let compute_shader = cuneus::compute_shader!(core, "shaders/tameimp.wgsl", config);
        compute_shader.set_custom_params(initial_params, &core.queue);

        Self {
            base,
            compute_shader,
            current_params: initial_params,
            should_reset_accumulation: true,
        }
    }

    fn update(&mut self, core: &Core) {
        self.base.update_current_texture(core, &core.queue);
        if let Some(tm) = self.base.get_current_texture_manager() {
            self.compute_shader.update_channel_texture(0, &tm.view, &tm.sampler, &core.device, &core.queue);
            self.current_params.use_hdri = 1;
        } else {
            self.current_params.use_hdri = 0;
        }
        
        self.compute_shader.handle_export(core, &mut self.base);
    }

    fn render(&mut self, core: &Core) -> Result<(), cuneus::SurfaceError> {
        let mut frame = self.base.begin_frame(core)?;

        let mut params = self.current_params;
        let mut changed = false;
        let mut should_start_export = false;
        
        let current_fps = self.base.fps_tracker.fps();
        let mut export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self
            .base
            .controls
            .get_ui_request(&self.base.start_time, &core.size, current_fps);

        let using_hdri = self.base.using_hdri_texture || self.current_params.use_hdri == 1;
        let hdri_info = self.base.get_hdri_info();
        let using_video = self.base.using_video_texture;
        let video_info = self.base.get_video_info();
        
        let current_frame_count = self.compute_shader.current_frame;
        let mut manual_reset = false;

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                ctx.global_style_mut(|style| {
                    style.visuals.window_fill = egui::Color32::from_black_alpha(220);
                    style.visuals.window_stroke = egui::Stroke::new(1.0, egui::Color32::from_gray(60));
                    style.text_styles.get_mut(&egui::TextStyle::Body).unwrap().size = 11.0;
                    style.text_styles.get_mut(&egui::TextStyle::Button).unwrap().size = 10.0;
                    style.text_styles.get_mut(&egui::TextStyle::Small).unwrap().size = 9.0;
                    style.text_styles.get_mut(&egui::TextStyle::Heading).unwrap().size = 12.0;
                    style.spacing.slider_width = 140.0;
                    style.spacing.item_spacing = egui::vec2(4.0, 3.0);
                });

                egui::Window::new("Currents")
                    .default_width(220.0)
                    .show(ctx, |ui| {
                        
                        ShaderControls::render_media_panel(
                            ui,
                            &mut controls_request,
                            using_video,
                            video_info,
                            using_hdri,
                            hdri_info,
                            false,
                            None,
                        );

                        ui.separator();

                        egui::CollapsingHeader::new("Anim?").default_open(true).show(ui, |ui| {
                            let mut animate_bool = params.animate_flow > 0;
                            if ui.checkbox(&mut animate_bool, "EMA Blend").changed() {
                                params.animate_flow = if animate_bool { 1 } else { 0 };
                                changed = true;
                                manual_reset = true; 
                            }
                            
                            ui.horizontal(|ui| {
                                ui.label("Time Offset");
                                changed |= ui.add(egui::Slider::new(&mut params.time_offset, 0.0..=50.0).show_value(true)).changed();
                            });
                        });
                        
                        ui.separator();

                        egui::CollapsingHeader::new("Scene").default_open(true).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.ball_offset_x, -1.0..=1.0).text("Ball X")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.ball_offset_y, -1.0..=1.0).text("Ball Z")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.ball_sink, -3.2..=3.2).text("Ball Sink")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.angle, -3.14..=3.14).text("Plane Angle")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.scale, 0.3..=3.0).text("Scale")).changed();
                        });
                        
                        egui::CollapsingHeader::new("Ball").default_open(true).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.ball_roughness, 0.01..=1.0).text("Roughness")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.ball_metalness, 0.0..=1.0).text("Metalness")).changed();
                        });

                        egui::CollapsingHeader::new("Distortion").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.distortion_amt, 0.0..=75.0).text("Distortion")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.stream_width, 0.01..=0.3).text("Stream Width")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.line_freq, 20.0..=200.0).text("Line Freq")).changed();
                        });

                        egui::CollapsingHeader::new("Cam").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.cam_height, 0.5..=5.0).text("Height")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.cam_distance, 3.0..=10.0).text("Distance")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.cam_fov, 0.5..=2.5).text("FOV")).changed();
                            
                            ui.separator();
                            changed |= ui.add(egui::Slider::new(&mut params.dof_strength, 0.0..=0.2).text("Depth of Field")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.focal_distance, 1.0..=10.0).text("Focal Distance")).changed();
                        });

                        egui::CollapsingHeader::new("Path Tracing").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.max_bounces, 1..=8).text("Max Bounces")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.samples_per_pixel, 1..=8).text("Samples/Frame")).changed();
                            
                            let mut accumulate_bool = params.accumulate > 0;
                            if ui.checkbox(&mut accumulate_bool, "Progressive Accumulation").changed() {
                                params.accumulate = if accumulate_bool { 1 } else { 0 };
                                changed = true;
                            }
                        });

                        egui::CollapsingHeader::new("Post").default_open(false).show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label("Background");
                                let mut rgb = [params.col_bg[0], params.col_bg[1], params.col_bg[2]];
                                if ui.color_edit_button_rgb(&mut rgb).changed() {
                                    params.col_bg[0] = rgb[0]; params.col_bg[1] = rgb[1]; params.col_bg[2] = rgb[2];
                                    changed = true;
                                }
                            });
                            ui.horizontal(|ui| {
                                ui.label("Lines");
                                let mut rgb = [params.col_line[0], params.col_line[1], params.col_line[2]];
                                if ui.color_edit_button_rgb(&mut rgb).changed() {
                                    params.col_line[0] = rgb[0]; params.col_line[1] = rgb[1]; params.col_line[2] = rgb[2];
                                    changed = true;
                                }
                            });
                            ui.horizontal(|ui| {
                                ui.label("Stream Core");
                                let mut rgb = [params.col_core[0], params.col_core[1], params.col_core[2]];
                                if ui.color_edit_button_rgb(&mut rgb).changed() {
                                    params.col_core[0] = rgb[0]; params.col_core[1] = rgb[1]; params.col_core[2] = rgb[2];
                                    changed = true;
                                }
                            });
                            ui.separator();
                            changed |= ui.add(egui::Slider::new(&mut params.exposure, 0.1..=5.0).text("Exposure")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.gamma, 0.1..=2.0).text("Gamma")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.contrast, 0.5..=2.0).text("Contrast")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.saturation, 0.0..=2.5).text("Saturation")).changed();
                        });

                        ui.separator();
                        
                        if ui.button("Reset").clicked() {
                            manual_reset = true;
                        }

                        egui::CollapsingHeader::new("Controls").default_open(false).show(ui, |ui| {
                            ShaderControls::render_controls_widget(ui, &mut controls_request);
                        });

                        egui::CollapsingHeader::new("Export").default_open(false).show(ui, |ui| {
                            should_start_export = ExportManager::render_export_ui_widget(ui, &mut export_request);
                        });

                        ui.label(format!("Samples: {}", current_frame_count));
                        ui.label(format!("FPS: {:.1}", current_fps));
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        if manual_reset {
            self.should_reset_accumulation = true;
        }

        self.base.apply_media_requests(core, &controls_request);
        self.base.export_manager.apply_ui_request(export_request);
        
        if controls_request.should_clear_buffers || self.should_reset_accumulation || changed {
            self.reset_accumulation();
        }
        self.base.apply_control_request(controls_request);

        let current_time = self.base.controls.get_time(&self.base.start_time);
        self.compute_shader.set_time(current_time, 1.0 / 60.0, &core.queue);

        if changed || self.should_reset_accumulation {
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
        }

        if should_start_export {
            self.base.export_manager.start_export();
        }

        self.compute_shader.dispatch(&mut frame.encoder, core);

        self.base.renderer.render_to_view(
            &mut frame.encoder,
            &frame.view,
            &self.compute_shader.get_output_texture().bind_group,
        );

        self.base.end_frame(core, frame, full_output);
        Ok(())
    }

    fn resize(&mut self, core: &Core) {
        self.base.default_resize(core, &mut self.compute_shader);
        self.should_reset_accumulation = true;
    }

    fn handle_input(&mut self, core: &Core, event: &cuneus::WindowEvent) -> bool {
        if self.base.default_handle_input(core, event) {
            return true;
        }
        
        if let cuneus::WindowEvent::DroppedFile(path) = event {
            if let Err(e) = self.base.load_media(core, path) {
                error!("Failed to load dropped file: {e:?}");
            }
            self.should_reset_accumulation = true;
            return true;
        }
        
        false
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    cuneus::gst::init()?;
    env_logger::init();
    let (app, event_loop) = cuneus::ShaderApp::new("Currents Path Tracer", 800, 800);
    app.run(event_loop, ExperimentShader::init)
}
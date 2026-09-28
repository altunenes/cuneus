use cuneus::compute::{ComputeShader, ComputeShaderBuilder, PassDescription, StorageBufferSpec, COMPUTE_TEXTURE_FORMAT_RGBA16};
use cuneus::{Core, RenderKit, ShaderApp, ShaderControls, ShaderManager};
use cuneus::ExportManager;
use log::error;
use cuneus::WindowEvent;

cuneus::uniform_params! {
    struct GaussianParams {
        num_gaussians: u32,
        learning_rate: f32,
        color_learning_rate: f32,
        reset_training: u32,
        show_target: u32,
        show_error: u32,
        opacity_learning_rate: f32,
        error_scale: f32,
        min_sigma: f32,
        max_sigma: f32,
        freq_max: f32,
        random_seed: u32,
        iteration: u32,
        sigma_learning_rate: f32,
        draw_progress: f32,
        draw_grow: f32,
        oil_enable: u32,
        hardness: f32,
        bristle_amt: f32,
        canvas_amt: f32,
        edge_rag: f32,
        impasto: f32,
        curve_amt: f32,
        mode: u32,
        draw_prepare: u32,
        lr_decay_rate: f32,
        l1_mix: f32,
        dens: f32,
        _pd: u32,
        par: f32,
        pal_k: f32,
        pal_amt: f32,
        _pe: f32,
        _pf: f32,
        _pg: f32,
        _ph: f32,
    }
}

impl Default for GaussianParams {
    fn default() -> Self {
        Self {
            num_gaussians: 20000,

            learning_rate: 0.01,


            color_learning_rate: 0.008,

            reset_training: 0,
            show_target: 0,
            show_error: 0,

            opacity_learning_rate: 0.02,

            error_scale: 2.0,

            min_sigma: 0.001,

            max_sigma: 0.15,

            freq_max: 200.0,

            random_seed: 42,
            iteration: 0,

            sigma_learning_rate: 0.003,

            draw_progress: -1.0,
            draw_grow: 0.6,

            oil_enable: 1,
            hardness: 3.0,
            bristle_amt: 0.5,
            canvas_amt: 0.3,
            edge_rag: 0.4,
            impasto: 0.3,
            curve_amt: 0.3,
            mode: 0,
            draw_prepare: 0,
            lr_decay_rate: 0.0008,
            l1_mix: 0.5,
            dens: 1.0,
            _pd: 0,
            par: 0.0,
            pal_k: 8.0,
            pal_amt: 0.0,
            _pe: 0.0,
            _pf: 0.0,
            _pg: 0.0,
            _ph: 0.0,
        }
    }
}

struct GaussianShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: GaussianParams,
    drawing: bool,
    draw_t: f32,
    draw_speed: f32,
}

impl ShaderManager for GaussianShader {
    fn init(core: &Core) -> Self {
        let base = RenderKit::new(core);

        // 1. init_gaussians: Initialize/reset Gaussian parameters
        // 2. clear_gradients: Clear gradient buffer before each iteration
        // 3. render_display: Render Gaussians + compute gradients via backprop
        // 4. update_gaussians: Adam to update parameters
        let max_gaussians = 40000u32;
        let wg_1d = max_gaussians.div_ceil(256);
        let wg_clear = (max_gaussians * 12).div_ceil(256);

        let passes = vec![
            PassDescription::new("init_gaussians", &[]).with_workgroup_size([wg_1d, 1, 1]),
            PassDescription::new("clear_gradients", &[]).with_workgroup_size([wg_clear, 1, 1]),
            PassDescription::new("compute_draw_rank", &[]).with_workgroup_size([wg_1d, 1, 1]),
            PassDescription::new("bin_clear", &[]).with_workgroup_size([17, 1, 1]),
            PassDescription::new("bin_splats", &[]).with_workgroup_size([wg_1d, 1, 1]),
            PassDescription::new("pal_assign", &[]).with_workgroup_size([wg_1d, 1, 1]),
            PassDescription::new("pal_update", &[]).with_workgroup_size([1, 1, 1]),
            PassDescription::new("render_display", &[]),
            PassDescription::new("update_gaussians", &[]).with_workgroup_size([wg_1d, 1, 1]),
        ];

        let gaussian_buffer_size = (max_gaussians * 48) as u64;
        let gradient_buffer_size = (max_gaussians * 48) as u64;
        let adam_buffer_size = (max_gaussians * 48) as u64;
        let rank_buffer_size = (max_gaussians * 4) as u64;
        let err_buffer_size = (1u32 << 18) as u64 * 4;
        let bin_cnt_size = 4097u64 * 4;
        let bin_idx_size = (4096u64 * 2048 + max_gaussians as u64) * 4;

        let config = ComputeShaderBuilder::new()
            .with_label("Gaussian Splatting Training")
            .with_workgroup_size([8, 8, 1])
            .with_multi_pass(&passes)
            .with_channels(1)
            .with_custom_uniforms::<GaussianParams>()
            .with_storage_buffer(StorageBufferSpec::new("gaussian_params", gaussian_buffer_size))
            .with_storage_buffer(StorageBufferSpec::new("gradient_buffer", gradient_buffer_size))
            .with_storage_buffer(StorageBufferSpec::new("adam_first_moment", adam_buffer_size))
            .with_storage_buffer(StorageBufferSpec::new("adam_second_moment", adam_buffer_size))
            .with_storage_buffer(StorageBufferSpec::new("draw_rank", rank_buffer_size))
            .with_storage_buffer(StorageBufferSpec::new("err_grid", err_buffer_size))
            .with_storage_buffer(StorageBufferSpec::new("bin_cnt", bin_cnt_size))
            .with_storage_buffer(StorageBufferSpec::new("bin_idx", bin_idx_size))
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .build();

        let compute_shader = cuneus::compute_shader!(core, "shaders/gaussian.wgsl", config);

        let initial_params = GaussianParams::default();
        let shader = Self {
            base,
            compute_shader,
            current_params: initial_params,
            drawing: false,
            draw_t: 0.0,
            draw_speed: 0.35,
        };

        shader
            .compute_shader
            .set_custom_params(initial_params, &core.queue);

        shader
    }

    fn update(&mut self, core: &Core) {
        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0 / 60.0;
        self.compute_shader
            .set_time(current_time, delta, &core.queue);

        // Update target texture from media
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

        if self.drawing {
            self.draw_t = (self.draw_t + self.draw_speed * delta).min(1.0);
            self.current_params.draw_progress = self.draw_t;
            // painting order is built once, on the frame Draw starts
            self.current_params.draw_prepare = 0;
            self.compute_shader.set_custom_params(self.current_params, &core.queue);
        } else if self.current_params.reset_training == 0 {
            // Auto-increment iteration counter (live training)
            self.current_params.iteration = self.current_params.iteration.wrapping_add(1);
            self.current_params.draw_progress = -1.0;
            self.current_params.draw_prepare = 0;
            self.compute_shader.set_custom_params(self.current_params, &core.queue);
        }
        self.compute_shader.handle_export(core, &mut self.base);
    }

    fn resize(&mut self, core: &Core) {
        self.base.default_resize(core, &mut self.compute_shader);
    }

    fn render(&mut self, core: &Core) -> Result<(), cuneus::SurfaceError> {
        let mut frame = self.base.begin_frame(core)?;

        let mut controls_request = self
            .base
            .controls
            .get_ui_request(&self.base.start_time, &core.size, self.base.fps_tracker.fps());

        let mut params = self.current_params;
        let mut changed = false;
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();

        let mut start_draw = false;
        let mut stop_draw = false;
        let mut draw_speed_local = self.draw_speed;
        let drawing_now = self.drawing;
        let draw_t_now = self.draw_t;

        let using_video_texture = self.base.using_video_texture;
        let using_hdri_texture = self.base.using_hdri_texture;
        let using_webcam_texture = self.base.using_webcam_texture;
        let video_info = self.base.get_video_info();
        let hdri_info = self.base.get_hdri_info();
        let webcam_info = self.base.get_webcam_info();

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                RenderKit::apply_default_style(ctx);

                egui::Window::new("gaussian splatting")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(280.0)
                    .vscroll(true)   // never clip the lower sections (Export) on tall windows
                    .show(ctx, |ui| {
                        ui.label(format!("Iteration: {}", params.iteration));

                        egui::CollapsingHeader::new("Mode").default_open(true).show(ui, |ui| {
                                let mut m = params.mode;
                                ui.horizontal(|ui| {
                                    if ui.selectable_label(m == 0, "Gaussian").clicked() { m = 0; }
                                    if ui.selectable_label(m == 1, "Gabor").clicked() { m = 1; }
                                });
                                if m != params.mode {
                                    params.mode = m;
                                    params.reset_training = 1;
                                    params.iteration = 0;
                                    changed = true;
                                }
                                if params.mode == 1 {
                                    changed |= ui.add(egui::Slider::new(&mut params.freq_max, 10.0..=400.0).text("Max Frequency")).changed();
                                    ui.label("tip: lower N Splats (~5k) for gabor");
                                }
                            });

                        egui::CollapsingHeader::new("Draw").default_open(true).show(ui, |ui| {
                                if !drawing_now {
                                    if ui.button("Draw").clicked() { start_draw = true; }
                                    ui.label("freezes training, paints strokes big → fine");
                                } else {
                                    ui.label(format!("drawing… {:.0}%", draw_t_now * 100.0));
                                    ui.horizontal(|ui| {
                                        if ui.button("↺ Replay").clicked() { start_draw = true; }
                                        if ui.button("■ Resume training").clicked() { stop_draw = true; }
                                    });
                                }
                                ui.add(egui::Slider::new(&mut draw_speed_local, 0.05..=1.5).text("Draw speed"));
                                changed |= ui.add(egui::Slider::new(&mut params.draw_grow, 0.0..=1.0).text("Brush draw")).changed();
                            });

                        egui::CollapsingHeader::new("Shading").default_open(true).show(ui, |ui| {
                                let mut oil = params.oil_enable != 0;
                                if ui.checkbox(&mut oil, "Shading").changed() {
                                    params.oil_enable = if oil { 1 } else { 0 };
                                    changed = true;
                                }
                                changed |= ui.add(egui::Slider::new(&mut params.hardness, 1.0..=6.0).text("Hardness")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.bristle_amt, 0.0..=1.0).text("Bristle")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.canvas_amt, 0.0..=1.0).text("Canvas")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.edge_rag, 0.0..=1.5).text("Ragged edge")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.impasto, 0.0..=1.0).text("Impasto")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.curve_amt, 0.0..=1.0).text("Curve")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.pal_k, 2.0..=16.0).step_by(1.0).text("Palette colours")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.pal_amt, 0.0..=1.0).text("Palette")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.par, 0.0..=2.0).text("Parallax")).changed();
                            });

                        egui::CollapsingHeader::new("Training").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.num_gaussians, 100..=40000).text("N Gauss").logarithmic(true)).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.learning_rate, 0.0001..=0.1).text("pos LR").logarithmic(true)).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.color_learning_rate, 0.001..=0.2).text("col LR").logarithmic(true)).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.opacity_learning_rate, 0.001..=0.2).text("opacity LR").logarithmic(true)).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.l1_mix, 0.0..=1.0).text("L1 mix")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.dens, 0.0..=1.0).text("Error-guided respawn")).changed();
                            ui.separator();
                            if ui.button("res training").clicked() {
                                params.reset_training = 1;
                                params.iteration = 0;
                                changed = true;
                            }
                        });

                        egui::CollapsingHeader::new("vis").default_open(false).show(ui, |ui| {
                            let mut show_target = params.show_target != 0;
                            if ui.checkbox(&mut show_target, "Show Target").changed() {
                                params.show_target = if show_target { 1 } else { 0 };
                                changed = true;
                            }
                            let mut show_error = params.show_error != 0;
                            if ui.checkbox(&mut show_error, "Show Error").changed() {
                                params.show_error = if show_error { 1 } else { 0 };
                                changed = true;
                            }
                            if params.show_error != 0 {
                                changed |= ui.add(egui::Slider::new(&mut params.error_scale, 0.5..=10.0).text("Error Scale")).changed();
                            }
                        });

                        egui::CollapsingHeader::new("Advanced").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.sigma_learning_rate, 0.001..=0.1).text("Sigma LR").logarithmic(true)).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.min_sigma, 0.001..=0.05).text("Min Sigma").logarithmic(true)).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.max_sigma, 0.02..=0.3).text("Max Sigma").logarithmic(true)).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.lr_decay_rate, 0.0002..=0.005).text("LR decay").logarithmic(true)).changed();
                        });

                        ui.separator();
                        ShaderControls::render_controls_widget(ui, &mut controls_request);

                        ui.separator();
                        should_start_export =
                            ExportManager::render_export_ui_widget(ui, &mut export_request);

                        // media at the bottom, collapsed
                        ui.separator();
                        egui::CollapsingHeader::new("Source (image / video / webcam)").default_open(false).show(ui, |ui| {
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
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        self.base.export_manager.apply_ui_request(export_request);
        self.base.apply_media_requests(core, &controls_request);

        self.draw_speed = draw_speed_local;
        if start_draw {
            self.drawing = true;
            self.draw_t = 0.0;
            params.draw_progress = 0.0;
            params.draw_prepare = 1;
            changed = true;
        }
        if stop_draw {
            self.drawing = false;
            self.draw_t = 0.0;
            params.draw_progress = -1.0;
            params.draw_prepare = 0;
            changed = true;
        }

        if controls_request.should_clear_buffers || params.reset_training != 0 {
            self.compute_shader.current_frame = 0;
            self.compute_shader.time_uniform.data.frame = 0;
            self.compute_shader.time_uniform.update(&core.queue);

            // Fresh random layout on every reset
            params.random_seed = (std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0) % 10000) as u32;

            params.iteration = 0;
            params.reset_training = 0;
            // a reset also leaves any active "Draw" 
            self.drawing = false;
            self.draw_t = 0.0;
            params.draw_progress = -1.0;
            changed = true;
        }
        params.min_sigma = params.min_sigma.min(params.max_sigma);

        if changed {
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
        }

        if should_start_export {
            self.base.export_manager.start_export();
            //this section releated with the draw = export section. otherwise, buffers will be cleared and you can't export in the draw state... 
            if self.drawing {
                self.draw_t = 0.0;
                params.draw_progress = 0.0;
                params.draw_prepare = 1;
            } else {
                params.iteration = 0;
                params.random_seed = (std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis())
                    .unwrap_or(0) % 10000) as u32;
            }
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
        }

        if !self.base.export_manager.is_exporting() {
            self.compute_shader.dispatch(&mut frame.encoder, core);
        }

        self.base.renderer.render_to_view(&mut frame.encoder, &frame.view, &self.compute_shader.get_output_texture().bind_group);

        self.base.end_frame(core, frame, full_output);

        Ok(())
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        if let WindowEvent::DroppedFile(path) = event {
            if let Err(e) = self.base.load_media(core, path) {
                error!("Failed to load dropped file: {e:?}");
            }
            self.current_params.reset_training = 1;
            self.current_params.iteration = 0;
            self.compute_shader.set_custom_params(self.current_params, &core.queue);
            return true;
        }
        self.base.default_handle_input(core, event)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    cuneus::gst::init()?;
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("Splatting", 450, 350);
    app.run(event_loop, GaussianShader::init)
}

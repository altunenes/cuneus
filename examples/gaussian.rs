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
        _reserved1: f32,
        random_seed: u32,
        iteration: u32,
        sigma_learning_rate: f32,
        _padding0: u32,
        _padding1: u32,
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

            _reserved1: 0.0,

            random_seed: 42,
            iteration: 0,

            sigma_learning_rate: 0.003,

            _padding0: 0,
            _padding1: 0,
        }
    }
}

struct GaussianShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: GaussianParams,
}

impl ShaderManager for GaussianShader {
    fn init(core: &Core) -> Self {
        let base = RenderKit::new(core);

        // 1. init_gaussians: Initialize/reset Gaussian parameters
        // 2. clear_gradients: Clear gradient buffer before each iteration
        // 3. render_display: Render Gaussians + compute gradients via backprop
        // 4. update_gaussians: Adam to update parameters
        let max_gaussians = 20000u32;
        let wg_1d = max_gaussians.div_ceil(256); 
        let wg_clear = (max_gaussians * 9).div_ceil(256);

        let passes = vec![
            PassDescription::new("init_gaussians", &[]).with_workgroup_size([wg_1d, 1, 1]),
            PassDescription::new("clear_gradients", &[]).with_workgroup_size([wg_clear, 1, 1]),
            PassDescription::new("render_display", &[]),
            PassDescription::new("update_gaussians", &[]).with_workgroup_size([wg_1d, 1, 1]),
        ];

        // Storage buffers for training
        // Each Gaussian: position(2f32) + sigma(3f32) + color(3f32) + opacity(1f32) = 9 f32 (gradient data)
        let gaussian_buffer_size = (max_gaussians * 48) as u64;
        let gradient_buffer_size = (max_gaussians * 36) as u64;
        let adam_buffer_size = (max_gaussians * 36) as u64;

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
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .build();

        let compute_shader = cuneus::compute_shader!(core, "shaders/gaussian.wgsl", config);

        let initial_params = GaussianParams::default();
        let shader = Self {
            base,
            compute_shader,
            current_params: initial_params,
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

        // Auto-increment iteration counter
        if self.current_params.reset_training == 0 {
            self.current_params.iteration = self.current_params.iteration.wrapping_add(1);
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
                    .show(ctx, |ui| {
                        ui.label(format!("Iteration: {}", params.iteration));

                        egui::CollapsingHeader::new("Training")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.num_gaussians, 100..=20000)
                                            .text("N Gauss")
                                            .logarithmic(true),
                                    )
                                    .changed();

                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.learning_rate, 0.0001..=0.1)
                                            .text("pos LR")
                                            .logarithmic(true),
                                    )
                                    .changed();

                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.color_learning_rate, 0.001..=0.2)
                                            .text("col LR")
                                            .logarithmic(true),
                                    )
                                    .changed();

                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.opacity_learning_rate, 0.001..=0.2)
                                            .text("opacity LR")
                                            .logarithmic(true),
                                    )
                                    .changed();

                                ui.separator();

                                if ui.button("res training").clicked() {
                                    params.reset_training = 1;
                                    params.iteration = 0;
                                    changed = true;
                                }
                            });

                        egui::CollapsingHeader::new("vis")
                            .default_open(false)
                            .show(ui, |ui| {
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
                                    changed |= ui
                                        .add(
                                            egui::Slider::new(&mut params.error_scale, 0.5..=10.0)
                                                .text("Error Scale"),
                                        )
                                        .changed();
                                }
                            });

                        egui::CollapsingHeader::new("Advanced")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.sigma_learning_rate, 0.001..=0.1)
                                            .text("Sigma LR")
                                            .logarithmic(true),
                                    )
                                    .changed();

                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.min_sigma, 0.001..=0.05)
                                            .text("Min Sigma")
                                            .logarithmic(true),
                                    )
                                    .changed();

                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.max_sigma, 0.02..=0.3)
                                            .text("Max Sigma")
                                            .logarithmic(true),
                                    )
                                    .changed();
                            });

                        ui.separator();

                        ui.separator();

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
            changed = true;
        }
        params.min_sigma = params.min_sigma.min(params.max_sigma);

        if changed {
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
        }

        if should_start_export {
            self.base.export_manager.start_export();
            params.iteration = 0;
            params.random_seed = (std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0) % 10000) as u32;
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
    let (app, event_loop) = ShaderApp::new("2D Gaussian Splatting", 450, 350);
    app.run(event_loop, GaussianShader::init)
}

use cuneus::compute::{
    ComputeShader, PassDescription, StorageBufferSpec, COMPUTE_TEXTURE_FORMAT_RGBA16,
};
use cuneus::{Core, ExportManager, RenderKit, ShaderControls, ShaderManager, UniformProvider};
use winit::event::WindowEvent;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct FFTParams {
    filter_type: i32,
    filter_strength: f32,
    filter_direction: f32,
    filter_radius: f32,
    show_freqs: i32,
    resolution: u32,
    _padding1: u32,
    _padding2: u32,
}

impl UniformProvider for FFTParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct FFTShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    should_initialize: bool,
    current_params: FFTParams, // Store current parameters
}

impl ShaderManager for FFTShader {
    fn init(core: &Core) -> Self {
        let initial_params = FFTParams {
            filter_type: 1,
            filter_strength: 0.3,
            filter_direction: 0.0,
            filter_radius: 3.0,
            show_freqs: 0,
            resolution: 1024,
            _padding1: 0,
            _padding2: 0,
        };

        let texture_bind_group_layout = RenderKit::create_standard_texture_layout(&core.device);
        let base = RenderKit::new(core, &texture_bind_group_layout, None);

        // Define the FFT multi-pass pipeline
        let passes = vec![
            PassDescription::new("initialize_data", &[]), // Stage 0: Initialize from input texture
            PassDescription::new("fft_horizontal", &["initialize_data"]), // Stage 1: FFT horizontal pass
            PassDescription::new("fft_vertical", &["fft_horizontal"]), // Stage 2: FFT vertical pass
            PassDescription::new("modify_frequencies", &["fft_vertical"]), // Stage 3: Apply frequency domain filters
            PassDescription::new("ifft_horizontal", &["modify_frequencies"]), // Stage 4: Inverse FFT horizontal
            PassDescription::new("ifft_vertical", &["ifft_horizontal"]), // Stage 5: Inverse FFT vertical
            PassDescription::new("main_image", &["ifft_vertical"]),      // Stage 6: Final display
        ];

        let config = ComputeShader::builder()
            .with_entry_point("initialize_data") // Start with data initialization
            .with_multi_pass(&passes)
            .with_input_texture() // Re-enable input texture support
            .with_custom_uniforms::<FFTParams>()
            .with_storage_buffer(StorageBufferSpec::new("image_data", 1024 * 1024 * 3 * 8)) // FFT working memory: 3 channels × 8 bytes per complex number (vec2f)
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("FFT Multi-Pass")
            .build();

        let mut compute_shader =
            ComputeShader::from_builder(core, include_str!("shaders/fft.wgsl"), config);

        // Enable hot reload
        if let Err(e) = compute_shader.enable_hot_reload(
            core.device.clone(),
            std::path::PathBuf::from("examples/shaders/fft.wgsl"),
            core.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("FFT Compute Shader"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("shaders/fft.wgsl").into()),
                }),
        ) {
            eprintln!("Failed to enable FFT compute shader hot reload: {}", e);
        }

        // Initialize custom uniform with initial parameters
        compute_shader.set_custom_params(initial_params, &core.queue);

        Self {
            base,
            compute_shader,
            should_initialize: true,
            current_params: initial_params,
        }
    }

    fn update(&mut self, core: &Core) {
        // Update time
        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0 / 60.0;
        self.compute_shader
            .set_time(current_time, delta, &core.queue);

        // Update input textures for image proc.
        self.base.update_current_texture(core, &core.queue);
        if let Some(texture_manager) = self.base.get_current_texture_manager() {
            // Update input texture in unified ComputeShader
            self.compute_shader.update_input_texture(
                &texture_manager.view,
                &texture_manager.sampler,
                &core.device,
            );
        }

        self.base.fps_tracker.update();
        // Handle export
        self.compute_shader.handle_export(core, &mut self.base);
    }

    fn resize(&mut self, core: &Core) {
        self.compute_shader
            .resize(core, core.size.width, core.size.height);
    }

    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = core
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("FFT Render Encoder"),
            });

        // Handle UI and controls - using original transparent UI design
        let mut params = self.current_params;
        let mut changed = false;
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self
            .base
            .controls
            .get_ui_request(&self.base.start_time, &core.size);

        let using_video_texture = self.base.using_video_texture;
        let using_hdri_texture = self.base.using_hdri_texture;
        let using_webcam_texture = self.base.using_webcam_texture;
        let video_info = self.base.get_video_info();
        let hdri_info = self.base.get_hdri_info();
        let webcam_info = self.base.get_webcam_info();

        controls_request.current_fps = Some(self.base.fps_tracker.fps());
        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                ctx.style_mut(|style| {
                    style.visuals.window_fill =
                        egui::Color32::from_rgba_premultiplied(0, 0, 0, 180);
                    style
                        .text_styles
                        .get_mut(&egui::TextStyle::Body)
                        .unwrap()
                        .size = 11.0;
                    style
                        .text_styles
                        .get_mut(&egui::TextStyle::Button)
                        .unwrap()
                        .size = 10.0;
                });

                egui::Window::new("fourier workflow")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(250.0)
                    .show(ctx, |ui| {
                        // Media controls
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

                        egui::CollapsingHeader::new("FFT Settings")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.label("Resolution:");

                                ui.horizontal(|ui| {
                                    changed |= ui
                                        .radio_value(&mut params.resolution, 256, "256")
                                        .changed();
                                    changed |= ui
                                        .radio_value(&mut params.resolution, 512, "512")
                                        .changed();
                                    changed |= ui
                                        .radio_value(&mut params.resolution, 1024, "1024")
                                        .changed();
                                    changed |= ui
                                        .radio_value(&mut params.resolution, 2048, "2048")
                                        .changed();
                                });

                                if changed {
                                    self.should_initialize = true;
                                }

                                ui.separator();
                                ui.label("View Mode:");
                                changed |= ui
                                    .radio_value(&mut params.show_freqs, 0, "Filtered")
                                    .changed();
                                changed |= ui
                                    .radio_value(&mut params.show_freqs, 1, "Frequency Domain")
                                    .changed();

                                ui.separator();
                            });

                        egui::CollapsingHeader::new("Filter Settings")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.label("Filter Type:");
                                // Keep the improved ComboBox as requested
                                changed |= egui::ComboBox::from_label("")
                                    .selected_text(match params.filter_type {
                                        0 => "LP",
                                        1 => "HP",
                                        2 => "BP",
                                        3 => "Directional",
                                        _ => "None",
                                    })
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(&mut params.filter_type, 0, "LP")
                                            .changed()
                                            || ui
                                                .selectable_value(&mut params.filter_type, 1, "HP")
                                                .changed()
                                            || ui
                                                .selectable_value(&mut params.filter_type, 2, "BP")
                                                .changed()
                                            || ui
                                                .selectable_value(
                                                    &mut params.filter_type,
                                                    3,
                                                    "Directional",
                                                )
                                                .changed()
                                    })
                                    .inner
                                    .unwrap_or(false);

                                ui.separator();

                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.filter_strength, 0.0..=1.0)
                                            .text("Filter Strength"),
                                    )
                                    .changed();

                                if params.filter_type == 2 {
                                    changed |= ui
                                        .add(
                                            egui::Slider::new(
                                                &mut params.filter_radius,
                                                0.0..=6.28,
                                            )
                                            .text("Band Radius"),
                                        )
                                        .changed();
                                }

                                if params.filter_type == 3 {
                                    changed |= ui
                                        .add(
                                            egui::Slider::new(
                                                &mut params.filter_direction,
                                                0.0..=6.28,
                                            )
                                            .text("Direction"),
                                        )
                                        .changed();
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

        // Keep current parameters - don't reset to defaults
        // The UI will modify 'params' directly, and we'll apply changes at the end

        // Apply controls
        self.base.apply_control_request(controls_request.clone());
        self.base.handle_video_requests(core, &controls_request);
        self.base.handle_webcam_requests(core, &controls_request);

        // Handle export requests
        self.base.export_manager.apply_ui_request(export_request);
        if should_start_export {
            self.base.export_manager.start_export();
        }

        if controls_request.load_media_path.is_some() {
            self.should_initialize = true;
        }
        if controls_request.start_webcam {
            self.should_initialize = true;
        }

        // Apply parameter changes
        if changed {
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
            self.should_initialize = true; // Trigger FFT reprocessing
        }

        // FFT dispatch - only run full pipeline when needed, otherwise just display
        let mut should_run_full_fft = self.should_initialize
            || self.base.using_video_texture
            || self.base.using_webcam_texture
            || changed; // Also run when parameters change

        // FORCE run FFT if there's any texture to debug the issue
        let has_any_texture = self.base.get_current_texture_manager().is_some();
        if has_any_texture && !should_run_full_fft {
            should_run_full_fft = true;
        }
        // Get FFT resolution for proper workgroup calculation
        let n = params.resolution;
        if should_run_full_fft {
            // Stage 0: Initialize data from input texture (16x16 workgroups)
            self.compute_shader.dispatch_stage_with_workgroups(
                &mut encoder,
                0,
                [n.div_ceil(16), n.div_ceil(16), 1],
            );

            // Stage 1: FFT horizontal (Nx1 workgroups)
            self.compute_shader
                .dispatch_stage_with_workgroups(&mut encoder, 1, [n, 1, 1]);

            // Stage 2: FFT vertical (Nx1 workgroups)
            self.compute_shader
                .dispatch_stage_with_workgroups(&mut encoder, 2, [n, 1, 1]);

            // Stage 3: Modify frequencies - apply filter (16x16 workgroups)
            self.compute_shader.dispatch_stage_with_workgroups(
                &mut encoder,
                3,
                [n.div_ceil(16), n.div_ceil(16), 1],
            );

            // Stage 4: Inverse FFT horizontal (Nx1 workgroups)
            self.compute_shader
                .dispatch_stage_with_workgroups(&mut encoder, 4, [n, 1, 1]);

            // Stage 5: Inverse FFT vertical (Nx1 workgroups)
            self.compute_shader
                .dispatch_stage_with_workgroups(&mut encoder, 5, [n, 1, 1]);

            self.should_initialize = false;
            log::info!("Completed full FFT pipeline");
        } else {
            log::debug!("Skipping full FFT pipeline - using cached result");
        }

        // Stage 6: Main rendering - always run for display (uses screen size)
        self.compute_shader.dispatch_stage(&mut encoder, core, 6);

        // Display result using unified ComputeShader
        {
            let mut render_pass = cuneus::Renderer::begin_render_pass(
                &mut encoder,
                &view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("FFT Render Pass"),
            );

            let compute_texture = self.compute_shader.get_output_texture();
            render_pass.set_pipeline(&self.base.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &compute_texture.bind_group, &[]);
            render_pass.draw(0..4, 0..1);
        }

        self.base
            .handle_render_output(core, &view, full_output, &mut encoder);
        core.queue.submit(Some(encoder.finish()));
        output.present();

        Ok(())
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        if self
            .base
            .egui_state
            .on_window_event(core.window(), event)
            .consumed
        {
            return true;
        }

        if let WindowEvent::KeyboardInput { event, .. } = event {
            return self
                .base
                .key_handler
                .handle_keyboard_input(core.window(), event);
        }

        if let WindowEvent::DroppedFile(path) = event {
            if let Err(e) = self.base.load_media(core, path) {
                eprintln!("Failed to load dropped file: {:?}", e);
            } else {
                self.should_initialize = true;
            }
            return true;
        }

        false
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    cuneus::gst::init()?;
    env_logger::init();
    let (app, event_loop) = cuneus::ShaderApp::new("FFT", 800, 600);
    app.run(event_loop, |core| FFTShader::init(core))
}

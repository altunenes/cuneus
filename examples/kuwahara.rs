use cuneus::compute::*;
use cuneus::prelude::*;
use winit::event::WindowEvent;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct KuwaharaParams {
    radius: f32,
    q: f32,
    alpha: f32,
    filter_strength: f32,

    sigma_d: f32,
    sigma_r: f32,

    edge_threshold: f32,
    color_enhance: f32,

    blur_samples: f32,
    blur_lod: f32,
    blur_slod: f32,

    filter_mode: i32,
    show_tensors: i32,

    _padding: [u32; 3],
}

impl UniformProvider for KuwaharaParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct KuwaharaShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: KuwaharaParams,
}

impl ShaderManager for KuwaharaShader {
    fn init(core: &Core) -> Self {
        let initial_params = KuwaharaParams {
            radius: 5.0,
            q: 1.5,
            alpha: 4.0,
            filter_strength: 0.8,
            sigma_d: 0.8,
            sigma_r: 1.2,
            edge_threshold: 0.2,
            color_enhance: 1.0,
            blur_samples: 15.0,
            blur_lod: 2.0,
            blur_slod: 4.0,
            filter_mode: 1,
            show_tensors: 0,
            _padding: [0; 3],
        };

        let texture_bind_group_layout = RenderKit::create_standard_texture_layout(&core.device);
        let base = RenderKit::new(core, &texture_bind_group_layout, None);

        let passes = vec![
            PassDescription::new("structure_tensor", &[]),
            PassDescription::new("tensor_field", &["structure_tensor"]),
            PassDescription::new("kuwahara_filter", &["tensor_field"]),
            PassDescription::new("main_image", &["kuwahara_filter"]),
        ];

        let config = ComputeShader::builder()
            .with_entry_point("structure_tensor")
            .with_multi_pass(&passes)
            .with_custom_uniforms::<KuwaharaParams>()
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_channels(2)
            .with_label("Kuwahara Multi-Pass")
            .build();

        let mut compute_shader =
            ComputeShader::from_builder(core, include_str!("shaders/kuwahara.wgsl"), config);

        if let Err(e) = compute_shader.enable_hot_reload(
            core.device.clone(),
            std::path::PathBuf::from("examples/shaders/kuwahara.wgsl"),
            core.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Kuwahara Hot Reload"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("shaders/kuwahara.wgsl").into()),
                }),
        ) {
            eprintln!("Failed to enable hot reload for Kuwahara shader: {}", e);
        }

        compute_shader.set_custom_params(initial_params, &core.queue);

        Self {
            base,
            compute_shader,
            current_params: initial_params,
        }
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

        self.base.fps_tracker.update();

        self.compute_shader.check_hot_reload(&core.device);

        self.compute_shader.handle_export(core, &mut self.base);
    }

    fn resize(&mut self, core: &Core) {
        self.base.update_resolution(&core.queue, core.size);
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
                label: Some("Kuwahara Render Encoder"),
            });

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

        let current_fps = self.base.fps_tracker.fps();
        controls_request.current_fps = Some(current_fps);

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

                        let mut anisotropy_enabled = params.filter_mode == 1;
                        if ui
                            .checkbox(&mut anisotropy_enabled, "Anisotropy?")
                            .changed()
                        {
                            params.filter_mode = if anisotropy_enabled { 1 } else { 0 };
                            changed = true;
                        }

                        egui::CollapsingHeader::new("Filter Parameters")
                            .default_open(true)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.radius, 2.0..=16.0)
                                            .text("Radius"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.filter_strength, 0.0..=16.0)
                                            .text("Filter Strength"),
                                    )
                                    .changed();

                                if params.filter_mode == 1 {
                                    ui.separator();
                                    ui.label("Anisotropic Controls:");
                                    changed |= ui
                                        .add(
                                            egui::Slider::new(&mut params.alpha, 0.1..=16.0)
                                                .text("Anisotropy"),
                                        )
                                        .changed();
                                }
                            });
                        egui::CollapsingHeader::new("Blur Settings")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.blur_samples, 5.0..=25.0)
                                            .text("Samples"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.blur_lod, 0.0..=5.0)
                                            .text("LOD"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.blur_slod, 2.0..=5.0)
                                            .text("Step"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Post-Processing")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.color_enhance, 0.5..=2.0)
                                            .text("Color Filter"),
                                    )
                                    .changed();

                                ui.separator();
                                if ui.button("Reset to Defaults").clicked() {
                                    params = KuwaharaParams {
                                        radius: 8.0,
                                        q: 8.0,
                                        alpha: 1.0,
                                        filter_strength: 1.0,
                                        sigma_d: 1.0,
                                        sigma_r: 2.0,
                                        edge_threshold: 0.2,
                                        color_enhance: 1.0,
                                        blur_samples: 35.0,
                                        blur_lod: 2.0,
                                        blur_slod: 4.0,
                                        filter_mode: params.filter_mode,
                                        show_tensors: 0,
                                        _padding: [0; 3],
                                    };
                                    changed = true;
                                }
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
                        ui.label(format!("FPS: {:.1}", current_fps));
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        self.base.apply_control_request(controls_request.clone());
        self.base.handle_video_requests(core, &controls_request);
        self.base.handle_webcam_requests(core, &controls_request);
        self.base.handle_hdri_requests(core, &controls_request);

        self.base.export_manager.apply_ui_request(export_request);
        if should_start_export {
            self.base.export_manager.start_export();
        }

        if changed {
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
        }

        self.compute_shader.dispatch(&mut encoder, core);

        {
            let mut render_pass = cuneus::Renderer::begin_render_pass(
                &mut encoder,
                &view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("Kuwahara Display Pass"),
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
            }
            return true;
        }

        false
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    cuneus::gst::init()?;
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("Kuwahara Filter", 800, 600);

    app.run(event_loop, |core| KuwaharaShader::init(core))
}

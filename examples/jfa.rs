use cuneus::compute::*;
use cuneus::prelude::*;
use winit::event::WindowEvent;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct JfaParams {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    scale: f32,
    n: f32,
    gamma: f32,
    color_intensity: f32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
    color_w: f32,
    accumulation_speed: f32,
    fade_speed: f32,
    freeze_accumulation: f32,
    pattern_floor_add: f32,
    pattern_temp_add: f32,
    pattern_v_offset: f32,
    pattern_temp_mul1: f32,
    pattern_temp_mul2_3: f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
}

impl UniformProvider for JfaParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct JfaShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: JfaParams,
}

impl ShaderManager for JfaShader {
    fn init(core: &Core) -> Self {
        let initial_params = JfaParams {
            a: -2.7,
            b: 0.7,
            c: 0.2,
            d: 0.2,
            scale: 0.3,
            n: 10.0,
            gamma: 2.1,
            color_intensity: 1.0,
            color_r: 1.0,
            color_g: 2.0,
            color_b: 3.0,
            color_w: 4.0,
            accumulation_speed: 0.1,
            fade_speed: 0.99,
            freeze_accumulation: 0.0,
            pattern_floor_add: 1.0,
            pattern_temp_add: 0.1,
            pattern_v_offset: 0.7,
            pattern_temp_mul1: 0.7,
            pattern_temp_mul2_3: 3.0,
            _padding0: 0.0,
            _padding1: 0.0,
            _padding2: 0.0,
        };

        let texture_bind_group_layout = RenderKit::create_standard_texture_layout(&core.device);
        let base = RenderKit::new(core, &texture_bind_group_layout, None);

        // Create multipass system: buffer_a -> buffer_b -> buffer_c -> main_image
        let passes = vec![
            PassDescription::new("buffer_a", &["buffer_a"]), // self-feedback
            PassDescription::new("buffer_b", &["buffer_a", "buffer_b"]), // reads buffer_a + self-feedback
            PassDescription::new("buffer_c", &["buffer_a", "buffer_b", "buffer_c"]), // reads ALL 3 buffers
            PassDescription::new("main_image", &["buffer_c"]),
        ];

        let config = ComputeShader::builder()
            .with_entry_point("buffer_a")
            .with_multi_pass(&passes)
            .with_custom_uniforms::<JfaParams>()
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("JFA Unified")
            .build();

        let mut compute_shader =
            ComputeShader::from_builder(core, include_str!("shaders/jfa.wgsl"), config);

        // Enable hot reload
        if let Err(e) = compute_shader.enable_hot_reload(
            core.device.clone(),
            std::path::PathBuf::from("examples/shaders/jfa.wgsl"),
            core.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("JFA Hot Reload"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("shaders/jfa.wgsl").into()),
                }),
        ) {
            eprintln!("Failed to enable hot reload for JFA shader: {}", e);
        }

        compute_shader.set_custom_params(initial_params, &core.queue);

        Self {
            base,
            compute_shader,
            current_params: initial_params,
        }
    }

    fn update(&mut self, core: &Core) {
        // Check for hot reload updates
        self.compute_shader.check_hot_reload(&core.device);
        // Handle export
        self.compute_shader.handle_export(core, &mut self.base);

        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0 / 60.0;
        self.compute_shader
            .set_time(current_time, delta, &core.queue);

        self.base.fps_tracker.update();
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
                label: Some("JFA Render Encoder"),
            });

        // Execute multi-pass compute shader: buffer_a -> buffer_b -> buffer_c -> main_image
        self.compute_shader.dispatch(&mut encoder, core);

        // Render compute output to screen
        {
            let compute_texture = self.compute_shader.get_output_texture();
            let mut render_pass = cuneus::Renderer::begin_render_pass(
                &mut encoder,
                &view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("JFA Display Pass"),
            );

            render_pass.set_pipeline(&self.base.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &compute_texture.bind_group, &[]);
            render_pass.draw(0..4, 0..1);
        }

        // Handle UI and controls
        let mut params = self.current_params;
        let mut changed = false;
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self
            .base
            .controls
            .get_ui_request(&self.base.start_time, &core.size);
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

                egui::Window::new("JFA - Simplified")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(280.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("JFA Parameters")
                            .default_open(true)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.n, 1.0..=50.0)
                                            .text("N (Frame Cycle)"),
                                    )
                                    .changed();
                                ui.separator();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(
                                            &mut params.accumulation_speed,
                                            0.0..=3.0,
                                        )
                                        .text("Accumulation Speed"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.fade_speed, 0.9..=1.0)
                                            .text("Fade Speed"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Clifford Attractor")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(egui::Slider::new(&mut params.a, -5.0..=5.0).text("a"))
                                    .changed();
                                changed |= ui
                                    .add(egui::Slider::new(&mut params.b, -5.0..=5.0).text("b"))
                                    .changed();
                                changed |= ui
                                    .add(egui::Slider::new(&mut params.c, -5.0..=5.0).text("c"))
                                    .changed();
                                changed |= ui
                                    .add(egui::Slider::new(&mut params.d, -5.0..=5.0).text("d"))
                                    .changed();
                                ui.separator();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.scale, 0.1..=1.0)
                                            .text("Scale"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Colors")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Color Pattern:");
                                    let mut color =
                                        [params.color_r, params.color_g, params.color_b];
                                    if ui.color_edit_button_rgb(&mut color).changed() {
                                        params.color_r = color[0];
                                        params.color_g = color[1];
                                        params.color_b = color[2];
                                        changed = true;
                                    }
                                });
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.color_w, 0.0..=10.0)
                                            .text("Color W"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.color_intensity, 0.1..=3.0)
                                            .text("Color Intensity"),
                                    )
                                    .changed();
                                ui.separator();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.gamma, 0.1..=4.0)
                                            .text("Gamma"),
                                    )
                                    .changed();
                            });

                        ui.separator();
                        ShaderControls::render_controls_widget(ui, &mut controls_request);

                        ui.separator();
                        should_start_export =
                            ExportManager::render_export_ui_widget(ui, &mut export_request);

                        ui.separator();
                        ui.label(format!("Frame: {}", self.compute_shader.current_frame));
                        ui.label("JFA with Clifford Attractor (Simplified)");
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        // Handle control requests
        if controls_request.is_paused != (params.freeze_accumulation > 0.5) {
            params.freeze_accumulation = if controls_request.is_paused { 1.0 } else { 0.0 };
            changed = true;
        }

        self.base.export_manager.apply_ui_request(export_request);
        if controls_request.should_clear_buffers {
            // Reset frame count to restart accumulation
            self.compute_shader.current_frame = 0;
        }
        self.base.apply_control_request(controls_request);

        if changed {
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
        }

        if should_start_export {
            self.base.export_manager.start_export();
        }

        self.base
            .handle_render_output(core, &view, full_output, &mut encoder);
        core.queue.submit(std::iter::once(encoder.finish()));
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
        false
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("JFA", 800, 600);

    app.run(event_loop, |core| JfaShader::init(core))
}

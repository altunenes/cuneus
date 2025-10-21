use cuneus::compute::*;
use cuneus::prelude::*;
use winit::event::WindowEvent;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SpiralParams {
    a: f32,
    b: f32,
    c: f32,
    dof_amount: f32,
    dof_focal_dist: f32,
    rotation_x: f32,
    rotation_y: f32,
    click_state: i32,
    brightness: f32,
    color1_r: f32,
    color1_g: f32,
    color1_b: f32,
    color2_r: f32,
    color2_g: f32,
    color2_b: f32,
    _padding: u32,
}

impl UniformProvider for SpiralParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct SpiralShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: SpiralParams,
}

impl SpiralShader {
    fn clear_buffers(&mut self, core: &Core) {
        self.compute_shader.clear_all_buffers(core);
    }
}

impl ShaderManager for SpiralShader {
    fn init(core: &Core) -> Self {
        let texture_bind_group_layout = RenderKit::create_standard_texture_layout(&core.device);

        let initial_params = SpiralParams {
            a: 1.0,
            b: 1.0,
            c: 1.0,
            dof_amount: 1.0,
            dof_focal_dist: 1.0,
            rotation_x: 0.0,
            rotation_y: 0.0,
            click_state: 0,
            brightness: 0.00004,
            color1_r: 0.0,
            color1_g: 0.7,
            color1_b: 1.0,
            color2_r: 1.0,
            color2_g: 0.3,
            color2_b: 0.5,
            _padding: 0,
        };

        let base = RenderKit::new(core, &texture_bind_group_layout, None);

        let mut config = ComputeShader::builder()
            .with_entry_point("Splat")
            .with_custom_uniforms::<SpiralParams>()
            .with_atomic_buffer()
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Spiralchaos Unified")
            .build();

        // Add second entry point manually
        config.entry_points.push("main_image".to_string());

        let mut compute_shader =
            ComputeShader::from_builder(core, include_str!("shaders/spiralchaos.wgsl"), config);

        // Enable hot reload
        if let Err(e) = compute_shader.enable_hot_reload(
            core.device.clone(),
            std::path::PathBuf::from("examples/shaders/spiralchaos.wgsl"),
            core.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Spiralchaos Hot Reload"),
                    source: wgpu::ShaderSource::Wgsl(
                        include_str!("shaders/spiralchaos.wgsl").into(),
                    ),
                }),
        ) {
            eprintln!("Failed to enable hot reload for Spiralchaos shader: {}", e);
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
        // Handle export with custom dispatch pattern
        self.compute_shader.handle_export_dispatch(
            core,
            &mut self.base,
            |shader, encoder, core| {
                shader.dispatch_stage_with_workgroups(encoder, 0, [4096, 1, 1]);
                shader.dispatch_stage(encoder, core, 1);
            },
        );

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
                label: Some("Render Encoder"),
            });

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

                egui::Window::new("Chaos Spiral")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(250.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("Spiral Parameters")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.a, 0.0..=3.0)
                                            .text("Tightness"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(egui::Slider::new(&mut params.b, 0.0..=3.0).text("Speed"))
                                    .changed();
                                changed |= ui
                                    .add(egui::Slider::new(&mut params.c, 0.0..=3.0).text("N Arms"))
                                    .changed();
                            });

                        egui::CollapsingHeader::new("DOF")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.dof_amount, 0.0..=3.0)
                                            .text("Amount"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.dof_focal_dist, 0.0..=3.0)
                                            .text("Focal Distance"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Rotation")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.rotation_x, -1.0..=1.0)
                                            .text("X"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.rotation_y, -1.0..=1.0)
                                            .text("Y"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Colors")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.brightness, 0.00001..=0.0001)
                                            .logarithmic(true)
                                            .text("Brightness"),
                                    )
                                    .changed();

                                ui.horizontal(|ui| {
                                    ui.label("Color 1:");
                                    let mut color =
                                        [params.color1_r, params.color1_g, params.color1_b];
                                    if ui.color_edit_button_rgb(&mut color).changed() {
                                        params.color1_r = color[0];
                                        params.color1_g = color[1];
                                        params.color1_b = color[2];
                                        changed = true;
                                    }
                                });

                                ui.horizontal(|ui| {
                                    ui.label("Color 2:");
                                    let mut color =
                                        [params.color2_r, params.color2_g, params.color2_b];
                                    if ui.color_edit_button_rgb(&mut color).changed() {
                                        params.color2_r = color[0];
                                        params.color2_g = color[1];
                                        params.color2_b = color[2];
                                        changed = true;
                                    }
                                });
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
        if controls_request.should_clear_buffers {
            self.clear_buffers(core);
        }
        self.base.apply_control_request(controls_request);

        let current_time = self.base.controls.get_time(&self.base.start_time);

        let delta = 1.0 / 60.0;
        self.compute_shader
            .set_time(current_time, delta, &core.queue);

        // Mouse data integration
        params.rotation_x = self.base.mouse_tracker.uniform.position[0];
        params.rotation_y = self.base.mouse_tracker.uniform.position[1];
        params.click_state = if self.base.mouse_tracker.uniform.buttons[0] & 1 > 0 {
            1
        } else {
            0
        };
        changed = true;

        if changed {
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
        }

        if should_start_export {
            self.base.export_manager.start_export();
        }

        // Note: in cuneus, individual stage dispatch methods need manual frame management (if you need of course!)

        // Stage 0: Splat particles (workgroup size [256, 1, 1])
        self.compute_shader
            .dispatch_stage_with_workgroups(&mut encoder, 0, [4096, 1, 1]);

        // Stage 1: Render to screen (workgroup size [16, 16, 1])
        self.compute_shader.dispatch_stage(&mut encoder, core, 1);

        {
            let mut render_pass = cuneus::Renderer::begin_render_pass(
                &mut encoder,
                &view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("Display Pass"),
            );

            render_pass.set_pipeline(&self.base.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &self.compute_shader.output_texture.bind_group, &[]);

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

        false
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = cuneus::ShaderApp::new("Chaos Spiral", 800, 600);

    app.run(event_loop, |core| SpiralShader::init(core))
}

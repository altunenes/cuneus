use cuneus::compute::*;
use cuneus::prelude::*;
use winit::event::WindowEvent;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct RorschachParams {
    // Matrix 1
    m1_scale: f32,
    m1_y_scale: f32,
    // Matrix 2
    m2_scale: f32,
    m2_shear: f32,
    m2_shift: f32,
    // Matrix 3
    m3_scale: f32,
    m3_shear: f32,
    m3_shift: f32,
    // Matrix 4
    m4_scale: f32,
    m4_shift: f32,
    // Matrix 5
    m5_scale: f32,
    m5_shift: f32,
    time_scale: f32,
    decay: f32,
    intensity: f32,
    rotation_x: f32,
    rotation_y: f32,
    click_state: i32,
    brightness: f32,
    exposure: f32,
    gamma: f32,
    particle_count: f32,
    scale: f32,
    dof_amount: f32,
    dof_focal_dist: f32,
    color1_r: f32,
    color1_g: f32,
    color1_b: f32,
    color2_r: f32,
    color2_g: f32,
    color2_b: f32,
}

impl UniformProvider for RorschachParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct RorschachShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: RorschachParams,
}

impl RorschachShader {
    fn clear_buffers(&mut self, core: &Core) {
        self.compute_shader.clear_all_buffers(core);
    }
}

impl ShaderManager for RorschachShader {
    fn init(core: &Core) -> Self {
        // Create texture bind group layout for displaying compute shader output
        let texture_bind_group_layout = RenderKit::create_standard_texture_layout(&core.device);
        let mut base = RenderKit::new(core, &texture_bind_group_layout, None);
        base.setup_mouse_uniform(core);

        let initial_params = RorschachParams {
            m1_scale: 0.8,
            m1_y_scale: 0.5,
            m2_scale: 0.4,
            m2_shear: 0.2,
            m2_shift: 0.3,
            m3_scale: 0.4,
            m3_shear: 0.2,
            m3_shift: 0.3,
            m4_scale: 0.3,
            m4_shift: 0.2,
            m5_scale: 0.2,
            m5_shift: 0.4,
            time_scale: 0.5,
            decay: 0.0,
            intensity: 0.0,
            rotation_x: 0.0,
            rotation_y: 0.0,
            click_state: 0,
            brightness: 0.003,
            exposure: 1.5,
            gamma: 0.4,
            particle_count: 100000.0,
            scale: 1.0,
            dof_amount: 0.0,
            dof_focal_dist: 0.5,
            color1_r: 1.0,
            color1_g: 0.3,
            color1_b: 0.1,
            color2_r: 0.1,
            color2_g: 0.5,
            color2_b: 1.0,
        };

        let mut config = ComputeShader::builder()
            .with_entry_point("Splat")
            .with_custom_uniforms::<RorschachParams>()
            .with_atomic_buffer()
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Rorschach IFS Unified")
            .build();

        // Add second entry point manually (no ping-pong needed)
        config.entry_points.push("main_image".to_string());

        let mut compute_shader =
            ComputeShader::from_builder(core, include_str!("shaders/rorschach.wgsl"), config);

        // Enable hot reload
        if let Err(e) = compute_shader.enable_hot_reload(
            core.device.clone(),
            std::path::PathBuf::from("examples/shaders/rorschach.wgsl"),
            core.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Rorschach Hot Reload"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("shaders/rorschach.wgsl").into()),
                }),
        ) {
            eprintln!("Failed to enable hot reload for Rorschach shader: {}", e);
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

        self.base.update_mouse_uniform(&core.queue);
        self.base.fps_tracker.update();
    }
    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut params = self.current_params;
        let mut changed = false;
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self
            .base
            .controls
            .get_ui_request(&self.base.start_time, &core.size);
        controls_request.current_fps = Some(self.base.fps_tracker.fps());
        if self.base.mouse_tracker.uniform.buttons[0] & 1 != 0 {
            params.rotation_x = self.base.mouse_tracker.uniform.position[0];
            params.rotation_y = self.base.mouse_tracker.uniform.position[1];
            params.click_state = 1;
            changed = true;
        } else if self.base.mouse_tracker.uniform.buttons[0] & 2 != 0 {
            params.click_state = 0;
        } else {
            params.click_state = 0;
        }

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

                egui::Window::new("Rorschach IFS")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(280.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("IFS Matrices")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.label("Matrix 1:");
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.m1_scale, 0.1..=1.2)
                                            .text("Scale"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.m1_y_scale, 0.1..=1.2)
                                            .text("Y Scale"),
                                    )
                                    .changed();

                                ui.separator();
                                ui.label("Matrix 2:");
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.m2_scale, 0.1..=1.0)
                                            .text("Scale"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.m2_shear, -0.5..=0.5)
                                            .text("Shear"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.m2_shift, -0.5..=0.5)
                                            .text("Shift"),
                                    )
                                    .changed();

                                ui.separator();
                                ui.label("Matrix 3:");
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.m3_scale, 0.1..=1.0)
                                            .text("Scale"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.m3_shear, -0.5..=0.5)
                                            .text("Shear"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.m3_shift, -0.5..=0.5)
                                            .text("Shift"),
                                    )
                                    .changed();

                                ui.separator();
                                ui.label("Matrix 4 & 5:");
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.m4_scale, 0.1..=1.0)
                                            .text("M4 Scale"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.m4_shift, -0.5..=0.5)
                                            .text("M4 Shift"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.m5_scale, 0.1..=1.0)
                                            .text("M5 Scale"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.m5_shift, -0.5..=0.5)
                                            .text("M5 Shift"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Colors")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Primary:");
                                    let mut color1 =
                                        [params.color1_r, params.color1_g, params.color1_b];
                                    if ui.color_edit_button_rgb(&mut color1).changed() {
                                        params.color1_r = color1[0];
                                        params.color1_g = color1[1];
                                        params.color1_b = color1[2];
                                        changed = true;
                                    }
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Secondary:");
                                    let mut color2 =
                                        [params.color2_r, params.color2_g, params.color2_b];
                                    if ui.color_edit_button_rgb(&mut color2).changed() {
                                        params.color2_r = color2[0];
                                        params.color2_g = color2[1];
                                        params.color2_b = color2[2];
                                        changed = true;
                                    }
                                });
                            });

                        egui::CollapsingHeader::new("Rendering")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(
                                            &mut params.particle_count,
                                            10000.0..=200000.0,
                                        )
                                        .text("Particles"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.scale, 0.1..=3.0)
                                            .text("Scale"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.brightness, 0.001..=0.01)
                                            .text("Brightness"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.exposure, 0.5..=3.0)
                                            .text("Exposure"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.gamma, 0.1..=2.0)
                                            .text("Gamma"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Depth of Field")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.dof_amount, 0.0..=2.0)
                                            .text("DOF Amount"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.dof_focal_dist, 0.0..=1.0)
                                            .text("Focal Distance"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Animation")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.time_scale, 0.0..=2.0)
                                            .text("Animation Speed"),
                                    )
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
        if controls_request.should_clear_buffers {
            self.clear_buffers(core);
        }
        self.base.apply_control_request(controls_request);

        let current_time = self.base.controls.get_time(&self.base.start_time);

        let mut encoder = core
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        let delta = 1.0 / 60.0;
        self.compute_shader
            .set_time(current_time, delta, &core.queue);

        if changed {
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
        }

        if should_start_export {
            self.base.export_manager.start_export();
        }

        // Stage 0: Generate and splat particles (workgroup size [256, 1, 1])
        let workgroups = (self.current_params.particle_count as u32 / 256).max(1);
        self.compute_shader
            .dispatch_stage_with_workgroups(&mut encoder, 0, [workgroups, 1, 1]);

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
        let ui_handled = self
            .base
            .egui_state
            .on_window_event(core.window(), event)
            .consumed;

        if ui_handled {
            return true;
        }

        if self.base.handle_mouse_input(core, event, false) {
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

    fn resize(&mut self, core: &Core) {
        self.base.update_resolution(&core.queue, core.size);
        self.compute_shader
            .resize(core, core.size.width, core.size.height);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = cuneus::ShaderApp::new("Rorschach IFS", 800, 600);
    app.run(event_loop, |core| RorschachShader::init(core))
}

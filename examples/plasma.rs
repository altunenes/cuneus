use cuneus::compute::*;
use cuneus::prelude::*;
use winit::event::WindowEvent;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct NeuralParams {
    detail: f32,
    animation_speed: f32,
    pattern: f32,
    structure_smoothness: f32,
    saturation: f32,
    base_rotation: f32,
    rot_variation: f32,
    rotation_x: f32,
    rotation_y: f32,
    click_state: i32,
    brightness_mult: f32,
    color1_r: f32,
    color1_g: f32,
    color1_b: f32,
    color2_r: f32,
    color2_g: f32,
    color2_b: f32,
    dof_amount: f32,
    dof_focal_dist: f32,
}

impl UniformProvider for NeuralParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct Neural2Shader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: NeuralParams,
    mouse_look_enabled: bool,
}

impl Neural2Shader {
    fn clear_buffers(&mut self, core: &Core) {
        self.compute_shader.clear_all_buffers(core);
    }
}

impl ShaderManager for Neural2Shader {
    fn init(core: &Core) -> Self {
        let texture_bind_group_layout = RenderKit::create_standard_texture_layout(&core.device);

        let initial_params = NeuralParams {
            detail: 15.0,
            animation_speed: 0.1,
            pattern: 0.3,
            structure_smoothness: 1.0,
            saturation: 0.7,
            base_rotation: 7.6,
            rot_variation: 0.0070,
            rotation_x: -0.6,
            rotation_y: 0.15,
            click_state: 0,
            brightness_mult: 0.00004,
            color1_r: 0.5,
            color1_g: 0.1,
            color1_b: 0.8,
            color2_r: 0.0,
            color2_g: 0.7,
            color2_b: 1.0,
            dof_amount: 0.95,
            dof_focal_dist: 2.0,
        };

        let base = RenderKit::new(core, &texture_bind_group_layout, None);

        let mut config = ComputeShader::builder()
            .with_entry_point("Splat")
            .with_custom_uniforms::<NeuralParams>()
            .with_atomic_buffer()
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Neural Wave Unified")
            .build();

        // Add second entry point manually
        config.entry_points.push("main_image".to_string());

        let mut compute_shader =
            ComputeShader::from_builder(core, include_str!("shaders/plasma.wgsl"), config);

        // Enable hot reload
        if let Err(e) = compute_shader.enable_hot_reload(
            core.device.clone(),
            std::path::PathBuf::from("examples/shaders/plasma.wgsl"),
            core.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Plasma Hot Reload"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("shaders/plasma.wgsl").into()),
                }),
        ) {
            eprintln!("Failed to enable hot reload for Plasma shader: {}", e);
        }

        compute_shader.set_custom_params(initial_params, &core.queue);

        Self {
            base,
            compute_shader,
            current_params: initial_params,
            mouse_look_enabled: false,
        }
    }

    fn update(&mut self, core: &Core) {
        // Check for hot reload updates
        self.compute_shader.check_hot_reload(&core.device);
        // Handle export
        self.compute_shader.handle_export(core, &mut self.base);

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

                egui::Window::new("Neural Wave")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(250.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("Pattern")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.detail, 3.0..=45.0)
                                            .text("Detail"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.animation_speed, 0.1..=6.0)
                                            .text("v"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.pattern, 0.0..=1.0)
                                            .text("pattern"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(
                                            &mut params.structure_smoothness,
                                            1.0..=3.5,
                                        )
                                        .text("Smoothness"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.saturation, 0.1..=1.0)
                                            .text("Saturation"),
                                    )
                                    .changed();
                                ui.separator();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.base_rotation, 3.0..=12.0)
                                            .text("rot"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.rot_variation, 0.0..=0.1)
                                            .text("rot var"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Camera")
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.checkbox(&mut self.mouse_look_enabled, "Enable Mouse Look");
                                ui.separator();

                                if !self.mouse_look_enabled {
                                    changed |= ui
                                        .add(
                                            egui::Slider::new(&mut params.rotation_x, -1.0..=1.0)
                                                .text("Rotation X"),
                                        )
                                        .changed();
                                    changed |= ui
                                        .add(
                                            egui::Slider::new(&mut params.rotation_y, -1.0..=1.0)
                                                .text("Rotation Y"),
                                        )
                                        .changed();
                                } else {
                                    ui.label("Mouse Look Active - Move mouse to control camera");
                                }
                            });

                        egui::CollapsingHeader::new("Visual Settings")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(
                                            &mut params.brightness_mult,
                                            0.00001..=0.0001,
                                        )
                                        .logarithmic(true)
                                        .text("Brightness"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Depth of Field")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.dof_amount, 0.0..=3.0)
                                            .text("DOF Amount"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.dof_focal_dist, 0.0..=3.0)
                                            .text("Focal Distance"),
                                    )
                                    .changed();
                                params.click_state = 1;
                            });

                        egui::CollapsingHeader::new("Colors")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Base Color:");
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
                                    ui.label("Highlight Color:");
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

                        ui.separator();
                        ui.label("Controls:");
                        ui.horizontal(|ui| {
                            ui.label("• Mouse:");
                            if self.mouse_look_enabled {
                                ui.colored_label(egui::Color32::GREEN, "Active");
                            } else {
                                ui.colored_label(egui::Color32::RED, "Disabled");
                            }
                        });
                        ui.label("• Right click: Toggle mouse control");
                        ui.label("• H: Toggle UI");

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
        if self.mouse_look_enabled {
            params.rotation_x = self.base.mouse_tracker.uniform.position[0];
            params.rotation_y = self.base.mouse_tracker.uniform.position[1];
        }
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

        // Stage 0: Generate and splat particles (workgroup size [256, 1, 1])
        self.compute_shader
            .dispatch_stage_with_workgroups(&mut encoder, 0, [2048, 1, 1]);

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
        if let WindowEvent::MouseInput { state, button, .. } = event {
            if *button == winit::event::MouseButton::Right {
                if *state == winit::event::ElementState::Released {
                    self.mouse_look_enabled = !self.mouse_look_enabled;
                    return true;
                }
            }
        }
        if self.mouse_look_enabled && self.base.handle_mouse_input(core, event, false) {
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
    let (app, event_loop) = cuneus::ShaderApp::new("Neural Wave", 800, 600);

    app.run(event_loop, |core| Neural2Shader::init(core))
}

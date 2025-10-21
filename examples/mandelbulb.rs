use cuneus::compute::ComputeShader;
use cuneus::{
    Core, ExportManager, MouseUniform, RenderKit, ShaderControls, ShaderManager, UniformProvider,
};
use winit::event::WindowEvent;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MandelbulbParams {
    power: f32,
    max_bounces: u32,
    samples_per_pixel: u32,
    accumulate: u32,

    animation_speed: f32,
    hold_duration: f32,
    transition_duration: f32,

    exposure: f32,
    focal_length: f32,
    dof_strength: f32,

    palette_a_r: f32,
    palette_a_g: f32,
    palette_a_b: f32,
    palette_b_r: f32,
    palette_b_g: f32,
    palette_b_b: f32,
    palette_c_r: f32,
    palette_c_g: f32,
    palette_c_b: f32,
    palette_d_r: f32,
    palette_d_g: f32,
    palette_d_b: f32,

    gamma: f32,
    zoom: f32,

    background_r: f32,
    background_g: f32,
    background_b: f32,
    sun_color_r: f32,
    sun_color_g: f32,
    sun_color_b: f32,
    fog_color_r: f32,
    fog_color_g: f32,
    fog_color_b: f32,
    glow_color_r: f32,
    glow_color_g: f32,
    glow_color_b: f32,
}

impl UniformProvider for MandelbulbParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct MandelbulbShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    frame_count: u32,
    should_reset_accumulation: bool,
    current_params: MandelbulbParams,
    // Mouse tracking for accumulation reset
    previous_mouse_pos: [f32; 2],
    mouse_enabled: bool,
}

impl MandelbulbShader {
    fn clear_atomic_buffer(&mut self, core: &Core) {
        self.compute_shader.clear_atomic_buffer(core);
        self.should_reset_accumulation = false;
        self.frame_count = 0;
    }
}

impl ShaderManager for MandelbulbShader {
    fn init(core: &Core) -> Self {
        let initial_params = MandelbulbParams {
            power: 8.0,
            max_bounces: 6,
            samples_per_pixel: 2,
            accumulate: 1,

            animation_speed: 1.0,
            hold_duration: 3.0,
            transition_duration: 3.0,

            exposure: 1.5,
            focal_length: 6.0,
            dof_strength: 0.02,

            palette_a_r: 0.5,
            palette_a_g: 0.5,
            palette_a_b: 0.5,
            palette_b_r: 0.5,
            palette_b_g: 0.1,
            palette_b_b: 0.1,
            palette_c_r: 1.0,
            palette_c_g: 1.0,
            palette_c_b: 1.0,
            palette_d_r: 0.0,
            palette_d_g: 0.33,
            palette_d_b: 0.67,

            gamma: 1.1,
            zoom: 1.0,

            background_r: 0.1,
            background_g: 0.1,
            background_b: 0.15,
            sun_color_r: 8.10,
            sun_color_g: 6.00,
            sun_color_b: 4.20,
            fog_color_r: 0.1,
            fog_color_g: 0.1,
            fog_color_b: 0.15,
            glow_color_r: 0.5,
            glow_color_g: 0.7,
            glow_color_b: 1.0,
        };
        let texture_bind_group_layout = RenderKit::create_standard_texture_layout(&core.device);
        let base = RenderKit::new(core, &texture_bind_group_layout, None);

        let config = ComputeShader::builder()
            .with_entry_point("main")
            .with_custom_uniforms::<MandelbulbParams>()
            .with_mouse() // Enable mouse backend integration
            .with_atomic_buffer() // Enable atomic buffer for path tracing
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(cuneus::compute::COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Mandelbulb Unified")
            .build();

        let mut compute_shader =
            ComputeShader::from_builder(core, include_str!("shaders/mandelbulb.wgsl"), config);

        // Enable hot reload
        if let Err(e) = compute_shader.enable_hot_reload(
            core.device.clone(),
            std::path::PathBuf::from("examples/shaders/mandelbulb.wgsl"),
            core.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Mandelbulb Hot Reload"),
                    source: wgpu::ShaderSource::Wgsl(
                        include_str!("shaders/mandelbulb.wgsl").into(),
                    ),
                }),
        ) {
            eprintln!("Failed to enable hot reload for mandelbulb shader: {}", e);
        }

        // Initialize custom uniform with initial parameters
        compute_shader.set_custom_params(initial_params, &core.queue);

        Self {
            base,
            compute_shader,
            frame_count: 0,
            should_reset_accumulation: true,
            current_params: initial_params,
            previous_mouse_pos: [0.0, 0.0],
            mouse_enabled: true,
        }
    }

    fn update(&mut self, core: &Core) {
        self.base.fps_tracker.update();

        // Check for hot reload updates
        self.compute_shader.check_hot_reload(&core.device);

        // Handle export
        self.compute_shader.handle_export(core, &mut self.base);
    }

    fn resize(&mut self, core: &Core) {
        println!("Resizing to {:?}", core.size);
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

        // Check if mouse moved and reset accumulation if needed
        let current_mouse_pos = self.base.mouse_tracker.uniform.position;
        if self.mouse_enabled {
            let moved = (current_mouse_pos[0] - self.previous_mouse_pos[0]).abs() > 0.001
                || (current_mouse_pos[1] - self.previous_mouse_pos[1]).abs() > 0.001;
            if moved {
                self.should_reset_accumulation = true;
                self.previous_mouse_pos = current_mouse_pos;
            }
        }

        // Update mouse uniform only if mouse is enabled, otherwise use static position
        if self.mouse_enabled {
            self.compute_shader
                .update_mouse_uniform(&self.base.mouse_tracker.uniform, &core.queue);
        } else {
            // Use a fixed mouse position when disabled
            let static_mouse = MouseUniform {
                position: [0.5, 0.5], // Center position
                click_position: [0.5, 0.5],
                wheel: [0.0, 0.0],
                buttons: [0, 0],
            };
            self.compute_shader
                .update_mouse_uniform(&static_mouse, &core.queue);
        }

        let mut params = self.current_params;
        let mut changed = false;
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self
            .base
            .controls
            .get_ui_request(&self.base.start_time, &core.size);
        controls_request.current_fps = Some(self.base.fps_tracker.fps());

        let current_fps = self.base.fps_tracker.fps();

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

                egui::Window::new("Mandelbulb PathTracer")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(350.0)
                    .show(ctx, |ui| {
                        ui.label("Mouse - Rotate camera (when enabled)");
                        ui.label("M key - Toggle mouse on/off");
                        ui.separator();

                        egui::CollapsingHeader::new("Camera&View")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.zoom, 0.1..=5.0).text("Zoom"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.focal_length, 2.0..=20.0)
                                            .text("Focal Length"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.dof_strength, 0.0..=1.0)
                                            .text("DoF"),
                                    )
                                    .changed();

                                ui.separator();
                                let old_mouse_enabled = self.mouse_enabled;
                                ui.checkbox(&mut self.mouse_enabled, "Mouse Camera Control");
                                if self.mouse_enabled != old_mouse_enabled {
                                    if !self.mouse_enabled {
                                        // When disabling mouse, reset accumulation one more time
                                        self.should_reset_accumulation = true;
                                    }
                                }
                                if !self.mouse_enabled {
                                    ui.colored_label(
                                        egui::Color32::GRAY,
                                        "Mouse disabled - camera locked",
                                    );
                                } else {
                                    ui.colored_label(
                                        egui::Color32::GREEN,
                                        "Mouse enabled - move to rotate camera",
                                    );
                                }
                            });

                        egui::CollapsingHeader::new("Mandelbulb")
                            .default_open(false)
                            .show(ui, |ui| {
                                let old_power = params.power;
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.power, 2.0..=12.0)
                                            .text("Power"),
                                    )
                                    .changed();
                                if params.power != old_power {
                                    self.should_reset_accumulation = true;
                                }
                            });

                        egui::CollapsingHeader::new("Render")
                            .default_open(false)
                            .show(ui, |ui| {
                                let old_samples = params.samples_per_pixel;
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.samples_per_pixel, 1..=8)
                                            .text("Samples/pixel"),
                                    )
                                    .changed();
                                if params.samples_per_pixel != old_samples {
                                    self.should_reset_accumulation = true;
                                }

                                let old_bounces = params.max_bounces;
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.max_bounces, 1..=12)
                                            .text("Max Bounces"),
                                    )
                                    .changed();
                                if params.max_bounces != old_bounces {
                                    self.should_reset_accumulation = true;
                                }

                                let old_accumulate = params.accumulate;
                                let mut accumulate_bool = params.accumulate > 0;
                                changed |= ui
                                    .checkbox(&mut accumulate_bool, "Progressive Rendering")
                                    .changed();
                                params.accumulate = if accumulate_bool { 1 } else { 0 };
                                if params.accumulate != old_accumulate {
                                    self.should_reset_accumulation = true;
                                }

                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.exposure, 0.1..=5.0)
                                            .text("Exposure"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.gamma, 0.1..=2.0)
                                            .text("Gamma"),
                                    )
                                    .changed();

                                if ui.button("Reset Accumulation").clicked() {
                                    self.should_reset_accumulation = true;
                                    changed = true;
                                }
                            });

                        egui::CollapsingHeader::new("env")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("bg:");
                                    let mut bg_color = [
                                        params.background_r,
                                        params.background_g,
                                        params.background_b,
                                    ];
                                    if ui.color_edit_button_rgb(&mut bg_color).changed() {
                                        params.background_r = bg_color[0];
                                        params.background_g = bg_color[1];
                                        params.background_b = bg_color[2];
                                        changed = true;
                                    }
                                });

                                ui.horizontal(|ui| {
                                    ui.label("Sun:");
                                    let mut sun_color = [
                                        params.sun_color_r,
                                        params.sun_color_g,
                                        params.sun_color_b,
                                    ];
                                    if ui.color_edit_button_rgb(&mut sun_color).changed() {
                                        params.sun_color_r = sun_color[0];
                                        params.sun_color_g = sun_color[1];
                                        params.sun_color_b = sun_color[2];
                                        changed = true;
                                    }
                                });

                                ui.horizontal(|ui| {
                                    ui.label("Fog:");
                                    let mut fog_color = [
                                        params.fog_color_r,
                                        params.fog_color_g,
                                        params.fog_color_b,
                                    ];
                                    if ui.color_edit_button_rgb(&mut fog_color).changed() {
                                        params.fog_color_r = fog_color[0];
                                        params.fog_color_g = fog_color[1];
                                        params.fog_color_b = fog_color[2];
                                        changed = true;
                                    }
                                });

                                ui.horizontal(|ui| {
                                    ui.label("Sky Glow:");
                                    let mut glow_color = [
                                        params.glow_color_r,
                                        params.glow_color_g,
                                        params.glow_color_b,
                                    ];
                                    if ui.color_edit_button_rgb(&mut glow_color).changed() {
                                        params.glow_color_r = glow_color[0];
                                        params.glow_color_g = glow_color[1];
                                        params.glow_color_b = glow_color[2];
                                        changed = true;
                                    }
                                });

                                if ui.button("Reset env cols").clicked() {
                                    params.background_r = 0.1;
                                    params.background_g = 0.1;
                                    params.background_b = 0.15;
                                    params.sun_color_r = 8.10;
                                    params.sun_color_g = 6.00;
                                    params.sun_color_b = 4.20;
                                    params.fog_color_r = 0.1;
                                    params.fog_color_g = 0.1;
                                    params.fog_color_b = 0.15;
                                    params.glow_color_r = 0.5;
                                    params.glow_color_g = 0.7;
                                    params.glow_color_b = 1.0;
                                    changed = true;
                                }
                            });

                        egui::CollapsingHeader::new("Color Palette")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Base Color:");
                                    let mut color_a = [
                                        params.palette_a_r,
                                        params.palette_a_g,
                                        params.palette_a_b,
                                    ];
                                    if ui.color_edit_button_rgb(&mut color_a).changed() {
                                        params.palette_a_r = color_a[0];
                                        params.palette_a_g = color_a[1];
                                        params.palette_a_b = color_a[2];
                                        changed = true;
                                    }
                                });

                                ui.horizontal(|ui| {
                                    ui.label("Amplitude:");
                                    let mut color_b = [
                                        params.palette_b_r,
                                        params.palette_b_g,
                                        params.palette_b_b,
                                    ];
                                    if ui.color_edit_button_rgb(&mut color_b).changed() {
                                        params.palette_b_r = color_b[0];
                                        params.palette_b_g = color_b[1];
                                        params.palette_b_b = color_b[2];
                                        changed = true;
                                    }
                                });

                                ui.horizontal(|ui| {
                                    ui.label("Frequency:");
                                    let mut color_c = [
                                        params.palette_c_r,
                                        params.palette_c_g,
                                        params.palette_c_b,
                                    ];
                                    if ui.color_edit_button_rgb(&mut color_c).changed() {
                                        params.palette_c_r = color_c[0];
                                        params.palette_c_g = color_c[1];
                                        params.palette_c_b = color_c[2];
                                        changed = true;
                                    }
                                });

                                ui.horizontal(|ui| {
                                    ui.label("Phase:");
                                    let mut color_d = [
                                        params.palette_d_r,
                                        params.palette_d_g,
                                        params.palette_d_b,
                                    ];
                                    if ui.color_edit_button_rgb(&mut color_d).changed() {
                                        params.palette_d_r = color_d[0];
                                        params.palette_d_g = color_d[1];
                                        params.palette_d_b = color_d[2];
                                        changed = true;
                                    }
                                });
                                if ui.button("Reset to Default Palette").clicked() {
                                    params.palette_a_r = 0.5;
                                    params.palette_a_g = 0.5;
                                    params.palette_a_b = 0.5;
                                    params.palette_b_r = 0.5;
                                    params.palette_b_g = 0.1;
                                    params.palette_b_b = 0.1;
                                    params.palette_c_r = 1.0;
                                    params.palette_c_g = 1.0;
                                    params.palette_c_b = 1.0;
                                    params.palette_d_r = 0.0;
                                    params.palette_d_g = 0.33;
                                    params.palette_d_b = 0.67;
                                    changed = true;
                                }

                                ui.separator();
                            });

                        ui.separator();

                        ShaderControls::render_controls_widget(ui, &mut controls_request);

                        ui.separator();

                        should_start_export =
                            ExportManager::render_export_ui_widget(ui, &mut export_request);

                        ui.separator();
                        ui.label(format!("Accumulated Samples: {}", self.frame_count));
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

        self.base.export_manager.apply_ui_request(export_request);
        if controls_request.should_clear_buffers || self.should_reset_accumulation {
            self.clear_atomic_buffer(core);
        }
        self.base.apply_control_request(controls_request);

        let current_time = self.base.controls.get_time(&self.base.start_time);

        self.base.time_uniform.data.time = current_time;
        self.base.time_uniform.data.frame = self.frame_count;
        self.base.time_uniform.update(&core.queue);

        // Update compute shader with the same time data
        self.compute_shader
            .set_time(current_time, 1.0 / 60.0, &core.queue);
        self.compute_shader.time_uniform.data.frame = self.frame_count;
        self.compute_shader.time_uniform.update(&core.queue);

        if changed {
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
            self.should_reset_accumulation = true;
        }

        if should_start_export {
            self.base.export_manager.start_export();
        }

        self.compute_shader.dispatch(&mut encoder, core);

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

        if self.current_params.accumulate > 0 {
            self.frame_count += 1;
        }

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

        if self.base.handle_mouse_input(core, event, false) {
            return true;
        }

        if let WindowEvent::KeyboardInput { event, .. } = event {
            if let winit::keyboard::Key::Character(ch) = &event.logical_key {
                match ch.as_str() {
                    " " => {
                        if event.state == winit::event::ElementState::Released {
                            self.current_params.accumulate = 1 - self.current_params.accumulate;
                            self.should_reset_accumulation = true;
                            self.compute_shader
                                .set_custom_params(self.current_params, &core.queue);
                            return true;
                        }
                    }
                    "m" | "M" => {
                        if event.state == winit::event::ElementState::Released {
                            self.mouse_enabled = !self.mouse_enabled;
                            if !self.mouse_enabled {
                                self.should_reset_accumulation = true;
                            }
                            return true;
                        }
                    }
                    _ => {}
                }
            }
        }

        if let WindowEvent::KeyboardInput { event, .. } = event {
            if self
                .base
                .key_handler
                .handle_keyboard_input(core.window(), event)
            {
                return true;
            }
        }

        false
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = cuneus::ShaderApp::new("Mandelbulb Path Tracer", 800, 600);

    app.run(event_loop, |core| MandelbulbShader::init(core))
}

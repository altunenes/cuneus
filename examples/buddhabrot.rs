use cuneus::compute::*;
use cuneus::prelude::*;
use winit::event::WindowEvent;

cuneus::uniform_params! {
    struct BuddhabrotParams {
    max_iterations: u32,
    escape_radius: f32,
    zoom: f32,
    offset_x: f32,
    offset_y: f32,
    rotation: f32,
    exposure: f32,
    low_iterations: u32,
    high_iterations: u32,
    motion_speed: f32,
    color1_r: f32,
    color1_g: f32,
    color1_b: f32,
    color2_r: f32,
    color2_g: f32,
    color2_b: f32,
    sample_density: f32,
    dithering: f32,
    _pad_m1: f32,
    _pad_m2: f32,
    }
}

struct BuddhabrotShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    frame_count: u32,
    accumulated_rendering: bool,
    current_params: BuddhabrotParams}

impl BuddhabrotShader {
    fn clear_buffers(&mut self, core: &Core) {
        // Clear atomic buffer (by recreating it)
        self.compute_shader.clear_atomic_buffer(core);

        self.compute_shader.current_frame = 0;
        self.frame_count = 0;
        self.accumulated_rendering = false;
    }
}

impl ShaderManager for BuddhabrotShader {
    fn init(core: &Core) -> Self {
        let base = RenderKit::new(core);

        let initial_params = BuddhabrotParams {
            max_iterations: 500,
            escape_radius: 4.0,
            zoom: 0.5,
            offset_x: -0.5,
            offset_y: 0.0,
            rotation: 1.5,
            exposure: 0.0005,
            low_iterations: 20,
            high_iterations: 100,
            motion_speed: 0.0,
            color1_r: 1.0,
            color1_g: 0.5,
            color1_b: 0.2,
            color2_r: 0.2,
            color2_g: 0.5,
            color2_b: 1.0,
            sample_density: 0.5,
            dithering: 0.2,
            _pad_m1: 0.0,
            _pad_m2: 0.0,
        };

        let mut config = ComputeShader::builder()
            .with_entry_point("Splat")
            .with_custom_uniforms::<BuddhabrotParams>()
            .with_atomic_buffer()
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Buddhabrot Unified")
            .build();

        // Add second entry point
        config.entry_points.push("main_image".to_string());

        let compute_shader = cuneus::compute_shader!(core, "shaders/buddhabrot.wgsl", config);


        compute_shader.set_custom_params(initial_params, &core.queue);

        Self {
            base,
            compute_shader,
            frame_count: 0,
            accumulated_rendering: false,
            current_params: initial_params}
    }

    fn update(&mut self, core: &Core) {
        // Handle export
        self.compute_shader.handle_export_dispatch(
            core,
            &mut self.base,
            |shader, encoder, core| {
                shader.dispatch_stage_with_workgroups(encoder, 0, [2048, 1, 1]);
                shader.dispatch_stage(encoder, core, 1);
            },
        );
    }

    fn resize(&mut self, core: &Core) {
        self.base.default_resize(core, &mut self.compute_shader);
    }

    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let mut frame = self.base.begin_frame(core)?;

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
                RenderKit::apply_default_style(ctx);

                egui::Window::new("Buddhabrot Explorer")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(300.0)
                    .min_width(250.0)
                    .max_width(500.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("Fractal Parameters")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.max_iterations, 100..=500)
                                            .text("Max Iterations"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.escape_radius, 2.0..=10.0)
                                            .text("Escape Radius"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.low_iterations, 5..=50)
                                            .text("Low Iterations"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.high_iterations, 50..=500)
                                            .text("High Iterations"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("View Controls")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.zoom, 0.1..=5.0)
                                            .logarithmic(true)
                                            .text("Zoom"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.offset_x, -2.0..=1.0)
                                            .text("Offset X"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.offset_y, -1.5..=1.5)
                                            .text("Offset Y"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.rotation, -3.14159..=3.14159)
                                            .text("Rotation"),
                                    )
                                    .changed();
                                ui.add_space(10.0);
                                ui.separator();

                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.exposure, 0.00005..=0.001)
                                            .logarithmic(true)
                                            .text("Exposure"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.sample_density, 0.1..=2.0)
                                            .text("Sample Density"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.dithering, 0.0..=1.0)
                                            .text("Dithering"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Colors")
                            .default_open(false)
                            .show(ui, |ui| {
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

                        egui::CollapsingHeader::new("Rendering Options")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Accumulated?:");
                                    ui.checkbox(&mut self.accumulated_rendering, "");
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

        if changed {
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);

            // Clear buffers when parameters change (unless in accumulated mode)
            if !self.accumulated_rendering {
                self.clear_buffers(core);
            }
        }

        if should_start_export {
            self.base.export_manager.start_export();
        }

        // Only generate new samples if we're not in accumulated mode
        // or if we're still accumulating (frame count < 500) - use frame counter
        let should_generate_samples =
            !self.accumulated_rendering || self.compute_shader.current_frame < 500;

        if should_generate_samples {
            self.compute_shader
                .dispatch_stage_with_workgroups(&mut frame.encoder, 0, [2048, 1, 1]);
        }

        // Always dispatch stage 1 (main_image) for rendering with screen-based workgroups
        // Note: in cuneus, individual stage dispatch methods need manual frame management (if you need of course!)

        self.compute_shader.dispatch_stage(&mut frame.encoder, core, 1);

        //Manual frame increment since dispatch_stage() doesn't auto-increment
        self.compute_shader.current_frame += 1;

        self.base.renderer.render_to_view(&mut frame.encoder, &frame.view, &self.compute_shader);

        self.base.end_frame(core, frame, full_output);
        self.frame_count = self.frame_count.wrapping_add(1);

        Ok(())
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        self.base.default_handle_input(core, event)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = cuneus::ShaderApp::new("Buddhabrot", 800, 600);

    app.run(event_loop, BuddhabrotShader::init)
}

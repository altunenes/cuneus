use cuneus::compute::*;
use cuneus::prelude::*;

cuneus::uniform_params! {
    struct BuddhabrotParams {
        max_iterations: u32,
        escape_radius: f32,
        zoom: f32,
        offset_x: f32,
        offset_y: f32,
        rotation: f32,
        exposure: f32,
        sample_density: f32,
        dithering: f32,
        wavelength_min: f32,
        wavelength_max: f32,
        gamma: f32,
        saturation: f32,
        color_shift: f32,
        intensity_scale: f32,
        white_balance_r: f32,
        white_balance_g: f32,
        white_balance_b: f32,
        min_trajectory_len: u32,
        sampling: u32,
        rot_a: f32,
        rot_b: f32,
        spin_a: f32,
        spin_b: f32,
        persist: f32,
        z0x: f32,
        z0y: f32,
        mut_size: f32,
        rot_c: f32,
        rot_d: f32,
        _p0: f32,
        _p1: f32,
    }
}

// one Metropolis chain per sampling thread
const CHAINS: usize = 2048 * 64;

struct BuddhabrotShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: BuddhabrotParams,
    dragging: bool,
    last_cursor: Option<(f64, f64)>,
}

impl BuddhabrotShader {
    fn clear_buffers(&mut self, core: &Core) {
        self.compute_shader.clear_atomic_buffer(core);
        self.compute_shader.current_frame = 0;
    }

    // complex_to_screen inverted
    fn to_plane(&self, core: &Core, x: f64, y: f64, zoom: f32) -> (f32, f32) {
        let (w, h) = (core.size.width as f32, core.size.height as f32);
        let (u, v) = ((x as f32 / w - 0.5) * 2.0 * w / h, (y as f32 / h - 0.5) * 2.0);
        let (s, c) = self.current_params.rotation.sin_cos();
        ((c * u + s * v) / zoom, (-s * u + c * v) / zoom)
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
            rotation: 1.55,
            exposure: 3.5,
            sample_density: 0.5,
            dithering: 0.2,
            wavelength_min: 485.0,
            wavelength_max: 660.0,
            gamma: 0.6,
            saturation: 1.2,
            color_shift: 1.1,
            intensity_scale: 5.0,
            white_balance_r: 1.2,
            white_balance_g: 1.0,
            white_balance_b: 1.08,
            min_trajectory_len: 20,
            sampling: 1,
            rot_a: 0.0,
            rot_b: 0.0,
            spin_a: 0.0,
            spin_b: 0.0,
            persist: 0.9,
            z0x: 0.0,
            z0y: 0.0,
            mut_size: 1.0,
            rot_c: 0.0,
            rot_d: 0.0,
            _p0: 0.0,
            _p1: 0.0,
        };

        let passes = vec![
            PassDescription::new("Splat", &[]).with_workgroup_size([(CHAINS / 64) as u32, 1, 1]),
            PassDescription::new("main_image", &[]),
        ];
        let config = ComputeShader::builder()
            .with_entry_point("Splat")
            .with_multi_pass(&passes)
            .with_custom_uniforms::<BuddhabrotParams>()
            // the Metropolis chains, kept between frames (audio buffer as storage)
            .with_audio(CHAINS * 4)
            .with_atomic_buffer(3)
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Spectral Buddhabrot")
            .build();

        let compute_shader = cuneus::compute_shader!(core, "shaders/buddhabrot.wgsl", config);
        compute_shader.set_custom_params(initial_params, &core.queue);

        Self { base, compute_shader, current_params: initial_params, dragging: false, last_cursor: None }
    }

    fn update(&mut self, core: &Core) {
        self.compute_shader.handle_export(core, &mut self.base);
    }

    fn resize(&mut self, core: &Core) {
        self.base.default_resize(core, &mut self.compute_shader);
    }

    fn render(&mut self, core: &Core) -> Result<(), cuneus::SurfaceError> {
        let mut frame = self.base.begin_frame(core)?;

        let mut params = self.current_params;
        let mut changed = false;
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self
            .base
            .controls
            .get_ui_request(&self.base.start_time, &core.size, self.base.fps_tracker.fps());
        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                RenderKit::apply_default_style(ctx);

                egui::Window::new("Spectral Buddhabrot")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(300.0)
                    .show(ctx, |ui| {
                        ui.label("Drag to pan, wheel to zoom");
                        egui::CollapsingHeader::new("4D rotation").default_open(true).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.rot_a, -180.0..=180.0).text("Re z ↔ Re c")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.rot_b, -180.0..=180.0).text("Im z ↔ Im c")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.rot_c, -180.0..=180.0).text("Re z ↔ Im c")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.rot_d, -180.0..=180.0).text("Im z ↔ Re c")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.spin_a, -30.0..=30.0).text("Spin Re")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.spin_b, -30.0..=30.0).text("Spin Im")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.persist, 0.5..=0.99).text("Trail while spinning")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.z0x, -1.0..=1.0).text("Start z (re)")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.z0y, -1.0..=1.0).text("Start z (im)")).changed();
                        });

                        egui::CollapsingHeader::new("Sampling").default_open(true).show(ui, |ui| {
                            ui.horizontal(|ui| {
                                for (i, name) in ["Uniform", "Metropolis"].iter().enumerate() {
                                    changed |= ui.selectable_value(&mut params.sampling, i as u32, *name).changed();
                                }
                            });
                            if params.sampling == 1 {
                                changed |= ui.add(egui::Slider::new(&mut params.mut_size, 0.05..=4.0).logarithmic(true).text("Step size")).changed();
                            }
                            changed |= ui.add(egui::Slider::new(&mut params.sample_density, 0.1..=2.0).text("Samples per frame")).changed();
                        });

                        egui::CollapsingHeader::new("Fractal").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.max_iterations, 100..=5000).logarithmic(true).text("Max iterations")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.escape_radius, 2.0..=20.0).text("Escape radius")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.min_trajectory_len, 5..=200).text("Min trajectory")).changed();
                        });

                        egui::CollapsingHeader::new("View").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.zoom, 0.1..=1000.0).logarithmic(true).text("Zoom")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.offset_x, -2.0..=1.0).text("Offset X")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.offset_y, -1.5..=1.5).text("Offset Y")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.rotation, -3.14159..=3.14159).text("Rotation")).changed();
                            if ui.button("Reset view").clicked() {
                                (params.zoom, params.offset_x, params.offset_y) = (0.5, -0.5, 0.0);
                                changed = true;
                            }
                        });

                        egui::CollapsingHeader::new("Spectral").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.wavelength_min, 390.0..=700.0).text("Min (nm)")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.wavelength_max, 390.0..=700.0).text("Max (nm)")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.color_shift, 0.0..=2.0).text("Color curve")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.saturation, 0.0..=3.0).text("Saturation")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.intensity_scale, 0.1..=10.0).logarithmic(true).text("Intensity")).changed();
                        });

                        egui::CollapsingHeader::new("Tone mapping").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.exposure, 0.5..=12.0).text("Exposure")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.gamma, 0.2..=2.2).text("Gamma")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.dithering, 0.0..=1.0).text("Dithering")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.white_balance_r, 0.5..=2.0).text("White balance R")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.white_balance_g, 0.5..=2.0).text("White balance G")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.white_balance_b, 0.5..=2.0).text("White balance B")).changed();
                        });

                        ui.separator();
                        ShaderControls::render_controls_widget(ui, &mut controls_request);
                        ui.separator();
                        should_start_export = ExportManager::render_export_ui_widget(ui, &mut export_request);
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

        if changed {
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
            self.clear_buffers(core);
        }

        if should_start_export {
            self.base.export_manager.start_export();
        }

        // after any reset, so the frame counter the shader sees is current
        let current_time = self.base.controls.get_time(&self.base.start_time);
        self.compute_shader.set_time(current_time, 1.0 / 60.0, &core.queue);
        self.compute_shader.dispatch(&mut frame.encoder, core);

        self.base
            .renderer
            .render_to_view(&mut frame.encoder, &frame.view, &self.compute_shader.get_output_texture().bind_group);

        self.base.end_frame(core, frame, full_output);
        Ok(())
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        if self.base.default_handle_input(core, event) {
            return true;
        }
        match event {
            WindowEvent::MouseInput { state, button: winit::event::MouseButton::Left, .. } => {
                self.dragging = *state == winit::event::ElementState::Pressed;
                true
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let (true, Some((x, y))) = (self.dragging, self.last_cursor) {
                    let z = self.current_params.zoom;
                    let (a, b) = self.to_plane(core, position.x, position.y, z);
                    let (c, d) = self.to_plane(core, x, y, z);
                    self.current_params.offset_x -= a - c;
                    self.current_params.offset_y -= b - d;
                    self.compute_shader.set_custom_params(self.current_params, &core.queue);
                    self.clear_buffers(core);
                }
                self.last_cursor = Some((position.x, position.y));
                self.dragging
            }
            // zoom toward the cursor: the point under it stays put
            WindowEvent::MouseWheel { delta, .. } => {
                let steps = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => *y,
                    winit::event::MouseScrollDelta::PixelDelta(d) => d.y as f32 / 40.0,
                };
                let (x, y) = self.last_cursor.unwrap_or((core.size.width as f64 * 0.5, core.size.height as f64 * 0.5));
                let z0 = self.current_params.zoom;
                let z1 = (z0 * 1.15f32.powf(steps)).clamp(0.1, 100_000.0);
                let (a, b) = self.to_plane(core, x, y, z0);
                let (c, d) = self.to_plane(core, x, y, z1);
                let p = &mut self.current_params;
                p.zoom = z1;
                p.offset_x += a - c;
                p.offset_y += b - d;
                self.compute_shader.set_custom_params(*p, &core.queue);
                self.clear_buffers(core);
                true
            }
            _ => false,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = cuneus::ShaderApp::new("Spectral Buddhabrot", 800, 600);
    app.run(event_loop, BuddhabrotShader::init)
}

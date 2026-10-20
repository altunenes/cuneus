use cuneus::compute::*;
use cuneus::prelude::*;

cuneus::uniform_params! {
    struct CNNParams {
        brush_size: f32, clear_canvas: i32, drawing: i32, mode: u32,
        steps: u32, batch: u32, step_base: u32, reset: u32,
        lr: f32, teach: i32, teach_slot: u32, rep_n: u32,
        fibers: f32, back: f32, aug: f32, dream: f32,
        yaw: f32, elev: f32, zoom: f32, cam: f32,
        bloom: f32, expo: f32, height: f32, kernels: f32,
        conf: f32, seed: u32, _p0: f32, _p1: f32,
    }
}

// largest window the 3D view's HDR buffer holds
const MAXPIX: u64 = 3840 * 2400;
// floats in the shader's net buffer (weights, optimiser, batch, activations, gradients, history, stats, pad, dreams)
const NET: u64 = 1_162_720;
// EMNIST balanced classes
const CLASSES: &str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt";
const SPEEDS: [&str; 4] = ["Paused", "Slow", "Normal", "Fast"];

struct CNN {
    base: RenderKit,
    compute_shader: ComputeShader,
    params: CNNParams,
    speed: usize,
    step: u32,
    frame: u64,
    reset: bool,
    rep_head: u32,
    // testing with training off; after a teach it still trains this many frames
    train_in_test: bool,
    burst: u32,
    dragging: bool,
    last_cursor: Option<(f64, f64)>,
}

// must match the shader's pad_hit and pad2d_rect
fn over_pad(p: &CNNParams, core: &Core, x: f64, y: f64) -> bool {
    let (yw, el) = (p.yaw.to_radians(), p.elev.clamp(5.0, 85.0).to_radians());
    let d = 150.0 / p.zoom.max(0.1);
    let ro = [74.0 + d * yw.sin() * el.cos(), -d * yw.cos() * el.cos(), 4.0 + d * el.sin()];
    let n = (ro[0] * ro[0] + ro[1] * ro[1] + (ro[2] - 4.0) * (ro[2] - 4.0)).sqrt();
    let f = [(74.0 - ro[0]) / n, -ro[1] / n, (4.0 - ro[2]) / n];
    let rl = (f[1] * f[1] + f[0] * f[0]).sqrt();
    let rt = [f[1] / rl, -f[0] / rl, 0.0];
    let up = [rt[1] * f[2] - rt[2] * f[1], rt[2] * f[0] - rt[0] * f[2], rt[0] * f[1] - rt[1] * f[0]];
    let (w, h) = (core.size.width as f32, core.size.height as f32);
    let sp = [(x as f32 - 0.5 * w) / h * 0.8, (y as f32 - 0.5 * h) / h * 0.8];
    let rd: Vec<f32> = (0..3).map(|i| f[i] + sp[0] * rt[i] - sp[1] * up[i]).collect();
    if rd[2] >= 0.0 {
        return false;
    }
    let t = (0.3 - ro[2]) / rd[2];
    let (qx, qy) = (ro[0] + rd[0] * t, ro[1] + rd[1] * t + 14.0);
    let on_3d = (0.0..28.0).contains(&qx) && (0.0..28.0).contains(&qy);
    let side = (0.3 * h).floor();
    let (lx, ly) = (x as f32 - 16.0, y as f32 - (h - side - 16.0));
    on_3d || ((0.0..side).contains(&lx) && (0.0..side).contains(&ly))
}

impl ShaderManager for CNN {
    fn init(core: &Core) -> Self {
        let base = RenderKit::new(core);

        // a training step: generate, forward, loss, backward, Adam, stats; four copies gated by `steps`
        let mut passes = vec![
            PassDescription::new("train_init", &[]).with_workgroup_size([2176, 1, 1]),
            PassDescription::new("teach", &[]).with_workgroup_size([13, 1, 1]),
        ];
        let step_passes: [(&str, u32); 11] = [
            ("gen", 784), ("fwd1", 2304), ("fwd2", 512), ("head", 1), ("bfc", 25), ("bgap", 32),
            ("bw2", 201), ("bda1", 2304), ("bw1", 7), ("adam", 232), ("tstat", 1),
        ];
        for r in 0..4 {
            for (name, groups) in step_passes {
                passes.push(PassDescription::new(&format!("{name}_{r}"), &[]).with_workgroup_size([groups, 1, 1]));
            }
        }
        passes.extend([
            PassDescription::new("gstat", &[]).with_workgroup_size([1, 1, 1]),
            // one ascent step on every class's dream
            PassDescription::new("d_in", &[]).with_workgroup_size([576, 1, 1]),
            PassDescription::new("d_fwd1", &[]).with_workgroup_size([1692, 1, 1]),
            PassDescription::new("d_fwd2", &[]).with_workgroup_size([376, 1, 1]),
            PassDescription::new("d_head", &[]).with_workgroup_size([1, 1, 1]),
            PassDescription::new("d_bda1", &[]).with_workgroup_size([1692, 1, 1]),
            PassDescription::new("d_bin", &[]).with_workgroup_size([576, 1, 1]),
            PassDescription::new("d_norm", &[]).with_workgroup_size([1, 1, 1]),
            PassDescription::new("d_step", &[]).with_workgroup_size([576, 1, 1]),
            PassDescription::new("d_blur", &[]).with_workgroup_size([576, 1, 1]),
            // the pad through the current network, for the view
            PassDescription::new("canvas_update", &[]).with_workgroup_size([28, 28, 1]),
            PassDescription::new("conv_layer1", &[]).with_workgroup_size([12, 12, 16]),
            PassDescription::new("conv_layer2", &[]).with_workgroup_size([4, 4, 32]),
            PassDescription::new("fully_connected", &[]).with_workgroup_size([47, 1, 1]),
            PassDescription::new("analyze", &[]).with_workgroup_size([1, 1, 1]),
            PassDescription::new("scene3d", &[]),
            PassDescription::new("fiber_clear", &[]),
            // 12512 fibers and 47 dream threads
            PassDescription::new("fibers", &[]).with_workgroup_size([197, 1, 1]),
            PassDescription::new("bloom_down", &[]),
            PassDescription::new("bloom_h", &[]),
            PassDescription::new("bloom_v", &[]),
            PassDescription::new("main_image", &[]),
        ]);

        let config = ComputeShaderBuilder::new()
            .with_label("CNN")
            .with_multi_pass(&passes)
            .with_custom_uniforms::<CNNParams>()
            .with_mouse()
            .with_fonts()
            .with_storage_buffer(StorageBufferSpec::new("net", NET * 4))
            .with_storage_buffer(StorageBufferSpec::new("hdr", MAXPIX * 8))
            .with_storage_buffer(StorageBufferSpec::new("b1", MAXPIX / 16 * 8 + 64))
            .with_storage_buffer(StorageBufferSpec::new("b2", MAXPIX / 16 * 8 + 64))
            .with_storage_buffer(StorageBufferSpec::new("fib", (MAXPIX / 4 + 4096) * 3 * 4))
            .build();
        let compute_shader = cuneus::compute_shader!(core, "shaders/cnn.wgsl", config);

        let params = CNNParams {
            brush_size: 0.007, clear_canvas: 0, drawing: 0, mode: 0,
            steps: 0, batch: 32, step_base: 0, reset: 1,
            lr: 0.003, teach: -1, teach_slot: 0, rep_n: 0,
            fibers: 1.0, back: 1.0, aug: 1.0, dream: 1.0,
            yaw: -20.0, elev: 32.0, zoom: 1.3, cam: 0.6,
            bloom: 0.45, expo: 1.0, height: 1.0, kernels: 1.0,
            conf: 1.0, seed: 1, _p0: 0.0, _p1: 0.0,
        };
        Self { base, compute_shader, params, speed: 2, step: 0, frame: 0, reset: true, rep_head: 0, train_in_test: true, burst: 0, dragging: false, last_cursor: None }
    }

    fn update(&mut self, core: &Core) {
        let current_time = self.base.controls.get_time(&self.base.start_time);
        self.compute_shader.set_time(current_time, 1.0 / 60.0, &core.queue);
        self.compute_shader.handle_export(core, &mut self.base);
    }

    fn resize(&mut self, core: &Core) {
        self.compute_shader.resize(core, core.size.width, core.size.height);
    }

    fn render(&mut self, core: &Core) -> Result<(), cuneus::SurfaceError> {
        let mut frame = self.base.begin_frame(core)?;
        let mut p = self.params;
        let mut speed = self.speed;
        let mut train_in_test = self.train_in_test;
        let mut reset = false;
        let mut teach = None;
        let mut clear = false;
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self.base.controls.get_ui_request(&self.base.start_time, &core.size, self.base.fps_tracker.fps());
        let (step, taught) = (self.step, p.rep_n);

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                RenderKit::apply_default_style(ctx);
                egui::Window::new("CNN").collapsible(true).resizable(true).default_width(300.0).show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        for (i, name) in ["Watch it learn", "Test & teach"].iter().enumerate() {
                            ui.selectable_value(&mut p.mode, i as u32, *name);
                        }
                    });
                    ui.horizontal(|ui| {
                        for (i, name) in SPEEDS.iter().enumerate() {
                            ui.selectable_value(&mut speed, i, *name);
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label(format!("step {step}"));
                        if ui.button("Reset to random weights").clicked() {
                            reset = true;
                        }
                    });
                    egui::CollapsingHeader::new("Training").default_open(true).show(ui, |ui| {
                        ui.add(egui::Slider::new(&mut p.lr, 0.0001..=0.02).logarithmic(true).text("Learning rate"));
                        ui.add(egui::Slider::new(&mut p.batch, 8..=64).text("Batch"));
                        ui.add(egui::Slider::new(&mut p.aug, 0.0..=2.0).text("Distortion"));
                    });
                    if p.mode == 1 {
                        egui::CollapsingHeader::new("Teach").default_open(true).show(ui, |ui| {
                            ui.label("Draw on the pad: the hologram shows the network's guess. If it's wrong, click the right character to teach it. Every taught drawing is kept (up to 64) and keeps being trained on.");
                            ui.label(format!("{taught} taught"));
                            ui.checkbox(&mut train_in_test, "Keep training while testing");
                            ui.horizontal_wrapped(|ui| {
                                for (i, ch) in CLASSES.chars().enumerate() {
                                    if ui.small_button(ch.to_string()).clicked() {
                                        teach = Some(i as i32);
                                    }
                                }
                            });
                            ui.horizontal(|ui| {
                                ui.add(egui::Slider::new(&mut p.brush_size, 0.001..=0.015).text("Brush"));
                                clear = ui.button("Clear").clicked();
                            });
                        });
                    }
                    egui::CollapsingHeader::new("Show").default_open(true).show(ui, |ui| {
                        ui.add(egui::Slider::new(&mut p.fibers, 0.0..=3.0).text("Signal (forward)"));
                        ui.add(egui::Slider::new(&mut p.back, 0.0..=3.0).text("Gradient (backward)"));
                        ui.add(egui::Slider::new(&mut p.dream, 0.0..=3.0).text("Dreams")).on_hover_text("What each output neuron imagines its character looks like: an image pushed by gradient ascent to raise that class");
                        ui.add(egui::Slider::new(&mut p.kernels, 0.0..=3.0).text("Layer 1 kernels"));
                        ui.add(egui::Slider::new(&mut p.conf, 0.0..=3.0).text("Confusion matrix"));
                        ui.add(egui::Slider::new(&mut p.cam, 0.0..=1.0).text("Class activation map"));
                        ui.add(egui::Slider::new(&mut p.height, 0.2..=3.0).text("Tower height"));
                    });
                    egui::CollapsingHeader::new("Camera").show(ui, |ui| {
                        ui.label("Drag to orbit, wheel to zoom, hover a neuron to see what it sees");
                        ui.add(egui::Slider::new(&mut p.yaw, -180.0..=180.0).text("Orbit"));
                        ui.add(egui::Slider::new(&mut p.elev, 5.0..=85.0).text("Tilt"));
                        ui.add(egui::Slider::new(&mut p.zoom, 0.5..=4.0).logarithmic(true).text("Zoom"));
                        ui.add(egui::Slider::new(&mut p.bloom, 0.0..=3.0).text("Bloom"));
                        ui.add(egui::Slider::new(&mut p.expo, 0.2..=4.0).logarithmic(true).text("Exposure"));
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

        self.base.apply_control_request(controls_request.clone());
        self.base.export_manager.apply_ui_request(export_request);
        if should_start_export {
            self.base.export_manager.start_export();
        }

        self.speed = speed;
        self.train_in_test = train_in_test;
        if reset {
            self.reset = true;
        }
        // one-frame flags: re-initialise, store a taught drawing, clear the pad
        p.reset = self.reset as u32;
        if self.reset {
            (self.step, p.rep_n, self.rep_head) = (0, 0, 0);
            p.seed = p.seed.wrapping_add(1);
            self.reset = false;
        }
        p.teach = -1;
        if let Some(t) = teach {
            self.burst = 120;
            (p.teach, p.teach_slot) = (t, self.rep_head);
            self.rep_head = (self.rep_head + 1) % 64;
            p.rep_n = (p.rep_n + 1).min(64);
        }
        p.clear_canvas = clear as i32;
        // slow: one step every twelve frames
        p.steps = match self.speed {
            0 => 0,
            1 => (self.frame % 12 == 0) as u32,
            2 => 1,
            _ => 4,
        };
        // frozen while testing, except a short burst after teaching
        if p.mode == 1 && !self.train_in_test {
            p.steps = if self.burst > 0 { p.steps.max(1) } else { 0 };
        }
        self.burst = self.burst.saturating_sub(1);
        p.step_base = self.step;
        self.params = p;
        self.compute_shader.set_custom_params(self.params, &core.queue);
        self.compute_shader.update_mouse_uniform(&self.base.mouse_tracker.uniform, &core.queue);
        self.compute_shader.dispatch(&mut frame.encoder, core);
        self.step += p.steps;
        self.frame += 1;

        self.base.renderer.render_to_view(&mut frame.encoder, &frame.view, &self.compute_shader.get_output_texture().bind_group);
        self.base.end_frame(core, frame, full_output);
        Ok(())
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        if self.base.default_handle_input(core, event) {
            return true;
        }
        match event {
            // in draw mode a press on the pad draws; anywhere else it orbits
            WindowEvent::MouseInput { state, button: winit::event::MouseButton::Left, .. } => {
                let pressed = *state == winit::event::ElementState::Pressed;
                let on_pad = self.params.mode == 1 && self.last_cursor.is_some_and(|(x, y)| over_pad(&self.params, core, x, y));
                self.dragging = pressed && !on_pad;
                self.params.drawing = (pressed && on_pad) as i32;
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let steps = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => *y,
                    winit::event::MouseScrollDelta::PixelDelta(d) => d.y as f32 / 40.0,
                };
                self.params.zoom = (self.params.zoom * 1.1f32.powf(steps)).clamp(0.5, 4.0);
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let (true, Some((x, y))) = (self.dragging, self.last_cursor) {
                    let p = &mut self.params;
                    p.yaw = (p.yaw - (position.x - x) as f32 * 0.3 + 180.0).rem_euclid(360.0) - 180.0;
                    p.elev = (p.elev + (position.y - y) as f32 * 0.3).clamp(5.0, 85.0);
                }
                self.last_cursor = Some((position.x, position.y));
            }
            _ => {}
        }
        self.base.handle_mouse_input(core, event, false)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (app, event_loop) = ShaderApp::new("CNN", 1280, 720);
    app.run(event_loop, CNN::init)
}

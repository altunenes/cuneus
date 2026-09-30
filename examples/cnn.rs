use cuneus::compute::*;
use cuneus::prelude::*;

cuneus::uniform_params! {
    struct CNNParams {
    brush_size: f32,
    input_resolution: f32,
    clear_canvas: i32,
    drawing: i32,
    feature_maps_1: f32,
    feature_maps_2: f32,
    num_classes: f32,
    normalization_mean: f32,
    normalization_std: f32,
    show_frequencies: i32,
    conv1_pool_size: f32,
    conv2_pool_size: f32,
    fibers: f32,
    yaw: f32,
    elev: f32,
    zoom: f32,
    cam: f32,
    bloom: f32,
    expo: f32,
    height: f32,
    }
}

struct CNNDigitRecognizer {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: CNNParams,
    first_frame: bool,
    dragging: bool,
    last_cursor: Option<(f64, f64)>}

// largest window the 3D view's HDR buffer holds
const MAXPIX: u64 = 3840 * 2400;

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

impl CNNDigitRecognizer {}

impl ShaderManager for CNNDigitRecognizer {
    fn init(core: &Core) -> Self {
        let base = RenderKit::new(core);

        // Configure multi-pass CNN with 5 stages: canvas_update -> conv_layer1 -> conv_layer2 -> fully_connected -> main_image
        let passes = vec![
            PassDescription::new("canvas_update", &[]).with_workgroup_size([28, 28, 1]),
            
            PassDescription::new("conv_layer1", &["canvas_update"])
                .with_workgroup_size([12, 12, 16]), // 16 Feature Maps
            
            PassDescription::new("conv_layer2", &["conv_layer1"])
                .with_workgroup_size([4, 4, 32]),   // 32 Feature Maps
            
            PassDescription::new("fully_connected", &["conv_layer2"])
                .with_workgroup_size([47, 1, 1]),   // 47 Classes
            
            PassDescription::new("analyze", &[]).with_workgroup_size([1, 1, 1]),
            PassDescription::new("scene3d", &[]),
            PassDescription::new("fiber_clear", &[]),
            // 12512 fibers
            PassDescription::new("fibers", &[]).with_workgroup_size([196, 1, 1]),
            PassDescription::new("bloom_down", &[]),
            PassDescription::new("bloom_h", &[]),
            PassDescription::new("bloom_v", &[]),
            PassDescription::new("main_image", &["fully_connected"]),
        ];

        let compute_shader = ComputeShaderBuilder::new()
            .with_label("CNN Digit Recognizer")
            .with_multi_pass(&passes)
            .with_custom_uniforms::<CNNParams>()
            .with_mouse()
            .with_fonts()
            .with_storage_buffer(StorageBufferSpec::new("canvas_data", (28 * 28 * 4) as u64))
            .with_storage_buffer(StorageBufferSpec::new(
                "conv1_data",
                (12 * 12 * 16 * 4) as u64,
            )) 
            .with_storage_buffer(StorageBufferSpec::new(
                "conv2_data", 
                (4 * 4 * 32 * 4) as u64
            )) 
            .with_storage_buffer(StorageBufferSpec::new("fc_data", 128 * 4))
            .with_storage_buffer(StorageBufferSpec::new("hdr", MAXPIX * 8))
            .with_storage_buffer(StorageBufferSpec::new("b1", MAXPIX / 16 * 8 + 64))
            .with_storage_buffer(StorageBufferSpec::new("b2", MAXPIX / 16 * 8 + 64))
            .with_storage_buffer(StorageBufferSpec::new("fib", (MAXPIX / 4 + 4096) * 3 * 4))
            .build();

        let compute_shader = cuneus::compute_shader!(core, "shaders/cnn.wgsl", compute_shader);

        let current_params = CNNParams {
            brush_size: 0.007,
            input_resolution: 28.0,
            clear_canvas: 0,
            drawing: 0,
            feature_maps_1: 16.0,
            feature_maps_2: 32.0,
            num_classes: 47.0,
            normalization_mean: 0.175,
            normalization_std: 0.33,
            show_frequencies: 0,
            conv1_pool_size: 12.0,
            conv2_pool_size: 4.0,
            fibers: 1.0,
            yaw: -20.0,
            elev: 32.0,
            zoom: 1.3,
            cam: 0.8,
            bloom: 0.45,
            expo: 1.0,
            height: 1.0,
        };

        Self {
            base,
            compute_shader,
            current_params,
            first_frame: true,
            dragging: false,
            last_cursor: None}
    }

    fn update(&mut self, core: &Core) {
        let current_time = self.base.controls.get_time(&self.base.start_time);
        self.compute_shader.set_time(current_time, 1.0 / 60.0, &core.queue);
        self.compute_shader.handle_export(core, &mut self.base);
    }

    fn resize(&mut self, core: &Core) {
        self.compute_shader
            .resize(core, core.size.width, core.size.height);
    }

    fn render(&mut self, core: &Core) -> Result<(), cuneus::SurfaceError> {
        let mut frame = self.base.begin_frame(core)?;

        let mut params = self.current_params;
        let mut changed = self.first_frame; // Update params on first frame
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self
            .base
            .controls
            .get_ui_request(&self.base.start_time, &core.size, self.base.fps_tracker.fps());

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                RenderKit::apply_default_style(ctx);

                egui::Window::new("CNN chr Recognizer")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(280.0)
                    .show(ctx, |ui| {
                        ui.label("Draw a character in the canvas area and watch the CNN predict it!");
                        ui.separator();
                        ui.label("The CNN will predict the character using pre-trained weights");
                        ui.separator();

                        egui::CollapsingHeader::new("Brush")
                            .default_open(true)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.brush_size, 0.001..=0.015)
                                            .text("Brush Size"),
                                    )
                                    .changed();
                                if ui.button("Clear Canvas").clicked() {
                                    params.clear_canvas = 1;
                                    changed = true;
                                } else {
                                    params.clear_canvas = 0;
                                }
                            });

                        ui.label("Draw on the pad (bottom left, or the one in the scene), right-click to clear. Drag elsewhere to orbit, wheel to zoom, hover a neuron to see what it sees.");
                        changed |= ui.add(egui::Slider::new(&mut params.cam, 0.0..=1.0).text("Class activation map")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.fibers, 0.0..=3.0).text("Fibers")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.height, 0.2..=3.0).text("Tower height")).changed();
                        egui::CollapsingHeader::new("Camera").show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.yaw, -180.0..=180.0).text("Orbit")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.elev, 5.0..=85.0).text("Tilt")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.zoom, 0.5..=4.0).logarithmic(true).text("Zoom")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.bloom, 0.0..=3.0).text("Bloom")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.expo, 0.2..=4.0).logarithmic(true).text("Exposure")).changed();
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

        // Update mouse uniform for drawing interaction
        self.compute_shader
            .update_mouse_uniform(&self.base.mouse_tracker.uniform, &core.queue);

        // Execute CNN pipeline
        // Note: our backend automatically uses custom workgroup sizes from PassDescription
        self.compute_shader.dispatch(&mut frame.encoder, core);

        self.base.renderer.render_to_view(&mut frame.encoder, &frame.view, &self.compute_shader.get_output_texture().bind_group);

        // Apply UI changes
        self.base.apply_control_request(controls_request.clone());

        self.base.export_manager.apply_ui_request(export_request);
        if should_start_export {
            self.base.export_manager.start_export();
        }

        if changed {
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
            self.first_frame = false;
        }

        self.base.end_frame(core, frame, full_output);

        Ok(())
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        if self.base.default_handle_input(core, event) {
            return true;
        }
        {
            match event {
                // a press on the pad draws; anywhere else it orbits
                WindowEvent::MouseInput { state, button: winit::event::MouseButton::Left, .. } => {
                    let pressed = *state == winit::event::ElementState::Pressed;
                    let on_pad = self.last_cursor.is_some_and(|(x, y)| over_pad(&self.current_params, core, x, y));
                    self.dragging = pressed && !on_pad;
                    self.current_params.drawing = (pressed && on_pad) as i32;
                    self.compute_shader.set_custom_params(self.current_params, &core.queue);
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let steps = match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, y) => *y,
                        winit::event::MouseScrollDelta::PixelDelta(d) => d.y as f32 / 40.0,
                    };
                    let p = &mut self.current_params;
                    p.zoom = (p.zoom * 1.1f32.powf(steps)).clamp(0.5, 4.0);
                    self.compute_shader.set_custom_params(*p, &core.queue);
                }
                WindowEvent::CursorMoved { position, .. } => {
                    if let (true, Some((x, y))) = (self.dragging, self.last_cursor) {
                        let p = &mut self.current_params;
                        p.yaw = (p.yaw - (position.x - x) as f32 * 0.3 + 180.0).rem_euclid(360.0) - 180.0;
                        p.elev = (p.elev + (position.y - y) as f32 * 0.3).clamp(5.0, 85.0);
                        self.compute_shader.set_custom_params(*p, &core.queue);
                    }
                    self.last_cursor = Some((position.x, position.y));
                }
                _ => {}
            }
        }
        self.base.handle_mouse_input(core, event, false)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (app, event_loop) = ShaderApp::new("EMNIST", 800, 600);

    app.run(event_loop, CNNDigitRecognizer::init)
}

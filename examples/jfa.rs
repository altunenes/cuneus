use cuneus::compute::*;
use cuneus::prelude::*;

cuneus::uniform_params! {
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
    _pad_m: f32,
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
            n: 0.1,
            gamma: 2.1,
            color_intensity: 1.0,
            color_r: 1.0,
            color_g: 2.0,
            color_b: 3.0,
            color_w: 4.0,
            accumulation_speed: 0.01,
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
            _pad_m: 0.0,
        };
        let base = RenderKit::new(core);

        // The fully unrolled JFA Pipeline
        let passes = vec![
            PassDescription::new("seed_points", &["seed_points"]),
            PassDescription::new("flood_init", &[]),
            
            // JFA Unroll: input_texture0 = "seed_points", input_texture1 = previous step
            PassDescription::new("flood_1024", &["seed_points", "flood_init"]),
            PassDescription::new("flood_512",  &["seed_points", "flood_1024"]),
            PassDescription::new("flood_256",  &["seed_points", "flood_512"]),
            PassDescription::new("flood_128",  &["seed_points", "flood_256"]),
            PassDescription::new("flood_64",   &["seed_points", "flood_128"]),
            PassDescription::new("flood_32",   &["seed_points", "flood_64"]),
            PassDescription::new("flood_16",   &["seed_points", "flood_32"]),
            PassDescription::new("flood_8",    &["seed_points", "flood_16"]),
            PassDescription::new("flood_4",    &["seed_points", "flood_8"]),
            PassDescription::new("flood_2",    &["seed_points", "flood_4"]),
            PassDescription::new("flood_1",    &["seed_points", "flood_2"]),

            // Reads seed_points, completed JFA (flood_1), and its own feedback
            PassDescription::new("color_accumulate", &["seed_points", "flood_1", "color_accumulate"]),
            PassDescription::new("main_image", &["color_accumulate"]),
        ];

        let config = ComputeShader::builder()
            .with_entry_point("seed_points")
            .with_multi_pass(&passes)
            .with_custom_uniforms::<JfaParams>()
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("JFA Unified")
            .build();

        let compute_shader = cuneus::compute_shader!(core, "shaders/jfa.wgsl", config);

        compute_shader.set_custom_params(initial_params, &core.queue);

        Self {
            base,
            compute_shader,
            current_params: initial_params,
        }
    }

    fn update(&mut self, core: &Core) {
        self.compute_shader.handle_export(core, &mut self.base);

        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0 / 60.0;
        self.compute_shader
            .set_time(current_time, delta, &core.queue);
    }

    fn resize(&mut self, core: &Core) {
        self.base.default_resize(core, &mut self.compute_shader);
    }

    fn render(&mut self, core: &Core) -> Result<(), cuneus::SurfaceError> {
        let mut frame = self.base.begin_frame(core)?;

        // Executes the entire unrolled JFA per frame automatically
        self.compute_shader.dispatch(&mut frame.encoder, core);

        self.base.renderer.render_to_view(
            &mut frame.encoder,
            &frame.view,
            &self.compute_shader.get_output_texture().bind_group,
        );

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

                egui::Window::new("JFA - Fully Unrolled")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(280.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("JFA Parameters")
                            .default_open(true)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.n, 0.01..=1.0)
                                            .text("Pattern Speed (N)"),
                                    )
                                    .changed();
                                ui.separator();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(
                                            &mut params.accumulation_speed,
                                            0.01..=0.1,
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
                        ui.label("15-Pass Realtime Unrolled JFA");
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        if controls_request.is_paused != (params.freeze_accumulation > 0.5) {
            params.freeze_accumulation = if controls_request.is_paused { 1.0 } else { 0.0 };
            changed = true;
        }

        self.base.export_manager.apply_ui_request(export_request);
        if controls_request.should_clear_buffers {
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

        self.base.end_frame(core, frame, full_output);

        Ok(())
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        self.base.default_handle_input(core, event)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("JFA", 800, 600);

    app.run(event_loop, JfaShader::init)
}
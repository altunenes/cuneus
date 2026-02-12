use cuneus::compute::*;
use cuneus::prelude::*;
use winit::event::WindowEvent;

cuneus::uniform_params! {
    struct NeuronParams {
    pixel_offset: f32,
    pixel_offset2: f32,
    lights: f32,
    exp: f32,
    frame: f32,
    col1: f32,
    col2: f32,
    decay: f32}
}

struct NeuronShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: NeuronParams}

impl ShaderManager for NeuronShader {
    fn init(core: &Core) -> Self {
        let initial_params = NeuronParams {
            pixel_offset: -1.0,
            pixel_offset2: 1.0,
            lights: 2.2,
            exp: 4.0,
            frame: 1.0,
            col1: 100.0,
            col2: 1.0,
            decay: 1.0};
        let base = RenderKit::new(core);

        // Create multipass system: buffer_a -> buffer_b -> buffer_c -> main_image
        let passes = vec![
            PassDescription::new("buffer_a", &[]), // no dependencies, generates pattern
            PassDescription::new("buffer_b", &["buffer_a"]), // reads buffer_a
            PassDescription::new("buffer_c", &["buffer_c", "buffer_b"]), // self-feedback + buffer_b
            PassDescription::new("main_image", &["buffer_c"]),
        ];

        let config = ComputeShader::builder()
            .with_entry_point("buffer_a")
            .with_multi_pass(&passes)
            .with_custom_uniforms::<NeuronParams>()
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("2D Neuron Unified")
            .build();

        let compute_shader = cuneus::compute_shader!(core, "shaders/2dneuron.wgsl", config);


        compute_shader.set_custom_params(initial_params, &core.queue);

        Self {
            base,
            compute_shader,
            current_params: initial_params}
    }

    fn update(&mut self, core: &Core) {
        // Handle export
        self.compute_shader.handle_export(core, &mut self.base);

        // Update time uniform - this is crucial for accumulation!
        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0 / 60.0;
        self.compute_shader
            .set_time(current_time, delta, &core.queue);
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
            .get_ui_request(&self.base.start_time, &core.size, self.base.fps_tracker.fps());

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                RenderKit::apply_default_style(ctx);

                egui::Window::new("2D Neuron")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(280.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("Neuron Parameters")
                            .default_open(true)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.pixel_offset, -3.14..=3.14)
                                            .text("Pixel Offset Y"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.pixel_offset2, -3.14..=3.14)
                                            .text("Pixel Offset X"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.lights, 0.0..=12.2)
                                            .text("Lights"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.exp, 1.0..=120.0).text("Exp"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Visual Settings")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.frame, 0.0..=5.2)
                                            .text("Frame"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.col1, 0.0..=150.0)
                                            .text("Iterations"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.col2, 0.0..=20.0)
                                            .text("Color 2"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.decay, 0.0..=1.0)
                                            .text("Feedback"),
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
                        ui.label("Multi-buffer neuron with particle tracing");
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        // Handle controls and clear buffers if requested
        if controls_request.should_clear_buffers {
            // Reset frame count to restart accumulation - this is crucial
            self.compute_shader.current_frame = 0;
        }

        // Execute multi-pass compute shader: buffer_a -> buffer_b -> buffer_c -> main_image
        self.compute_shader.dispatch(&mut frame.encoder, core);

        self.base.renderer.render_to_view(&mut frame.encoder, &frame.view, &self.compute_shader);

        self.base.apply_control_request(controls_request);
        self.base.export_manager.apply_ui_request(export_request);

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
    let (app, event_loop) = ShaderApp::new("2D Neuron", 600, 800);
    app.run(event_loop, NeuronShader::init)
}

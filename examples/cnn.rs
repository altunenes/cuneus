use cuneus::compute::*;
use cuneus::prelude::*;
use winit::event::WindowEvent;

cuneus::uniform_params! {
    struct CNNParams {
    brush_size: f32,
    input_resolution: f32,
    clear_canvas: i32,
    show_debug: i32,
    feature_maps_1: f32,
    feature_maps_2: f32,
    num_classes: f32,
    normalization_mean: f32,
    normalization_std: f32,
    show_frequencies: i32,
    conv1_pool_size: f32,
    conv2_pool_size: f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
    _padding4: f32,
    _padding5: f32,
    _padding6: f32,
    _pad_m1: f32,
    _pad_m2: f32,
    }
}

struct CNNDigitRecognizer {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: CNNParams,
    first_frame: bool}

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
            .with_storage_buffer(StorageBufferSpec::new(
                "fc_data", 
                (47 * 4) as u64
            ))
            .build();

        let compute_shader = cuneus::compute_shader!(core, "shaders/cnn.wgsl", compute_shader);


        let current_params = CNNParams {
            brush_size: 0.007,
            input_resolution: 28.0,
            clear_canvas: 0,
            show_debug: 0,
            feature_maps_1: 16.0,
            feature_maps_2: 32.0,
            num_classes: 47.0,
            normalization_mean: 0.175,
            normalization_std: 0.33,
            show_frequencies: 0,
            conv1_pool_size: 12.0,
            conv2_pool_size: 4.0,
            _padding1: 0.0,
            _padding2: 0.0,
            _padding3: 0.0,
            _padding4: 0.0,
            _padding5: 0.0,
            _padding6: 0.0,
            _pad_m1: 0.0,
            _pad_m2: 0.0,
        };

        Self {
            base,
            compute_shader,
            current_params,
            first_frame: true}
    }

    fn update(&mut self, _core: &Core) {
    }

    fn resize(&mut self, core: &Core) {
        self.compute_shader
            .resize(core, core.size.width, core.size.height);
    }

    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
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

        self.base.renderer.render_to_view(&mut frame.encoder, &frame.view, &self.compute_shader);

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
        self.base.handle_mouse_input(core, event, false)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (app, event_loop) = ShaderApp::new("EMNIST", 800, 600);

    app.run(event_loop, CNNDigitRecognizer::init)
}

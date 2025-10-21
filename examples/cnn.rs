use cuneus::compute::*;
use cuneus::prelude::*;
use winit::event::WindowEvent;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CNNParams {
    canvas_size: f32,
    brush_size: f32,
    input_resolution: f32,
    clear_canvas: i32,
    show_debug: i32,
    prediction_threshold: f32,
    canvas_offset_x: f32,
    canvas_offset_y: f32,
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
}

impl UniformProvider for CNNParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct CNNDigitRecognizer {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: CNNParams,
    first_frame: bool,
}

impl CNNDigitRecognizer {}

impl ShaderManager for CNNDigitRecognizer {
    fn init(core: &Core) -> Self {
        let texture_bind_group_layout = RenderKit::create_standard_texture_layout(&core.device);
        let base = RenderKit::new(core, &texture_bind_group_layout, None);

        // Configure multi-pass CNN with 5 stages: canvas_update -> conv_layer1 -> conv_layer2 -> fully_connected -> main_image
        let passes = vec![
            PassDescription::new("canvas_update", &[]).with_workgroup_size([28, 28, 1]), // 28x28 canvas pixels
            PassDescription::new("conv_layer1", &["canvas_update"])
                .with_workgroup_size([12, 12, 8]), // 12x12 output × 8 feature maps
            PassDescription::new("conv_layer2", &["conv_layer1"]).with_workgroup_size([4, 4, 5]), // 4x4 output × 5 feature maps
            PassDescription::new("fully_connected", &["conv_layer2"])
                .with_workgroup_size([10, 1, 1]), // 10 output classes
            PassDescription::new("main_image", &["fully_connected"]), // Screen size dispatch handled automatically
        ];

        let compute_shader = ComputeShaderBuilder::new()
            .with_label("CNN Digit Recognizer")
            .with_multi_pass(&passes)
            .with_custom_uniforms::<CNNParams>()
            .with_mouse()
            .with_fonts()
            .with_storage_buffer(StorageBufferSpec::new("canvas_data", (28 * 28 * 4) as u64)) // Canvas data for 28x28 pixels
            .with_storage_buffer(StorageBufferSpec::new(
                "conv1_data",
                (12 * 12 * 8 * 4) as u64,
            )) // First convolution layer: 12x12x8 feature maps
            .with_storage_buffer(StorageBufferSpec::new("conv2_data", (4 * 4 * 5 * 4) as u64)) // Second convolution layer: 4x4x5 feature maps
            .with_storage_buffer(StorageBufferSpec::new("fc_data", (10 * 4) as u64)) // Fully connected layer: 10 classes
            .build();

        let mut compute_shader =
            ComputeShader::from_builder(core, include_str!("shaders/cnn.wgsl"), compute_shader);

        // Enable hot reload
        if let Err(e) = compute_shader.enable_hot_reload(
            core.device.clone(),
            std::path::PathBuf::from("examples/shaders/cnn.wgsl"),
            core.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("CNN Hot Reload"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("shaders/cnn.wgsl").into()),
                }),
        ) {
            eprintln!("Failed to enable hot reload for cnn shader: {}", e);
        }

        let current_params = CNNParams {
            canvas_size: 0.6,
            brush_size: 0.007,
            input_resolution: 28.0,
            clear_canvas: 0,
            show_debug: 0,
            prediction_threshold: 0.1,
            canvas_offset_x: 0.1,
            canvas_offset_y: 0.1,
            feature_maps_1: 8.0,
            feature_maps_2: 5.0,
            num_classes: 10.0,
            normalization_mean: 0.1307,
            normalization_std: 0.3081,
            show_frequencies: 0,
            conv1_pool_size: 12.0,
            conv2_pool_size: 4.0,
            _padding1: 0.0,
            _padding2: 0.0,
            _padding3: 0.0,
            _padding4: 0.0,
            _padding5: 0.0,
            _padding6: 0.0,
        };

        Self {
            base,
            compute_shader,
            current_params,
            first_frame: true,
        }
    }

    fn update(&mut self, core: &Core) {
        self.base.fps_tracker.update();

        // Check for hot reload updates
        self.compute_shader.check_hot_reload(&core.device);
    }

    fn resize(&mut self, core: &Core) {
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
                label: Some("CNN Frame"),
            });

        let mut params = self.current_params;
        let mut changed = self.first_frame; // Update params on first frame
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

                egui::Window::new("CNN Digit Recognizer")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(280.0)
                    .show(ctx, |ui| {
                        ui.label("Draw a digit in the canvas area and watch the CNN predict it!");
                        ui.separator();
                        ui.label("The CNN will predict the digit using pre-trained weights");
                        ui.separator();

                        egui::CollapsingHeader::new("Canvas Settings")
                            .default_open(true)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.canvas_size, 0.3..=0.8)
                                            .text("Canvas Size"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.brush_size, 0.001..=0.015)
                                            .text("Brush Size"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.canvas_offset_x, 0.0..=0.5)
                                            .text("Canvas X"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.canvas_offset_y, 0.0..=0.5)
                                            .text("Canvas Y"),
                                    )
                                    .changed();

                                if ui.button("Clear Canvas").clicked() {
                                    params.clear_canvas = 1;
                                    changed = true;
                                } else {
                                    params.clear_canvas = 0;
                                }
                            });

                        egui::CollapsingHeader::new("CNN Settings")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(
                                            &mut params.prediction_threshold,
                                            0.0..=0.5,
                                        )
                                        .text("Prediction Threshold"),
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

        // Update mouse uniform for drawing interaction
        self.compute_shader
            .update_mouse_uniform(&self.base.mouse_tracker.uniform, &core.queue);

        // Execute CNN pipeline
        // Note: our backend automatically uses custom workgroup sizes from PassDescription
        self.compute_shader.dispatch(&mut encoder, core);

        // Render compute output to screen
        {
            let compute_texture = self.compute_shader.get_output_texture();
            let mut render_pass = cuneus::Renderer::begin_render_pass(
                &mut encoder,
                &view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("CNN Display Pass"),
            );

            render_pass.set_pipeline(&self.base.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &compute_texture.bind_group, &[]);
            render_pass.draw(0..4, 0..1);
        }

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

        self.base
            .handle_render_output(core, &view, full_output, &mut encoder);
        core.queue.submit(std::iter::once(encoder.finish()));
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

        // Handle mouse input for drawing on canvas
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
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (app, event_loop) = ShaderApp::new("CNN Digit Recognizer", 800, 600);

    app.run(event_loop, |core| CNNDigitRecognizer::init(core))
}

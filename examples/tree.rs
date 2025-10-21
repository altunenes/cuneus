use cuneus::compute::*;
use cuneus::prelude::*;
use winit::event::WindowEvent;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TreeParams {
    pixel_offset: f32,
    pixel_offset2: f32,
    lights: f32,
    exp: f32,
    frame: f32,
    col1: f32,
    col2: f32,
    decay: f32,
}

impl UniformProvider for TreeParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct TreeShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: TreeParams,
}

impl ShaderManager for TreeShader {
    fn init(core: &Core) -> Self {
        let initial_params = TreeParams {
            pixel_offset: 1.35,
            pixel_offset2: 1.0,
            lights: 2.2,
            exp: 4.0,
            frame: 0.5,
            col1: 205.0,
            col2: 5.5,
            decay: 0.96,
        };

        let texture_bind_group_layout = RenderKit::create_standard_texture_layout(&core.device);
        let base = RenderKit::new(core, &texture_bind_group_layout, None);

        // Create multipass system: buffer_a -> buffer_b -> buffer_c -> main_image
        let passes = vec![
            PassDescription::new("buffer_a", &[]), // no dependencies, generates fractal
            PassDescription::new("buffer_b", &["buffer_a"]), // reads buffer_a
            PassDescription::new("buffer_c", &["buffer_c", "buffer_b"]), // self-feedback + buffer_b
            PassDescription::new("main_image", &["buffer_c"]),
        ];

        let config = ComputeShader::builder()
            .with_entry_point("buffer_a")
            .with_multi_pass(&passes)
            .with_custom_uniforms::<TreeParams>()
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Tree Unified")
            .build();

        let mut compute_shader =
            ComputeShader::from_builder(core, include_str!("shaders/tree.wgsl"), config);

        // Enable hot reload
        if let Err(e) = compute_shader.enable_hot_reload(
            core.device.clone(),
            std::path::PathBuf::from("examples/shaders/tree.wgsl"),
            core.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Tree Hot Reload"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("shaders/tree.wgsl").into()),
                }),
        ) {
            eprintln!("Failed to enable hot reload for tree shader: {}", e);
        }

        compute_shader.set_custom_params(initial_params, &core.queue);

        Self {
            base,
            compute_shader,
            current_params: initial_params,
        }
    }

    fn update(&mut self, core: &Core) {
        // Check for hot reload updates
        self.compute_shader.check_hot_reload(&core.device);
        // Handle export
        self.compute_shader.handle_export(core, &mut self.base);

        // Update time uniform - this is crucial for accumulation!
        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0 / 60.0; // Approximate delta time
        self.compute_shader
            .set_time(current_time, delta, &core.queue);

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
                label: Some("Tree Render Encoder"),
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

                egui::Window::new("Fractal Tree")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(280.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("Fractal Parameters")
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
                                        egui::Slider::new(&mut params.frame, 0.0..=2.2)
                                            .text("Frame"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.col1, 0.0..=300.0)
                                            .text("Iterations"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.col2, 0.0..=10.0)
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
                        ui.label("Multi-buffer fractal tree with particle tracing");
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        // Handle controls and clear buffers if requested
        if controls_request.should_clear_buffers {
            // Reset frame count to restart accumulation
            self.compute_shader.current_frame = 0;
        }

        // Execute multi-pass compute shader: buffer_a -> buffer_b -> buffer_c -> main_image
        self.compute_shader.dispatch(&mut encoder, core);

        // Render compute output to screen
        {
            let compute_texture = self.compute_shader.get_output_texture();
            let mut render_pass = cuneus::Renderer::begin_render_pass(
                &mut encoder,
                &view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("Tree Display Pass"),
            );

            render_pass.set_pipeline(&self.base.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &compute_texture.bind_group, &[]);
            render_pass.draw(0..4, 0..1);
        }

        // Apply UI changes
        self.base.apply_control_request(controls_request);
        self.base.export_manager.apply_ui_request(export_request);

        if changed {
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
        }

        if should_start_export {
            self.base.export_manager.start_export();
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
    let (app, event_loop) = ShaderApp::new("Fractal Tree", 800, 600);
    app.run(event_loop, |core| TreeShader::init(core))
}

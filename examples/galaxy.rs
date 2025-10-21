use cuneus::prelude::*;
use cuneus::{Core, RenderKit, ShaderApp, ShaderManager, UniformProvider};
use winit::event::WindowEvent;
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ShaderParams {
    max_iterations: i32,
    max_sub_iterations: i32,
    point_intensity: f32,
    center_scale: f32,
    time_scale: f32,
    dist_offset: f32,
    _pad1: [f32; 2],
}

impl UniformProvider for ShaderParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct GalaxyShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: ShaderParams,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("Galaxy", 800, 600);
    app.run(event_loop, |core| GalaxyShader::init(core))
}
impl ShaderManager for GalaxyShader {
    fn init(core: &Core) -> Self {
        let texture_bind_group_layout = RenderKit::create_standard_texture_layout(&core.device);

        let initial_params = ShaderParams {
            max_iterations: 150,
            max_sub_iterations: 11,
            point_intensity: 0.000828,
            center_scale: 1.0,
            time_scale: 0.1,
            dist_offset: 0.07,
            _pad1: [0.0; 2],
        };

        let base = RenderKit::new(core, &texture_bind_group_layout, None);

        let config = ComputeShader::builder()
            .with_entry_point("main")
            .with_custom_uniforms::<ShaderParams>()
            .build();

        let mut compute_shader =
            ComputeShader::from_builder(core, include_str!("shaders/galaxy.wgsl"), config);

        // Enable hot reload
        if let Err(e) = compute_shader.enable_hot_reload(
            core.device.clone(),
            std::path::PathBuf::from("examples/shaders/galaxy.wgsl"),
            core.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Galaxy Hot Reload"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("shaders/galaxy.wgsl").into()),
                }),
        ) {
            eprintln!("Failed to enable hot reload for Galaxy shader: {}", e);
        }

        compute_shader.set_custom_params(initial_params, &core.queue);

        Self {
            base,
            compute_shader,
            current_params: initial_params,
        }
    }

    fn update(&mut self, core: &Core) {
        self.compute_shader.check_hot_reload(&core.device);
        // Handle export
        self.compute_shader.handle_export(core, &mut self.base);
        self.base.fps_tracker.update();
    }
    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = core
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Galaxy Render Encoder"),
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
                });

                egui::Window::new("Galaxy Settings").show(ctx, |ui| {
                    ui.collapsing("Parameters", |ui| {
                        changed |= ui
                            .add(
                                egui::Slider::new(&mut params.max_iterations, 50..=300)
                                    .text("Max Iterations"),
                            )
                            .changed();

                        changed |= ui
                            .add(
                                egui::Slider::new(&mut params.max_sub_iterations, 5..=20)
                                    .text("Max Sub Iterations"),
                            )
                            .changed();

                        changed |= ui
                            .add(
                                egui::Slider::new(&mut params.point_intensity, 0.0001..=0.01)
                                    .logarithmic(true)
                                    .text("Point Intensity"),
                            )
                            .changed();

                        changed |= ui
                            .add(
                                egui::Slider::new(&mut params.center_scale, 0.1..=5.0)
                                    .text("Center Scale"),
                            )
                            .changed();

                        changed |= ui
                            .add(
                                egui::Slider::new(&mut params.time_scale, 0.01..=1.0)
                                    .text("Time Scale"),
                            )
                            .changed();

                        changed |= ui
                            .add(
                                egui::Slider::new(&mut params.dist_offset, 0.01..=0.5)
                                    .text("Distance Offset"),
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

        self.base.export_manager.apply_ui_request(export_request);
        self.base.apply_control_request(controls_request);

        let current_time = self.base.controls.get_time(&self.base.start_time);

        let delta = 1.0 / 60.0;
        self.compute_shader
            .set_time(current_time, delta, &core.queue);

        if changed {
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
        }

        if should_start_export {
            self.base.export_manager.start_export();
        }

        // Dispatch compute shader
        self.compute_shader.dispatch(&mut encoder, core);

        // Render compute output to screen
        {
            let mut render_pass = cuneus::Renderer::begin_render_pass(
                &mut encoder,
                &view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("Main Render Pass"),
            );

            let compute_texture = self.compute_shader.get_output_texture();
            render_pass.set_pipeline(&self.base.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &compute_texture.bind_group, &[]);
            render_pass.draw(0..4, 0..1);
        }

        self.base
            .handle_render_output(core, &view, full_output, &mut encoder);
        core.queue.submit(Some(encoder.finish()));
        output.present();
        Ok(())
    }
    fn resize(&mut self, core: &Core) {
        self.base.update_resolution(&core.queue, core.size);
        self.compute_shader
            .resize(core, core.size.width, core.size.height);
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

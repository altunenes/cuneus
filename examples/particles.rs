use cuneus::compute::*;
use cuneus::prelude::*;
use winit::event::WindowEvent;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ParticleParams {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    num_circles: f32,
    num_points: f32,
    particle_intensity: f32,
    gamma: f32,
    feedback_mix: f32,
    feedback_decay: f32,
    scale: f32,
    rotation: f32,
    bloom_scale: f32,
    animation_speed: f32,
    color_shift_speed: f32,
    color_scale: f32,
}

impl UniformProvider for ParticleParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct ParticleShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: ParticleParams,
    frame_count: u32,
}

impl ParticleShader {
    fn clear_buffers(&mut self, core: &Core) {
        self.compute_shader.clear_all_buffers(core);
        self.frame_count = 0;
    }
}

impl ShaderManager for ParticleShader {
    fn init(core: &Core) -> Self {
        let texture_bind_group_layout = RenderKit::create_standard_texture_layout(&core.device);
        let base = RenderKit::new(core, &texture_bind_group_layout, None);

        let initial_params = ParticleParams {
            a: -1.8,
            b: -2.0,
            c: -0.5,
            d: -0.9,
            num_circles: 6.0,
            num_points: 7.0,
            particle_intensity: 1.0,
            gamma: 0.5,
            feedback_mix: 0.5,
            feedback_decay: 2.0,
            scale: 3.0,
            rotation: 0.0,
            bloom_scale: 7.0,
            animation_speed: 0.1,
            color_shift_speed: 0.1,
            color_scale: 1.2,
        };

        let config = ComputeShader::builder()
            .with_entry_point("main_image")
            .with_custom_uniforms::<ParticleParams>()
            .with_storage_buffer(StorageBufferSpec::new(
                "atomic_buffer",
                (core.size.width * core.size.height * 3 * 4) as u64,
            )) // 3 channels * u32 per pixel
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Particles Unified")
            .build();

        let mut compute_shader =
            ComputeShader::from_builder(core, include_str!("shaders/particles.wgsl"), config);

        // Enable hot reload
        if let Err(e) = compute_shader.enable_hot_reload(
            core.device.clone(),
            std::path::PathBuf::from("examples/shaders/particles.wgsl"),
            core.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Particles Hot Reload"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("shaders/particles.wgsl").into()),
                }),
        ) {
            eprintln!("Failed to enable hot reload for Particles shader: {}", e);
        }

        compute_shader.set_custom_params(initial_params, &core.queue);

        Self {
            base,
            compute_shader,
            current_params: initial_params,
            frame_count: 0,
        }
    }

    fn update(&mut self, core: &Core) {
        // Check for hot reload updates
        self.compute_shader.check_hot_reload(&core.device);
        // Handle export
        self.compute_shader.handle_export(core, &mut self.base);

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
                label: Some("Render Encoder"),
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

                egui::Window::new("Particle System")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(280.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("Attractor Parameters")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.a, -3.0..=3.0)
                                            .text("Parameter A"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.b, -3.0..=3.0)
                                            .text("Parameter B"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.c, -3.0..=3.0)
                                            .text("Parameter C"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.d, -3.0..=3.0)
                                            .text("Parameter D"),
                                    )
                                    .changed();
                                ui.separator();
                            });

                        egui::CollapsingHeader::new("Visual Settings")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.num_circles, 1.0..=10.0)
                                            .text("Number of Circles"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.num_points, 1.0..=10.0)
                                            .text("Points per Circle"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(
                                            &mut params.particle_intensity,
                                            0.1..=5.0,
                                        )
                                        .text("Intensity"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.gamma, 0.1..=0.5)
                                            .text("Gamma"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.bloom_scale, 1.0..=20.0)
                                            .text("Scale"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.scale, 0.1..=10.0)
                                            .text("Attractor Scale"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Animation & Feedback")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.animation_speed, 0.0..=1.0)
                                            .text("Animation Speed"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.feedback_decay, 0.0..=2.5)
                                            .text("Feedback Decay"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.feedback_mix, 0.0..=1.0)
                                            .text("Feedback Mix"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.color_shift_speed, 0.0..=1.0)
                                            .text("Color Shift Speed"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.color_scale, 0.0..=3.24)
                                            .text("Color Scale"),
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
        if controls_request.should_clear_buffers {
            self.clear_buffers(core);
        }
        self.base.apply_control_request(controls_request);

        if changed {
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
        }

        if should_start_export {
            self.base.export_manager.start_export();
        }

        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0 / 60.0;
        self.compute_shader
            .set_time(current_time, delta, &core.queue);
        self.compute_shader.time_uniform.data.frame = self.frame_count;
        self.compute_shader.time_uniform.update(&core.queue);

        // Single stage dispatch
        self.compute_shader.dispatch(&mut encoder, core);

        {
            let mut render_pass = cuneus::Renderer::begin_render_pass(
                &mut encoder,
                &view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("Display Pass"),
            );

            render_pass.set_pipeline(&self.base.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &self.compute_shader.output_texture.bind_group, &[]);
            render_pass.draw(0..4, 0..1);
        }

        self.base
            .handle_render_output(core, &view, full_output, &mut encoder);
        core.queue.submit(Some(encoder.finish()));
        output.present();

        self.frame_count = self.frame_count.wrapping_add(1);

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
    let (app, event_loop) = cuneus::ShaderApp::new("Particles", 800, 600);
    app.run(event_loop, |core| ParticleShader::init(core))
}

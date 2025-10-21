use cuneus::compute::{ComputeShader, COMPUTE_TEXTURE_FORMAT_RGBA16};
use cuneus::{Core, RenderKit, ShaderApp, ShaderControls, ShaderManager};
use cuneus::{ExportManager, UniformProvider};
use winit::event::*;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ShaderParams {
    background_color: f32,
    _pad0: f32,
    _pad00: f32,
    _pad000: f32,
    hue_color: [f32; 3],
    _pad1: f32,

    light_intensity: f32,
    rim_power: f32,
    ao_strength: f32,
    env_light_strength: f32,

    iridescence_power: f32,
    falloff_distance: f32,
    global_light: f32,
    alpha_threshold: f32,

    mix_factor_scale: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,

    _pad5: f32,
    _pad6: f32,
    _pad7: f32,
    _pad8: f32,
    _pad9: f32,
    _pad10: f32,
}

impl UniformProvider for ShaderParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct Shader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: ShaderParams,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("cuneus", 800, 600);
    app.run(event_loop, |core| Shader::init(core))
}

impl Shader {
    fn clear_buffers(&mut self, core: &Core) {
        self.compute_shader.clear_all_buffers(core);
    }
}

impl ShaderManager for Shader {
    fn init(core: &Core) -> Self {
        let texture_bind_group_layout = RenderKit::create_standard_texture_layout(&core.device);
        let base = RenderKit::new(core, &texture_bind_group_layout, None);

        let initial_params = ShaderParams {
            background_color: 0.4,
            _pad0: 0.0,
            _pad00: 0.0,
            _pad000: 0.0,
            hue_color: [1.0, 2.0, 3.0],
            _pad1: 0.0,

            light_intensity: 1.8,
            rim_power: 3.0,
            ao_strength: 0.1,
            env_light_strength: 0.5,

            iridescence_power: 0.2,
            falloff_distance: 1.0,
            global_light: 1.0,
            alpha_threshold: 1.0,

            mix_factor_scale: 0.3,
            _pad2: 0.0,
            _pad3: 0.0,
            _pad4: 0.0,

            _pad5: 0.0,
            _pad6: 0.0,
            _pad7: 0.0,
            _pad8: 0.0,
            _pad9: 0.0,
            _pad10: 0.0,
        };

        // Entry point configuration
        let config = ComputeShader::builder()
            .with_entry_point("main")
            .with_custom_uniforms::<ShaderParams>()
            .with_audio(1024) // Automatically goes to @group(2)
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Cuneus Compute")
            .build();

        let mut compute_shader =
            ComputeShader::from_builder(core, include_str!("shaders/cuneus.wgsl"), config);

        // Enable hot reload
        if let Err(e) = compute_shader.enable_hot_reload(
            core.device.clone(),
            std::path::PathBuf::from("examples/shaders/cuneus.wgsl"),
            core.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Cuneus Hot Reload"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("shaders/cuneus.wgsl").into()),
                }),
        ) {
            eprintln!("Failed to enable hot reload for cuneus shader: {}", e);
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

                egui::Window::new("Cuneus")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(250.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("Colors")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.background_color, 0.0..=1.0)
                                            .text("Background"),
                                    )
                                    .changed();

                                changed |=
                                    ui.color_edit_button_rgb(&mut params.hue_color).changed();
                                ui.label("Base Color");
                            });

                        egui::CollapsingHeader::new("Lighting")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.light_intensity, 0.0..=3.2)
                                            .text("Light Intensity"),
                                    )
                                    .changed();

                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.ao_strength, 0.0..=10.0)
                                            .text("AO Strength"),
                                    )
                                    .changed();

                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.global_light, 0.1..=2.0)
                                            .text("Global Light"),
                                    )
                                    .changed();

                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.rim_power, 0.1..=10.0)
                                            .text("Rim Power"),
                                    )
                                    .changed();

                                changed |= ui
                                    .add(
                                        egui::Slider::new(
                                            &mut params.env_light_strength,
                                            0.0..=1.0,
                                        )
                                        .text("Environment Light"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Effects")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.alpha_threshold, 0.0..=3.0)
                                            .text("Alpha Threshold"),
                                    )
                                    .changed();

                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.mix_factor_scale, 0.0..=1.5)
                                            .text("Mix Factor Scale"),
                                    )
                                    .changed();

                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.iridescence_power, 0.0..=1.0)
                                            .text("Iridescence"),
                                    )
                                    .changed();

                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.falloff_distance, 0.5..=5.0)
                                            .text("Light Falloff"),
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

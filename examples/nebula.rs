use cuneus::compute::*;
use cuneus::prelude::*;
use winit::event::WindowEvent;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct NebulaParams {
    iterations: i32,
    formuparam: f32,
    volsteps: i32,
    stepsize: f32,
    zoom: f32,
    tile: f32,
    speed: f32,
    brightness: f32,
    dust_intensity: f32,
    distfading: f32,
    color_variation: f32,
    n_boxes: f32,
    rotation: i32,
    depth: f32,
    color_mode: i32,
    _padding1: f32,

    rotation_x: f32,
    rotation_y: f32,
    click_state: i32,
    scale: f32,

    exposure: f32,
    gamma: f32,

    _padding4: f32,
    _padding5: f32,
    _padding6: f32,
    _padding7: f32,
    _padding8: f32,
    _padding9: f32,
    _padding10: f32,

    time_scale: f32,
    visual_mode: i32,
    _padding2: f32,
    _padding3: f32,
}

impl UniformProvider for NebulaParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct NebulaShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: NebulaParams,
    frame_count: u32,
}

impl NebulaShader {
    fn clear_buffers(&mut self, core: &Core) {
        self.compute_shader.clear_all_buffers(core);
        self.frame_count = 0;
    }
}

impl ShaderManager for NebulaShader {
    fn init(core: &Core) -> Self {
        let texture_bind_group_layout = RenderKit::create_standard_texture_layout(&core.device);
        let base = RenderKit::new(core, &texture_bind_group_layout, None);

        let initial_params = NebulaParams {
            iterations: 17,
            formuparam: 0.52,
            volsteps: 6,
            stepsize: 0.31,
            zoom: 5.0,
            tile: 0.35,
            speed: 0.020,
            brightness: 0.00062,
            dust_intensity: 1.0,
            distfading: 0.95,
            color_variation: 0.51,
            n_boxes: 10.0,
            rotation: 1,
            depth: 5.0,
            color_mode: 1,
            _padding1: 0.0,

            rotation_x: 0.0,
            rotation_y: 0.0,
            click_state: 0,
            scale: 1.0,

            exposure: 1.6,
            gamma: 0.400,

            _padding4: 0.0,
            _padding5: 0.0,
            _padding6: 0.0,
            _padding7: 0.0,
            _padding8: 0.0,
            _padding9: 0.0,
            _padding10: 0.0,

            time_scale: 1.0,
            visual_mode: 0,
            _padding2: 0.0,
            _padding3: 0.0,
        };

        let mut config = ComputeShader::builder()
            .with_entry_point("volumetric_render")
            .with_custom_uniforms::<NebulaParams>()
            .with_atomic_buffer()
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Nebula Unified")
            .build();

        // Add second entry point manually
        config.entry_points.push("main_image".to_string());

        let mut compute_shader =
            ComputeShader::from_builder(core, include_str!("shaders/nebula.wgsl"), config);

        // Enable hot reload
        if let Err(e) = compute_shader.enable_hot_reload(
            core.device.clone(),
            std::path::PathBuf::from("examples/shaders/nebula.wgsl"),
            core.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Nebula Hot Reload"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("shaders/nebula.wgsl").into()),
                }),
        ) {
            eprintln!("Failed to enable hot reload for Nebula shader: {}", e);
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

        // Mouse interaction
        if self.base.mouse_tracker.uniform.buttons[0] & 1 != 0 {
            params.rotation_x = self.base.mouse_tracker.uniform.position[0];
            params.rotation_y = self.base.mouse_tracker.uniform.position[1];
            params.click_state = 1;
            changed = true;
        } else {
            params.click_state = 0;
        }

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

                egui::Window::new("universe")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(320.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("Volumetric Parameters")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.iterations, 5..=30)
                                            .text("Iterations"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.formuparam, 0.1..=1.0)
                                            .text("Form Parameter"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.volsteps, 1..=20)
                                            .text("Volume Steps"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.stepsize, 0.05..=0.5)
                                            .text("Step Size"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.zoom, 0.1..=112.0)
                                            .text("Zoom"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.tile, 0.1..=3.0).text("Tile"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Appearance")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.brightness, 0.0001..=0.015)
                                            .logarithmic(true)
                                            .text("Brightness"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.dust_intensity, 0.0..=2.0)
                                            .text("Dust Intensity"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.distfading, 0.1..=3.0)
                                            .text("Distance Fading"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.color_variation, 0.2..=5.0)
                                            .text("Color Variation"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.exposure, 0.2..=3.0)
                                            .text("Exposure"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.gamma, 0.1..=1.2)
                                            .text("Gamma"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Animation")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.speed, -0.1..=0.1)
                                            .text("Galaxy Speed"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.time_scale, 0.1..=2.0)
                                            .text("Animation Speed"),
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

        // Stage 0: Volumetric render (not doing anything in this case, just placeholder)
        self.compute_shader.dispatch_stage(&mut encoder, core, 0);

        // Stage 1: Main image render
        self.compute_shader.dispatch_stage(&mut encoder, core, 1);

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

        self.frame_count = self.frame_count.wrapping_add(1);

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
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("universe", 800, 600);
    app.run(event_loop, |core| NebulaShader::init(core))
}

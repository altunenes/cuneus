use cuneus::compute::*;
use cuneus::prelude::*;
use winit::event::WindowEvent;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct RorschachParams {
    seed: f32,
    zoom: f32,
    threshold: f32,
    distortion: f32,
    
    particle_speed: f32,
    particle_life: f32,
    trace_steps: f32,
    contrast: f32,
    
    color_r: f32,
    color_g: f32,
    color_b: f32,
    gamma: f32,

    style: f32, 
    fbm_octaves: f32,
    tint_x: f32,
    tint_y: f32,

    tint_z: f32,
    _pad_final1: f32,
    _pad_final2: f32,
    _pad_final3: f32,
}

impl UniformProvider for RorschachParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct RorschachShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: RorschachParams,
}

impl ShaderManager for RorschachShader {
    fn init(core: &Core) -> Self {
        let initial_params = RorschachParams {
            seed: 87.0,
            zoom: 5.2,
            threshold: 0.383,
            distortion: 2.63,
            particle_speed: 0.45,
            particle_life: 0.99,
            trace_steps: 22.0,
            contrast: 6.0,
            color_r: 0.58,
            color_g: 0.12,
            color_b: 0.12,
            gamma: 0.2,
            style: 1.0, 
            
            fbm_octaves: 5.0,
            tint_x: 0.3,
            tint_y: 0.04,
            tint_z: 0.28,

            _pad_final1: 0.0,
            _pad_final2: 0.0,
            _pad_final3: 0.0,
        };

        let texture_bind_group_layout = RenderKit::create_standard_texture_layout(&core.device);
        let base = RenderKit::new(core, &texture_bind_group_layout, None);

        let passes = vec![
            PassDescription::new("buffer_a", &[]), 
            PassDescription::new("buffer_b", &["buffer_a"]), 
            PassDescription::new("buffer_c", &["buffer_c", "buffer_b"]), 
            PassDescription::new("main_image", &["buffer_c"]),
        ];

        let config = ComputeShader::builder()
            .with_entry_point("buffer_a")
            .with_multi_pass(&passes)
            .with_custom_uniforms::<RorschachParams>()
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Rorschach Unified")
            .build();

        let compute_shader = cuneus::compute_shader!(core, "shaders/rorschach.wgsl", config);

        compute_shader.set_custom_params(initial_params, &core.queue);

        Self {
            base,
            compute_shader,
            current_params: initial_params,
        }
    }

    fn update(&mut self, core: &Core) {
        self.compute_shader.check_hot_reload(&core.device);
        self.compute_shader.handle_export(core, &mut self.base);

        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0 / 60.0;
        self.compute_shader.set_time(current_time, delta, &core.queue);
        self.base.fps_tracker.update();
    }

    fn resize(&mut self, core: &Core) {
        self.base.update_resolution(&core.queue, core.size);
        self.compute_shader.resize(core, core.size.width, core.size.height);
    }

    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Rorschach Render Encoder"),
        });

        let mut params = self.current_params;
        let mut changed = false;
        let mut should_reset = false;
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self.base.controls.get_ui_request(&self.base.start_time, &core.size);
        controls_request.current_fps = Some(self.base.fps_tracker.fps());

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                ctx.style_mut(|style| {
                    style.visuals.window_fill = egui::Color32::from_rgba_premultiplied(0, 0, 0, 180);
                    style.text_styles.get_mut(&egui::TextStyle::Body).unwrap().size = 11.0;
                    style.text_styles.get_mut(&egui::TextStyle::Button).unwrap().size = 10.0;
                });

                egui::Window::new("Rorschach")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(280.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("Shape")
                            .default_open(true)
                            .show(ui, |ui| {
                                if ui.add(egui::Slider::new(&mut params.seed, 0.0..=100.0).text("Seed")).changed() { changed = true; should_reset = true; }
                                if ui.add(egui::Slider::new(&mut params.zoom, 1.0..=10.0).text("Zoom")).changed() { changed = true; should_reset = true; }
                                if ui.add(egui::Slider::new(&mut params.threshold, 0.3..=0.6).text("Ink Amount")).changed() { changed = true; should_reset = true; }
                                if ui.add(egui::Slider::new(&mut params.distortion, 0.0..=3.0).text("Warping")).changed() { changed = true; should_reset = true; }
                                if ui.add(egui::Slider::new(&mut params.fbm_octaves, 1.0..=25.0).text("Detail Octaves")).changed() { 
                                    params.fbm_octaves = params.fbm_octaves.round();
                                    changed = true; 
                                    should_reset = true; 
                                }
                            });

                        egui::CollapsingHeader::new("Particle Tracer")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.particle_speed, 0.0..=5.0).text("brush")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.trace_steps, 1.0..=100.0).text("Density")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.particle_life, 0.8..=0.999).text("Trail Life")).changed();
                            });

                        egui::CollapsingHeader::new("Visual Settings")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.label("Primary Ink Color:");
                                let mut color = [params.color_r, params.color_g, params.color_b];
                                if ui.color_edit_button_rgb(&mut color).changed() {
                                    params.color_r = color[0];
                                    params.color_g = color[1];
                                    params.color_b = color[2];
                                    changed = true;
                                }
                                changed |= ui.add(egui::Slider::new(&mut params.contrast, 0.5..=6.0).text("Contrast")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.gamma, 0.1..=2.0).text("Gamma")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.style, 0.0..=1.0).text("Blend")).changed();

                                ui.separator();
                                ui.horizontal(|ui| {
                                    ui.label("Phase");
                                    let mut tint_color = [params.tint_x, params.tint_y, params.tint_z];
                                    if ui.color_edit_button_rgb(&mut tint_color).changed() {
                                        params.tint_x = tint_color[0];
                                        params.tint_y = tint_color[1];
                                        params.tint_z = tint_color[2];
                                        changed = true;
                                    }
                                });
                            });

                        ui.separator();
                        ShaderControls::render_controls_widget(ui, &mut controls_request);
                        ui.separator();
                        should_start_export = ExportManager::render_export_ui_widget(ui, &mut export_request);
                        ui.separator();
                        
                        ui.horizontal(|ui| {
                           ui.label(format!("Frame: {}", self.compute_shader.current_frame));
                           if ui.button("Clear").clicked() { should_reset = true; }
                        });
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        if controls_request.should_clear_buffers || should_reset {
            self.compute_shader.current_frame = 0;
            self.compute_shader.time_uniform.data.frame = 0;
            self.compute_shader.time_uniform.update(&core.queue);
        }

        self.compute_shader.dispatch(&mut encoder, core);

        self.base.renderer.render_to_view(&mut encoder, &view, &self.compute_shader);

        self.base.apply_control_request(controls_request);
        self.base.export_manager.apply_ui_request(export_request);

        if changed {
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
        }

        if should_start_export {
            self.base.export_manager.start_export();
        }

        self.base.handle_render_output(core, &view, full_output, &mut encoder);
        core.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        if self.base.egui_state.on_window_event(core.window(), event).consumed { return true; }
        if let WindowEvent::KeyboardInput { event, .. } = event {
            return self.base.key_handler.handle_keyboard_input(core.window(), event);
        }
        false
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("Rorschach Tracer", 700, 500);
    app.run(event_loop, RorschachShader::init)
}
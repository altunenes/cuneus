use cuneus::prelude::*;
use cuneus::{Core, RenderKit, ShaderApp, ShaderManager, UniformProvider};
use winit::event::WindowEvent;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ExperimentParams {

    col_bg: [f32; 4],
    col_line: [f32; 4],
    col_core: [f32; 4],
    col_amber: [f32; 4],

    ball_offset_x: f32,
    ball_offset_y: f32,
    ball_sink: f32,
    
    distortion_amt: f32,
    noise_amt: f32,
    stream_width: f32,
    
    speed: f32,
    scale: f32,
    angle: f32,
    line_freq: f32,
    
    cam_height: f32,
    cam_distance: f32,
    cam_fov: f32,
    
    rim_intensity: f32,
    spotlight_intensity: f32,
    spotlight_height: f32,
    
    gamma: f32,
    saturation: f32,
    contrast: f32,
    
    orbital_enabled: u32,
    orbital_speed: f32,
    orbital_radius: f32,
    _pad: [f32; 2],
}

impl Default for ExperimentParams {
    fn default() -> Self {
        Self {
            col_bg: [0.05, 0.02, 0.10, 1.0],
            col_line: [0.55, 0.40, 0.85, 1.0],
            col_core: [1.0, 0.1, 0.2, 1.0],
            col_amber: [1.0, 0.6, 0.1, 1.0],

            ball_offset_x: 0.0,
            ball_offset_y: 0.0,
            ball_sink: 1.0,
            
            distortion_amt: 50.0,
            noise_amt: 300.0,
            stream_width: 0.08,
            
            speed: 1.0,
            scale: 1.35,
            angle: -1.785398,
            line_freq: 100.0,
            
            cam_height: 3.58,
            cam_distance: 6.0,
            cam_fov: 1.7,
            
            rim_intensity: 1.2,
            spotlight_intensity: 1.5,
            spotlight_height: 1.0,
            
            gamma: 0.66,
            saturation: 0.71,
            contrast: 1.1,
            
            orbital_enabled: 1,
            orbital_speed: 0.3,
            orbital_radius: 2.12,
            _pad: [0.0; 2],
        }
    }
}

impl UniformProvider for ExperimentParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct ExperimentShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: ExperimentParams,
    orbital_enabled: bool,
}

impl ShaderManager for ExperimentShader {
    fn init(core: &Core) -> Self {
        let base = RenderKit::new(core);
        let initial_params = ExperimentParams::default();

        let config = ComputeShader::builder()
            .with_entry_point("main")
            .with_custom_uniforms::<ExperimentParams>()
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(wgpu::TextureFormat::Rgba16Float)
            .with_label("Currents Port")
            .build();

        let compute_shader = cuneus::compute_shader!(core, "shaders/tameimp.wgsl", config);

        compute_shader.set_custom_params(initial_params, &core.queue);

        Self {
            base,
            compute_shader,
            current_params: initial_params,
            orbital_enabled: true,
        }
    }

    fn update(&mut self, core: &Core) {
        self.compute_shader.handle_export(core, &mut self.base);
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
            .get_ui_request(&self.base.start_time, &core.size);
        
        let current_fps = self.base.fps_tracker.fps();
        controls_request.current_fps = Some(current_fps);
        
        let mut orbital_enabled = self.orbital_enabled;

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                ctx.style_mut(|style| {
                    style.visuals.window_fill = egui::Color32::from_black_alpha(220);
                    style.visuals.window_stroke = egui::Stroke::new(1.0, egui::Color32::from_gray(60));
                    style.text_styles.get_mut(&egui::TextStyle::Body).unwrap().size = 11.0;
                    style.text_styles.get_mut(&egui::TextStyle::Button).unwrap().size = 10.0;
                    style.text_styles.get_mut(&egui::TextStyle::Small).unwrap().size = 9.0;
                    style.text_styles.get_mut(&egui::TextStyle::Heading).unwrap().size = 12.0;
                    style.spacing.slider_width = 140.0;
                    style.spacing.item_spacing = egui::vec2(4.0, 3.0);
                });

                egui::Window::new("Currents")
                    .default_width(220.0)
                    .show(ctx, |ui| {
                        
                        egui::CollapsingHeader::new("🔮 ball").default_open(true).show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label("X");
                                changed |= ui.add(egui::Slider::new(&mut params.ball_offset_x, -1.0..=1.0).show_value(true)).changed();
                            });
                            ui.horizontal(|ui| {
                                ui.label("Y");
                                changed |= ui.add(egui::Slider::new(&mut params.ball_offset_y, -1.0..=1.0).show_value(true)).changed();
                            });
                            ui.horizontal(|ui| {
                                ui.label("Sink");
                                changed |= ui.add(egui::Slider::new(&mut params.ball_sink, -3.2..=3.2).show_value(true)).changed();
                            });
                            ui.horizontal(|ui| {
                                ui.label("Angle");
                                changed |= ui.add(egui::Slider::new(&mut params.angle, -3.14..=3.14).show_value(true)).changed();
                            });
                        });

                        egui::CollapsingHeader::new("🌊 noise").default_open(false).show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label("Distortion");
                                changed |= ui.add(egui::Slider::new(&mut params.distortion_amt, 0.0..=75.0).show_value(true)).changed();
                            });
                            ui.horizontal(|ui| {
                                ui.label("Stream W");
                                changed |= ui.add(egui::Slider::new(&mut params.stream_width, 0.01..=0.3).show_value(true)).changed();
                            });
                            ui.horizontal(|ui| {
                                ui.label("Line Freq");
                                changed |= ui.add(egui::Slider::new(&mut params.line_freq, 20.0..=200.0).show_value(true)).changed();
                            });
                        });
                        egui::CollapsingHeader::new("📷 cam").default_open(false).show(ui, |ui| {
                            ui.horizontal(|ui| {
                                if ui.checkbox(&mut orbital_enabled, "🔄 Orbital").changed() {
                                    changed = true;
                                }
                            });
                            if orbital_enabled {
                                ui.horizontal(|ui| {
                                    ui.label("Orb Speed");
                                    changed |= ui.add(egui::Slider::new(&mut params.orbital_speed, 0.01..=2.0).show_value(true)).changed();
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Orb Radius");
                                    changed |= ui.add(egui::Slider::new(&mut params.orbital_radius, 0.1..=3.0).show_value(true)).changed();
                                });
                            }
                            
                            ui.separator();
                            
                            ui.horizontal(|ui| {
                                ui.label("Height");
                                changed |= ui.add(egui::Slider::new(&mut params.cam_height, 0.5..=5.0).show_value(true)).changed();
                            });
                            ui.horizontal(|ui| {
                                ui.label("FOV");
                                changed |= ui.add(egui::Slider::new(&mut params.cam_fov, 0.5..=2.5).show_value(true)).changed();
                            });
                            ui.horizontal(|ui| {
                                ui.label("Scale");
                                changed |= ui.add(egui::Slider::new(&mut params.scale, 0.3..=3.0).show_value(true)).changed();
                            });
                        });

                        egui::CollapsingHeader::new("💡 light").default_open(false).show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label("Spot Int");
                                changed |= ui.add(egui::Slider::new(&mut params.spotlight_intensity, 0.0..=5.0).show_value(true)).changed();
                            });
                            ui.horizontal(|ui| {
                                ui.label("Spot H");
                                changed |= ui.add(egui::Slider::new(&mut params.spotlight_height, 0.2..=3.0).show_value(true)).changed();
                            });
                            ui.horizontal(|ui| {
                                ui.label("fog");
                                changed |= ui.add(egui::Slider::new(&mut params.cam_distance, 3.0..=8.0).show_value(true)).changed();
                            });
                        });

                        egui::CollapsingHeader::new("✨ post").default_open(false).show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label("Gamma");
                                changed |= ui.add(egui::Slider::new(&mut params.gamma, 0.2..=2.0).show_value(true)).changed();
                            });
                            ui.horizontal(|ui| {
                                ui.label("Saturation");
                                changed |= ui.add(egui::Slider::new(&mut params.saturation, 0.0..=2.5).show_value(true)).changed();
                            });
                            ui.horizontal(|ui| {
                                ui.label("Contrast");
                                changed |= ui.add(egui::Slider::new(&mut params.contrast, 0.5..=2.0).show_value(true)).changed();
                            });
                        });
                        egui::CollapsingHeader::new("🎨 cols").default_open(false).show(ui, |ui| {
                            egui::Grid::new("colors_grid")
                                .num_columns(2)
                                .spacing([10.0, 4.0])
                                .striped(true)
                                .show(ui, |ui| {
                                    ui.label("Background");
                                    let mut rgb = [params.col_bg[0], params.col_bg[1], params.col_bg[2]];
                                    if ui.color_edit_button_rgb(&mut rgb).changed() {
                                        params.col_bg[0] = rgb[0];
                                        params.col_bg[1] = rgb[1];
                                        params.col_bg[2] = rgb[2];
                                        changed = true;
                                    }
                                    ui.end_row();
                                    ui.label("Line");
                                    let mut rgb = [params.col_line[0], params.col_line[1], params.col_line[2]];
                                    if ui.color_edit_button_rgb(&mut rgb).changed() {
                                        params.col_line[0] = rgb[0];
                                        params.col_line[1] = rgb[1];
                                        params.col_line[2] = rgb[2];
                                        changed = true;
                                    }
                                    ui.end_row();
                                    ui.label("Stream Core");
                                    let mut rgb = [params.col_core[0], params.col_core[1], params.col_core[2]];
                                    if ui.color_edit_button_rgb(&mut rgb).changed() {
                                        params.col_core[0] = rgb[0];
                                        params.col_core[1] = rgb[1];
                                        params.col_core[2] = rgb[2];
                                        changed = true;
                                    }
                                    ui.end_row();
                                    ui.label("Stream Amber");
                                    let mut rgb = [params.col_amber[0], params.col_amber[1], params.col_amber[2]];
                                    if ui.color_edit_button_rgb(&mut rgb).changed() {
                                        params.col_amber[0] = rgb[0];
                                        params.col_amber[1] = rgb[1];
                                        params.col_amber[2] = rgb[2];
                                        changed = true;
                                    }
                                    ui.end_row();
                                });
                        });
                        egui::CollapsingHeader::new("⏱").default_open(false).show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label("Speed");
                                changed |= ui.add(egui::Slider::new(&mut params.speed, 0.0..=2.0).show_value(true)).changed();
                            });
                        });

                        ui.separator();
                        
                        egui::CollapsingHeader::new("▶").default_open(false).show(ui, |ui| {
                            cuneus::ShaderControls::render_controls_widget(ui, &mut controls_request);
                        });

                        egui::CollapsingHeader::new("💾").default_open(false).show(ui, |ui| {
                            should_start_export = cuneus::ExportManager::render_export_ui_widget(ui, &mut export_request);
                        });

                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.separator();
                            ui.small("H: toggle UI");
                        });
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        // Update orbital state
        self.orbital_enabled = orbital_enabled;
        params.orbital_enabled = if orbital_enabled { 1 } else { 0 };

        self.base.export_manager.apply_ui_request(export_request);
        self.base.apply_control_request(controls_request);

        let current_time = self.base.controls.get_time(&self.base.start_time);
        self.compute_shader
            .set_time(current_time, 1.0 / 60.0, &core.queue);

        if changed {
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
        }

        if should_start_export {
            self.base.export_manager.start_export();
        }

        self.compute_shader.dispatch(&mut frame.encoder, core);

        self.base.renderer.render_to_view(&mut frame.encoder, &frame.view, &self.compute_shader);

        self.base.end_frame(core, frame, full_output);
        Ok(())
    }

    fn resize(&mut self, core: &Core) {
        self.base.default_resize(core, &mut self.compute_shader);
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        self.base.default_handle_input(core, event)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("Currents", 600, 800);
    app.run(event_loop, ExperimentShader::init)
}
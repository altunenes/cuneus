use cuneus::prelude::ComputeShader;
use cuneus::{Core, ExportManager, RenderKit, ShaderApp, ShaderControls, ShaderManager, UniformProvider};
use winit::event::*;
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShaderParams {
    base_color: [f32; 3],
    x: f32,
    rim_color: [f32; 3],
    y: f32,
    accent_color: [f32; 3],
    gamma_correction: f32,
    travel_speed: f32,
    iteration: i32,
    col_ext: f32,
    zoom: f32,
    trap_pow: f32,
    trap_x: f32,
    trap_y: f32,
    trap_c1: f32,
    aa: i32,
    trap_s1: f32,
    wave_speed: f32,
    fold_intensity: f32,
}

impl UniformProvider for ShaderParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct Shader {
    base: RenderKit,
    compute_shader: ComputeShader,
    mouse_dragging: bool,
    drag_start: [f32; 2],
    drag_start_pos: [f32; 2],
    zoom_level: f32,
    current_params: ShaderParams,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("orbits", 800, 600);
    app.run(event_loop, Shader::init)
}
impl ShaderManager for Shader {
    fn init(core: &Core) -> Self {
        let initial_zoom = 0.0004;
        let initial_x = 2.14278;
        let initial_y = 2.14278;

        // Create texture display layout
        let base = RenderKit::new(core);

        let config = ComputeShader::builder()
            .with_entry_point("main")
            .with_custom_uniforms::<ShaderParams>()
            .with_mouse()
            .build();

        let compute_shader = cuneus::compute_shader!(core, "shaders/orbits.wgsl", config);

        let initial_params = ShaderParams {
            base_color: [0.0, 0.5, 1.0],
            x: initial_x,
            rim_color: [0.0, 0.5, 1.0],
            y: initial_y,
            accent_color: [0.018, 0.018, 0.018],
            gamma_correction: 0.4,
            travel_speed: 1.0,
            iteration: 355,
            col_ext: 2.0,
            zoom: initial_zoom,
            trap_pow: 1.0,
            trap_x: -0.5,
            trap_y: 2.0,
            trap_c1: 0.2,
            aa: 1,
            trap_s1: 0.8,
            wave_speed: 0.1,
            fold_intensity: 1.0,
        };

        compute_shader.set_custom_params(initial_params, &core.queue);

        Self {
            base,
            compute_shader,
            mouse_dragging: false,
            drag_start: [0.0, 0.0],
            drag_start_pos: [initial_x, initial_y],
            zoom_level: initial_zoom,
            current_params: initial_params,
        }
    }

    fn update(&mut self, core: &Core) {
        // Handle export
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
            .get_ui_request(&self.base.start_time, &core.size, self.base.fps_tracker.fps());

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                RenderKit::apply_default_style(ctx);
                egui::Window::new("Orbits")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(280.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("Colors")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Base:");
                                    changed |=
                                        ui.color_edit_button_rgb(&mut params.base_color).changed();
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Orbit:");
                                    changed |=
                                        ui.color_edit_button_rgb(&mut params.rim_color).changed();
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Exterior:");
                                    changed |= ui
                                        .color_edit_button_rgb(&mut params.accent_color)
                                        .changed();
                                });
                            });

                        egui::CollapsingHeader::new("Rendering")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.iteration, 50..=500)
                                            .text("Iterations"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.aa, 1..=4)
                                            .text("Anti-aliasing"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.gamma_correction, 0.1..=2.0)
                                            .text("Gamma"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Traps")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.trap_x, -5.0..=5.0)
                                            .text("Trap X"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.trap_y, -5.0..=5.0)
                                            .text("Trap Y"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.trap_pow, 0.0..=3.0)
                                            .text("Trap Power"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.trap_c1, 0.0..=1.0)
                                            .text("Trap Mix"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.trap_s1, 0.0..=2.0)
                                            .text("Trap Blend"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Animation")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.travel_speed, 0.0..=2.0)
                                            .text("Travel Speed"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.wave_speed, 0.0..=2.0)
                                            .text("Wave Speed"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.fold_intensity, 0.0..=3.0)
                                            .text("Fold Intensity"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.col_ext, 0.0..=10.0)
                                            .text("Color Extension"),
                                    )
                                    .changed();
                            });

                        egui::CollapsingHeader::new("Navigation")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.label("Left-click + drag: Pan view");
                                ui.label("Mouse wheel: Zoom");
                                ui.separator();
                                let old_zoom = params.zoom;
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.zoom, 0.0001..=1.0)
                                            .text("Zoom")
                                            .logarithmic(true),
                                    )
                                    .changed();
                                if old_zoom != params.zoom {
                                    self.zoom_level = params.zoom;
                                }
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.x, 0.0..=3.0)
                                            .text("X Position"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut params.y, 0.0..=6.0)
                                            .text("Y Position"),
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

        if changed {
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
        }

        if should_start_export {
            self.base.export_manager.start_export();
        }

        // Create command encoder

        // Update time uniform
        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta_time = 1.0 / 60.0;
        self.compute_shader
            .set_time(current_time, delta_time, &core.queue);

        // Update mouse uniform
        self.compute_shader
            .update_mouse_uniform(&self.base.mouse_tracker.uniform, &core.queue);

        // Dispatch compute shader
        self.compute_shader.dispatch(&mut frame.encoder, core);

        self.base.renderer.render_to_view(&mut frame.encoder, &frame.view, &self.compute_shader.get_output_texture().bind_group);

        self.base.end_frame(core, frame, full_output);

        Ok(())
    }
    fn resize(&mut self, core: &Core) {
        self.base.default_resize(core, &mut self.compute_shader);
    }
    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        if self.base.default_handle_input(core, event) {
            return true;
        }
        match event {
            WindowEvent::MouseInput { state, button, .. } => {
                if button == &MouseButton::Left {
                    match state {
                        ElementState::Pressed => {
                            let mouse_pos = self.base.mouse_tracker.uniform.position;
                            self.mouse_dragging = true;
                            self.drag_start = mouse_pos;
                            self.drag_start_pos = [self.current_params.x, self.current_params.y];
                            return true;
                        }
                        ElementState::Released => {
                            self.mouse_dragging = false;
                            return true;
                        }
                    }
                }
                false
            }
            WindowEvent::CursorMoved { .. } => {
                if self.mouse_dragging {
                    let current_pos = self.base.mouse_tracker.uniform.position;
                    let dx = (current_pos[0] - self.drag_start[0]) * 3.0 * self.zoom_level;
                    let dy = (current_pos[1] - self.drag_start[1]) * 6.0 * self.zoom_level;
                    let mut new_x = self.drag_start_pos[0] + dx;
                    let mut new_y = self.drag_start_pos[1] + dy;
                    new_x = new_x.clamp(0.0, 3.0);
                    new_y = new_y.clamp(0.0, 6.0);
                    self.current_params.x = new_x;
                    self.current_params.y = new_y;
                    self.compute_shader
                        .set_custom_params(self.current_params, &core.queue);
                }
                self.base.handle_mouse_input(core, event, false)
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let zoom_delta = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y * 0.1,
                    MouseScrollDelta::PixelDelta(pos) => (pos.y as f32) * 0.001,
                };

                if zoom_delta != 0.0 {
                    let mouse_pos = self.base.mouse_tracker.uniform.position;
                    let center_x = self.current_params.x;
                    let center_y = self.current_params.y;

                    let rel_x = mouse_pos[0] - 0.5;
                    let rel_y = mouse_pos[1] - 0.5;

                    let zoom_factor = if zoom_delta > 0.0 { 0.9 } else { 1.1 };
                    self.zoom_level = (self.zoom_level * zoom_factor).clamp(0.0001, 1.5);

                    let scale_change = 1.0 - zoom_factor;
                    let dx = rel_x * scale_change * 3.0 * self.zoom_level;
                    let dy = rel_y * scale_change * 6.0 * self.zoom_level;
                    self.current_params.zoom = self.zoom_level;
                    self.current_params.x = (center_x + dx).clamp(0.0, 3.0);
                    self.current_params.y = (center_y + dy).clamp(0.0, 6.0);
                    self.compute_shader
                        .set_custom_params(self.current_params, &core.queue);
                }
                self.base.handle_mouse_input(core, event, false)
            }

            _ => self.base.handle_mouse_input(core, event, false),
        }
    }
}

use cuneus::prelude::*;
use cuneus::compute::{ComputeShader, PassDescription};
use cuneus::WindowEvent;
use std::ops::RangeInclusive;
use winit::event::{ElementState, MouseButton, MouseScrollDelta};

cuneus::uniform_params! {
    pub struct ShaderParams {
        base_color: [f32; 3],
        pixel_size: f32,
        palette_a: [f32; 3],
        gamma_correction: f32,
        palette_b: [f32; 3],
        iteration: i32,
        light_color: [f32; 3],
        aa: i32,
        rim_color: [f32; 3],
        ref_len: i32,
        col_ext: f32,
        trap_pow: f32,
        trap_x: f32,
        trap_y: f32,
        trap_c1: f32,
        trap_s1: f32,
        wave_speed: f32,
        fold_intensity: f32,
        light_az: f32,
        light_el: f32,
        spec_str: f32,
        rim_str: f32,
        ao_str: f32,
        relief: f32,
        ridge_amp: f32,
        ridge_freq: f32,
        plateau: f32,
        shadow_soft: f32,
        shadow_len: f32,
        bounce_str: f32,
        roughness: f32,
        metallic: f32,
        reflection: f32,
        interior: f32,
    }
}

const MAX_ITER: usize = 4000;
// the opening view: centre in the complex plane and the size of one pixel at 600 px height
const START_CENTER: [f64; 2] = [-0.1847195, -0.6499969];
const START_SCALE: f64 = 0.0004 * 2.033 / 600.0;
// f64 reference orbit: deeper than this the view turns blocky
const MIN_SCALE: f64 = 1e-14;

impl Default for ShaderParams {
    fn default() -> Self {
        Self {
            base_color: [0.0, 0.5, 1.0],
            pixel_size: START_SCALE as f32,
            palette_a: [0.0, 0.5, 1.0],
            gamma_correction: 0.6,
            palette_b: [0.018, 0.018, 0.018],
            iteration: 355,
            light_color: [1.0, 0.5, 0.0],
            aa: 1,
            rim_color: [0.8, 0.9, 1.0],
            ref_len: 2,
            col_ext: 2.0,
            trap_pow: 1.0,
            trap_x: -0.5,
            trap_y: 2.0,
            trap_c1: 0.2,
            trap_s1: 0.8,
            wave_speed: 0.1,
            fold_intensity: 1.0,
            light_az: 220.0,
            light_el: 35.0,
            spec_str: 8.0,
            rim_str: 0.3,
            ao_str: 0.36,
            relief: 0.1,
            ridge_amp: 0.5,
            ridge_freq: 1.8,
            plateau: 5.0,
            shadow_soft: 8.0,
            shadow_len: 60.0,
            bounce_str: 1.3,
            roughness: 0.2,
            metallic: 0.0,
            reflection: 1.47,
            interior: 2.0,
        }
    }
}

struct Shader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: ShaderParams,
    // view in f64: centre and complex size of one pixel
    center: [f64; 2],
    scale: f64,
    // slow drift in pixels per second, opt in
    drift: f32,
    ref_dirty: bool,
    dragging: bool,
    last_mouse: [f32; 2],
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("GGXbrot", 800, 600);
    app.run(event_loop, Shader::init)
}

fn slider(ui: &mut egui::Ui, v: &mut f32, range: RangeInclusive<f32>, label: &str) -> bool {
    ui.add(egui::Slider::new(v, range).text(label)).changed()
}

fn colour(ui: &mut egui::Ui, v: &mut [f32; 3], label: &str) -> bool {
    ui.horizontal(|ui| {
        let c = ui.color_edit_button_rgb(v).changed();
        ui.label(label);
        c
    })
    .inner
}

impl Shader {
    /// Reference orbit at the view centre in f64, stored as f32 pairs for the perturbation
    fn upload_reference(&mut self, core: &Core) {
        let (cx, cy) = (self.center[0], self.center[1]);
        let (mut x, mut y) = (0.0f64, 0.0f64);
        let mut data = vec![0.0f32; 2];
        let iters = (self.current_params.iteration as usize).min(MAX_ITER);
        for _ in 0..iters {
            let nx = x * x - y * y + cx;
            y = 2.0 * x * y + cy;
            x = nx;
            data.push(x as f32);
            data.push(y as f32);
            if x * x + y * y > 1e8 {
                break;
            }
        }
        self.current_params.ref_len = (data.len() / 2) as i32;
        if let Some(buf) = self.compute_shader.get_audio_buffer() {
            core.queue.write_buffer(buf, 0, bytemuck::cast_slice(&data));
        }
        self.ref_dirty = false;
    }

    /// Complex offset of a window position from the centre (y up)
    fn offset(&self, core: &Core, m: [f32; 2]) -> [f64; 2] {
        let (w, h) = (core.size.width as f64, core.size.height as f64);
        [(m[0] as f64 * w - 0.5 * w) * self.scale, (0.5 * h - m[1] as f64 * h) * self.scale]
    }

    fn zoom_at(&mut self, core: &Core, m: [f32; 2], factor: f64) {
        let before = self.offset(core, m);
        self.scale = (self.scale * factor).clamp(MIN_SCALE, 0.01);
        let after = self.offset(core, m);
        self.center[0] += before[0] - after[0];
        self.center[1] += before[1] - after[1];
        self.ref_dirty = true;
    }
}

impl ShaderManager for Shader {
    fn init(core: &Core) -> Self {
        let base = RenderKit::new(core);

        let passes = vec![
            PassDescription::new("compute_fractal", &[]),
            PassDescription::new("main_image", &["compute_fractal"]),
        ];

        let config = ComputeShader::builder()
            .with_multi_pass(&passes)
            .with_custom_uniforms::<ShaderParams>()
            .with_mouse()
            // reference orbit (group 2 binding 1) and the per-pixel slope (binding 2)
            .with_audio((MAX_ITER + 2) * 2)
            .with_atomic_buffer(1)
            .with_label("Orbits 3D")
            .build();

        let compute_shader = cuneus::compute_shader!(core, "shaders/orbits.wgsl", config);
        let current_params = ShaderParams::default();
        compute_shader.set_custom_params(current_params, &core.queue);

        Self {
            base,
            compute_shader,
            current_params,
            center: START_CENTER,
            scale: START_SCALE,
            drift: 0.0,
            ref_dirty: true,
            dragging: false,
            last_mouse: [0.0; 2],
        }
    }

    fn update(&mut self, core: &Core) {
        self.compute_shader.handle_export(core, &mut self.base);
    }

    fn render(&mut self, core: &Core) -> Result<(), cuneus::SurfaceError> {
        let mut frame = self.base.begin_frame(core)?;

        let mut params = self.current_params;
        let mut changed = false;
        let mut reset_view = false;
        let mut zoom_exp = (START_SCALE / self.scale).log10() as f32;
        let zoom_before = zoom_exp;
        let mut drift = self.drift;
        let (cx, cy) = (self.center[0], self.center[1]);
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();

        let mut controls_request = self
            .base
            .controls
            .get_ui_request(&self.base.start_time, &core.size, self.base.fps_tracker.fps());

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                RenderKit::apply_default_style(ctx);
                egui::Window::new("GGXbrot")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(320.0)
                    .show(ctx, |ui| {
                        ui.label("Drag: move   wheel: zoom at cursor");
                        egui::CollapsingHeader::new("Surface").default_open(true).show(ui, |ui| {
                            changed |= slider(ui, &mut params.relief, 0.01..=1.0, "Relief");
                            changed |= slider(ui, &mut params.plateau, 0.0..=10.0, "Thickness");
                            changed |= slider(ui, &mut params.ridge_amp, 0.0..=1.0, "Ridge amplitude");
                            changed |= slider(ui, &mut params.ridge_freq, 0.1..=10.0, "Ridge frequency");
                            changed |= slider(ui, &mut params.interior, 0.0..=6.0, "Interior relief");
                        });
                        egui::CollapsingHeader::new("Light").default_open(true).show(ui, |ui| {
                            changed |= colour(ui, &mut params.light_color, "Key colour");
                            changed |= slider(ui, &mut params.light_az, 0.0..=360.0, "Azimuth");
                            changed |= slider(ui, &mut params.light_el, 5.0..=85.0, "Elevation");
                            changed |= slider(ui, &mut params.shadow_len, 0.0..=200.0, "Shadow reach (px)");
                            changed |= slider(ui, &mut params.shadow_soft, 1.0..=32.0, "Shadow hardness");
                            changed |= slider(ui, &mut params.ao_str, 0.0..=1.0, "Ambient occlusion");
                            changed |= slider(ui, &mut params.bounce_str, 0.0..=3.0, "Bounce into shadow");
                            changed |= slider(ui, &mut params.spec_str, 0.0..=8.0, "Specular");
                            changed |= slider(ui, &mut params.rim_str, 0.0..=3.0, "Back light");
                            changed |= colour(ui, &mut params.rim_color, "Back light colour");
                        });
                        egui::CollapsingHeader::new("Material").show(ui, |ui| {
                            changed |= slider(ui, &mut params.metallic, 0.0..=1.0, "Metallic");
                            changed |= slider(ui, &mut params.roughness, 0.04..=1.0, "Roughness");
                            changed |= slider(ui, &mut params.reflection, 0.0..=2.0, "Reflection");
                        });
                        egui::CollapsingHeader::new("Colour").show(ui, |ui| {
                            changed |= colour(ui, &mut params.base_color, "Base");
                            changed |= colour(ui, &mut params.palette_a, "Palette A");
                            changed |= colour(ui, &mut params.palette_b, "Palette B");
                            changed |= slider(ui, &mut params.col_ext, 0.0..=10.0, "Colour extension");
                            changed |= slider(ui, &mut params.gamma_correction, 0.1..=3.0, "Gamma");
                        });
                        egui::CollapsingHeader::new("Traps").show(ui, |ui| {
                            changed |= slider(ui, &mut params.trap_x, -5.0..=5.0, "Trap X");
                            changed |= slider(ui, &mut params.trap_y, -5.0..=5.0, "Trap Y");
                            changed |= slider(ui, &mut params.trap_pow, 0.0..=3.0, "Trap power");
                            changed |= slider(ui, &mut params.trap_c1, 0.0..=1.0, "Trap mix");
                            changed |= slider(ui, &mut params.trap_s1, 0.0..=2.0, "Trap blend");
                            changed |= slider(ui, &mut params.fold_intensity, 0.0..=3.0, "Fold");
                            changed |= slider(ui, &mut params.wave_speed, 0.0..=2.0, "Wave speed");
                        });
                        egui::CollapsingHeader::new("View").show(ui, |ui| {
                            ui.add(egui::Slider::new(&mut zoom_exp, -3.0..=9.5).text("Zoom (log10)"));
                            ui.label(format!("Centre {cx:.12}, {cy:.12}"));
                            ui.add(egui::Slider::new(&mut drift, 0.0..=120.0).text("Drift (px/s)"));
                            changed |= ui.add(egui::Slider::new(&mut params.iteration, 50..=MAX_ITER as i32).text("Iterations")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.aa, 1..=4).text("Anti-aliasing")).changed();
                            if ui.button("Reset view").clicked() {
                                reset_view = true;
                            }
                        });

                        ui.separator();
                        ShaderControls::render_controls_widget(ui, &mut controls_request);
                        ui.separator();
                        should_start_export = ExportManager::render_export_ui_widget(ui, &mut export_request);
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        self.base.export_manager.apply_ui_request(export_request);
        self.base.apply_control_request(controls_request);

        if changed {
            if params.iteration != self.current_params.iteration {
                self.ref_dirty = true;
            }
            params.pixel_size = self.current_params.pixel_size;
            params.ref_len = self.current_params.ref_len;
            self.current_params = params;
        }
        if zoom_exp != zoom_before {
            let target = START_SCALE / 10f64.powf(zoom_exp as f64);
            self.zoom_at(core, [0.5, 0.5], target / self.scale);
        }
        if reset_view {
            self.center = START_CENTER;
            self.scale = START_SCALE;
            self.ref_dirty = true;
        }
        self.drift = drift;
        if self.drift > 0.0 {
            // a slowly turning heading, so a long drift wanders instead of running straight
            let t = self.base.controls.get_time(&self.base.start_time) as f64;
            let step = self.drift as f64 / 60.0 * self.scale;
            self.center[0] += step * (0.05 * t).cos();
            self.center[1] += step * (0.05 * t).sin();
            self.ref_dirty = true;
        }
        if self.ref_dirty {
            self.upload_reference(core);
        }
        self.current_params.pixel_size = self.scale as f32;
        self.compute_shader.set_custom_params(self.current_params, &core.queue);

        if should_start_export {
            self.base.export_manager.start_export();
        }

        let current_time = self.base.controls.get_time(&self.base.start_time);
        self.compute_shader.set_time(current_time, 1.0 / 60.0, &core.queue);
        self.compute_shader.update_mouse_uniform(&self.base.mouse_tracker.uniform, &core.queue);

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
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                self.dragging = *state == ElementState::Pressed;
                self.last_mouse = self.base.mouse_tracker.uniform.position;
                self.base.handle_mouse_input(core, event, false);
                true
            }
            WindowEvent::CursorMoved { .. } => {
                let handled = self.base.handle_mouse_input(core, event, false);
                if self.dragging {
                    let m = self.base.mouse_tracker.uniform.position;
                    let (w, h) = (core.size.width as f64, core.size.height as f64);
                    self.center[0] -= (m[0] - self.last_mouse[0]) as f64 * w * self.scale;
                    self.center[1] += (m[1] - self.last_mouse[1]) as f64 * h * self.scale;
                    self.last_mouse = m;
                    self.ref_dirty = true;
                }
                handled
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let steps = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y as f64,
                    MouseScrollDelta::PixelDelta(p) => p.y / 40.0,
                };
                if steps != 0.0 {
                    let m = self.base.mouse_tracker.uniform.position;
                    self.zoom_at(core, m, 0.9f64.powf(steps));
                }
                self.base.handle_mouse_input(core, event, false)
            }
            _ => self.base.handle_mouse_input(core, event, false),
        }
    }
}

// Enes Altun, 2026;
// This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 Unported License.

use cuneus::compute::*;
use cuneus::prelude::*;
use winit::event::WindowEvent;

cuneus::uniform_params! {
    struct BlackHoleParams {
        disk_inner: f32, disk_outer: f32, disk_thickness: f32, disk_brightness: f32,
        disk_density: f32, noise_scale: f32, swirl_speed: f32, temperature: f32,
        doppler: f32, redshift: f32, beaming: f32, ring_glow: f32,
        pan_x: f32, pan_y: f32, pan_z: f32, elev: f32,
        azim: f32, cam_roll: f32, fov: f32, taa_weight: f32,
        exposure: f32, bloom: f32, star_density: f32, gamma: f32,
        spectral_shift: f32, saturation: f32, reddening: f32, sharpen: f32,
        vividness: f32, opacity: f32, highlight: f32, spectral: f32,
        relief: f32, layering: f32, flare: f32, irradiation: f32,
        aperture: f32, focus: f32, focus_range: f32, bokeh_radius: f32,
        bokeh: f32, bloom_threshold: f32, self_shadow: f32, dist: f32,
        cinematic: f32, prev_azim: f32, prev_elev: f32, prev_dist: f32,
    }
}

// baked disc noise volumes (radius x spiral angle x height)
const DISC_VOLUME: [u32; 3] = [256, 128, 64];

struct BlackHoleShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: BlackHoleParams,
    dragging: bool,
    last_cursor: Option<(f64, f64)>,
    // last frame's orbit + time (TAA)
    prev_orbit: (f32, f32, f32, f32),
}

impl ShaderManager for BlackHoleShader {
    fn init(core: &Core) -> Self {
        let initial_params = BlackHoleParams {
            disk_inner: 3.0, disk_outer: 14.0, disk_thickness: 0.35, disk_brightness: 1.0,
            disk_density: 1.8, noise_scale: 1.0, swirl_speed: 0.25, temperature: 2.5,
            doppler: 0.75, redshift: 2.5, beaming: 4.0, ring_glow: 3.0,
            pan_x: 0.0, pan_y: 0.0, pan_z: 0.0, elev: 0.0886,
            azim: 0.0, cam_roll: 0.0, fov: 80.0, taa_weight: 0.8,
            exposure: 1.0, bloom: 2.0, star_density: 1.0, gamma: 1.2,
            spectral_shift: 0.0, saturation: 1.2, reddening: 0.6, sharpen: 0.7,
            vividness: 0.35, opacity: 0.4, highlight: 0.08, spectral: 0.0,
            relief: 0.6, layering: 0.55, flare: 0.5, irradiation: 0.7,
            aperture: 0.0, focus: 1.0, focus_range: 0.0, bokeh_radius: 2.2,
            bokeh: 1.0, bloom_threshold: 0.6, self_shadow: 0.7, dist: 18.07,
            cinematic: 0.0, prev_azim: 0.0, prev_elev: 0.0886, prev_dist: 18.07,
        };

        let base = RenderKit::new(core);

        let [vw, vh, vd] = DISC_VOLUME;
        let passes = vec![
            // disc noise bake (run_once)
            PassDescription::new("bake_a", &[]).with_volume(vw, vh, vd).run_once(),
            PassDescription::new("bake_b", &[]).with_volume(vw, vh, vd).run_once(),
            PassDescription::new("bb_lut", &[]),
            PassDescription::new("scene", &["bb_lut"]),
            PassDescription::new("dof", &["scene"]),
            PassDescription::new("resolve", &["dof", "resolve"]),
            PassDescription::new("bloom_pre", &["resolve"]).with_resolution_scale(0.5),
            PassDescription::new("bd2", &["bloom_pre"]).with_resolution_scale(0.25),
            PassDescription::new("bd3", &["bd2"]).with_resolution_scale(0.125),
            PassDescription::new("bd4", &["bd3"]).with_resolution_scale(0.0625),
            PassDescription::new("bd5", &["bd4"]).with_resolution_scale(0.03125),
            PassDescription::new("bu4", &["bd5", "bd4"]).with_resolution_scale(0.0625),
            PassDescription::new("bu3", &["bu4", "bd3"]).with_resolution_scale(0.125),
            PassDescription::new("bu2", &["bu3", "bd2"]).with_resolution_scale(0.25),
            PassDescription::new("bu1", &["bu2", "bloom_pre"]).with_resolution_scale(0.5),
            PassDescription::new("main_image", &["resolve", "bu1"]),
        ];

        let config = ComputeShader::builder()
            .with_entry_point("bake_a")
            .with_multi_pass(&passes)
            .with_custom_uniforms::<BlackHoleParams>()
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Schwarzschild Black Hole")
            .build();

        let compute_shader = cuneus::compute_shader!(core, "shaders/blackhole.wgsl", config);
        compute_shader.set_custom_params(initial_params, &core.queue);

        Self {
            base,
            compute_shader,
            current_params: initial_params,
            dragging: false,
            last_cursor: None,
            prev_orbit: (0.0, 0.0886, 18.07, 0.0),
        }
    }

    fn update(&mut self, core: &Core) {
        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = (current_time - self.prev_orbit.3).clamp(0.0, 0.25);
        self.compute_shader.set_time(current_time, delta, &core.queue);
        // last frame's camera (TAA)
        let p = &mut self.current_params;
        (p.prev_azim, p.prev_elev, p.prev_dist) = (self.prev_orbit.0, self.prev_orbit.1, self.prev_orbit.2);
        self.compute_shader.set_custom_params(*p, &core.queue);
        self.prev_orbit = (p.azim, p.elev, p.dist, current_time);
        self.compute_shader.handle_export(core, &mut self.base);
    }

    fn resize(&mut self, core: &Core) {
        self.base.default_resize(core, &mut self.compute_shader);
    }

    fn render(&mut self, core: &Core) -> Result<(), cuneus::SurfaceError> {
        let mut frame = self.base.begin_frame(core)?;
        let mut params = self.current_params;
        let mut changed = false;
        // camera sliders don't reset TAA or rebake
        let mut camera = false;
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self.base.controls.get_ui_request(
            &self.base.start_time,
            &core.size,
            self.base.fps_tracker.fps(),
        );

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                RenderKit::apply_default_style(ctx);
                egui::Window::new("Schwarzschild")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(330.0)
                    .show(ctx, |ui| {
                        egui::ScrollArea::vertical().show(ui, |ui| {
                        egui::CollapsingHeader::new("Camera").default_open(false).show(ui, |ui| {
                                ui.label("Drag to orbit, wheel to fly in and out");
                                camera |= ui.add(egui::Slider::new(&mut params.dist, 2.5..=60.0).logarithmic(true).text("Distance")).changed();
                                camera |= ui.add(egui::Slider::new(&mut params.pan_x, -15.0..=15.0).text("Camera X (left / right)")).changed();
                                camera |= ui.add(egui::Slider::new(&mut params.pan_y, -15.0..=15.0).text("Camera Y (down / up)")).changed();
                                camera |= ui.add(egui::Slider::new(&mut params.pan_z, -15.0..=15.0).text("Camera Z (back / forward)")).changed();
                                camera |= ui.add(egui::Slider::new(&mut params.cinematic, 0.0..=2.0).text("Cinematic camera")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.cam_roll, -3.14..=3.14).text("Roll")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.fov, 15.0..=90.0).text("FOV")).changed();
                            });

                        egui::CollapsingHeader::new("Disk").default_open(false).show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.disk_inner, 1.5..=8.0).text("Inner")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.disk_outer, 5.0..=40.0).text("Outer")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.disk_thickness, 0.05..=2.0).text("Thickness")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.disk_brightness, 0.1..=5.0).text("Brightness")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.disk_density, 0.2..=4.0).text("Density")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.opacity, 0.0..=1.0).text("Opacity")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.noise_scale, 0.2..=3.0).text("Detail")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.swirl_speed, 0.0..=1.5).text("Swirl")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.ring_glow, 0.0..=3.0).text("Ring glow")).changed();
                            });

                        egui::CollapsingHeader::new("Volume").default_open(false).show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.relief, 0.0..=1.0).text("Relief")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.layering, 0.0..=1.0).text("Layering")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.flare, 0.0..=1.5).text("Flare")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.irradiation, 0.0..=1.5).text("Irradiation")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.self_shadow, 0.0..=1.0).text("Self-shadow")).changed();
                            });

                        egui::CollapsingHeader::new("Color").default_open(false).show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.temperature, 0.3..=3.5).text("Temperature")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.spectral, 0.0..=1.0).text("Spectral")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.spectral_shift, -120.0..=120.0).text("Hue shift")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.vividness, 0.0..=1.0).text("Vividness")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.saturation, 0.0..=3.0).text("Saturation")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.reddening, 0.0..=1.0).text("Reddening")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.highlight, 0.02..=0.25).text("Highlight rolloff")).changed();
                            });

                        egui::CollapsingHeader::new("Relativity").default_open(false).show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.doppler, 0.0..=3.0).text("Doppler")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.redshift, 0.0..=3.0).text("Redshift color")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.beaming, 0.0..=5.0).text("Beaming")).changed();
                            });

                        egui::CollapsingHeader::new("Lens").default_open(false).show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.aperture, 0.0..=3.0).text("Aperture (DoF)")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.focus, 0.1..=4.0).logarithmic(true).text("Focus (1 = hole)")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.focus_range, 0.0..=20.0).text("Focus range (rs)")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.bokeh_radius, 0.3..=6.0).text("Max blur (% height)")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.bokeh, 0.0..=4.0).text("Bokeh highlights")).changed();
                            });

                        egui::CollapsingHeader::new("Post").default_open(false).show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.exposure, 0.1..=3.0).text("Exposure")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.bloom, 0.0..=2.0).text("Bloom")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.bloom_threshold, 0.0..=2.0).text("Bloom threshold")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.gamma, 0.6..=2.2).text("Gamma")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.sharpen, 0.0..=2.0).text("Sharpen")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.taa_weight, 0.0..=0.95).text("TAA")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.star_density, 0.0..=2.0).text("Background")).changed();
                            });

                        ui.separator();
                        ShaderControls::render_controls_widget(ui, &mut controls_request);
                        ui.separator();
                        should_start_export = ExportManager::render_export_ui_widget(ui, &mut export_request);
                        });
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        self.base.export_manager.apply_ui_request(export_request);

        // if slider moved? -> so lets wipe TAA buffers so it doesn't smear
        if changed || controls_request.should_clear_buffers {
            self.current_params = params;
            self.restart(core);
            // rebake on disk changes
            self.compute_shader.clear_all_buffers(core);
        } else if camera {
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
        }

        self.base.apply_control_request(controls_request);

        if should_start_export {
            self.base.export_manager.start_export();
        }

        if !self.base.export_manager.is_exporting() {
            self.compute_shader.dispatch(&mut frame.encoder, core);
        }

        self.base.renderer.render_to_view(
            &mut frame.encoder,
            &frame.view,
            &self.compute_shader.get_output_texture().bind_group,
        );
        self.base.end_frame(core, frame, full_output);

        Ok(())
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        if self.base.default_handle_input(core, event) {
            return true;
        }
        match event {
            WindowEvent::MouseInput { state, button: winit::event::MouseButton::Left, .. } => {
                self.dragging = *state == winit::event::ElementState::Pressed;
                self.last_cursor = None;
                self.dragging
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let steps = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => *y,
                    winit::event::MouseScrollDelta::PixelDelta(d) => d.y as f32 / 40.0,
                };
                let p = &mut self.current_params;
                p.dist = (p.dist * 1.1f32.powf(-steps)).clamp(2.5, 60.0);
                self.compute_shader.set_custom_params(*p, &core.queue);
                true
            }
            WindowEvent::CursorMoved { position, .. } if self.dragging => {
                if let Some((x, y)) = self.last_cursor {
                    let p = &mut self.current_params;
                    p.azim = (p.azim - (position.x - x) as f32 * 0.006).rem_euclid(std::f32::consts::TAU);
                    p.elev = (p.elev + (position.y - y) as f32 * 0.005).clamp(-1.5, 1.5);
                    self.compute_shader.set_custom_params(*p, &core.queue);
                }
                self.last_cursor = Some((position.x, position.y));
                true
            }
            _ => false,
        }
    }
}

impl BlackHoleShader {
    fn restart(&mut self, core: &Core) {
        self.compute_shader.current_frame = 0;
        self.compute_shader.time_uniform.data.frame = 0;
        self.compute_shader.time_uniform.update(&core.queue);
        self.compute_shader.set_custom_params(self.current_params, &core.queue);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = cuneus::ShaderApp::new("Schwarzschild Black Hole", 800, 500);
    app.run(event_loop, |core| BlackHoleShader::init(core))
}

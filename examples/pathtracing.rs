use cuneus::compute::*;
use cuneus::prelude::*;
use log::error;
use winit::event::{ElementState, MouseButton};
use winit::keyboard::Key;

struct CameraMovement {
    fwd: bool, back: bool, left: bool, right: bool, up: bool, down: bool,
    speed: f32, last_update: std::time::Instant,
    yaw: f32, pitch: f32, sensitivity: f32,
    last_mouse: (f32, f32), mouse_init: bool, mouse_look: bool,
}

impl CameraMovement {
    fn new(pos: [f32; 3], target: [f32; 3]) -> Self {
        let d = [target[0] - pos[0], target[1] - pos[1], target[2] - pos[2]];
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-6);
        Self {
            fwd: false, back: false, left: false, right: false, up: false, down: false,
            speed: 2.0, last_update: std::time::Instant::now(),
            yaw: d[2].atan2(d[0]), pitch: (d[1] / len).asin(), sensitivity: 0.005,
            last_mouse: (0.0, 0.0), mouse_init: false, mouse_look: false,
        }
    }

    fn update(&mut self, p: &mut PathTracingParams) {
        let now = std::time::Instant::now();
        let dt = now.duration_since(self.last_update).as_secs_f32();
        self.last_update = now;
        let f = [self.pitch.cos() * self.yaw.cos(), self.pitch.sin(), self.pitch.cos() * self.yaw.sin()];
        let r = { let x = -f[2]; let z = f[0]; let l = (x * x + z * z).sqrt().max(1e-6); [x / l, 0.0, z / l] };
        let s = self.speed * dt;
        let axis = |a: bool, b: bool| (a as i32 - b as i32) as f32 * s;
        let (mf, mr, mu) = (axis(self.fwd, self.back), axis(self.right, self.left), axis(self.up, self.down));
        p.cam_x += f[0] * mf + r[0] * mr;
        p.cam_y += f[1] * mf + mu;
        p.cam_z += f[2] * mf + r[2] * mr;
        p.tgt_x = p.cam_x + f[0]; p.tgt_y = p.cam_y + f[1]; p.tgt_z = p.cam_z + f[2];
    }

    fn mouse(&mut self, x: f32, y: f32) {
        if !self.mouse_look { return; }
        if !self.mouse_init { self.last_mouse = (x, y); self.mouse_init = true; return; }
        self.yaw += (x - self.last_mouse.0) * self.sensitivity;
        self.pitch = (self.pitch - (y - self.last_mouse.1) * self.sensitivity).clamp(-1.54, 1.54);
        self.last_mouse = (x, y);
    }
}

cuneus::uniform_params! {
    struct PathTracingParams {
        cam_x: f32, cam_y: f32, cam_z: f32, fov: f32,
        tgt_x: f32, tgt_y: f32, tgt_z: f32, aperture: f32,
        pcam_x: f32, pcam_y: f32, pcam_z: f32, focus_dist: f32,
        ptgt_x: f32, ptgt_y: f32, ptgt_z: f32, exposure: f32,
        max_bounces: u32, diffuse_bounces: u32, accumulate: u32, cam_moved: u32,
        num_lights: u32, ris_candidates: u32, restir: u32, spatial_count: u32,
        spatial_radius: f32, c_cap: f32, hist_realtime: f32, hist_move: f32,
        denoise: u32, atrous_iters: u32, sigma_l: f32, sigma_n: f32,
        sigma_z: f32, firefly: f32, dispersion: f32, rotation_speed: f32,
        bloom: f32, use_hdri: u32, sky_strength: f32, open_roof: u32,
        regularize: f32, clip_gamma: f32, debug_view: u32, side_mode: u32,
        gamma: f32, obj_speed: f32, _p3: f32, _p4: f32,
    }
}

struct PathTracingShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    params: PathTracingParams,
    cam: CameraMovement,
    last_cam: [f32; 6],
    frame: u32,
    reset: bool,
    last_t: f32,
}

impl PathTracingShader {
    fn cam_now(&self) -> [f32; 6] {
        let p = &self.params;
        [p.cam_x, p.cam_y, p.cam_z, p.tgt_x, p.tgt_y, p.tgt_z]
    }
}

fn toggle(ui: &mut egui::Ui, v: &mut u32, label: &str) -> bool {
    let mut b = *v > 0;
    let c = ui.checkbox(&mut b, label).changed();
    *v = b as u32;
    c
}

impl ShaderManager for PathTracingShader {
    fn init(core: &Core) -> Self {
        let base = RenderKit::new(core);
        let params = PathTracingParams {
            cam_x: 0.0, cam_y: 1.0, cam_z: 6.0, fov: 40.0,
            tgt_x: 0.0, tgt_y: 0.0, tgt_z: -1.0, aperture: 0.0,
            pcam_x: 0.0, pcam_y: 1.0, pcam_z: 6.0, focus_dist: 6.0,
            ptgt_x: 0.0, ptgt_y: 0.0, ptgt_z: -1.0, exposure: 1.1,
            max_bounces: 6, diffuse_bounces: 3, accumulate: 1, cam_moved: 0,
            num_lights: 3, ris_candidates: 16, restir: 1, spatial_count: 3,
            spatial_radius: 16.0, c_cap: 20.0, hist_realtime: 24.0, hist_move: 16.0,
            denoise: 1, atrous_iters: 4, sigma_l: 4.0, sigma_n: 64.0,
            sigma_z: 1.0, firefly: 10.0, dispersion: 0.06, rotation_speed: 1.0,
            bloom: 0.08, use_hdri: 0, sky_strength: 1.0, open_roof: 1,
            regularize: 0.3, clip_gamma: 1.5, debug_view: 0, side_mode: 1,
            gamma: 1.0, obj_speed: 1.0, _p3: 0.0, _p4: 0.0,
        };

        let hist = ["trace", "hprev", "lprev", "gbuf", "gprev"];
        let passes = vec![
            PassDescription::new("gprev", &["gbuf"]),
            PassDescription::new("gbuf", &[]),
            PassDescription::new("galb", &[]),
            PassDescription::new("ris", &[]),
            PassDescription::new("rtemp", &["ris", "rspat", "gbuf", "gprev"]),
            PassDescription::new("rspat", &["rtemp", "gbuf"]),
            PassDescription::new("trace", &["rspat"]),
            PassDescription::new("hprev", &["accum"]),
            PassDescription::new("lprev", &["accum_lo"]),
            PassDescription::new("accum", &hist),
            PassDescription::new("accum_lo", &hist),
            PassDescription::new("moments", &["trace", "moments", "gbuf", "gprev"]),
            PassDescription::new("svar", &["accum", "accum_lo", "moments", "gbuf"]),
            PassDescription::new("a1", &["svar", "gbuf", "galb"]),
            PassDescription::new("a2", &["a1", "gbuf", "galb"]),
            PassDescription::new("a3", &["a2", "gbuf", "galb"]),
            PassDescription::new("a4", &["a3", "gbuf", "galb"]),
            PassDescription::new("flr", &["svar", "gbuf", "galb"]),
            PassDescription::new("comp", &["a4", "flr"]),
            PassDescription::new("bloom_pre", &["comp"]).with_resolution_scale(0.5),
            PassDescription::new("bd2", &["bloom_pre"]).with_resolution_scale(0.25),
            PassDescription::new("bd3", &["bd2"]).with_resolution_scale(0.125),
            PassDescription::new("bd4", &["bd3"]).with_resolution_scale(0.0625),
            PassDescription::new("bd5", &["bd4"]).with_resolution_scale(0.03125),
            PassDescription::new("bu4", &["bd5", "bd4"]).with_resolution_scale(0.0625),
            PassDescription::new("bu3", &["bu4", "bd3"]).with_resolution_scale(0.125),
            PassDescription::new("bu2", &["bu3", "bd2"]).with_resolution_scale(0.25),
            PassDescription::new("bu1", &["bu2", "bloom_pre"]).with_resolution_scale(0.5),
            PassDescription::new("main_image", &["comp", "bu1", "accum"]),
        ];

        let config = ComputeShader::builder()
            .with_multi_pass(&passes)
            .with_custom_uniforms::<PathTracingParams>()
            .with_mouse()
            .with_channels(1)
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Path Tracer")
            .build();

        let compute_shader = cuneus::compute_shader!(core, "shaders/pathtracing.wgsl", config);
        compute_shader.set_custom_params(params, &core.queue);

        let cam = CameraMovement::new([params.cam_x, params.cam_y, params.cam_z], [params.tgt_x, params.tgt_y, params.tgt_z]);
        let mut s = Self { base, compute_shader, params, cam, last_cam: [0.0; 6], frame: 0, reset: true, last_t: 0.0 };
        s.cam.update(&mut s.params);
        s.last_cam = s.cam_now();
        s
    }

    fn update(&mut self, core: &Core) {
        let t = self.base.controls.get_time(&self.base.start_time);
        let dt = (t - self.last_t).clamp(0.0, 0.25);
        self.last_t = t;
        self.compute_shader.set_time(t, dt, &core.queue);
        self.base.update_current_texture(core, &core.queue);
        if let Some(tm) = self.base.get_current_texture_manager() {
            self.compute_shader.update_channel_texture(0, &tm.view, &tm.sampler, &core.device, &core.queue);
            self.params.use_hdri = 1;
        } else {
            self.params.use_hdri = 0;
        }
        self.cam.update(&mut self.params);
        self.compute_shader.handle_export(core, &mut self.base);
    }

    fn resize(&mut self, core: &Core) {
        self.base.default_resize(core, &mut self.compute_shader);
        self.reset = true;
    }

    fn render(&mut self, core: &Core) -> Result<(), cuneus::SurfaceError> {
        let mut frame = self.base.begin_frame(core)?;
        let mut p = self.params;
        let mut rs = false;
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self.base.controls.get_ui_request(&self.base.start_time, &core.size, self.base.fps_tracker.fps());
        let using_hdri = self.base.using_hdri_texture || p.use_hdri == 1;
        let hdri_info = self.base.get_hdri_info();
        let samples = self.frame;
        let moving = self.cam_now() != self.last_cam;
        let mouse_look = self.cam.mouse_look;

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                RenderKit::apply_default_style(ctx);
                egui::Window::new("Path Tracer").collapsible(true).resizable(true).default_width(300.0).show(ctx, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        egui::CollapsingHeader::new("Camera").default_open(false).show(ui, |ui| {
                            ui.label("WASD/QE move, right-click toggles mouse look");
                            rs |= ui.add(egui::Slider::new(&mut p.fov, 15.0..=90.0).text("FOV")).changed();
                            rs |= ui.add(egui::Slider::new(&mut p.aperture, 0.0..=0.5).text("Aperture")).changed();
                            rs |= ui.add(egui::Slider::new(&mut p.focus_dist, 0.5..=20.0).text("Focus distance")).changed();
                        });
                        egui::CollapsingHeader::new("Scene").default_open(true).show(ui, |ui| {
                            rs |= ui.add(egui::Slider::new(&mut p.num_lights, 1..=32).text("Lights")).changed();
                            let mut anim = p.accumulate == 0;
                            if ui.checkbox(&mut anim, "Animate (off = still image converges)").changed() { p.accumulate = (!anim) as u32; rs = true; }
                            ui.add(egui::Slider::new(&mut p.rotation_speed, 0.0..=3.0).text("Light spin"));
                            ui.add(egui::Slider::new(&mut p.obj_speed, 0.0..=3.0).text("Object speed"));
                            ui.horizontal(|ui| {
                                ui.label("Side walls");
                                rs |= ui.radio_value(&mut p.side_mode, 0, "Walls").changed();
                                rs |= ui.radio_value(&mut p.side_mode, 1, "Mirrors").changed();
                                rs |= ui.radio_value(&mut p.side_mode, 2, "Open").changed();
                            });
                            rs |= ui.add(egui::Slider::new(&mut p.dispersion, 0.0..=0.2).text("Prism dispersion")).changed();
                            rs |= toggle(ui, &mut p.open_roof, "Open roof (sky)");
                            rs |= ui.add(egui::Slider::new(&mut p.sky_strength, 0.0..=5.0).text("Sky / HDRI strength")).changed();
                            ShaderControls::render_media_panel(ui, &mut controls_request, false, None, using_hdri, hdri_info, false, None);
                        });
                        egui::CollapsingHeader::new("Sampling").default_open(false).show(ui, |ui| {
                            rs |= ui.add(egui::Slider::new(&mut p.max_bounces, 1..=16).text("Max bounces")).changed();
                            rs |= ui.add(egui::Slider::new(&mut p.diffuse_bounces, 1..=8).text("Diffuse bounces")).changed();
                            rs |= ui.add(egui::Slider::new(&mut p.firefly, 0.0..=100.0).text("Firefly clamp (0 = off)")).changed();
                            rs |= ui.add(egui::Slider::new(&mut p.regularize, 0.0..=0.6).text("Caustic regularization (0 = off)")).changed();
                            ui.add(egui::Slider::new(&mut p.hist_realtime, 2.0..=64.0).text("Realtime history"));
                            ui.add(egui::Slider::new(&mut p.hist_move, 2.0..=64.0).text("History while moving"));
                            ui.add(egui::Slider::new(&mut p.clip_gamma, 0.5..=4.0).text("Motion clip (γ)"));
                        });
                        egui::CollapsingHeader::new("ReSTIR DI").default_open(false).show(ui, |ui| {
                            rs |= toggle(ui, &mut p.restir, "Enable");
                            rs |= ui.add(egui::Slider::new(&mut p.ris_candidates, 1..=32).text("Candidates (M)")).changed();
                            rs |= ui.add(egui::Slider::new(&mut p.spatial_count, 0..=8).text("Spatial neighbours (K)")).changed();
                            rs |= ui.add(egui::Slider::new(&mut p.spatial_radius, 1.0..=48.0).text("Spatial radius (px)")).changed();
                            rs |= ui.add(egui::Slider::new(&mut p.c_cap, 1.0..=30.0).text("Temporal cap")).changed();
                        });
                        egui::CollapsingHeader::new("Denoiser").default_open(false).show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.radio_value(&mut p.denoise, 0, "Off");
                                ui.radio_value(&mut p.denoise, 1, "SVGF");
                                ui.radio_value(&mut p.denoise, 2, "FLR");
                            });
                            if p.denoise == 1 {
                                ui.add(egui::Slider::new(&mut p.atrous_iters, 1..=4).text("Iterations"));
                                ui.add(egui::Slider::new(&mut p.sigma_l, 0.5..=16.0).text("Luminance"));
                                ui.add(egui::Slider::new(&mut p.sigma_n, 1.0..=256.0).text("Normal"));
                                ui.add(egui::Slider::new(&mut p.sigma_z, 0.1..=8.0).text("Depth"));
                            }
                        });
                        egui::CollapsingHeader::new("Post").default_open(false).show(ui, |ui| {
                            ui.add(egui::Slider::new(&mut p.exposure, 0.1..=5.0).text("Exposure"));
                            ui.add(egui::Slider::new(&mut p.gamma, 0.5..=3.0).text("Gamma"));
                            ui.add(egui::Slider::new(&mut p.bloom, 0.0..=0.4).text("Bloom"));
                            if ui.button("Reset accumulation").clicked() { rs = true; }
                        });
                        ui.separator();
                        ShaderControls::render_controls_widget(ui, &mut controls_request);
                        ui.separator();
                        should_start_export = ExportManager::render_export_ui_widget(ui, &mut export_request);
                        ui.separator();
                        ui.label(format!("Samples: {samples}"));
                        ui.label(format!("Camera: {}   Mouse look: {}", if moving { "MOVING" } else { "still" }, if mouse_look { "on" } else { "off" }));
                        toggle(ui, &mut p.debug_view, "Debug: history heatmap");
                    });
                });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        self.base.export_manager.apply_ui_request(export_request);
        self.params = p;
        if rs || controls_request.should_clear_buffers || self.reset {
            self.compute_shader.clear_all_buffers(core);
            self.frame = 0;
            self.reset = false;
        }
        self.base.apply_media_requests(core, &controls_request);
        if should_start_export { self.base.export_manager.start_export(); }

        // previous camera for reprojection; moving the camera no longer resets the history
        let now = self.cam_now();
        let prev = self.last_cam;
        self.params.pcam_x = prev[0]; self.params.pcam_y = prev[1]; self.params.pcam_z = prev[2];
        self.params.ptgt_x = prev[3]; self.params.ptgt_y = prev[4]; self.params.ptgt_z = prev[5];
        self.params.cam_moved = (now != prev) as u32;
        self.compute_shader.set_custom_params(self.params, &core.queue);
        self.compute_shader.update_mouse_uniform(&self.base.mouse_tracker.uniform, &core.queue);
        self.compute_shader.time_uniform.data.frame = self.frame;
        self.compute_shader.time_uniform.update(&core.queue);

        self.compute_shader.dispatch(&mut frame.encoder, core);
        self.base.renderer.render_to_view(&mut frame.encoder, &frame.view, &self.compute_shader.get_output_texture().bind_group);
        self.base.end_frame(core, frame, full_output);

        self.last_cam = now;
        self.frame = self.frame.wrapping_add(1);
        Ok(())
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        if self.base.forward_to_egui(core, event) { return true; }
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                if let Key::Character(ch) = &event.logical_key {
                    let down = event.state == ElementState::Pressed;
                    match ch.as_str() {
                        "w" | "W" => { self.cam.fwd = down; return true; }
                        "s" | "S" => { self.cam.back = down; return true; }
                        "a" | "A" => { self.cam.left = down; return true; }
                        "d" | "D" => { self.cam.right = down; return true; }
                        "q" | "Q" => { self.cam.down = down; return true; }
                        "e" | "E" => { self.cam.up = down; return true; }
                        " " if !down => { self.params.accumulate = 1 - self.params.accumulate; self.reset = true; return true; }
                        _ => {}
                    }
                }
                if self.base.key_handler.handle_keyboard_input(core.window(), event) { return true; }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.base.handle_mouse_input(core, event, false);
                self.cam.mouse(position.x as f32, position.y as f32);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if *button == MouseButton::Right && *state == ElementState::Released {
                    self.cam.mouse_look = !self.cam.mouse_look;
                    self.cam.mouse_init = false;
                    return true;
                }
            }
            WindowEvent::DroppedFile(path) => {
                if let Err(e) = self.base.load_media(core, path) { error!("Failed to load dropped file: {e:?}"); }
                return true;
            }
            _ => {}
        }
        false
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    cuneus::gst::init()?;
    let (app, event_loop) = ShaderApp::new("Path Tracer", 960, 600);
    app.run(event_loop, PathTracingShader::init)
}

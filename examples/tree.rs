use cuneus::compute::*;
use cuneus::prelude::*;
use cuneus::{OrbitCamera, Vec3};

cuneus::uniform_params! {
    struct Params {
        eye_x: f32, eye_y: f32, eye_z: f32, fov: f32,
        fwd_x: f32, fwd_y: f32, fwd_z: f32, aperture: f32,
        rt_x: f32, rt_y: f32, rt_z: f32, focus: f32,
        up_x: f32, up_y: f32, up_z: f32, exposure: f32,
        n: u32, view_n: u32, walkers: u32, iterations: u32,
        steps: u32, zoom: f32, pan_x: f32, pan_y: f32,
        power: f32, offset: f32, bend: f32, align: f32,
        light: f32, light_dir: f32, phase: f32, phase_off: f32,
        darken: f32, gamma: f32, jewel: f32, jsat: f32,
        shine: f32, bloom: f32, twinkle: f32, _q0: f32,
        amp: f32, gem_h: f32, _q1: f32, _q2: f32,
        metal: f32, sun_az: f32, sun_el: f32, sun_power: f32,
    }
}

struct Tree {
    base: RenderKit,
    compute_shader: ComputeShader,
    params: Params,
    camera: OrbitCamera,
    focus_offset: f32,
    last_pose: [f32; 6],
    last_t: f32,
}

impl Tree {
    fn pose_key(&self) -> [f32; 6] {
        let p = self.camera.pose;
        [p.yaw, p.pitch, p.distance, p.target.x, p.target.y, p.target.z]
    }
    fn write_camera(&mut self) {
        let pose = self.camera.pose;
        let (e, f, r, u) = (pose.eye(), pose.forward(), pose.right(), pose.up());
        let q = &mut self.params;
        (q.eye_x, q.eye_y, q.eye_z, q.fov) = (e.x, e.y, e.z, self.camera.fov);
        (q.fwd_x, q.fwd_y, q.fwd_z) = (f.x, f.y, f.z);
        (q.rt_x, q.rt_y, q.rt_z) = (r.x, r.y, r.z);
        (q.up_x, q.up_y, q.up_z) = (u.x, u.y, u.z);
        q.focus = (pose.distance + self.focus_offset).max(0.05);
    }
}

impl ShaderManager for Tree {
    fn init(core: &Core) -> Self {
        let base = RenderKit::new(core);
        let params = Params {
            eye_x: 0.0, eye_y: 0.0, eye_z: 0.0, fov: 40.0,
            fwd_x: 0.0, fwd_y: 0.0, fwd_z: -1.0, aperture: 0.015,
            rt_x: 1.0, rt_y: 0.0, rt_z: 0.0, focus: 3.0,
            up_x: 0.0, up_y: 1.0, up_z: 0.0, exposure: 1.0,
            n: 0, view_n: 0, walkers: 1, iterations: 200,
            steps: 100, zoom: 0.3, pan_x: 0.8, pan_y: 1.7,
            power: 1.5, offset: 0.2, bend: 30.0, align: 10.0,
            light: 0.4, light_dir: -90.0, phase: 6.0, phase_off: -2.0,
            darken: 0.0005, gamma: 0.4, jewel: 0.5, jsat: 0.35,
            shine: 0.2, bloom: 1.5, twinkle: 0.0, _q0: 0.0,
            amp: 0.12, gem_h: 0.03, _q1: 0.0, _q2: 0.0,
            metal: 0.6, sun_az: 30.0, sun_el: 45.0, sun_power: 2.0,
        };
        let passes = vec![
            PassDescription::new("relief", &[]),
            PassDescription::new("slope", &["relief"]),
            PassDescription::new("walk", &["walk", "slope"]),
            PassDescription::new("height", &["walk"]),
            PassDescription::new("view", &["walk", "height", "view"]),
            PassDescription::new("bloom_d1", &["view"]).with_resolution_scale(0.5),
            PassDescription::new("bloom_d2", &["bloom_d1"]).with_resolution_scale(0.25),
            PassDescription::new("bloom_d3", &["bloom_d2"]).with_resolution_scale(0.125),
            PassDescription::new("bloom_d4", &["bloom_d3"]).with_resolution_scale(0.0625),
            PassDescription::new("bloom_u3", &["bloom_d4", "bloom_d3"]).with_resolution_scale(0.125),
            PassDescription::new("bloom_u2", &["bloom_u3", "bloom_d2"]).with_resolution_scale(0.25),
            PassDescription::new("bloom_u1", &["bloom_u2", "bloom_d1"]).with_resolution_scale(0.5),
            PassDescription::new("main_image", &["view", "bloom_u1"]),
        ];
        let config = ComputeShader::builder()
            .with_entry_point("relief")
            .with_multi_pass(&passes)
            .with_custom_uniforms::<Params>()
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Tree")
            .build();
        let compute_shader = cuneus::compute_shader!(core, "shaders/tree.wgsl", config);

        let mut camera = OrbitCamera::new();
        camera.fov = 40.0;
        camera.frame(Vec3::ZERO, 1.5);
        camera.home.pitch = 1.0;
        (camera.goal, camera.pose) = (camera.home, camera.home);
        let mut s = Self { base, compute_shader, params, camera, focus_offset: 0.0, last_pose: [0.0; 6], last_t: 0.0 };
        s.write_camera();
        s.last_pose = s.pose_key();
        s
    }

    fn update(&mut self, core: &Core) {
        let t = self.base.controls.get_time(&self.base.start_time);
        let dt = (t - self.last_t).clamp(0.0, 0.1);
        self.last_t = t;
        self.camera.update(dt);
        self.compute_shader.handle_export(core, &mut self.base);
    }

    fn resize(&mut self, core: &Core) {
        self.base.default_resize(core, &mut self.compute_shader);
        (self.params.n, self.params.view_n) = (0, 0);
    }

    fn render(&mut self, core: &Core) -> Result<(), cuneus::SurfaceError> {
        let mut frame = self.base.begin_frame(core)?;
        let mut p = self.params;
        // walk changes restart everything; glass, jewel and camera changes only the view
        let mut rewalk = false;
        let mut review = false;
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self.base.controls.get_ui_request(&self.base.start_time, &core.size, self.base.fps_tracker.fps());
        let (mut fly, mut fov, mut turn, mut focus_offset) = (self.camera.fly, self.camera.fov, self.camera.auto_rotate, self.focus_offset);

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                RenderKit::apply_default_style(ctx);
                egui::Window::new("Tree").collapsible(true).resizable(true).default_width(300.0).show(ctx, |ui| {
                    ui.label("Drag to orbit, right-drag to pan, wheel to zoom, R to reset");
                    egui::CollapsingHeader::new("Tree").show(ui, |ui| {
                        rewalk |= ui.add(egui::Slider::new(&mut p.power, 1.2..=2.2).text("Exponent")).changed();
                        rewalk |= ui.add(egui::Slider::new(&mut p.offset, 0.0..=0.6).text("Offset")).changed();
                        rewalk |= ui.add(egui::Slider::new(&mut p.zoom, 0.01..=1.0).logarithmic(true).text("Zoom")).changed();
                        rewalk |= ui.add(egui::Slider::new(&mut p.pan_x, -1.0..=3.0).text("Pan X")).changed();
                        rewalk |= ui.add(egui::Slider::new(&mut p.pan_y, -1.0..=4.0).text("Pan Y")).changed();
                        rewalk |= ui.add(egui::Slider::new(&mut p.iterations, 20..=400).text("Iterations")).changed();
                    });
                    egui::CollapsingHeader::new("Light walk").show(ui, |ui| {
                        rewalk |= ui.add(egui::Slider::new(&mut p.bend, 0.0..=120.0).text("Bend")).changed();
                        rewalk |= ui.add(egui::Slider::new(&mut p.steps, 10..=300).text("Steps")).changed();
                        rewalk |= ui.add(egui::Slider::new(&mut p.light_dir, -180.0..=180.0).text("Light direction")).changed();
                        rewalk |= ui.add(egui::Slider::new(&mut p.align, 1.0..=40.0).text("Alignment")).changed();
                        rewalk |= ui.add(egui::Slider::new(&mut p.light, 0.05..=2.0).logarithmic(true).text("Brightness")).changed();
                        rewalk |= ui.add(egui::Slider::new(&mut p.phase, 0.0..=20.0).text("Colour spread")).changed();
                        rewalk |= ui.add(egui::Slider::new(&mut p.phase_off, -6.3..=6.3).text("Colour shift")).changed();
                        rewalk |= ui.add(egui::Slider::new(&mut p.darken, 0.0..=0.003).text("Depth darkening")).changed();
                        rewalk |= ui.add(egui::Slider::new(&mut p.walkers, 1..=8).text("Walkers per frame")).changed();
                    });
                    egui::CollapsingHeader::new("Jewels").default_open(true).show(ui, |ui| {
                        review |= ui.add(egui::Slider::new(&mut p.jewel, 0.0..=12.0).text("Glow")).changed();
                        review |= ui.add(egui::Slider::new(&mut p.jsat, 0.0..=0.9).text("Colour threshold")).changed();
                        review |= ui.add(egui::Slider::new(&mut p.shine, 0.0..=2.0).text("Silver sheen")).changed();
                        review |= ui.add(egui::Slider::new(&mut p.twinkle, 0.0..=1.0).text("Twinkle")).changed();
                    });
                    egui::CollapsingHeader::new("Relief").default_open(true).show(ui, |ui| {
                        review |= ui.add(egui::Slider::new(&mut p.amp, 0.0..=0.5).text("Height")).changed();
                        review |= ui.add(egui::Slider::new(&mut p.gem_h, 0.0..=0.2).text("Gem height")).changed();
                        review |= ui.add(egui::Slider::new(&mut p.metal, 0.0..=2.0).text("Metal")).changed();
                        review |= ui.add(egui::Slider::new(&mut p.sun_az, -180.0..=180.0).text("Sun angle")).changed();
                        review |= ui.add(egui::Slider::new(&mut p.sun_el, 5.0..=90.0).text("Sun height")).changed();
                        review |= ui.add(egui::Slider::new(&mut p.sun_power, 0.0..=10.0).text("Sun")).changed();
                    });
                    egui::CollapsingHeader::new("Camera").show(ui, |ui| {
                        ui.checkbox(&mut fly, "Fly");
                        ui.add(egui::Slider::new(&mut fov, 15.0..=90.0).text("FOV"));
                        ui.add(egui::Slider::new(&mut turn, -0.5..=0.5).text("Turntable"));
                        review |= ui.add(egui::Slider::new(&mut p.aperture, 0.0..=0.1).text("Aperture")).changed();
                        ui.add(egui::Slider::new(&mut focus_offset, -1.5..=1.5).text("Focus"));
                        review |= ui.add(egui::Slider::new(&mut p.exposure, 0.1..=4.0).logarithmic(true).text("Exposure")).changed();
                    });
                    egui::CollapsingHeader::new("Post").show(ui, |ui| {
                        ui.add(egui::Slider::new(&mut p.bloom, 0.0..=4.0).text("Bloom"));
                        ui.add(egui::Slider::new(&mut p.gamma, 0.2..=3.0).text("Gamma"));
                    });
                    ui.label(format!("{} frames", p.n));
                    ui.separator();
                    ShaderControls::render_controls_widget(ui, &mut controls_request);
                    ui.separator();
                    should_start_export = ExportManager::render_export_ui_widget(ui, &mut export_request);
                });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        if controls_request.should_clear_buffers {
            rewalk = true;
        }
        self.base.apply_control_request(controls_request);
        self.base.export_manager.apply_ui_request(export_request);
        if should_start_export {
            self.base.export_manager.start_export();
        }

        (self.camera.fly, self.camera.fov, self.camera.auto_rotate) = (fly, fov, turn);
        review |= fov != self.params.fov || focus_offset != self.focus_offset;
        self.focus_offset = focus_offset;
        self.params = p;
        self.write_camera();
        let pose = self.pose_key();
        review |= pose != self.last_pose;
        self.last_pose = pose;
        let q = &mut self.params;
        if rewalk {
            q.n = 0;
        }
        if rewalk || review {
            q.view_n = 0;
        }
        let t = self.base.controls.get_time(&self.base.start_time);
        self.compute_shader.set_time(t, 1.0 / 60.0, &core.queue);
        self.compute_shader.set_custom_params(self.params, &core.queue);
        self.compute_shader.dispatch(&mut frame.encoder, core);
        let q = &mut self.params;
        (q.n, q.view_n) = (q.n.saturating_add(1), q.view_n.saturating_add(1));

        self.base.renderer.render_to_view(&mut frame.encoder, &frame.view, &self.compute_shader.get_output_texture().bind_group);
        self.base.end_frame(core, frame, full_output);
        Ok(())
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        if self.base.default_handle_input(core, event) {
            return true;
        }
        self.camera.handle_event(event)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("Tree", 1280, 720);
    app.run(event_loop, Tree::init)
}

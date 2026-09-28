use cuneus::compute::{ComputeShader, ComputeShaderBuilder, StorageBufferSpec};
use cuneus::prelude::*;
use cuneus::{GaussianCamera, GaussianCloud, GaussianExporter, GaussianRenderer, GaussianSorter};
use log::{error, info};
use std::collections::HashSet;

const MAX_GAUSSIANS: u32 = 2_000_000;

cuneus::uniform_params! {
    struct GaussianParams {
    num_gaussians: u32,
    gaussian_size: f32,
    scene_scale: f32,
    gamma: f32,
    depth_shift: u32,
    up_mode: u32,
    sh_amt: f32,
    opacity_scale: f32,
    depth_near: f32,
    depth_far: f32,
    near_cull: f32,
    sh_degree: u32,
    focus_dist: f32,
    aperture: f32,
    view_mode: u32,
    oil_enable: u32,
    hardness: f32,
    bristle: f32,
    canvas: f32,
    impasto: f32,
    edge_rag: f32,
    edge_blur: f32,
    sharp_radius: f32,
    focus_x: f32,
    focus_y: f32,
    anim_prog: f32,
    anim_order: u32,
    anim_band: f32,
    anim_grow: f32,
    scene_cx: f32,
    scene_cy: f32,
    scene_cz: f32,
    scene_r: f32,
    _p0: f32,
    _p1: f32,
    _p2: f32}
}

impl Default for GaussianParams {
    fn default() -> Self {
        Self {
            num_gaussians: 0,
            gaussian_size: 1.0,
            scene_scale: 1.0,
            gamma: 1.0,
            depth_shift: 16,
            up_mode: 1,
            sh_amt: 1.0,
            opacity_scale: 1.0,
            depth_near: 0.1,
            depth_far: 100.0,
            near_cull: 0.01,
            sh_degree: 0,
            focus_dist: 1.0,
            aperture: 0.0,
            view_mode: 0,
            oil_enable: 0,
            hardness: 3.0,
            bristle: 0.5,
            canvas: 0.3,
            impasto: 0.4,
            edge_rag: 0.4,
            edge_blur: 0.0,
            sharp_radius: 0.3,
            focus_x: 0.5,
            focus_y: 0.5,
            anim_prog: -1.0,
            anim_order: 1,
            anim_band: 0.15,
            anim_grow: 0.6,
            scene_cx: 0.0,
            scene_cy: 0.0,
            scene_cz: 0.0,
            scene_r: 1.0,
            _p0: 0.0,
            _p1: 0.0,
            _p2: 0.0}
    }
}

#[derive(Clone, Copy)]
struct Cam { yaw: f32, pitch: f32, dist: f32, target: [f32; 3] }

impl Cam {
    fn off(&self) -> [f32; 3] {
        let (sy, cy, sp, cp) = (self.yaw.sin(), self.yaw.cos(), self.pitch.sin(), self.pitch.cos());
        [cp * sy, sp, cp * cy]
    }
    fn eye(&self) -> [f32; 3] {
        let o = self.off();
        [self.target[0] + self.dist * o[0], self.target[1] + self.dist * o[1], self.target[2] + self.dist * o[2]]
    }
    // forward, right, up
    fn basis(&self) -> ([f32; 3], [f32; 3], [f32; 3]) {
        let o = self.off();
        let f = [-o[0], -o[1], -o[2]];
        let r = norm([-f[2], 0.0, f[0]]);
        let u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]];
        (f, r, u)
    }
}

fn norm(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
    [v[0] / l, v[1] / l, v[2] / l]
}

fn orient(up_mode: u32, p: [f32; 3]) -> [f32; 3] {
    match up_mode {
        1 => [p[0], -p[1], -p[2]],
        2 => [p[0], p[2], -p[1]],
        _ => p,
    }
}

fn srgb_to_linear(c: f32) -> f64 {
    let c = c as f64;
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

struct CameraState {
    goal: Cam,
    cur: Cam,
    home: Cam,
    fov: f32,
    fly: bool,
    auto_rotate: f32,
    drag_l: bool,
    drag_r: bool,
    shift: bool,
    last_mouse: [f32; 2],
    keys_held: HashSet<String>,
    scene_center: [f32; 3],
    scene_radius: f32}

impl CameraState {
    fn new() -> Self {
        let c = Cam { yaw: 0.0, pitch: 0.0, dist: 4.0, target: [0.0; 3] };
        Self {
            goal: c, cur: c, home: c, fov: 50.0, fly: false, auto_rotate: 0.0,
            drag_l: false, drag_r: false, shift: false, last_mouse: [0.0; 2],
            keys_held: HashSet::new(), scene_center: [0.0; 3], scene_radius: 1.0}
    }

    // frame the robust scene bounds
    fn frame(&mut self, center: [f32; 3], radius: f32) {
        self.scene_center = center;
        self.scene_radius = radius.max(1e-4);
        let dist = self.scene_radius / (self.fov.to_radians() * 0.5).tan() * 1.15;
        self.home = Cam { yaw: 0.0, pitch: 0.0, dist, target: center };
        self.goal = self.home;
        self.cur = self.home;
    }

    fn reset(&mut self) { self.goal = self.home; }

    fn rotate(&mut self, dx: f32, dy: f32) {
        if self.fly {
            // look around the eye
            let e = self.goal.eye();
            self.goal.yaw -= dx * 0.004;
            self.goal.pitch = (self.goal.pitch + dy * 0.004).clamp(-1.55, 1.55);
            let o = self.goal.off();
            self.goal.target = [e[0] - self.goal.dist * o[0], e[1] - self.goal.dist * o[1], e[2] - self.goal.dist * o[2]];
        } else {
            self.goal.yaw += dx * 0.008;
            self.goal.pitch = (self.goal.pitch + dy * 0.008).clamp(-1.55, 1.55);
        }
    }

    fn pan(&mut self, dx: f32, dy: f32) {
        let (_, r, u) = self.goal.basis();
        let k = self.goal.dist * 0.0012;
        for i in 0..3 { self.goal.target[i] += (-dx * r[i] + dy * u[i]) * k; }
    }

    fn zoom(&mut self, d: f32) {
        if self.fly {
            let (f, _, _) = self.goal.basis();
            let s = self.scene_radius * 0.08 * d;
            for i in 0..3 { self.goal.target[i] += f[i] * s; }
        } else {
            self.goal.dist = (self.goal.dist * (1.0 - d * 0.1).clamp(0.5, 2.0)).clamp(self.scene_radius * 0.01, self.scene_radius * 200.0);
        }
    }

    fn update(&mut self, dt: f32) {
        let mut speed = self.scene_radius * if self.fly { 0.6 } else { 0.8 } * dt;
        if self.shift { speed *= 4.0; }
        let (f, r, _) = self.goal.basis();
        let flat = norm([f[0], 0.0, f[2]]);
        let fwd = if self.fly { f } else { flat };
        for key in &self.keys_held {
            let m: [f32; 3] = match key.as_str() {
                "w" => fwd, "s" => [-fwd[0], -fwd[1], -fwd[2]],
                "d" => r, "a" => [-r[0], -r[1], -r[2]],
                "q" => [0.0, 1.0, 0.0], "e" => [0.0, -1.0, 0.0],
                _ => [0.0; 3]};
            for i in 0..3 { self.goal.target[i] += m[i] * speed; }
        }
        self.goal.yaw += self.auto_rotate * dt;

        // smooth follow
        let k = 1.0 - (-dt * 14.0).exp();
        let l = |a: f32, b: f32| if (b - a).abs() < 1e-6 { b } else { a + (b - a) * k };
        self.cur.yaw = l(self.cur.yaw, self.goal.yaw);
        self.cur.pitch = l(self.cur.pitch, self.goal.pitch);
        self.cur.dist = l(self.cur.dist, self.goal.dist);
        for i in 0..3 { self.cur.target[i] = l(self.cur.target[i], self.goal.target[i]); }
    }

    fn camera(&self, c: &Cam, viewport: [f32; 2]) -> GaussianCamera {
        GaussianCamera::from_orbit(c.yaw, c.pitch, c.dist, c.target, self.fov.to_radians(), viewport)
    }
}

struct Gaussian3DShader {
    base: RenderKit,
    preprocess: ComputeShader,
    sorter: GaussianSorter,
    renderer: GaussianRenderer,
    render_bind_group: Option<wgpu::BindGroup>,
    camera_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    params: GaussianParams,
    camera: CameraState,
    surface_format: wgpu::TextureFormat,
    // raw scene bounds, before orientation and scale
    bounds: ([f32; 3], f32),
    focus_rel: f32,
    bg: [f32; 3],
    last_state: Vec<u8>,
    last_tick: std::time::Instant,
    anim_on: bool,
    anim_loop: bool,
    anim_t: f32,
    anim_dur: f32}

impl Gaussian3DShader {
    fn load_ply(&mut self, core: &Core, path: &std::path::Path) {
        info!("Loading: {:?}", path);
        let t0 = std::time::Instant::now();
        match GaussianCloud::from_ply(path) {
            Ok(cloud) => {
                let count = cloud.metadata.num_gaussians.min(MAX_GAUSSIANS);
                info!("Loaded {} Gaussians (sh degree {}) in {:.2}s", count, cloud.sh_degree, t0.elapsed().as_secs_f32());

                let bytes = cloud.as_bytes();
                core.queue.write_buffer(&self.preprocess.storage_buffers[0], 0, &bytes[..(count as usize * 64).min(bytes.len())]);
                let sh = cloud.sh_bytes();
                core.queue.write_buffer(&self.preprocess.storage_buffers[5], 0, &sh[..(count as usize * 48).min(sh.len())]);

                self.params.num_gaussians = count;
                self.params.sh_degree = cloud.sh_degree;
                self.bounds = (cloud.bounds_center, cloud.bounds_radius);
                self.frame_scene();
                self.sync_params(core);

                self.sorter.prepare_with_buffers(
                    &core.device,
                    &self.preprocess.storage_buffers[2],
                    &self.preprocess.storage_buffers[3],
                    count,
                );

                self.render_bind_group = Some(self.renderer.create_bind_group(
                    &core.device,
                    &self.params_buffer,
                    &self.camera_buffer,
                    &self.preprocess.storage_buffers[1],
                    &self.preprocess.storage_buffers[3],
                ));
                self.last_state.clear();
            }
            Err(e) => error!("Load error: {:?}", e)}
    }

    fn frame_scene(&mut self) {
        let s = self.params.scene_scale;
        let c = orient(self.params.up_mode, self.bounds.0);
        self.camera.frame([c[0] * s, c[1] * s, c[2] * s], self.bounds.1 * s);
        self.params.scene_cx = c[0] * s;
        self.params.scene_cy = c[1] * s;
        self.params.scene_cz = c[2] * s;
        self.params.scene_r = self.bounds.1 * s;
    }

    fn sync_params(&self, core: &Core) {
        self.preprocess.set_custom_params(self.params, &core.queue);
        core.queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));
    }

    // sort range, near cull and focus from the current eye
    fn update_depth(&mut self, cam: &Cam) {
        let e = cam.eye();
        let c = self.camera.scene_center;
        let r = self.camera.scene_radius;
        let d = ((e[0] - c[0]).powi(2) + (e[1] - c[1]).powi(2) + (e[2] - c[2]).powi(2)).sqrt();
        self.params.depth_near = (d - r * 1.5).max(r * 0.002);
        self.params.depth_far = d + r * 2.5;
        self.params.near_cull = r * 0.005;
        self.params.focus_dist = cam.dist * self.focus_rel;
    }

    // build over 80% of the loop, hold the finished scene for the rest
    fn anim_progress(&self, t: f32) -> f32 {
        if !self.anim_on { return -1.0; }
        let ph = t / self.anim_dur.max(0.5);
        let ph = if self.anim_loop { ph.fract() } else { ph.min(1.0) };
        (ph / 0.8).min(1.0)
    }

    fn clear_color(&self) -> wgpu::Color {
        wgpu::Color { r: srgb_to_linear(self.bg[0]), g: srgb_to_linear(self.bg[1]), b: srgb_to_linear(self.bg[2]), a: 1.0 }
    }

    fn export_frame(&mut self, core: &Core, frame: u32, time: f32) {
        let settings = self.base.export_manager.settings().clone();
        let mut c = self.camera.cur;
        c.yaw += time * self.camera.auto_rotate;
        let camera = self.camera.camera(&c, [settings.width as f32, settings.height as f32]);
        self.update_depth(&c);
        self.params.anim_prog = self.anim_progress(time);
        self.sync_params(core);
        core.queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&camera));
        core.queue.write_buffer(&self.preprocess.storage_buffers[4], 0, bytemuck::bytes_of(&camera));
        self.preprocess.set_time(time, 1.0 / settings.fps as f32, &core.queue);

        let clear = self.clear_color();
        if let Some(ref bg) = self.render_bind_group {
            GaussianExporter::export_frame(
                core, &mut self.preprocess, &self.sorter, &self.renderer,
                bg, self.params.num_gaussians, frame, &settings, self.surface_format, clear,
            );
        }
        self.last_state.clear();
    }

    // true when the view or params changed since the last sort
    fn upload_camera(&mut self, core: &Core) -> bool {
        let cam = self.camera.cur;
        let camera = self.camera.camera(&cam, [core.size.width as f32, core.size.height as f32]);
        self.update_depth(&cam);
        let mut state = bytemuck::bytes_of(&camera).to_vec();
        state.extend_from_slice(bytemuck::bytes_of(&self.params));
        if state == self.last_state { return false; }
        self.last_state = state;
        self.sync_params(core);
        core.queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&camera));
        core.queue.write_buffer(&self.preprocess.storage_buffers[4], 0, bytemuck::bytes_of(&camera));
        true
    }
}

impl ShaderManager for Gaussian3DShader {
    fn init(core: &Core) -> Self {
        let base = RenderKit::new(core);

        let gaussian_size = (MAX_GAUSSIANS as u64) * 64;
        let gaussian_2d_size = (MAX_GAUSSIANS as u64) * 48;
        let keys_size = (MAX_GAUSSIANS as u64) * 4;
        let indices_size = (MAX_GAUSSIANS as u64) * 4;
        let camera_size = std::mem::size_of::<GaussianCamera>() as u64;
        let sh_size = (MAX_GAUSSIANS as u64) * 48;

        let config = ComputeShaderBuilder::new()
            .with_label("Gaussian Preprocess")
            .with_entry_point("preprocess")
            .with_custom_uniforms::<GaussianParams>()
            .with_workgroup_size([256, 1, 1])
            .with_storage_buffer(StorageBufferSpec::new("gaussians", gaussian_size))
            .with_storage_buffer(StorageBufferSpec::new("gaussian_2d", gaussian_2d_size))
            .with_storage_buffer(StorageBufferSpec::new("depth_keys", keys_size))
            .with_storage_buffer(StorageBufferSpec::new("sorted_indices", indices_size))
            .with_storage_buffer(StorageBufferSpec::new("camera", camera_size))
            .with_storage_buffer(StorageBufferSpec::new("sh", sh_size))
            .build();

        let preprocess = cuneus::compute_shader!(core, "shaders/gaussian3d.wgsl", config);

        let camera_buffer = core.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gaussian Camera"),
            size: std::mem::size_of::<GaussianCamera>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false});

        let params_buffer = core.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gaussian Params"),
            size: std::mem::size_of::<GaussianParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false});

        let params = GaussianParams::default();
        let sorter = GaussianSorter::for_depth_shift(&core.device, params.depth_shift);
        let mut renderer = GaussianRenderer::new(
            &core.device,
            core.config.format,
            include_str!("shaders/gaussian3d.wgsl"),
        );
        if let Err(e) = renderer.enable_hot_reload(
            core.device.clone(),
            std::path::PathBuf::from("examples/shaders/gaussian3d.wgsl"),
        ) {
            log::warn!("Failed to enable gaussian render hot reload: {e}");
        }

        Self {
            base,
            preprocess,
            sorter,
            renderer,
            render_bind_group: None,
            camera_buffer,
            params_buffer,
            params,
            camera: CameraState::new(),
            surface_format: core.config.format,
            bounds: ([0.0; 3], 1.0),
            focus_rel: 1.0,
            bg: [0.0; 3],
            last_state: Vec::new(),
            last_tick: std::time::Instant::now(),
            anim_on: false,
            anim_loop: true,
            anim_t: 0.0,
            anim_dur: 6.0}
    }

    fn update(&mut self, core: &Core) {
        if self.preprocess.check_hot_reload(&core.device) { self.last_state.clear(); }
        self.renderer.check_hot_reload(&core.device);

        if let Some((frame, time)) = self.base.export_manager.try_get_next_frame() {
            self.export_frame(core, frame, time);
        } else {
            self.base.export_manager.complete_export();
        }

        // real frame time; the fps tracker resets right before update
        let now = std::time::Instant::now();
        let dt = now.duration_since(self.last_tick).as_secs_f32().min(0.1);
        self.last_tick = now;
        if self.anim_on { self.anim_t += dt; }
        self.params.anim_prog = self.anim_progress(self.anim_t);
        self.camera.update(dt);

        let current_time = self.base.controls.get_time(&self.base.start_time);
        self.preprocess.set_time(current_time, dt, &core.queue);
    }

    fn resize(&mut self, core: &Core) {
        self.base.update_resolution(&core.queue, core.size);
        self.last_state.clear();
    }

    fn render(&mut self, core: &Core) -> Result<(), cuneus::SurfaceError> {
        let output = match core.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded => {
                return Err(cuneus::SurfaceError::SkipFrame);
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                return Err(cuneus::SurfaceError::Outdated);
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                return Err(cuneus::SurfaceError::Lost);
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err(cuneus::SurfaceError::Lost);
            }
        };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut params = self.params;
        let previous_depth_shift = params.depth_shift;
        let previous_up = params.up_mode;
        let previous_scale = params.scene_scale;
        let mut changed = false;
        let mut load_ply_path: Option<std::path::PathBuf> = None;
        let mut frame_scene = false;
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self.base.controls.get_ui_request(&self.base.start_time, &core.size, self.base.fps_tracker.fps());
        let mut fly = self.camera.fly;
        let mut fov = self.camera.fov;
        let mut auto_rotate = self.camera.auto_rotate;
        let mut focus_rel = self.focus_rel;
        let mut bg = self.bg;
        let mut anim_on = self.anim_on;
        let mut anim_loop = self.anim_loop;
        let mut anim_dur = self.anim_dur;
        let mut anim_restart = false;

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                RenderKit::apply_default_style(ctx);

                egui::Window::new("3D Gaussian Splatting")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(300.0)
                    .show(ctx, |ui| {
                        if params.num_gaussians > 0 {
                            ui.label(format!("Gaussians: {}  (view colour: degree {})", params.num_gaussians, params.sh_degree));
                        } else {
                            ui.label("Drag & drop a .ply file");
                        }
                        ui.small("drag: orbit | right drag / shift drag: pan | wheel: zoom | WASD QE: move | R: reset");
                        if ui.button("Load PLY...").clicked() {
                            if let Some(p) = rfd::FileDialog::new().add_filter("PLY", &["ply"]).pick_file() {
                                load_ply_path = Some(p);
                            }
                        }

                        ui.separator();

                        egui::CollapsingHeader::new("Camera").default_open(true).show(ui, |ui| {
                            ui.checkbox(&mut fly, "Fly mode");
                            ui.add(egui::Slider::new(&mut fov, 20.0..=110.0).text("FOV"));
                            ui.add(egui::Slider::new(&mut auto_rotate, -1.0..=1.0).text("Auto Rotate"));
                            let mut up = params.up_mode as f32;
                            if ui.add(egui::Slider::new(&mut up, 0.0..=2.0).step_by(1.0).text("Up Axis (0 file, 1 flip, 2 z-up)")).changed() { params.up_mode = up as u32; changed = true; }
                            changed |= ui.add(egui::Slider::new(&mut params.scene_scale, 0.01..=100.0).logarithmic(true).text("Scene Scale")).changed();
                            if ui.button("Frame Scene (R)").clicked() { frame_scene = true; }
                        });

                        egui::CollapsingHeader::new("Look").default_open(true).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.gamma, 0.5..=2.0).text("Gamma")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.gaussian_size, 0.1..=2.0).text("Splat Size")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.opacity_scale, 0.1..=2.0).text("Opacity")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.sh_amt, 0.0..=1.5).text("View-dependent Colour")).changed();
                            let mut vm = params.view_mode as f32;
                            if ui.add(egui::Slider::new(&mut vm, 0.0..=3.0).step_by(1.0).text("View (0 splats, 1 depth, 2 ellipses, 3 points)")).changed() { params.view_mode = vm as u32; changed = true; }
                            ui.horizontal(|ui| { ui.color_edit_button_rgb(&mut bg); ui.label("Background"); });
                            let mut precise = params.depth_shift < 16;
                            if ui.checkbox(&mut precise, "Precise sort (24-bit)").changed() { params.depth_shift = if precise { 8 } else { 16 }; changed = true; }
                        });

                        egui::CollapsingHeader::new("Animate").default_open(false).show(ui, |ui| {
                            ui.horizontal(|ui| {
                                if ui.button(if anim_on { "■ Stop" } else { "▶ Play" }).clicked() { anim_on = !anim_on; anim_restart = true; }
                                if ui.button("↺ Restart").clicked() { anim_restart = true; }
                                ui.checkbox(&mut anim_loop, "Loop");
                            });
                            ui.add(egui::Slider::new(&mut anim_dur, 1.0..=30.0).suffix(" s").text("Duration"));
                            let mut ord = params.anim_order as f32;
                            if ui.add(egui::Slider::new(&mut ord, 0.0..=3.0).step_by(1.0).text("Order (0 big first, 1 centre out, 2 bottom up, 3 random)")).changed() { params.anim_order = ord as u32; changed = true; }
                            changed |= ui.add(egui::Slider::new(&mut params.anim_band, 0.02..=0.8).text("Band")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.anim_grow, 0.0..=1.0).text("Grow")).changed();
                        });

                        egui::CollapsingHeader::new("Depth of Field").default_open(false).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.aperture, 0.0..=10.0).text("Depth Blur")).changed();
                            ui.add(egui::Slider::new(&mut focus_rel, 0.1..=3.0).logarithmic(true).text("Focus Distance (1 = orbit centre)"));
                            changed |= ui.add(egui::Slider::new(&mut params.edge_blur, 0.0..=2.0).text("Edge Blur")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.sharp_radius, 0.0..=1.0).text("Sharp Radius")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.focus_x, 0.0..=1.0).text("Focus X")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.focus_y, 0.0..=1.0).text("Focus Y")).changed();
                        });

                        egui::CollapsingHeader::new("Shading").default_open(false).show(ui, |ui| {
                            let mut oil = params.oil_enable != 0;
                            if ui.checkbox(&mut oil, "Shading").changed() { params.oil_enable = oil as u32; changed = true; }
                            changed |= ui.add(egui::Slider::new(&mut params.hardness, 1.0..=6.0).text("Hardness")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.bristle, 0.0..=1.0).text("Bristle")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.impasto, 0.0..=1.0).text("Impasto")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.edge_rag, 0.0..=1.5).text("Ragged Edge")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.canvas, 0.0..=1.0).text("Canvas")).changed();
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

        self.camera.fly = fly;
        self.camera.fov = fov;
        self.camera.auto_rotate = auto_rotate;
        self.focus_rel = focus_rel;
        self.bg = bg;
        self.anim_on = anim_on;
        self.anim_loop = anim_loop;
        self.anim_dur = anim_dur;
        if anim_restart { self.anim_t = 0.0; }

        self.base.export_manager.apply_ui_request(export_request);
        self.base.apply_control_request(controls_request);

        if should_start_export {
            self.base.export_manager.start_export();
        }

        if let Some(path) = load_ply_path {
            self.load_ply(core, &path);
            params.num_gaussians = self.params.num_gaussians;
            params.sh_degree = self.params.sh_degree;
        }
        if changed {
            if (GaussianSorter::required_key_bits(previous_depth_shift) <= 16)
                != (GaussianSorter::required_key_bits(params.depth_shift) <= 16)
            {
                self.sorter = GaussianSorter::for_depth_shift(&core.device, params.depth_shift);
                if params.num_gaussians > 0 {
                    self.sorter.prepare_with_buffers(
                        &core.device,
                        &self.preprocess.storage_buffers[2],
                        &self.preprocess.storage_buffers[3],
                        params.num_gaussians,
                    );
                }
            }
            self.params = params;
            if params.up_mode != previous_up || params.scene_scale != previous_scale { frame_scene = true; }
        }
        if frame_scene { self.frame_scene(); }

        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Gaussian3D")});

        let count = self.params.num_gaussians;
        if count > 0 && self.render_bind_group.is_some() {
            // preprocess and sort only when the view or settings moved
            if self.upload_camera(core) {
                let workgroups = (count + 255) / 256;
                self.preprocess.dispatch_stage_with_workgroups(&mut encoder, 0, [workgroups, 1, 1]);
                self.sorter.sort(&mut encoder, count);
                encoder = core.flush_encoder(encoder);
            }
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Gaussian Render"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(self.clear_color()),
                            store: wgpu::StoreOp::Store},
                        depth_slice: None})],
                    ..Default::default()
                });
                self.renderer.render(&mut pass, self.render_bind_group.as_ref().unwrap(), count);
            }
        } else {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color()),
                        store: wgpu::StoreOp::Store},
                    depth_slice: None})],
                ..Default::default()
            });
        }

        self.base.handle_render_output(core, &view, full_output, &mut encoder);
        core.queue.submit(Some(encoder.finish()));
        core.queue.present(output);
        self.base.fps_tracker.update();
        Ok(())
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        if self.base.forward_to_egui(core, event) {
            return true;
        }

        if let WindowEvent::ModifiersChanged(m) = event {
            self.camera.shift = m.state().shift_key();
        }

        if let WindowEvent::KeyboardInput { event, .. } = event {
            if self.base.key_handler.handle_keyboard_input(core.window(), event) {
                return true;
            }
            if let winit::keyboard::Key::Character(ch) = &event.logical_key {
                let key = ch.as_str().to_lowercase();
                match event.state {
                    winit::event::ElementState::Pressed => {
                        if key == "r" { self.camera.reset(); return true; }
                        if matches!(key.as_str(), "w" | "a" | "s" | "d" | "q" | "e") {
                            self.camera.keys_held.insert(key);
                            return true;
                        }
                    }
                    winit::event::ElementState::Released => {
                        self.camera.keys_held.remove(&key);
                    }
                }
            }
        }

        if let WindowEvent::MouseInput { state, button, .. } = event {
            let down = *state == winit::event::ElementState::Pressed;
            match button {
                winit::event::MouseButton::Left => { self.camera.drag_l = down; return true; }
                winit::event::MouseButton::Right | winit::event::MouseButton::Middle => { self.camera.drag_r = down; return true; }
                _ => {}
            }
        }

        if let WindowEvent::CursorMoved { position, .. } = event {
            let (x, y) = (position.x as f32, position.y as f32);
            let (dx, dy) = (x - self.camera.last_mouse[0], y - self.camera.last_mouse[1]);
            self.camera.last_mouse = [x, y];
            if self.camera.drag_r || (self.camera.drag_l && self.camera.shift) {
                self.camera.pan(dx, dy);
                return true;
            }
            if self.camera.drag_l {
                self.camera.rotate(dx, dy);
                return true;
            }
            return false;
        }

        if let WindowEvent::MouseWheel { delta, .. } = event {
            let d = match delta {
                winit::event::MouseScrollDelta::LineDelta(_, y) => *y,
                winit::event::MouseScrollDelta::PixelDelta(p) => (p.y as f32 / 100.0).clamp(-3.0, 3.0)};
            self.camera.zoom(d);
            return true;
        }

        if let WindowEvent::DroppedFile(path) = event {
            if path.extension().map(|e| e == "ply").unwrap_or(false) {
                self.load_ply(core, path);
            }
            return true;
        }

        false
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("3D Gaussian Splatting", 800, 600);
    app.run(event_loop, Gaussian3DShader::init)
}

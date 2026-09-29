use cuneus::compute::*;
use cuneus::prelude::*;
use cuneus::{Blend, Cull, GpuInstances, Instance, InstanceAnim, Light, Mat4, MaterialId, MaterialOptions, MeshData, MeshId, MeshScene, MeshView, ObjectId, OrbitCamera, Vec3, GPU_INSTANCE_SIZE};
use log::{error, info};

const MAX_SWARM: u32 = 100_000;

// layout must match MeshParams in meshlab.wgsl
cuneus::uniform_params! {
    struct MeshParams {
    ambient: f32,
    gloss: f32,
    rim: f32,
    rim_power: f32,
    rim_r: f32,
    rim_g: f32,
    rim_b: f32,
    pattern: f32,
    pattern_scale: f32,
    pattern_speed: f32,
    wobble: f32,
    wobble_speed: f32,
    emissive: f32,
    opacity: f32,
    hue_spread: f32,
    shadow: f32,
    env: f32,
    _q0: f32,
    _q1: f32,
    _q2: f32}
}

// matches Swarm in meshlab_swarm.wgsl
cuneus::uniform_params! {
    struct SwarmParams {
    count: u32,
    spread: f32,
    scale: f32,
    clips: u32,
    cx: f32,
    cy: f32,
    cz: f32,
    _q: f32}
}

cuneus::uniform_params! {
    struct PostParams {
    bloom: f32,
    threshold: f32,
    exposure: f32,
    gamma: f32,
    vignette: f32,
    bg_top_r: f32,
    bg_top_g: f32,
    bg_top_b: f32,
    bg_bot_r: f32,
    bg_bot_g: f32,
    bg_bot_b: f32,
    view_mode: f32,
    fog: f32,
    outline: f32,
    env_bg: f32,
    crx: f32, cry: f32, crz: f32,
    cux: f32, cuy: f32, cuz: f32,
    cfx: f32, cfy: f32, cfz: f32}
}

impl Default for MeshParams {
    fn default() -> Self {
        Self {
            ambient: 0.6, gloss: 1.0,
            rim: 1.5, rim_power: 3.0, rim_r: 0.3, rim_g: 0.7, rim_b: 1.6,
            pattern: 0.6, pattern_scale: 6.0, pattern_speed: 1.5,
            wobble: 0.0, wobble_speed: 3.0, emissive: 1.0,
            opacity: 1.0, hue_spread: 0.0, shadow: 1.0, env: 0.0, _q0: 0.0, _q1: 0.0, _q2: 0.0}
    }
}

impl Default for PostParams {
    fn default() -> Self {
        Self {
            bloom: 1.0, threshold: 0.9, exposure: 1.0, gamma: 1.0, vignette: 0.6,
            bg_top_r: 0.02, bg_top_g: 0.025, bg_top_b: 0.04,
            bg_bot_r: 0.005, bg_bot_g: 0.005, bg_bot_b: 0.01,
            view_mode: 0.0, fog: 0.0, outline: 0.0, env_bg: 0.0,
            crx: 0.0, cry: 0.0, crz: 0.0, cux: 0.0, cuy: 0.0, cuz: 0.0, cfx: 0.0, cfy: 0.0, cfz: 1.0}
    }
}

struct MeshExample {
    base: RenderKit,
    post: ComputeShader,
    scene: MeshScene,
    model: MeshId,
    material_id: MaterialId,
    object: ObjectId,
    model_xf: Mat4,
    bottom: f32,
    copies: u32,
    floor: ObjectId,
    floor_mat: MaterialId,
    show_floor: bool,
    sun_yaw: f32,
    sun_pitch: f32,
    sun_power: f32,
    // name, seconds, animates only blend shapes (blink, breath...)
    clips: Vec<(String, f32, bool)>,
    clip: usize,
    layer: usize,
    anim_speed: f32,
    anim_time: f32,
    playing: bool,
    // extra engine features: HDRI in a material texture, lights, GPU swarm, picking
    model_center: Vec3,
    model_radius: f32,
    swarm: ComputeShader,
    swarm_on: bool,
    swarm_count: u32,
    crowd: bool,
    lights_on: bool,
    spot_on: bool,
    light_shadows: bool,
    light_power: f32,
    cursor: [f32; 2],
    press: Option<[f32; 2]>,
    picked: String,
    env_name: String,
    camera: OrbitCamera,
    material: MeshParams,
    params: PostParams,
    info: String,
    last_tick: std::time::Instant}

impl MeshExample {
    fn set_model(&mut self, core: &Core, data: &MeshData) {
        self.scene.replace_mesh(core, self.model, data);
        self.model_xf = data.normalize_transform();
        (self.model_center, self.model_radius) = (data.center, data.radius);
        self.bottom = data.normalized_bottom();
        self.clips = data.skinning.as_ref().map(|s| s.animations.iter()
            .map(|a| (a.name.clone(), a.duration, a.morph_only()))
            .collect()).unwrap_or_default();
        // body clip first, a face-only clip (blink) as the layer; 0 = none
        self.clip = self.clips.iter().position(|c| !c.2).map_or(0, |i| i + 1);
        self.layer = self.clips.iter().position(|c| c.2).map_or(0, |i| i + 1);
        self.anim_time = 0.0;
        self.layout_copies();
        self.info = format!("{} triangles, {} materials, {} textures", data.indices.len() / 3, data.materials.len(), data.images.len());
    }

    // copies on a square grid, each with a random value for the shader
    fn layout_copies(&mut self) {
        let n = self.copies.max(1) as usize;
        let side = (n as f32).sqrt().ceil() as usize;
        let off = (side - 1) as f32 * 0.5 * 2.4;
        let inst = (0..n).map(|i| Instance {
            transform: Mat4::from_translation(Vec3::new((i % side) as f32 * 2.4 - off, 0.0, (i / side) as f32 * 2.4 - off)) * self.model_xf,
            data: [hash(i), 0.0, 0.0, 0.0],
            anim: InstanceAnim::default(),
        }).collect();
        if let Some(v) = self.scene.instances_mut(self.object) { *v = inst; }
        self.camera.frame(Vec3::ZERO, off * 1.42 + 1.0);
        self.place_floor(off);
    }

    // clip indices that move the body (blend-shape-only clips excluded)
    fn body_clips(&self) -> Vec<usize> {
        self.clips.iter().enumerate().filter(|(_, c)| !c.2).map(|(i, _)| i).collect()
    }

    // floor under the models, a little wider than the grid
    fn place_floor(&mut self, off: f32) {
        let xf = Mat4::from_translation(Vec3::new(0.0, self.bottom - 1e-3, 0.0)) * Mat4::from_scale(Vec3::splat((off + 3.0) * 2.0));
        let show = self.show_floor;
        if let Some(v) = self.scene.instances_mut(self.floor) {
            v.clear();
            if show { v.push(xf.into()); }
        }
    }

    fn load(&mut self, core: &Core, path: &std::path::Path) {
        info!("Loading: {path:?}");
        match MeshData::from_gltf(path) {
            Ok(data) => {
                self.set_model(core, &data);
                self.info = format!("{}: {}", path.file_name().map(|n| n.to_string_lossy()).unwrap_or_default(), self.info);
            }
            Err(e) => error!("Load error: {e:?}"),
        }
    }
}

impl ShaderManager for MeshExample {
    fn init(core: &Core) -> Self {
        let base = RenderKit::new(core);

        let passes = vec![
            PassDescription::new("bd1", &[]).with_resolution_scale(0.5),
            PassDescription::new("bd2", &["bd1"]).with_resolution_scale(0.25),
            PassDescription::new("bd3", &["bd2"]).with_resolution_scale(0.125),
            PassDescription::new("bd4", &["bd3"]).with_resolution_scale(0.0625),
            PassDescription::new("bu3", &["bd3", "bd4"]).with_resolution_scale(0.125),
            PassDescription::new("bu2", &["bd2", "bu3"]).with_resolution_scale(0.25),
            PassDescription::new("bu1", &["bd1", "bu2"]).with_resolution_scale(0.5),
            PassDescription::new("main_image", &["bu1"]),
        ];
        let config = ComputeShader::builder()
            .with_multi_pass(&passes)
            .with_channels(3)
            .with_custom_uniforms::<PostParams>()
            .with_label("Mesh Post")
            .build();
        let mut post = cuneus::compute_shader!(core, "shaders/meshlab_post.wgsl", config);
        let mut scene = MeshScene::new(core);
        let knot = MeshData::torus_knot(2.0, 3.0, 400, 48);
        let model = scene.add_mesh(core, &knot);
        let material_id = cuneus::mesh_material!(scene, core, "shaders/meshlab.wgsl", MeshParams);
        let object = scene.spawn(model, material_id, knot.normalize_transform());
        // floor: same shader, its own values, casts no shadow
        let plane = scene.add_mesh(core, &MeshData::plane(1.0, 1.0));
        let floor_mat = scene.material_variant(core, material_id);
        scene.set_options(floor_mat, MaterialOptions { shadows: false, ..Default::default() });
        let floor = scene.spawn_instances(plane, floor_mat, Vec::new());
        scene.attach(core, &mut post, 0);
        scene.attach_gbuffer(core, &mut post, 1);
        let swarm_config = ComputeShader::builder()
            .with_entry_point("main")
            .with_custom_uniforms::<SwarmParams>()
            .with_storage_buffer(StorageBufferSpec::new("instances", MAX_SWARM as u64 * GPU_INSTANCE_SIZE))
            .with_workgroup_size([64, 1, 1])
            .with_label("Swarm")
            .build();
        let swarm = cuneus::compute_shader!(core, "shaders/meshlab_swarm.wgsl", swarm_config);

        let mut s = Self {
            base, post, scene, model, material_id, object,
            model_xf: knot.normalize_transform(), bottom: knot.normalized_bottom(), copies: 1,
            floor, floor_mat, show_floor: true, sun_yaw: 0.6, sun_pitch: 0.9, sun_power: 1.0,
            clips: Vec::new(), clip: 0, layer: 0, anim_speed: 1.0, anim_time: 0.0, playing: true,
            model_center: knot.center, model_radius: knot.radius, swarm, swarm_on: false, swarm_count: 5000,
            crowd: false, lights_on: false, spot_on: false, light_shadows: true, light_power: 1.0,
            cursor: [0.0; 2], press: None, picked: String::new(), env_name: String::new(),
            camera: OrbitCamera::new(),
            material: MeshParams::default(),
            params: PostParams::default(),
            info: String::new(),
            last_tick: std::time::Instant::now()};
        s.layout_copies();
        s.info = format!("torus knot: {} triangles", knot.indices.len() / 3);
        s.post.set_custom_params(s.params, &core.queue);
        s
    }

    fn update(&mut self, core: &Core) {
        let now = std::time::Instant::now();
        let dt = now.duration_since(self.last_tick).as_secs_f32().min(0.1);
        self.last_tick = now;
        self.camera.update(dt);

        let time = self.base.controls.get_time(&self.base.start_time);
        self.post.set_time(time, dt, &core.queue);
        self.swarm.set_time(time, dt, &core.queue);

        // main clip + layer on top (0 = none for both)
        if self.playing { self.anim_time += dt * self.anim_speed; }
        let layers = |t: f32, clip: usize, layer: usize| [clip, layer].into_iter().filter_map(|c| c.checked_sub(1)).map(|c| (c, t, 1.0)).collect::<Vec<_>>();
        self.scene.animate_layers(self.object, &layers(self.anim_time, self.clip, self.layer));
        // crowd: each copy its own body clip, speed and phase
        if self.crowd {
            let (body, t) = (self.body_clips(), self.anim_time);
            if let Some(v) = self.scene.instances_mut(self.object) {
                for (i, inst) in v.iter_mut().enumerate() {
                    let h = hash(i);
                    inst.anim = match body.get(i % body.len().max(1)) {
                        Some(&c) => InstanceAnim::clip(c, t * (0.8 + h * 0.4) + h * 10.0),
                        None => InstanceAnim::default(),
                    };
                }
            }
        }

        let (camera, object, clip, layer, speed) = (&self.camera, self.object, self.clip, self.layer, self.anim_speed);
        self.scene.handle_export(core, &mut self.post, &mut self.base, |scene, t, aspect| {
            scene.animate_layers(object, &layers(t * speed, clip, layer));
            MeshView::orbit(camera, camera.turntable(t), aspect)
        });
    }

    fn resize(&mut self, core: &Core) {
        self.base.default_resize(core, &mut self.post);
    }

    fn render(&mut self, core: &Core) -> Result<(), cuneus::SurfaceError> {
        let mut frame = self.base.begin_frame(core)?;

        // picking answer: highlight the copy through its instance data .y
        if let Some(hit) = self.scene.take_pick() {
            let (object, floor, swarm) = (self.object, self.floor, self.swarm_on);
            self.picked = match hit {
                Some(h) if h.object == floor => "floor".into(),
                Some(h) if h.object == object => format!("{} #{} at ({:.2}, {:.2}, {:.2})", if swarm { "swarm copy" } else { "copy" }, h.instance, h.position.x, h.position.y, h.position.z),
                Some(h) => format!("object {:?}", h.object),
                None => "nothing".into(),
            };
            let target = hit.filter(|h| h.object == object).map(|h| h.instance as usize);
            if let Some(v) = self.scene.instances_mut(object) {
                for (i, inst) in v.iter_mut().enumerate() { inst.data[1] = if Some(i) == target { 1.0 } else { 0.0 }; }
            }
        }

        // panel values, edited below and applied after the ui
        // material and post
        let (mut m, mut p, mut changed) = (self.material, self.params, false);
        let mut opts = self.scene.options(self.material_id);
        let (mut blend, mut cull) = (opts.blend as u8 as f32, opts.cull as u8 as f32);
        // camera
        let (mut fly, mut fov, mut auto_rotate, mut reset_cam) = (self.camera.fly, self.camera.fov, self.camera.auto_rotate, false);
        // files and labels
        let (mut load_path, mut env_path): (Option<std::path::PathBuf>, Option<std::path::PathBuf>) = (None, None);
        let (info, picked, env_name) = (self.info.clone(), self.picked.clone(), self.env_name.clone());
        // copies and swarm
        let (mut copies, mut swarm_on, mut swarm_count) = (self.copies, self.swarm_on, self.swarm_count);
        // animation
        let (mut clip, mut layer, mut anim_speed, mut playing, mut restart) = (self.clip, self.layer, self.anim_speed, self.playing, false);
        let (mut crowd, clips) = (self.crowd, self.clips.clone());
        // sun, shadows, floor
        let (mut sun_yaw, mut sun_pitch, mut sun_power, mut show_floor) = (self.sun_yaw, self.sun_pitch, self.sun_power, self.show_floor);
        let (mut shadows, mut softness) = (self.scene.sun.shadows, self.scene.sun.softness);
        // lights
        let (mut lights_on, mut spot_on, mut light_shadows, mut light_power) = (self.lights_on, self.spot_on, self.light_shadows, self.light_power);
        // time controls and export
        let mut controls_request = self.base.controls.get_ui_request(&self.base.start_time, &core.size, self.base.fps_tracker.fps());
        let (mut export_request, mut should_start_export) = (self.base.export_manager.get_ui_request(), false);

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                RenderKit::apply_default_style(ctx);
                egui::Window::new("Mesh Lab").collapsible(true).resizable(true).default_width(300.0).show(ctx, |ui| {
                    ui.label(&info);
                    ui.small("drop a .glb/.gltf | drag: orbit | click: pick | right drag: pan | wheel: zoom | WASD/arrows QE | R: reset");
                    if !picked.is_empty() { ui.label(format!("Picked: {picked}")); }
                    if ui.button("Load model...").clicked() {
                        load_path = rfd::FileDialog::new().add_filter("glTF", &["glb", "gltf"]).pick_file();
                    }
                    ui.separator();

                    egui::CollapsingHeader::new("Camera").default_open(false).show(ui, |ui| {
                        ui.checkbox(&mut fly, "Fly mode");
                        ui.add(egui::Slider::new(&mut fov, 20.0..=110.0).text("FOV"));
                        ui.add(egui::Slider::new(&mut auto_rotate, -1.0..=1.0).text("Auto Rotate"));
                        if ui.button("Frame model (R)").clicked() { reset_cam = true; }
                    });

                    egui::CollapsingHeader::new("Sun & Shadows").default_open(true).show(ui, |ui| {
                        ui.add(egui::Slider::new(&mut sun_yaw, -std::f32::consts::PI..=std::f32::consts::PI).text("Sun Yaw"));
                        ui.add(egui::Slider::new(&mut sun_pitch, 0.05..=1.55).text("Sun Height"));
                        ui.add(egui::Slider::new(&mut sun_power, 0.0..=4.0).text("Sun Intensity"));
                        changed |= ui.add(egui::Slider::new(&mut m.ambient, 0.0..=2.0).text("Ambient")).changed();
                        changed |= ui.add(egui::Slider::new(&mut m.gloss, 0.1..=2.0).text("Roughness x")).changed();
                        ui.horizontal(|ui| { ui.checkbox(&mut shadows, "Shadows"); ui.checkbox(&mut show_floor, "Floor"); });
                        ui.add(egui::Slider::new(&mut softness, 0.0..=8.0).text("Softness"));
                        changed |= ui.add(egui::Slider::new(&mut m.shadow, 0.0..=1.0).text("Shadow Strength")).changed();
                    });

                    if !clips.is_empty() {
                        egui::CollapsingHeader::new("Animation").default_open(true).show(ui, |ui| {
                            let name = |c: usize, none: &str| c.checked_sub(1).and_then(|i| clips.get(i)).map(|(n, d, _)| format!("{n} ({d:.1}s)")).unwrap_or(none.into());
                            let (main, top) = (name(clip, "rest pose"), name(layer, "no layer"));
                            ui.add(egui::Slider::new(&mut clip, 0..=clips.len()).text(format!("Clip: {main}")));
                            // layering is for blend-shape clips (blink, breath) over a body clip
                            if clips.iter().any(|c| c.2) { ui.add(egui::Slider::new(&mut layer, 0..=clips.len()).text(format!("Face layer: {top}"))); }
                            ui.add(egui::Slider::new(&mut anim_speed, 0.0..=3.0).text("Speed"));
                            ui.horizontal(|ui| { ui.checkbox(&mut playing, "Play"); if ui.button("↺ Restart").clicked() { restart = true; } });
                            ui.checkbox(&mut crowd, "Crowd: every copy its own clip (use with Copies / GPU swarm)");
                        });
                    }

                    egui::CollapsingHeader::new("Material").default_open(false).show(ui, |ui| {
                        ui.add(egui::Slider::new(&mut blend, 0.0..=3.0).step_by(1.0).text("Blend (0 auto, 1 opaque, 2 alpha, 3 add)"));
                        ui.add(egui::Slider::new(&mut cull, 0.0..=3.0).step_by(1.0).text("Cull (0 auto, 1 none, 2 back, 3 front)"));
                        changed |= ui.add(egui::Slider::new(&mut m.opacity, 0.0..=1.0).text("Opacity (blend 2 / 3)")).changed();
                    });

                    egui::CollapsingHeader::new("Copies").default_open(false).show(ui, |ui| {
                        ui.add(egui::Slider::new(&mut copies, 1..=4096).logarithmic(true).text("Copies"));
                        changed |= ui.add(egui::Slider::new(&mut m.hue_spread, 0.0..=1.0).text("Glow Hue Spread")).changed();
                        ui.checkbox(&mut swarm_on, "GPU swarm (compute shader places the copies)");
                        ui.add(egui::Slider::new(&mut swarm_count, 100..=MAX_SWARM).logarithmic(true).text("Swarm Size"));
                    });

                    egui::CollapsingHeader::new("Lights").default_open(false).show(ui, |ui| {
                        ui.horizontal(|ui| { ui.checkbox(&mut lights_on, "3 point lights"); ui.checkbox(&mut spot_on, "Spot light"); });
                        ui.checkbox(&mut light_shadows, "Light shadows");
                        ui.add(egui::Slider::new(&mut light_power, 0.0..=4.0).text("Light Intensity"));
                    });

                    egui::CollapsingHeader::new("Environment").default_open(false).show(ui, |ui| {
                        if ui.button("Load HDRI / texture...").clicked() {
                            env_path = rfd::FileDialog::new().add_filter("image", &["hdr", "exr", "png", "jpg", "jpeg"]).pick_file();
                        }
                        if !env_name.is_empty() { ui.small(&env_name); }
                        changed |= ui.add(egui::Slider::new(&mut m.env, 0.0..=3.0).text("Environment")).changed();
                    });

                    egui::CollapsingHeader::new("Glow").default_open(true).show(ui, |ui| {
                        let mut c = [m.rim_r, m.rim_g, m.rim_b];
                        ui.horizontal(|ui| { if ui.color_edit_button_rgb(&mut c).changed() { [m.rim_r, m.rim_g, m.rim_b] = c; changed = true; } ui.label("Glow colour"); });
                        changed |= ui.add(egui::Slider::new(&mut m.rim, 0.0..=6.0).text("Rim")).changed();
                        changed |= ui.add(egui::Slider::new(&mut m.rim_power, 0.5..=8.0).text("Rim Falloff")).changed();
                        changed |= ui.add(egui::Slider::new(&mut m.pattern, 0.0..=3.0).text("Bands")).changed();
                        changed |= ui.add(egui::Slider::new(&mut m.pattern_scale, 1.0..=30.0).text("Band Scale")).changed();
                        changed |= ui.add(egui::Slider::new(&mut m.pattern_speed, -6.0..=6.0).text("Band Speed")).changed();
                        changed |= ui.add(egui::Slider::new(&mut m.emissive, 0.0..=10.0).text("Model Emissive")).changed();
                    });

                    egui::CollapsingHeader::new("Deform").default_open(false).show(ui, |ui| {
                        changed |= ui.add(egui::Slider::new(&mut m.wobble, 0.0..=3.0).text("Wobble")).changed();
                        changed |= ui.add(egui::Slider::new(&mut m.wobble_speed, 0.0..=10.0).text("Wobble Speed")).changed();
                    });

                    egui::CollapsingHeader::new("Post").default_open(false).show(ui, |ui| {
                        changed |= ui.add(egui::Slider::new(&mut p.bloom, 0.0..=4.0).text("Bloom")).changed();
                        changed |= ui.add(egui::Slider::new(&mut p.threshold, 0.0..=3.0).text("Threshold")).changed();
                        changed |= ui.add(egui::Slider::new(&mut p.exposure, 0.1..=4.0).logarithmic(true).text("Exposure")).changed();
                        changed |= ui.add(egui::Slider::new(&mut p.gamma, 0.5..=2.0).text("Gamma")).changed();
                        changed |= ui.add(egui::Slider::new(&mut p.vignette, 0.0..=2.0).text("Vignette")).changed();
                        changed |= ui.add(egui::Slider::new(&mut p.fog, 0.0..=2.0).text("Depth Fog")).changed();
                        changed |= ui.add(egui::Slider::new(&mut p.outline, 0.0..=1.0).text("Outline")).changed();
                        changed |= ui.add(egui::Slider::new(&mut p.view_mode, 0.0..=2.0).step_by(1.0).text("View (0 final, 1 normals, 2 depth)")).changed();
                        let mut top = [p.bg_top_r, p.bg_top_g, p.bg_top_b];
                        let mut bot = [p.bg_bot_r, p.bg_bot_g, p.bg_bot_b];
                        ui.horizontal(|ui| {
                            if ui.color_edit_button_rgb(&mut top).changed() { [p.bg_top_r, p.bg_top_g, p.bg_top_b] = top; changed = true; }
                            if ui.color_edit_button_rgb(&mut bot).changed() { [p.bg_bot_r, p.bg_bot_g, p.bg_bot_b] = bot; changed = true; }
                            ui.label("Background");
                        });
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

        // apply the panel
        // camera
        (self.camera.fly, self.camera.fov, self.camera.auto_rotate) = (fly, fov, auto_rotate);
        if reset_cam { self.camera.reset(); }

        // material and post
        const BLENDS: [Blend; 4] = [Blend::Auto, Blend::Opaque, Blend::Alpha, Blend::Additive];
        const CULLS: [Cull; 4] = [Cull::Auto, Cull::None, Cull::Back, Cull::Front];
        (opts.blend, opts.cull) = (BLENDS[blend as usize], CULLS[cull as usize]);
        if opts != self.scene.options(self.material_id) { self.scene.set_options(self.material_id, opts); }
        if changed {
            self.material = m;
            self.params = PostParams { env_bg: self.params.env_bg, ..p };
            self.post.set_custom_params(self.params, &core.queue);
        }

        // copies and swarm (the compute buffer becomes the object's copies)
        if copies != self.copies { self.copies = copies; self.layout_copies(); }
        let swarm_spread = (swarm_count as f32).sqrt() * 0.6;
        if swarm_on != self.swarm_on || (swarm_on && swarm_count != self.swarm_count) {
            let source = swarm_on.then(|| GpuInstances { buffer: self.swarm.storage_buffers[0].clone(), count: swarm_count, bounds: (Vec3::ZERO, swarm_spread + 1.0) });
            self.scene.set_gpu_instances(core, self.object, source);
            if swarm_on { self.camera.frame(Vec3::ZERO, swarm_spread + 1.0); } else { self.layout_copies(); }
        }
        (self.swarm_on, self.swarm_count) = (swarm_on, swarm_count);

        // animation
        if clip != self.clip || restart { self.anim_time = 0.0; }
        (self.clip, self.layer, self.anim_speed, self.playing) = (clip, layer, anim_speed, playing);
        if crowd != self.crowd { self.crowd = crowd; self.scene.set_crowd(self.object, crowd); }

        // sun, shadows, floor
        if show_floor != self.show_floor { self.show_floor = show_floor; self.layout_copies(); }
        (self.sun_yaw, self.sun_pitch, self.sun_power) = (sun_yaw, sun_pitch, sun_power);
        self.scene.sun.set_angles(sun_yaw, sun_pitch);
        self.scene.sun.color = [3.0 * sun_power; 3];
        (self.scene.sun.shadows, self.scene.sun.softness) = (shadows, softness);

        // lights
        (self.lights_on, self.spot_on, self.light_shadows, self.light_power) = (lights_on, spot_on, light_shadows, light_power);

        // environment: HDRI into the material, and behind the scene
        if let Some(path) = env_path {
            match self.scene.set_image_file(core, self.material_id, 0, &path, true) {
                Ok(()) => {
                    self.env_name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                    if let Some(view) = self.scene.material_texture(self.material_id, 0).cloned() {
                        self.post.update_channel_texture(2, &view, self.scene.output_sampler(), &core.device, &core.queue);
                        self.params.env_bg = 1.0;
                    }
                }
                Err(e) => error!("Texture load error: {e:?}"),
            }
        }

        // model
        if let Some(path) = load_path { self.load(core, &path); }

        // time controls and export
        self.base.apply_control_request(controls_request);
        self.base.export_manager.apply_ui_request(export_request);
        if should_start_export { self.base.export_manager.start_export(); }

        // orbiting coloured point lights and a spot from above
        let t = self.base.controls.get_time(&self.base.start_time);
        self.scene.lights.clear();
        if self.lights_on {
            for (k, c) in [[1.0, 0.45, 0.15], [0.15, 0.55, 1.0], [1.0, 0.2, 0.8]].iter().enumerate() {
                let a = t * 0.7 + k as f32 * 2.094;
                let pos = Vec3::new(a.cos() * 1.6, 0.5 + 0.3 * (t + k as f32).sin(), a.sin() * 1.6);
                let mut l = Light::point(pos, c.map(|x| x * 4.0 * self.light_power), 5.0);
                l.shadows = self.light_shadows;
                self.scene.lights.push(l);
            }
        }
        if self.spot_on {
            let pos = Vec3::new(0.0, 2.5, 1.5);
            let mut l = Light::spot(pos, -pos, [10.0 * self.light_power, 9.5 * self.light_power, 8.5 * self.light_power], 8.0, 0.5);
            l.shadows = self.light_shadows;
            self.scene.lights.push(l);
        }

        // camera basis for the HDRI background
        {
            let pose = self.camera.pose;
            let tan = (self.camera.fov.to_radians() * 0.5).tan();
            let aspect = core.size.width as f32 / core.size.height.max(1) as f32;
            let (r, u, f) = (pose.right() * (tan * aspect), pose.up() * tan, pose.forward());
            let p = &mut self.params;
            (p.crx, p.cry, p.crz, p.cux, p.cuy, p.cuz, p.cfx, p.cfy, p.cfz) = (r.x, r.y, r.z, u.x, u.y, u.z, f.x, f.y, f.z);
            self.post.set_custom_params(self.params, &core.queue);
        }

        if !self.base.export_manager.is_exporting() {
            if self.swarm_on {
                let k = 0.35 / self.model_radius;
                let c = self.model_center;
                self.swarm.set_custom_params(SwarmParams { count: self.swarm_count, spread: swarm_spread, scale: k, clips: self.body_clips().len() as u32, cx: c.x, cy: c.y, cz: c.z, _q: 0.0 }, &core.queue);
                self.swarm.dispatch_stage_with_workgroups(&mut frame.encoder, 0, [self.swarm_count.div_ceil(64), 1, 1]);
            }
            let view = MeshView::orbit(&self.camera, self.camera.pose, core.size.width as f32 / core.size.height.max(1) as f32);
            self.scene.set_params(&core.queue, self.material_id, &self.material);
            let floor = MeshParams { rim: 0.0, pattern: 0.0, wobble: 0.0, emissive: 0.0, hue_spread: 0.0, opacity: 1.0, env: self.material.env * 0.3, ..self.material };
            self.scene.set_params(&core.queue, self.floor_mat, &floor);
            self.scene.render(&mut frame.encoder, core, &mut self.post, &view);
            self.post.dispatch(&mut frame.encoder, core);
        }

        self.base.renderer.render_to_view(&mut frame.encoder, &frame.view, &self.post.get_output_texture().bind_group);
        self.base.end_frame(core, frame, full_output);
        Ok(())
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        if self.base.forward_to_egui(core, event) {
            return true;
        }
        if let WindowEvent::KeyboardInput { event, .. } = event {
            if self.base.key_handler.handle_keyboard_input(core.window(), event) {
                return true;
            }
        }
        // click without dragging: pick
        match event {
            WindowEvent::CursorMoved { position, .. } => self.cursor = [position.x as f32, position.y as f32],
            WindowEvent::MouseInput { state, button: winit::event::MouseButton::Left, .. } => match state {
                winit::event::ElementState::Pressed => self.press = Some(self.cursor),
                winit::event::ElementState::Released => {
                    if let Some(p) = self.press.take() {
                        if (p[0] - self.cursor[0]).abs() + (p[1] - self.cursor[1]).abs() < 4.0 {
                            self.scene.pick(self.cursor[0] as u32, self.cursor[1] as u32);
                        }
                    }
                }
            },
            _ => {}
        }
        if let WindowEvent::DroppedFile(path) = event {
            if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("glb") || e.eq_ignore_ascii_case("gltf")) {
                self.load(core, path);
            }
            return true;
        }
        self.camera.handle_event(event)
    }
}

fn hash(i: usize) -> f32 { ((i as f32 * 12.9898).sin() * 43758.545).fract().abs() }

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("Mesh Lab", 1280, 720);
    app.run(event_loop, MeshExample::init)
}

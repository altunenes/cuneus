use cuneus::compute::{ComputeShader, ComputeShaderBuilder, StorageBufferSpec};
use cuneus::prelude::*;
use cuneus::{GaussianCamera, GaussianCloud, GaussianRenderer, GaussianSorter};
use std::collections::HashSet;
use winit::event::WindowEvent;

const MAX_GAUSSIANS: u32 = 2_000_000;


#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GaussianParams {
    num_gaussians: u32,
    gaussian_size: f32,
    scene_scale: f32,
    gamma: f32,
    depth_shift: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

impl UniformProvider for GaussianParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

impl Default for GaussianParams {
    fn default() -> Self {
        Self {
            num_gaussians: 0,
            gaussian_size: 1.0,
            scene_scale: 10.0,
            gamma: 1.2,
            depth_shift: 16,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }
}

struct CameraState {
    yaw: f32,
    pitch: f32,
    distance: f32,
    fov: f32,
    target: [f32; 3],
    is_dragging: bool,
    last_mouse: [f32; 2],
    keys_held: HashSet<String>,
}

impl Default for CameraState {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            distance: 1.0,
            fov: 51.0,
            target: [0.0; 3],
            is_dragging: false,
            last_mouse: [0.0; 2],
            keys_held: HashSet::new(),
        }
    }
}

impl CameraState {
    fn new() -> Self {
        Self {
            yaw: 6.28,
            pitch: -0.05,
            distance: 4.0,
            fov: 51.0,
            target: [0.0, 0.0, -6.0],
            ..Default::default()
        }
    }

    fn reset(&mut self) {
        let keys = std::mem::take(&mut self.keys_held);
        *self = Self::new();
        self.keys_held = keys;
    }

    fn apply_held_keys(&mut self, dt: f32) {
        if self.keys_held.is_empty() {
            return;
        }
        let speed = 2.0 * self.distance * dt;
        let (sy, cy) = (self.yaw.sin(), self.yaw.cos());
        let forward = [sy, 0.0, cy];
        let right = [-cy, 0.0, sy];

        for key in &self.keys_held {
            match key.as_str() {
                "w" => { self.target[0] += forward[0] * speed; self.target[2] += forward[2] * speed; }
                "s" => { self.target[0] -= forward[0] * speed; self.target[2] -= forward[2] * speed; }
                "a" => { self.target[0] -= right[0] * speed; self.target[2] -= right[2] * speed; }
                "d" => { self.target[0] += right[0] * speed; self.target[2] += right[2] * speed; }
                "q" => { self.target[1] += speed; }
                "e" => { self.target[1] -= speed; }
                _ => {}
            }
        }
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
}

impl Gaussian3DShader {
    fn load_ply(&mut self, core: &Core, path: &std::path::Path) {
        println!("Loading: {:?}", path);
        match GaussianCloud::from_ply(path) {
            Ok(cloud) => {
                let count = cloud.metadata.num_gaussians.min(MAX_GAUSSIANS);
                println!("Loaded {} Gaussians", count);

                let bytes = cloud.as_bytes();
                let size = (count as usize * 64).min(bytes.len());
                core.queue.write_buffer(&self.preprocess.storage_buffers[0], 0, &bytes[..size]);

                self.params.num_gaussians = count;
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

                self.sorter.force_sort();
                self.camera.reset();
            }
            Err(e) => eprintln!("Load error: {:?}", e),
        }
    }

    fn sync_params(&self, core: &Core) {
        self.preprocess.set_custom_params(self.params, &core.queue);
        core.queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));
    }

    fn update_camera(&self, core: &Core) {
        let camera = GaussianCamera::from_orbit(
            self.camera.yaw,
            self.camera.pitch,
            self.camera.distance,
            self.camera.target,
            self.camera.fov.to_radians(),
            [core.size.width as f32, core.size.height as f32],
        );
        core.queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&camera));
        core.queue.write_buffer(&self.preprocess.storage_buffers[4], 0, bytemuck::bytes_of(&camera));
    }
}

impl ShaderManager for Gaussian3DShader {
    fn init(core: &Core) -> Self {
        let texture_bind_group_layout = RenderKit::create_standard_texture_layout(&core.device);
        let base = RenderKit::new(core, &texture_bind_group_layout, None);

        let gaussian_size = (MAX_GAUSSIANS as u64) * 64;
        let gaussian_2d_size = (MAX_GAUSSIANS as u64) * 48;
        let keys_size = (MAX_GAUSSIANS as u64) * 4;
        let indices_size = (MAX_GAUSSIANS as u64) * 4;
        let camera_size = std::mem::size_of::<GaussianCamera>() as u64;

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
            .build();

        let preprocess = ComputeShader::from_builder(
            core,
            include_str!("shaders/gaussian3d.wgsl"),
            config,
        );

        let camera_buffer = core.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gaussian Camera"),
            size: std::mem::size_of::<GaussianCamera>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params_buffer = core.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gaussian Params"),
            size: std::mem::size_of::<GaussianParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sorter = GaussianSorter::new_16bit(&core.device);
        let renderer = GaussianRenderer::new(
            &core.device,
            core.config.format,
            include_str!("shaders/gaussian3d.wgsl"),
        );

        Self {
            base,
            preprocess,
            sorter,
            renderer,
            render_bind_group: None,
            camera_buffer,
            params_buffer,
            params: GaussianParams::default(),
            camera: CameraState::new(),
        }
    }

    fn update(&mut self, core: &Core) {
        let dt = self.base.fps_tracker.delta_time();
        self.camera.apply_held_keys(dt);
        self.update_camera(core);

        let current_time = self.base.controls.get_time(&self.base.start_time);
        self.preprocess.set_time(current_time, dt, &core.queue);

        self.base.fps_tracker.update();
    }

    fn resize(&mut self, core: &Core) {
        self.base.update_resolution(&core.queue, core.size);
    }

    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut params = self.params;
        let mut changed = false;
        let mut load_ply_path: Option<std::path::PathBuf> = None;
        let mut controls_request = self.base.controls.get_ui_request(&self.base.start_time, &core.size);
        controls_request.current_fps = Some(self.base.fps_tracker.fps());

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                ctx.style_mut(|style| {
                    style.visuals.window_fill = egui::Color32::from_rgba_premultiplied(0, 0, 0, 180);
                    style.text_styles.get_mut(&egui::TextStyle::Body).unwrap().size = 11.0;
                    style.text_styles.get_mut(&egui::TextStyle::Button).unwrap().size = 10.0;
                });

                egui::Window::new("3D Gaussian Splatting")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(300.0)
                    .show(ctx, |ui| {
                        if params.num_gaussians > 0 {
                            ui.label(format!("Gaussians: {}", params.num_gaussians));
                        } else {
                            ui.label("Drag & drop a .ply file");
                        }
                        ui.small("WASD: move | QE: up/down | R: reset | Drag: rotate");

                        if ui.button("Load PLY...").clicked() {
                            if let Some(p) = rfd::FileDialog::new().add_filter("PLY", &["ply"]).pick_file() {
                                load_ply_path = Some(p);
                            }
                        }

                        ui.separator();

                        egui::CollapsingHeader::new("Visual Settings")
                            .default_open(true)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.scene_scale, 0.01..=100.0)
                                    .logarithmic(true).text("Scene Scale")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.gaussian_size, 0.1..=2.0)
                                    .text("Gaussian Size")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.gamma, 0.1..=2.2)
                                    .text("Gamma")).changed();

                                let mut depth_shift_f = params.depth_shift as f32;
                                if ui.add(egui::Slider::new(&mut depth_shift_f, 1.0..=30.0)
                                    .step_by(1.0)
                                    .text("Depth Blur")).changed() {
                                    params.depth_shift = depth_shift_f as u32;
                                    changed = true;
                                }
                            });

                        egui::CollapsingHeader::new("Camera Settings")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut self.camera.distance, 0.1..=100.0)
                                    .logarithmic(true).text("Distance")).changed();
                                changed |= ui.add(egui::Slider::new(&mut self.camera.fov, 20.0..=120.0)
                                    .text("FOV")).changed();
                                changed |= ui.add(egui::DragValue::new(&mut self.camera.yaw)
                                    .speed(0.05).prefix("Yaw: ")).changed();
                                changed |= ui.add(egui::Slider::new(&mut self.camera.pitch, -1.5..=1.5)
                                    .text("Pitch")).changed();

                                if ui.button("Reset Camera").clicked() {
                                    self.camera.reset();
                                    changed = true;
                                }
                            });

                        ui.separator();
                        ShaderControls::render_controls_widget(ui, &mut controls_request);
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        self.base.apply_control_request(controls_request);

        if let Some(path) = load_ply_path {
            self.load_ply(core, &path);
        }
        if changed {
            self.params = params;
            self.sync_params(core);
        }

        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Gaussian3D"),
        });

        let count = self.params.num_gaussians;
        if count > 0 && self.render_bind_group.is_some() {
            self.update_camera(core);

            // Compute preprocess
            let workgroups = (count + 255) / 256;
            self.preprocess.dispatch_stage_with_workgroups(&mut encoder, 0, [workgroups, 1, 1]);

            // GPU Radix Sort
            self.sorter.sort(&mut encoder, count);

            // Split submission: submit preprocess+sort, start new encoder for render
            encoder = core.flush_encoder(encoder);

            // Fragment render
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Gaussian Render"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
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
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                ..Default::default()
            });
        }

        self.base.handle_render_output(core, &view, full_output, &mut encoder);
        core.queue.submit(Some(encoder.finish()));
        output.present();
        Ok(())
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        if self.base.egui_state.on_window_event(core.window(), event).consumed {
            return true;
        }

        if let WindowEvent::KeyboardInput { event, .. } = event {
            if self.base.key_handler.handle_keyboard_input(core.window(), event) {
                return true;
            }
            if let winit::keyboard::Key::Character(ch) = &event.logical_key {
                let key = ch.as_str().to_lowercase();
                match event.state {
                    winit::event::ElementState::Pressed => {
                        if key == "r" {
                            self.camera.reset();
                            self.sorter.force_sort();
                            return true;
                        }
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
            if *button == winit::event::MouseButton::Left {
                self.camera.is_dragging = *state == winit::event::ElementState::Pressed;
                return true;
            }
        }

        if let WindowEvent::CursorMoved { position, .. } = event {
            let x = position.x as f32;
            let y = position.y as f32;
            if self.camera.is_dragging {
                let dx = x - self.camera.last_mouse[0];
                let dy = y - self.camera.last_mouse[1];
                self.camera.yaw += dx * 0.01;
                self.camera.pitch = (self.camera.pitch + dy * 0.01).clamp(-1.5, 1.5);
            }
            self.camera.last_mouse = [x, y];
            return self.camera.is_dragging;
        }

        if let WindowEvent::MouseWheel { delta, .. } = event {
            let d = match delta {
                winit::event::MouseScrollDelta::LineDelta(_, y) => *y,
                winit::event::MouseScrollDelta::PixelDelta(p) => (p.y as f32 / 100.0).clamp(-3.0, 3.0),
            };
            let factor = (1.0 + d * 0.1).clamp(0.5, 2.0);
            self.camera.distance = (self.camera.distance * factor).clamp(0.1, 500.0);
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

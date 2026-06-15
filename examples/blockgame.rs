// Block Game, Enes Altun, 2025, MIT License

use cuneus::audio::PcmStreamManager;
use cuneus::compute::*;
use cuneus::prelude::*;
use log::error;
use std::collections::HashSet;
use winit::event::ElementState;
use winit::keyboard::KeyCode;

const SAMPLE_RATE: u32 = 44100;
const MAX_SAMPLES_PER_FRAME: u32 = 1024;

cuneus::uniform_params! {
    struct GameUniform {
        camera_height: f32,
        camera_angle: f32,
        camera_scale: f32,
        volume: f32,
        sample_offset: u32,
        samples_to_generate: u32,
        sample_rate: f32,
        _pad: f32,
    }
}

impl Default for GameUniform {
    fn default() -> Self {
        Self {
            camera_height: 0.0,
            camera_angle: 0.0,
            camera_scale: 65.0,
            volume: 0.5,
            sample_offset: 0,
            samples_to_generate: 0,
            sample_rate: SAMPLE_RATE as f32,
            _pad: 0.0,
        }
    }
}

struct BlockTowerGame {
    base: RenderKit,
    compute_shader: ComputeShader,
    game: GameUniform,
    held_keys: HashSet<KeyCode>,
    pcm_stream: Option<PcmStreamManager>,
    audio_start: std::time::Instant,
    last_samples_generated: u32,
}

impl ShaderManager for BlockTowerGame {
    fn init(core: &Core) -> Self {
        let base = RenderKit::new(core);

        // Game state buffer: blocks (50 * 10 floats from index 100) + header. 1024 floats is plenty.
        let state_buffer_size = (1024 * std::mem::size_of::<f32>()) as u64;
        // Audio sample buffer: interleaved stereo f32 -> 2x samples.
        let audio_buffer_size = (MAX_SAMPLES_PER_FRAME * 2) as usize;

        let config = ComputeShader::builder()
            .with_entry_point("main")
            .with_custom_uniforms::<GameUniform>()
            .with_mouse()
            .with_fonts()
            .with_audio(audio_buffer_size)
            .with_storage_buffer(StorageBufferSpec::new("game_state", state_buffer_size))
            .with_workgroup_size([8, 8, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Block Tower Game")
            .build();

        let compute_shader = cuneus::compute_shader!(core, "shaders/blockgame.wgsl", config);

        let game = GameUniform::default();
        compute_shader.set_custom_params(game, &core.queue);

        // Real-time PCM stream for the synthesized sound effects.
        let pcm_stream = match PcmStreamManager::new(Some(SAMPLE_RATE)) {
            Ok(mut stream) => match stream.start() {
                Ok(()) => Some(stream),
                Err(e) => {
                    error!("Failed to start PCM stream: {e}");
                    None
                }
            },
            Err(e) => {
                error!("Failed to create PCM stream: {e}");
                None
            }
        };

        Self {
            base,
            compute_shader,
            game,
            held_keys: HashSet::new(),
            pcm_stream,
            audio_start: std::time::Instant::now(),
            last_samples_generated: 0,
        }
    }

    fn update(&mut self, core: &Core) {
        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0 / 60.0;
        self.compute_shader
            .set_time(current_time, delta, &core.queue);
        self.compute_shader
            .update_mouse_uniform(&self.base.mouse_tracker.uniform, &core.queue);

        let h_speed = 8.0;
        let a_speed = 1.5;
        if self.held_keys.contains(&KeyCode::KeyQ) { self.game.camera_height += h_speed * delta; }
        if self.held_keys.contains(&KeyCode::KeyE) { self.game.camera_height -= h_speed * delta; }
        if self.held_keys.contains(&KeyCode::KeyW) { self.game.camera_angle += a_speed * delta; }
        if self.held_keys.contains(&KeyCode::KeyS) { self.game.camera_angle -= a_speed * delta; }

        // Audio: push last frame's samples, then ask the shader for this frame's chunk.
        if let Some(ref mut stream) = self.pcm_stream {
            let prev = self.last_samples_generated;
            if prev > 0 {
                if let Ok(audio_data) = pollster::block_on(
                    self.compute_shader.read_audio_buffer(&core.device, &core.queue),
                ) {
                    let count = (prev * 2) as usize;
                    if audio_data.len() >= count {
                        let _ = stream.push_samples(&audio_data[..count]);
                    }
                }
            }

            let elapsed = self.audio_start.elapsed().as_secs_f64();
            let target = (elapsed * SAMPLE_RATE as f64) as u64;
            let written = stream.samples_written();
            let needed = (target.saturating_sub(written) as u32).min(MAX_SAMPLES_PER_FRAME);
            self.game.sample_offset = written as u32;
            self.game.samples_to_generate = needed;
            self.last_samples_generated = needed;
        }

        self.compute_shader.set_custom_params(self.game, &core.queue);
    }

    fn resize(&mut self, core: &Core) {
        self.base.default_resize(core, &mut self.compute_shader);
    }

    fn render(&mut self, core: &Core) -> Result<(), cuneus::SurfaceError> {
        let mut frame = self.base.begin_frame(core)?;
        let _controls_request = self
            .base
            .controls
            .get_ui_request(&self.base.start_time, &core.size, self.base.fps_tracker.fps());

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                RenderKit::apply_default_style(ctx);
                egui::Window::new("Block Tower")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(220.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("Camera")
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.add(egui::Slider::new(&mut self.game.camera_height, 0.0..=20.0).text("Height"));
                                ui.add(egui::Slider::new(&mut self.game.camera_angle, -3.14159..=3.14159).text("Angle"));
                                ui.add(egui::Slider::new(&mut self.game.camera_scale, 20.0..=200.0).text("Scale"));

                                ui.separator();
                                ui.add(egui::Slider::new(&mut self.game.volume, 0.0..=1.0).text("Volume"));

                                ui.separator();
                                ui.label("Controls:");
                                ui.label("Click: drop block");
                                ui.label("Q/E: Move up/down");
                                ui.label("W/S: Rotate left/right");

                                ui.separator();
                                ui.horizontal(|ui| {
                                    if ui.button("1080p").clicked() { self.game.camera_scale = 50.0; }
                                    if ui.button("1440p").clicked() { self.game.camera_scale = 65.0; }
                                    if ui.button("4K").clicked() { self.game.camera_scale = 100.0; }
                                });

                                if ui.button("Reset Camera").clicked() {
                                    self.game.camera_height = 8.0;
                                    self.game.camera_angle = 0.0;
                                    self.game.camera_scale = 65.0;
                                }
                            });
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        self.compute_shader.dispatch(&mut frame.encoder, core);

        self.base.renderer.render_to_view(&mut frame.encoder, &frame.view, &self.compute_shader.get_output_texture().bind_group);

        self.base.end_frame(core, frame, full_output);
        Ok(())
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        let ui_handled = self.base.forward_to_egui(core, event);

        if self.base.handle_mouse_input(core, event, ui_handled) {
            return true;
        }

        if let WindowEvent::KeyboardInput { event, .. } = event {
            if let winit::keyboard::PhysicalKey::Code(key_code) = event.physical_key {
                // Track held state for camera keys; movement is applied per-frame in update().
                match key_code {
                    KeyCode::KeyQ | KeyCode::KeyE | KeyCode::KeyW | KeyCode::KeyS => {
                        match event.state {
                            ElementState::Pressed => { self.held_keys.insert(key_code); }
                            ElementState::Released => { self.held_keys.remove(&key_code); }
                        }
                        return true;
                    }
                    _ => {}
                }
            }
            return self
                .base
                .key_handler
                .handle_keyboard_input(core.window(), event);
        }

        false
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    cuneus::gst::init()?;

    let (app, event_loop) = ShaderApp::new("Block Tower Game", 600, 800);

    app.run(event_loop, BlockTowerGame::init)
}

// Block Game, Enes Altun, 2025, MIT License

use cuneus::audio::PcmStreamManager;
use cuneus::compute::*;
use cuneus::prelude::*;
use log::error;
use std::collections::HashSet;
use winit::event::ElementState;
use winit::keyboard::KeyCode;

const SAMPLE_RATE: u32 = 44100;
// enough audio for a frame even at low frame rates
const MAX_SAMPLES_PER_FRAME: u32 = 4096;

cuneus::uniform_params! {
    struct GameUniform {
        camera_height: f32,
        camera_angle: f32,
        camera_scale: f32,
        volume: f32,
        sample_offset: u32,
        samples_to_generate: u32,
        sample_rate: f32,
        action: f32,
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
            action: 0.0,
        }
    }
}

struct BlockTowerGame {
    base: RenderKit,
    compute_shader: ComputeShader,
    game: GameUniform,
    held_keys: HashSet<KeyCode>,
    pcm_stream: Option<PcmStreamManager>,
    last_samples_generated: u32,
}

impl ShaderManager for BlockTowerGame {
    fn init(core: &Core) -> Self {
        let base = RenderKit::new(core);

        // game state (header, 50 blocks, 8 offcuts), fixed size so a resize keeps it
        let state_bytes = (1024 * std::mem::size_of::<u32>()) as u64;
        // Audio sample buffer: interleaved stereo f32 -> 2x samples.
        let audio_buffer_size = (MAX_SAMPLES_PER_FRAME * 2) as usize;
        let passes = vec![
            PassDescription::new("sim", &[]).with_resolution(1, 1).with_workgroup_size([1, 1, 1]),
            PassDescription::new("scene", &[]),
            PassDescription::new("bd1", &["scene"]).with_resolution_scale(0.5),
            PassDescription::new("bd2", &["bd1"]).with_resolution_scale(0.25),
            PassDescription::new("bd3", &["bd2"]).with_resolution_scale(0.125),
            PassDescription::new("bd4", &["bd3"]).with_resolution_scale(0.0625),
            PassDescription::new("bu3", &["bd3", "bd4"]).with_resolution_scale(0.125),
            PassDescription::new("bu2", &["bd2", "bu3"]).with_resolution_scale(0.25),
            PassDescription::new("bu1", &["bd1", "bu2"]).with_resolution_scale(0.5),
            PassDescription::new("db1", &["scene"]).with_resolution_scale(0.5),
            PassDescription::new("db2", &["db1"]).with_resolution_scale(0.25),
            PassDescription::new("du1", &["db1", "db2"]).with_resolution_scale(0.5),
            PassDescription::new("main_image", &["scene", "bu1", "du1"]),
        ];

        let config = ComputeShader::builder()
            .with_entry_point("sim")
            .with_multi_pass(&passes)
            .with_custom_uniforms::<GameUniform>()
            .with_mouse()
            .with_fonts()
            .with_audio(audio_buffer_size)
            .with_atomic_buffer_bytes(state_bytes)
            .with_workgroup_size([16, 16, 1])
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
        self.game.action = if self.held_keys.contains(&KeyCode::Space) { 1.0 } else { 0.0 };

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

            let (written, needed) = stream.next_block(MAX_SAMPLES_PER_FRAME);
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

        let full_output = self.base.render_ui(core, |_ctx| {});

        self.compute_shader.dispatch(&mut frame.encoder, core);

        self.base.renderer.render_to_view(&mut frame.encoder, &frame.view, &self.compute_shader.get_output_texture().bind_group);

        self.base.end_frame(core, frame, full_output);
        Ok(())
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        let ui_handled = self.base.forward_to_egui(core, event);

        if let WindowEvent::MouseWheel { delta, .. } = event {
            if !ui_handled {
                let dy = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => *y,
                    winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                };
                self.game.camera_scale = (self.game.camera_scale + dy * 5.0).clamp(20.0, 200.0);
            }
            return true;
        }

        if self.base.handle_mouse_input(core, event, ui_handled) {
            return true;
        }

        if let WindowEvent::KeyboardInput { event, .. } = event {
            if let winit::keyboard::PhysicalKey::Code(key_code) = event.physical_key {
                // Track held state for camera keys; movement is applied per-frame in update().
                match key_code {
                    KeyCode::KeyR => {
                        let d = GameUniform::default();
                        self.game.camera_height = d.camera_height;
                        self.game.camera_angle = d.camera_angle;
                        self.game.camera_scale = d.camera_scale;
                        return true;
                    }
                    KeyCode::KeyQ | KeyCode::KeyE | KeyCode::KeyW | KeyCode::KeyS | KeyCode::Space => {
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

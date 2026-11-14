// Bugberman, Enes Altun, 2026, CC0

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
        move_dir: u32,
        action_held: u32,
        volume: f32,
        sample_offset: u32,
        samples_to_generate: u32,
        sample_rate: f32,
        tilt_shift: f32,
        bloom: f32,
    }
}

impl Default for GameUniform {
    fn default() -> Self {
        Self {
            move_dir: 0,
            action_held: 0,
            volume: 0.5,
            sample_offset: 0,
            samples_to_generate: 0,
            sample_rate: SAMPLE_RATE as f32,
            tilt_shift: 1.0,
            bloom: 1.0,
        }
    }
}

struct Bugberman {
    base: RenderKit,
    compute_shader: ComputeShader,
    game: GameUniform,
    held_keys: HashSet<KeyCode>,
    pcm_stream: Option<PcmStreamManager>,
    last_samples_generated: u32,
}

impl Bugberman {
    fn current_dir(&self) -> u32 {
        let h = &self.held_keys;
        if h.contains(&KeyCode::ArrowUp) || h.contains(&KeyCode::KeyW) { 1 }
        else if h.contains(&KeyCode::ArrowDown) || h.contains(&KeyCode::KeyS) { 2 }
        else if h.contains(&KeyCode::ArrowLeft) || h.contains(&KeyCode::KeyA) { 3 }
        else if h.contains(&KeyCode::ArrowRight) || h.contains(&KeyCode::KeyD) { 4 }
        else { 0 }
    }
}

impl ShaderManager for Bugberman {
    fn init(core: &Core) -> Self {
        let base = RenderKit::new(core);

        // game state (header, 13x11 tiles and flames, bombs, bugs), fixed size so a resize keeps it
        let state_bytes = (2048 * std::mem::size_of::<u32>()) as u64;
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
            .with_fonts()
            .with_audio(audio_buffer_size)
            .with_atomic_buffer_bytes(state_bytes)
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Bugberman")
            .build();

        let compute_shader = cuneus::compute_shader!(core, "shaders/bugberman.wgsl", config);

        let game = GameUniform::default();
        compute_shader.set_custom_params(game, &core.queue);

        let pcm_stream = match PcmStreamManager::new(Some(SAMPLE_RATE)) {
            Ok(mut stream) => match stream.start() {
                Ok(()) => Some(stream),
                Err(e) => { error!("Failed to start PCM stream: {e}"); None }
            },
            Err(e) => { error!("Failed to create PCM stream: {e}"); None }
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
        self.compute_shader.set_time(current_time, delta, &core.queue);

        self.game.move_dir = self.current_dir();
        self.game.action_held = if self.held_keys.contains(&KeyCode::Space) { 1 } else { 0 };

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
        let _ui_handled = self.base.forward_to_egui(core, event);

        if let WindowEvent::KeyboardInput { event, .. } = event {
            if let winit::keyboard::PhysicalKey::Code(key_code) = event.physical_key {
                match key_code {
                    KeyCode::ArrowUp | KeyCode::ArrowDown | KeyCode::ArrowLeft | KeyCode::ArrowRight
                    | KeyCode::KeyW | KeyCode::KeyA | KeyCode::KeyS | KeyCode::KeyD
                    | KeyCode::Space => {
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

    let (app, event_loop) = ShaderApp::new("Bugberman", 832, 768);

    app.run(event_loop, Bugberman::init)
}

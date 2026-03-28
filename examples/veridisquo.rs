use cuneus::audio::PcmStreamManager;
use cuneus::compute::*;
use cuneus::prelude::*;
use log::{error, info};

const MAX_SAMPLES_PER_FRAME: u32 = 1024;
const SAMPLE_RATE: u32 = 44100;

cuneus::uniform_params! {
    struct SongParams {
        volume: f32,
        tempo_multiplier: f32,
        sample_offset: u32,
        samples_to_generate: u32,
        sample_rate: f32,
        _pad1: f32,
        _pad2: f32,
        _pad3: f32,
    }
}

struct VeridisQuo {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: SongParams,
    pcm_stream: Option<PcmStreamManager>,
    audio_start: std::time::Instant,
    last_samples_generated: u32,
}

impl ShaderManager for VeridisQuo {
    fn init(core: &Core) -> Self {
        let base = RenderKit::new(core);

        let initial_params = SongParams {
            volume: 0.5,
            tempo_multiplier: 1.0,
            sample_offset: 0,
            samples_to_generate: MAX_SAMPLES_PER_FRAME,
            sample_rate: SAMPLE_RATE as f32,
            _pad1: 0.0,
            _pad2: 0.0,
            _pad3: 0.0,
        };

        // Audio buffer: interleaved stereo f32 → need 2x samples
        let audio_buffer_size = (MAX_SAMPLES_PER_FRAME * 2) as usize;

        let config = ComputeShader::builder()
            .with_entry_point("main")
            .with_custom_uniforms::<SongParams>()
            .with_fonts()
            .with_audio(audio_buffer_size)
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Veridis Quo")
            .build();

        let compute_shader = cuneus::compute_shader!(core, "shaders/veridisquo.wgsl", config);
        compute_shader.set_custom_params(initial_params, &core.queue);

        let pcm_stream = match PcmStreamManager::new(Some(SAMPLE_RATE)) {
            Ok(mut stream) => {
                if let Err(e) = stream.start() {
                    error!("Failed to start PCM stream: {e}");
                    None
                } else {
                    info!("PCM audio stream started at {SAMPLE_RATE} Hz");
                    Some(stream)
                }
            }
            Err(e) => {
                error!("Failed to create PCM stream: {e}");
                None
            }
        };

        Self {
            base,
            compute_shader,
            current_params: initial_params,
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

        if let Some(ref mut stream) = self.pcm_stream {
            // Push previous frame's audio
            let prev = self.last_samples_generated;
            if prev > 0 {
                if let Ok(audio_data) = pollster::block_on(
                    self.compute_shader
                        .read_audio_buffer(&core.device, &core.queue),
                ) {
                    let count = (prev * 2) as usize;
                    if audio_data.len() >= count {
                        let _ = stream.push_samples(&audio_data[..count]);
                    }
                }
            }

            // Calculate this frame's needs
            let elapsed = self.audio_start.elapsed().as_secs_f64();
            let target_samples = (elapsed * SAMPLE_RATE as f64) as u64;
            let written = stream.samples_written();
            let needed = (target_samples.saturating_sub(written) as u32).min(MAX_SAMPLES_PER_FRAME);
            self.current_params.sample_offset = written as u32;
            self.current_params.samples_to_generate = needed;
            self.last_samples_generated = needed;
        }
        self.compute_shader
            .set_custom_params(self.current_params, &core.queue);

        self.compute_shader.handle_export(core, &mut self.base);
    }

    fn render(&mut self, core: &Core) -> Result<(), cuneus::SurfaceError> {
        let mut frame = self.base.begin_frame(core)?;

        let mut params = self.current_params;
        let mut changed = false;
        let mut controls_request = self
            .base
            .controls
            .get_ui_request(&self.base.start_time, &core.size, self.base.fps_tracker.fps());

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                RenderKit::apply_default_style(ctx);

                egui::Window::new("Veridis Quo")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(250.0)
                    .show(ctx, |ui| {
                        changed |= ui
                            .add(
                                egui::Slider::new(&mut params.volume, 0.0..=1.0).text("Volume"),
                            )
                            .changed();
                        changed |= ui
                            .add(
                                egui::Slider::new(&mut params.tempo_multiplier, 0.5..=2.0)
                                    .text("Tempo"),
                            )
                            .changed();

                        if let Some(ref mut stream) = self.pcm_stream {
                            let mut vol = params.volume as f64;
                            if ui
                                .add(
                                    egui::Slider::new(&mut vol, 0.0..=1.0)
                                        .text("Master Volume"),
                                )
                                .changed()
                            {
                                stream.set_master_volume(vol);
                            }
                        }

                        ui.separator();
                        ShaderControls::render_controls_widget(ui, &mut controls_request);
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        if changed {
            self.current_params = params;
        }

        self.base.apply_control_request(controls_request);

        self.compute_shader.dispatch(&mut frame.encoder, core);

        self.base.renderer.render_to_view(
            &mut frame.encoder,
            &frame.view,
            &self.compute_shader.get_output_texture().bind_group,
        );

        self.base.end_frame(core, frame, full_output);

        Ok(())
    }

    fn resize(&mut self, core: &Core) {
        self.base.default_resize(core, &mut self.compute_shader);
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        if self
            .base
            .egui_state
            .on_window_event(core.window(), event)
            .consumed
        {
            return true;
        }

        if let WindowEvent::KeyboardInput { event, .. } = event {
            if event.state == winit::event::ElementState::Pressed {
                if let winit::keyboard::Key::Character(ref s) = event.logical_key {
                    if s.as_str() == "r" || s.as_str() == "R" {
                        self.base.start_time = std::time::Instant::now();
                        // Reset audio stream
                        if let Some(ref mut stream) = self.pcm_stream {
                            let _ = stream.stop();
                            let _ = stream.start();
                        }
                        return true;
                    }
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

    let (app, event_loop) = ShaderApp::new("Veridis Quo", 800, 600);

    app.run(event_loop, VeridisQuo::init)
}

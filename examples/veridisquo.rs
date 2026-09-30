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
        mix_drums: f32,
        mix_bass: f32,
        mix_lead: f32,
        mix_guitar: f32,
        mix_pads: f32,
        mix_echo: f32,
        mix_space: f32,
        lead_type: f32,
        lead_tone: f32,
        lead_detune: f32,
        lead_glide: f32,
        bass_type: f32,
        bass_tone: f32,
        bass_drive: f32,
        pad_type: f32,
        pad_tone: f32,
        pad_width: f32,
        guitar_tone: f32,
        guitar_mute: f32,
        kick_tune: f32,
        kick_decay: f32,
        hat_decay: f32,
        drum_pattern: f32,
        echo_time: f32,
        echo_feedback: f32,
        swing: f32,
        play_sample: u32,
        song_origin: u32,
        _pad0: u32,
        _pad1: u32,
        _pad2: u32,
    }
}

struct VeridisQuo {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: SongParams,
    pcm_stream: Option<PcmStreamManager>,
    last_samples_generated: u32,
}

impl VeridisQuo {
    /// Back to bar one without touching the stream: the next block the GPU writes starts the song
    fn restart_song(&mut self) {
        self.current_params.song_origin = self.current_params.sample_offset + self.last_samples_generated;
    }
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
            mix_drums: 1.0,
            mix_bass: 1.0,
            mix_lead: 1.0,
            mix_guitar: 1.0,
            mix_pads: 1.0,
            mix_echo: 1.0,
            mix_space: 1.0,
            lead_type: 0.0,
            lead_tone: 1.0,
            lead_detune: 6.0,
            lead_glide: 0.0,
            bass_type: 0.0,
            bass_tone: 1.0,
            bass_drive: 1.3,
            pad_type: 0.0,
            pad_tone: 1.0,
            pad_width: 7.0,
            guitar_tone: 1.0,
            guitar_mute: 1.0,
            kick_tune: 1.0,
            kick_decay: 0.3,
            hat_decay: 1.0,
            drum_pattern: 0.0,
            echo_time: 0.15,
            echo_feedback: 1.0,
            swing: 0.0,
            play_sample: 0,
            song_origin: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
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
            let (written, needed) = stream.next_block(MAX_SAMPLES_PER_FRAME);
            self.current_params.play_sample = stream.playback_sample() as u32;
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

                        egui::CollapsingHeader::new("Mix").default_open(true).show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.mix_drums, 0.0..=2.0).text("Drums")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.mix_bass, 0.0..=2.0).text("Bass")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.mix_lead, 0.0..=2.0).text("Lead")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.mix_guitar, 0.0..=2.0).text("Guitar")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.mix_pads, 0.0..=2.0).text("Pads")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.mix_echo, 0.0..=2.0).text("Echo")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.mix_space, 0.0..=2.0).text("Space")).changed();
                        });
                        egui::CollapsingHeader::new("Lead").show(ui, |ui| {
                            changed |= kind(ui, &mut params.lead_type, &["Soft saw", "Organ", "Keys"]);
                            changed |= ui.add(egui::Slider::new(&mut params.lead_tone, 0.2..=3.0).text("Brightness")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.lead_detune, 0.0..=30.0).text("Detune").suffix(" ct")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.lead_glide, 0.0..=0.1).text("Glide").suffix(" s")).changed();
                        });
                        egui::CollapsingHeader::new("Bass").show(ui, |ui| {
                            changed |= kind(ui, &mut params.bass_type, &["Moog pluck", "Sub", "FM"]);
                            changed |= ui.add(egui::Slider::new(&mut params.bass_tone, 0.0..=3.0).text("Bite")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.bass_drive, 0.5..=4.0).text("Drive")).changed();
                        });
                        egui::CollapsingHeader::new("Pads").show(ui, |ui| {
                            changed |= kind(ui, &mut params.pad_type, &["Warm saw", "Organ", "Strings"]);
                            changed |= ui.add(egui::Slider::new(&mut params.pad_tone, 0.3..=3.0).text("Brightness")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.pad_width, 0.0..=25.0).text("Width").suffix(" ct")).changed();
                        });
                        egui::CollapsingHeader::new("Guitar").show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.guitar_tone, 0.3..=3.0).text("Brightness")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.guitar_mute, 0.3..=4.0).text("Ring")).changed();
                        });
                        egui::CollapsingHeader::new("Drums").show(ui, |ui| {
                            changed |= kind(ui, &mut params.drum_pattern, &["Full kit", "No clap", "Kick only"]);
                            changed |= ui.add(egui::Slider::new(&mut params.kick_tune, 0.6..=1.6).text("Kick tune")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.kick_decay, 0.1..=0.8).text("Kick decay")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.hat_decay, 0.3..=3.0).text("Hat decay")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.swing, 0.0..=0.6).text("Swing")).changed();
                        });
                        egui::CollapsingHeader::new("Echo").show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.echo_time, 0.05..=0.6).text("Time").suffix(" s")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.echo_feedback, 0.0..=1.5).text("Feedback")).changed();
                        });

                        ui.separator();
                        ShaderControls::render_controls_widget(ui, &mut controls_request);
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        if changed {
            params.play_sample = self.current_params.play_sample;
            params.song_origin = self.current_params.song_origin;
            self.current_params = params;
        }

        if controls_request.should_reset {
            self.restart_song();
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
        if self.base.forward_to_egui(core, event) {
            return true;
        }

        if let WindowEvent::KeyboardInput { event, .. } = event {
            if event.state == winit::event::ElementState::Pressed {
                if let winit::keyboard::Key::Character(ref s) = event.logical_key {
                    if s.as_str() == "r" || s.as_str() == "R" {
                        self.base.start_time = std::time::Instant::now();
                        self.restart_song();
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

// a stepped slider that names each position
fn kind(ui: &mut egui::Ui, v: &mut f32, names: &'static [&'static str]) -> bool {
    let last = names.len() - 1;
    ui.add(
        egui::Slider::new(v, 0.0..=last as f32)
            .step_by(1.0)
            .text("Type")
            .custom_formatter(move |x, _| names[(x as usize).min(last)].to_string()),
    )
    .changed()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    cuneus::gst::init()?;

    let (app, event_loop) = ShaderApp::new("Veridis Quo", 800, 600);

    app.run(event_loop, VeridisQuo::init)
}

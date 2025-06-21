// An example of a simple audio creation using Cuneus
use cuneus::{Core, ShaderApp, ShaderManager, RenderKit, ShaderControls, UniformBinding};
use cuneus::compute::{ComputeShaderConfig, COMPUTE_TEXTURE_FORMAT_RGBA16};
use cuneus::gst::synthesis::{AudioSynthManager, MusicalNote, AudioSynthUniform};
use winit::event::*;
use winit::keyboard::{KeyCode, PhysicalKey};
use log::info;
use std::collections::HashSet;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    cuneus::gst::init()?;
    env_logger::init();
    
    let (app, event_loop) = ShaderApp::new("Audio Synthesizer", 1200, 800);
    app.run(event_loop, |core| {
        SynthManager::init(core)
    })
}

struct SynthManager {
    base: RenderKit,
    audio_synth: AudioSynthManager,
    audio_synth_uniform: UniformBinding<AudioSynthUniform>,
    pressed_keys: HashSet<KeyCode>,
    pressed_notes: HashSet<MusicalNote>,
    master_volume: f64,
    current_waveform: cuneus::gst::synthesis::AudioWaveform,
}

impl SynthManager {
    fn handle_key_press(&mut self, keycode: KeyCode) -> bool {
        if self.pressed_keys.contains(&keycode) {
            return false;
        }
        
        self.pressed_keys.insert(keycode);
        
        let note = match keycode {
            KeyCode::Digit1 => Some(MusicalNote::C4),
            KeyCode::Digit2 => Some(MusicalNote::D4),
            KeyCode::Digit3 => Some(MusicalNote::E4),
            KeyCode::Digit4 => Some(MusicalNote::F4),
            KeyCode::Digit5 => Some(MusicalNote::G4),
            KeyCode::Digit6 => Some(MusicalNote::A4),
            KeyCode::Digit7 => Some(MusicalNote::B4),
            KeyCode::Digit8 => Some(MusicalNote::C5),
            KeyCode::Digit9 => Some(MusicalNote::CSharp4),
            KeyCode::Space => {
                if let Err(e) = self.audio_synth.stop_all_notes() {
                    eprintln!("Failed to stop all notes: {}", e);
                }
                self.pressed_notes.clear();
                return true;
            },
            KeyCode::ArrowUp => {
                // Volume up
                self.master_volume = (self.master_volume + 0.1).min(1.0);
                if let Err(e) = self.audio_synth.set_master_volume(self.master_volume) {
                    eprintln!("Failed to set master volume: {}", e);
                }
                info!("Master Volume: {:.1}%", self.master_volume * 100.0);
                return true;
            },
            KeyCode::ArrowDown => {
                // Volume down
                self.master_volume = (self.master_volume - 0.1).max(0.0);
                if let Err(e) = self.audio_synth.set_master_volume(self.master_volume) {
                    eprintln!("Failed to set master volume: {}", e);
                }
                info!("Master Volume: {:.1}%", self.master_volume * 100.0);
                return true;
            },
            _ => None,
        };
        
        if let Some(note) = note {
            if !self.pressed_notes.contains(&note) {
                self.pressed_notes.insert(note);
                if let Err(e) = self.audio_synth.play_note(note) {
                    eprintln!("Failed to play note: {}", e);
                } else {
                    info!("Playing note: {} ({:.2} Hz)", note.name(), note.to_frequency());
                }
            }
            return true;
        }
        
        false
    }
    
    fn handle_key_release(&mut self, keycode: KeyCode) -> bool {
        if !self.pressed_keys.contains(&keycode) {
            return false;
        }
        
        self.pressed_keys.remove(&keycode);
        
        // Stop specific note when releasing note keys
        let note = match keycode {
            KeyCode::Digit1 => Some(MusicalNote::C4),
            KeyCode::Digit2 => Some(MusicalNote::D4),
            KeyCode::Digit3 => Some(MusicalNote::E4),
            KeyCode::Digit4 => Some(MusicalNote::F4),
            KeyCode::Digit5 => Some(MusicalNote::G4),
            KeyCode::Digit6 => Some(MusicalNote::A4),
            KeyCode::Digit7 => Some(MusicalNote::B4),
            KeyCode::Digit8 => Some(MusicalNote::C5),
            KeyCode::Digit9 => Some(MusicalNote::CSharp4),
            _ => None,
        };
        
        if let Some(note) = note {
            if self.pressed_notes.contains(&note) {
                self.pressed_notes.remove(&note);
                if let Err(e) = self.audio_synth.stop_note(note) {
                    eprintln!("Failed to stop note: {}", e);
                } else {
                    info!("Stopped note: {}", note.name());
                }
            }
            return true;
        }
        
        false
    }
    
    fn update_audio_visualization(&mut self, queue: &wgpu::Queue) {
        // Get current active notes and master volume
        let active_notes = self.audio_synth.active_notes();
        let master_volume = self.audio_synth.master_volume();
        

        self.audio_synth_uniform.data.update_from_synthesis(
            &active_notes, 
            master_volume, 
            self.current_waveform
        );
        self.audio_synth_uniform.update(queue);
    }
}

impl ShaderManager for SynthManager {
    fn init(core: &Core) -> Self {
        info!("Initializing audio synthesizer");
        
        // init audio synth
        let mut audio_synth = AudioSynthManager::new(Some(44100))
            .expect("Failed to create audio synthesis manager");
        
        if let Err(e) = audio_synth.start() {
            eprintln!("Failed to start audio synthesis: {}", e);
        }
        
        if let Err(e) = audio_synth.set_master_volume(0.3) {
            eprintln!("Failed to set initial master volume: {}", e);
        }
        
        let texture_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Texture Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        
        let mut base = RenderKit::new(
            core,
            include_str!("../../shaders/vertex.wgsl"),
            include_str!("../../shaders/blit.wgsl"),
            &[&texture_bind_group_layout],
            None,
        );
        
        let audio_synth_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("audio_synth_bind_group_layout"),
        });
        
        // Create audio synthesis uniform
        let audio_synth_uniform = UniformBinding::new(
            &core.device,
            "Audio Synthesis Uniform",
            AudioSynthUniform::new(),
            &audio_synth_bind_group_layout,
            0,
        );

        let config = ComputeShaderConfig {
            workgroup_size: [16, 16, 1],
            workgroup_count: None,
            dispatch_once: false,
            storage_texture_format: COMPUTE_TEXTURE_FORMAT_RGBA16,
            enable_atomic_buffer: false,
            atomic_buffer_multiples: 4,
            entry_points: vec!["main".to_string()],
            sampler_address_mode: wgpu::AddressMode::ClampToEdge,
            sampler_filter_mode: wgpu::FilterMode::Linear,
            label: "Audio Synthesizer Compute".to_string(),
            mouse_bind_group_layout: Some(audio_synth_bind_group_layout.clone()),
            enable_fonts: false,
        };
        
        base.compute_shader = Some(cuneus::compute::ComputeShader::new_with_config(
            core,
            include_str!("../../shaders/synth.wgsl"),
            config,
        ));
        
        if let Some(compute_shader) = &mut base.compute_shader {
            // Add the audio synthesis uniform binding as "mouse" uniform (bind group 2)
            compute_shader.add_mouse_uniform_binding(&audio_synth_uniform.bind_group, 2);
        }
        
        //hot reload
        if let Some(compute_shader) = &mut base.compute_shader {
            let shader_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Synth Compute Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/synth.wgsl").into()),
            });
            if let Err(e) = compute_shader.enable_hot_reload(
                core.device.clone(),
                PathBuf::from("shaders/synth.wgsl"),
                shader_module,
            ) {
                eprintln!("Failed to enable compute shader hot reload: {}", e);
            }
        }
        
        
        info!("Audio synthesizer initialized successfully");
        info!("Controls:");
        info!("  Keys 1-9: Play musical notes");
        info!("  Space: Stop current note");
        info!("  Arrow Up/Down: Volume up/down");
        info!("  H: Toggle UI");
        
        Self {
            base,
            audio_synth,
            audio_synth_uniform,
            pressed_keys: HashSet::new(),
            pressed_notes: HashSet::new(),
            master_volume: 0.3,
            current_waveform: cuneus::gst::synthesis::AudioWaveform::Sine,
        }
    }
    
    fn update(&mut self, core: &Core) {
        self.base.fps_tracker.update();
        self.audio_synth.update();
        
        self.update_audio_visualization(&core.queue);
        
        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0 / 60.0;
        self.base.update_compute_shader_time(current_time, delta, &core.queue);
    }
    
    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        let mut controls_request = self.base.controls.get_ui_request(
            &self.base.start_time,
            &core.size
        );
        controls_request.current_fps = Some(self.base.fps_tracker.fps());
        
        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Synth Render Encoder"),
        });
        
        self.base.dispatch_compute_shader(&mut encoder, core);
        
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            
            if let Some(compute_texture) = self.base.get_compute_output_texture() {
                render_pass.set_pipeline(&self.base.renderer.render_pipeline);
                render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
                render_pass.set_bind_group(0, &compute_texture.bind_group, &[]);
                render_pass.draw(0..4, 0..1);
            }
        }
        
        let active_notes = self.audio_synth.active_notes();
        let master_volume = self.master_volume;
        
        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                ctx.style_mut(|style| {
                    style.visuals.window_fill = egui::Color32::from_rgba_premultiplied(0, 0, 0, 180);
                    style.text_styles.get_mut(&egui::TextStyle::Body).unwrap().size = 11.0;
                    style.text_styles.get_mut(&egui::TextStyle::Button).unwrap().size = 10.0;
                });
                
                egui::Window::new("Audio Synthesizer")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(250.0)
                    .show(ctx, |ui| {
                        ui.heading("Audio Synthesizer");
                        ui.separator();
                        
                        ShaderControls::render_controls_widget(ui, &mut controls_request);
                        
                        ui.separator();
                        
                        ui.label("Controls:");
                        ui.label("• Keys 1-9: Play musical notes");
                        ui.label("• Space: Stop current note");
                        ui.label("• Arrow Up/Down: Volume up/down");
                        ui.label("• H: Toggle this UI");
                        
                        ui.separator();
                        
                        ui.label("Waveform:");
                        ui.horizontal(|ui| {
                            if ui.radio_value(&mut self.current_waveform, cuneus::gst::synthesis::AudioWaveform::Sine, "Sine").changed() {
                                let _ = self.audio_synth.set_waveform(cuneus::gst::synthesis::AudioWaveform::Sine);
                            }
                            if ui.radio_value(&mut self.current_waveform, cuneus::gst::synthesis::AudioWaveform::Square, "Square").changed() {
                                let _ = self.audio_synth.set_waveform(cuneus::gst::synthesis::AudioWaveform::Square);
                            }
                            if ui.radio_value(&mut self.current_waveform, cuneus::gst::synthesis::AudioWaveform::Saw, "Saw").changed() {
                                let _ = self.audio_synth.set_waveform(cuneus::gst::synthesis::AudioWaveform::Saw);
                            }
                            if ui.radio_value(&mut self.current_waveform, cuneus::gst::synthesis::AudioWaveform::Triangle, "Triangle").changed() {
                                let _ = self.audio_synth.set_waveform(cuneus::gst::synthesis::AudioWaveform::Triangle);
                            }
                        });
                        
                        ui.separator();
                        
                        if !active_notes.is_empty() {
                            ui.label("Active Notes:");
                            for note in &active_notes {
                                ui.label(format!("• {} ({:.2} Hz)", note.name(), note.to_frequency()));
                            }
                        } else {
                            ui.label("Active Notes: None");
                        }
                        
                        ui.separator();
                        
                        ui.label(format!("Master Volume: {:.1}%", master_volume * 100.0));
                        ui.label(format!("Active Voices: {}", active_notes.len()));
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };
        
        self.base.apply_control_request(controls_request);
        
        self.base.handle_render_output(core, &view, full_output, &mut encoder);
        core.queue.submit(Some(encoder.finish()));
        output.present();
        Ok(())
    }
    
    fn resize(&mut self, core: &Core) {
        self.base.update_resolution(&core.queue, core.size);
        self.base.resize_compute_shader(core);
    }
    
    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        let ui_handled = self.base.egui_state.on_window_event(core.window(), event).consumed;
        
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                if self.base.key_handler.handle_keyboard_input(core.window(), event) {
                    return true;
                }
                
                if !ui_handled {
                    if let PhysicalKey::Code(keycode) = event.physical_key {
                        match event.state {
                            ElementState::Pressed => {
                                return self.handle_key_press(keycode);
                            },
                            ElementState::Released => {
                                return self.handle_key_release(keycode);
                            },
                        }
                    }
                }
            },
            _ => {},
        }
        
        false
    }
}
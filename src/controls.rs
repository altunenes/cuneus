#[cfg(feature = "media")]
use crate::gst::video::VideoTextureManager;
use crate::hdri::HdriMetadata;
use std::path::PathBuf;
#[derive(Clone)]
pub struct ControlsRequest {
    pub is_paused: bool,
    pub should_reset: bool,
    pub should_clear_buffers: bool,
    pub current_time: Option<f32>,
    pub window_size: Option<(u32, u32)>,

    pub current_fps: Option<f32>,

    // Video reqs
    pub load_media_path: Option<PathBuf>,
    pub play_video: bool,
    pub pause_video: bool,
    pub restart_video: bool,
    pub seek_position: Option<f64>,
    pub set_loop: Option<bool>,

    // Audio reqs
    pub set_volume: Option<f64>,
    pub mute_audio: Option<bool>,
    pub toggle_mute: bool,

    // HDRI reqs
    pub hdri_exposure: Option<f32>,
    pub hdri_gamma: Option<f32>,

    // Webcam reqs
    pub start_webcam: bool,
    pub stop_webcam: bool,
    pub webcam_device_index: Option<u32>,
}
impl Default for ControlsRequest {
    fn default() -> Self {
        let mut default_media = None;
        let mut should_play_video = false;
        if let Ok(media_dir) = std::env::var("CUNEUS_MEDIA") {
            println!("CUNEUS_MEDIA: {}", media_dir);
            if media_dir.starts_with('"') && media_dir.ends_with('"') {
                let unquoted = &media_dir[1..media_dir.len() - 1];
                default_media = Some(PathBuf::from(unquoted));
            } else {
                default_media = Some(PathBuf::from(media_dir));
            }
            should_play_video = true;
        }
        Self {
            is_paused: false,
            should_reset: false,
            should_clear_buffers: false,
            current_time: None,
            window_size: None,

            current_fps: None,

            // Video-related stuff
            load_media_path: default_media,
            play_video: should_play_video,
            pause_video: false,
            restart_video: false,
            seek_position: None,
            set_loop: None,

            // Audio-related stuff
            set_volume: None,
            mute_audio: None,
            toggle_mute: false,

            // HDRI-related stuff
            hdri_exposure: None,
            hdri_gamma: None,

            // Webcam-related stuff
            start_webcam: false,
            stop_webcam: false,
            webcam_device_index: None,
        }
    }
}

/// VideoInfo type alias
/// (duration, position, dimensions, framerate, is_looping, has_audio, volume, is_muted)
pub type VideoInfo = (
    Option<f32>,
    f32,
    (u32, u32),
    Option<f32>,
    bool,
    bool,
    f64,
    bool,
);

pub struct ShaderControls {
    is_paused: bool,
    pause_start: Option<std::time::Instant>,
    total_pause_duration: f32,
    current_frame: u32,
    media_loaded_once: bool,
}

impl Default for ShaderControls {
    fn default() -> Self {
        Self {
            is_paused: false,
            pause_start: None,
            total_pause_duration: 0.0,
            current_frame: 0,
            media_loaded_once: false,
        }
    }
}

impl ShaderControls {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_frame(&mut self) -> u32 {
        if !self.is_paused {
            self.current_frame = self.current_frame.wrapping_add(1);
        }
        self.current_frame
    }

    pub fn get_time(&self, start_time: &std::time::Instant) -> f32 {
        let raw_time = start_time.elapsed().as_secs_f32();
        if self.is_paused {
            if let Some(pause_start) = self.pause_start {
                raw_time - self.total_pause_duration - pause_start.elapsed().as_secs_f32()
            } else {
                raw_time - self.total_pause_duration
            }
        } else {
            raw_time - self.total_pause_duration
        }
    }

    pub fn get_ui_request(
        &mut self,
        start_time: &std::time::Instant,
        size: &winit::dpi::PhysicalSize<u32>,
    ) -> ControlsRequest {
        let mut load_media_path = None;
        let mut play_video = false;
        if !self.media_loaded_once {
            if let Ok(media_dir) = std::env::var("CUNEUS_MEDIA") {
                println!("CUNEUS_MEDIA: {}", media_dir);
                if media_dir.starts_with('"') && media_dir.ends_with('"') {
                    let unquoted = &media_dir[1..media_dir.len() - 1];
                    load_media_path = Some(PathBuf::from(unquoted));
                } else {
                    load_media_path = Some(PathBuf::from(media_dir));
                }
                play_video = true;
                self.media_loaded_once = true;
            }
        }
        ControlsRequest {
            is_paused: self.is_paused,
            should_reset: false,
            should_clear_buffers: false,
            current_time: Some(self.get_time(start_time)),
            window_size: Some((size.width, size.height)),
            current_fps: None,

            load_media_path,
            play_video,
            pause_video: false,
            restart_video: false,
            seek_position: None,
            set_loop: None,
            set_volume: None,
            mute_audio: None,
            toggle_mute: false,

            hdri_exposure: None,
            hdri_gamma: None,

            start_webcam: false,
            stop_webcam: false,
            webcam_device_index: None,
        }
    }

    pub fn apply_ui_request(&mut self, request: ControlsRequest) {
        if request.should_reset {
            self.is_paused = false;
            self.pause_start = None;
            self.total_pause_duration = 0.0;
            self.current_frame = 0;
            self.media_loaded_once = false;
        } else if request.is_paused && !self.is_paused {
            self.pause_start = Some(std::time::Instant::now());
        } else if !request.is_paused && self.is_paused {
            if let Some(pause_start) = self.pause_start {
                self.total_pause_duration += pause_start.elapsed().as_secs_f32();
            }
            self.pause_start = None;
        }
        self.is_paused = request.is_paused;
    }

    /// Extract video info from a video texture manager
    #[cfg(feature = "media")]
    pub fn get_video_info(
        using_video_texture: bool,
        video_manager: Option<&VideoTextureManager>,
    ) -> Option<VideoInfo> {
        if using_video_texture {
            if let Some(vm) = video_manager {
                Some((
                    vm.duration().map(|d| d.seconds() as f32),
                    vm.position().seconds() as f32,
                    vm.dimensions(),
                    vm.framerate().map(|(num, den)| num as f32 / den as f32),
                    vm.is_looping(),
                    vm.has_audio(),
                    vm.volume(),
                    vm.is_muted(),
                ))
            } else {
                None
            }
        } else {
            None
        }
    }
    ///media control panel (image, video, hdri)
    pub fn render_media_panel(
        ui: &mut egui::Ui,
        request: &mut ControlsRequest,
        using_video_texture: bool,
        video_info: Option<VideoInfo>,
        using_hdri_texture: bool,
        hdri_info: Option<HdriMetadata>,
        using_webcam_texture: bool,
        webcam_info: Option<(u32, u32)>,
    ) {
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.heading("Media");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if using_webcam_texture {
                        if ui.button("🔴 Stop Webcam").clicked() {
                            request.stop_webcam = true;
                        }
                    } else {
                        if ui.button("📹 Webcam").clicked() {
                            request.start_webcam = true;
                        }
                    }

                    if ui.button("Load").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter(
                                "Media Files",
                                &["png", "jpg", "jpeg", "mp4", "avi", "mkv", "webm", "mov"],
                            )
                            .add_filter("Images", &["png", "jpg", "jpeg", "webp", "bmp", "tiff"])
                            .add_filter("Videos", &["mp4", "avi", "mkv", "webm", "mov"])
                            .add_filter("HDRI", &["hdr", "exr"])
                            .pick_file()
                        {
                            request.load_media_path = Some(path);
                        }
                    }
                });
            });

            // Only show video controls if we're using a video texture
            if using_video_texture {
                ui.collapsing("Controls", |ui| {
                    // Main video controls
                    ui.horizontal(|ui| {
                        if ui.button("⏵").clicked() {
                            request.play_video = true;
                        }

                        if ui.button("⏸").clicked() {
                            request.pause_video = true;
                        }

                        if ui.button("⏮").clicked() {
                            request.restart_video = true;
                        }
                    });

                    if let Some((
                        duration_opt,
                        position_secs,
                        dimensions,
                        framerate_opt,
                        is_looping,
                        has_audio,
                        volume,
                        is_muted,
                    )) = video_info
                    {
                        ui.separator();

                        if let Some(duration_secs) = duration_opt {
                            ui.label(format!(
                                "Position: {:.1}s / {:.1}s",
                                position_secs, duration_secs
                            ));

                            let mut pos = position_secs;
                            if ui
                                .add(
                                    egui::Slider::new(&mut pos, 0.0..=duration_secs)
                                        .text("Timeline"),
                                )
                                .changed()
                            {
                                request.seek_position = Some(pos as f64);
                            }
                        }

                        // only show if video has audio
                        if has_audio {
                            ui.separator();
                            ui.heading("Audio");

                            let mut vol = volume;
                            if ui
                                .add(
                                    egui::Slider::new(&mut vol, 0.0..=1.0)
                                        .text("Volume")
                                        .show_value(true),
                                )
                                .changed()
                            {
                                request.set_volume = Some(vol);
                            }

                            ui.horizontal(|ui| {
                                let mut muted = is_muted;
                                if ui.checkbox(&mut muted, "Mute").changed() {
                                    request.mute_audio = Some(muted);
                                }
                            });
                        }

                        ui.separator();

                        ui.collapsing("Properties", |ui| {
                            ui.label(format!("Dimensions: {}x{}", dimensions.0, dimensions.1));

                            if let Some(fps) = framerate_opt {
                                ui.label(format!("Framerate: {:.2} fps", fps));
                            }

                            let mut looping = is_looping;
                            if ui.checkbox(&mut looping, "Loop").changed() {
                                request.set_loop = Some(looping);
                            }
                            if has_audio {
                                ui.label("Audio: Yes");
                            } else {
                                ui.label("Audio: No");
                            }
                        });
                    }
                });
            }
            if using_hdri_texture {
                ui.collapsing("HDRI Settings", |ui| {
                    if let Some(hdri_metadata) = &hdri_info {
                        ui.label(format!(
                            "Dimensions: {}x{}",
                            hdri_metadata.width, hdri_metadata.height
                        ));
                        ui.label("Type: High Dynamic Range Image");
                        let mut exposure = hdri_metadata.exposure;
                        if ui
                            .add(
                                egui::Slider::new(&mut exposure, 0.1..=6.28)
                                    .text("Exposure")
                                    .logarithmic(true),
                            )
                            .changed()
                        {
                            request.hdri_exposure = Some(exposure);
                        }
                        let mut gamma = hdri_metadata.gamma;
                        if ui
                            .add(egui::Slider::new(&mut gamma, 0.1..=6.28).text("Gamma"))
                            .changed()
                        {
                            request.hdri_gamma = Some(gamma);
                        }
                    } else {
                        ui.label("HDRI metadata not available");
                    }
                });
            }
            if using_webcam_texture {
                ui.collapsing("Webcam Settings", |ui| {
                    if let Some((width, height)) = webcam_info {
                        ui.label(format!("Resolution: {}x{}", width, height));
                        ui.label("Type: Live Camera Feed");
                        ui.label("Status: Active");
                    } else {
                        ui.label("Webcam information not available");
                    }
                });
            }
        });
    }

    pub fn render_controls_widget(ui: &mut egui::Ui, request: &mut ControlsRequest) {
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                if ui
                    .button(if request.is_paused {
                        "▶ Resume"
                    } else {
                        "⏸ Pause"
                    })
                    .clicked()
                {
                    request.is_paused = !request.is_paused;
                }
                if ui.button("↺ Reset").clicked() {
                    request.should_reset = true;
                    request.should_clear_buffers = true;
                }
                if let Some(time) = request.current_time {
                    ui.label(format!("Time: {:.2}s", time));
                }
                if let Some(fps) = request.current_fps {
                    ui.label(format!("FPS: {:.1}", fps));
                }
            });
            if let Some((width, height)) = request.window_size {
                ui.horizontal(|ui| {
                    ui.label(format!("Resolution: {}x{}", width, height));
                });
            }
        });
    }
}

use std::path::PathBuf;

#[cfg(not(target_os = "windows"))]
use crate::gst::video::VideoTextureManager;

#[derive(Clone)]
pub struct ControlsRequest {
    pub is_paused: bool,
    pub should_reset: bool,
    pub should_clear_buffers: bool,  
    pub current_time: Option<f32>, 
    pub window_size: Option<(u32, u32)>,
    
    // Video reqs
    pub load_media_path: Option<PathBuf>,
    pub play_video: bool,
    pub pause_video: bool,
    pub restart_video: bool,
    pub seek_position: Option<f64>,
    pub set_loop: Option<bool>,
}

impl Default for ControlsRequest {
    fn default() -> Self {
        Self {
            is_paused: false,
            should_reset: false,
            should_clear_buffers: false,
            current_time: None,
            window_size: None,
            
            // Video-related stuff
            load_media_path: None,
            play_video: false,
            pause_video: false,
            restart_video: false,
            seek_position: None,
            set_loop: None,
        }
    }
}

/// VideoInfo type alias
pub type VideoInfo = (Option<f32>, f32, (u32, u32), Option<f32>, bool);

pub struct ShaderControls {
    is_paused: bool,
    pause_start: Option<std::time::Instant>,
    total_pause_duration: f32,
    current_frame: u32,
}

impl Default for ShaderControls {
    fn default() -> Self {
        Self {
            is_paused: false,
            pause_start: None,
            total_pause_duration: 0.0,
            current_frame: 0,
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
    
    pub fn get_ui_request(&self, start_time: &std::time::Instant, size: &winit::dpi::PhysicalSize<u32>) -> ControlsRequest {
        ControlsRequest {
            is_paused: self.is_paused,
            should_reset: false,
            should_clear_buffers: false,
            current_time: Some(self.get_time(start_time)),
            window_size: Some((size.width, size.height)),
            
            load_media_path: None,
            play_video: false,
            pause_video: false,
            restart_video: false,
            seek_position: None,
            set_loop: None,
        }
    }
    
    pub fn apply_ui_request(&mut self, request: ControlsRequest) {
        if request.should_reset {
            self.is_paused = false;
            self.pause_start = None;
            self.total_pause_duration = 0.0;
            self.current_frame = 0;
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
    #[cfg(not(target_os = "windows"))]
    pub fn get_video_info(
        using_video_texture: bool,
        #[cfg(not(target_os = "windows"))]
        video_manager: Option<&VideoTextureManager>,
    ) -> Option<VideoInfo> {
        #[cfg(not(target_os = "windows"))]
        if using_video_texture {
            if let Some(vm) = video_manager {
                Some((
                    vm.duration().map(|d| d.seconds() as f32),
                    vm.position().seconds() as f32,
                    vm.dimensions(),
                    vm.framerate().map(|(num, den)| num as f32 / den as f32),
                    vm.is_looping()
                ))
            } else {
                None
            }
        } else {
            None
        }
    }
    ///media control panel (both image and video)
    pub fn render_media_panel(
        ui: &mut egui::Ui, 
        request: &mut ControlsRequest,
        using_video_texture: bool,
        video_info: Option<VideoInfo>,
    ) {
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.heading("Media");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Load").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Media Files", &["png", "jpg", "jpeg", "mp4", "avi", "mkv", "webm", "mov"])
                            .add_filter("Images", &["png", "jpg", "jpeg", "webp", "bmp", "tiff"])
                            .add_filter("Videos", &["mp4", "avi", "mkv", "webm", "mov"])
                            .pick_file() 
                        {
                            request.load_media_path = Some(path);
                        }
                    }
                });
            });
            
            // Only show video controls if we're using a video texture
            #[cfg(not(target_os = "windows"))]
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
                    
                    if let Some((duration_opt, position_secs, dimensions, framerate_opt, is_looping)) = video_info {
                        ui.separator();
                        
                        if let Some(duration_secs) = duration_opt {
                            ui.label(format!("Position: {:.1}s / {:.1}s", position_secs, duration_secs));
                            
                            let mut pos = position_secs;
                            if ui.add(egui::Slider::new(&mut pos, 0.0..=duration_secs)
                                .text("Timeline"))
                                .changed() 
                            {
                                request.seek_position = Some(pos as f64);
                            }
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
                        });
                    }
                });
            }
        });
    }

    pub fn render_controls_widget(ui: &mut egui::Ui, request: &mut ControlsRequest) {
        ui.vertical(|ui| { 
            ui.horizontal(|ui| {
                if ui.button(if request.is_paused { "▶ Resume" } else { "⏸ Pause" }).clicked() {
                    request.is_paused = !request.is_paused;
                }
                if ui.button("↺ Reset").clicked() {
                    request.should_reset = true;
                    request.should_clear_buffers = true;
                }
                if let Some(time) = request.current_time { 
                    ui.label(format!("Time: {:.2}s", time));
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

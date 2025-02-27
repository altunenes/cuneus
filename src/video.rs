//please see: https://github.com/nathanbabcock/ffmpeg-sidecar/blob/main/examples/ffprobe.rs
use crate::texture::TextureManager;
use ffmpeg_sidecar::{command::FfmpegCommand, event::FfmpegEvent};
use std::path::Path;
use std::time::{Duration, Instant};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
use std::thread::JoinHandle;
use anyhow::{Result, Context};
use image::RgbaImage;


pub struct VideoManager {
    // Current texture being displayed
    pub texture_manager: TextureManager,
    // Video playback state
    pub is_playing: bool,
    pub loop_video: bool,
    pub current_frame: usize,
    pub last_update_time: Instant,
    // Video metadata
    pub fps: f32, 
    pub frame_count: usize,
    pub duration: Duration,
    // Frame caching system
    frame_cache: VecDeque<RgbaImage>,
    max_cache_size: usize,
    // Frame dimensions
    pub width: u32,
    pub height: u32,
    batch_size: usize,
    next_batch_start: usize,
    batch_loading_thread: Option<JoinHandle<()>>,
    batch_ready: Arc<AtomicBool>,
    new_frames: Arc<Mutex<Vec<RgbaImage>>>,
    // Target size for performance
    target_width: u32,
}

impl VideoManager {
    pub fn test_ffmpeg() -> Result<()> {
        println!("Testing FFmpeg installation...");
        match ffmpeg_sidecar::download::auto_download() {
            Ok(_) => println!("FFmpeg auto-download successful"),
            Err(e) => println!("FFmpeg auto-download issue: {:?} - will try to use existing installation", e),
        }
        println!("Testing ffmpeg command...");
        let mut version_cmd = FfmpegCommand::new()
            .args(&["-version"])
            .spawn()
            .context("Failed to spawn FFmpeg command - is FFmpeg installed and in your PATH?")?;
            
        let mut iter = version_cmd.iter()?;
        let mut found_version = false;
        while let Some(event) = iter.next() {
            match event {
                FfmpegEvent::OutputChunk(data) => {
                    let version_str = String::from_utf8_lossy(&data);
                    println!("FFmpeg version info: {}", version_str);
                    found_version = true;
                },
                FfmpegEvent::Error(err) => {
                    println!("Error from FFmpeg: {}", err);
                },
                _ => {}
            }
        }
        
        if found_version {
            println!("FFmpeg test successful!");
            Ok(())
        } else {
            Err(anyhow::anyhow!("Failed to get FFmpeg version output"))
        }
    }
    
    ///a custom FFmpeg path
    pub fn set_ffmpeg_path(path: impl AsRef<Path>) {
        std::env::set_var("FFMPEG_BINARY_PATH", path.as_ref().to_string_lossy().to_string());
    }

    /// Creates a new VideoManager from a video file path
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: &Path, 
        layout: &wgpu::BindGroupLayout,
        max_cache_size: usize,
    ) -> Result<Self> {
        if !path.exists() {
            return Err(anyhow::anyhow!("Video file does not exist: {:?}", path));
        }
        
        println!("Loading video: {:?}", path);
        
        match ffmpeg_sidecar::download::auto_download() {
            Ok(_) => println!("FFmpeg is available"),
            Err(e) => println!("Warning: FFmpeg auto-download issue: {:?} - will try to use existing installation", e),
        }
        
        println!("Extracting video metadata...");
        let (width, height, fps, frame_count) = match Self::extract_metadata(path) {
            Ok((w, h, f, c)) => {
                println!("Successfully extracted metadata: {}x{} @ {:.2} fps, {} frames", w, h, f, c);
                (w, h, f, c)
            },
            Err(e) => {
                println!("Metadata extraction failed: {:?}. Using fallback approach...", e);
                
                match Self::extract_metadata_fallback(path) {
                    Ok((w, h, f, c)) => {
                        println!("Fallback metadata: {}x{} @ {:.2} fps, {} frames", w, h, f, c);
                        (w, h, f, c)
                    },
                    Err(e2) => {
                        println!("Fallback metadata also failed: {:?}. Using default values.", e2);
                        // Default values as last resort
                        (640, 480, 30.0, 300)
                    }
                }
            }
        };
        
        // reduce to 480p if larger
        let target_width = if width > 640 { 640 } else { width };
        let duration = Duration::from_secs_f32(frame_count as f32 / fps);
        println!("Extracting first video frame...");
        let first_frame = match Self::extract_first_frame(path, target_width) {
            Ok(frame) => {
                println!("Successfully extracted first frame");
                frame
            },
            Err(e) => {
                println!("Error extracting first frame: {:?}. Creating a placeholder frame.", e);
                //a placeholder frame with a checkerboard pattern
                let mut img = RgbaImage::new(target_width, target_width * height / width);
                for y in 0..img.height() {
                    for x in 0..img.width() {
                        let color = if (x / 32 + y / 32) % 2 == 0 {
                            [0, 0, 255, 255] // Blue
                        } else {
                            [255, 255, 255, 255] // White
                        };
                        img.put_pixel(x, y, image::Rgba(color));
                    }
                }
                
                img
            }
        };
        
        // a texture from first frame
        println!("Creating texture from first frame...");
        let texture_manager = TextureManager::new(
            device,
            queue,
            &first_frame,
            layout,
        );
        
        let mut frame_cache = VecDeque::with_capacity(max_cache_size);
        frame_cache.push_back(first_frame);
        
        let batch_size = std::cmp::min(max_cache_size / 2, 30);
        let batch_ready = Arc::new(AtomicBool::new(false));
        let new_frames = Arc::new(Mutex::new(Vec::new()));
        
        let mut manager = Self {
            texture_manager,
            is_playing: true,
            loop_video: true,
            current_frame: 0,
            last_update_time: Instant::now(),
            fps,
            frame_count,
            duration,
            frame_cache,
            max_cache_size,
            width,
            height,
            target_width,
            batch_size,
            next_batch_start: 1,
            batch_loading_thread: None,
            batch_ready,
            new_frames,
        };
        

        manager.start_batch_loading(path);
        
        println!("Video loaded successfully!");
        Ok(manager)
    }
    
    fn extract_metadata(path: &Path) -> Result<(u32, u32, f32, usize)> {
        match ffmpeg_sidecar::ffprobe::ffprobe_version() {
            Ok(version) => println!("ffprobe version: {}", version),
            Err(e) => println!("ffprobe not available: {:?}", e),
        }
        
        println!("Running ffprobe to extract metadata from {:?}", path);
        
        let output = std::process::Command::new("ffprobe")
            .args([
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
                "-of", "csv=p=0",
                path.to_str().unwrap()
            ])
            .output()
            .context("Failed to execute ffprobe command")?;
        
        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            println!("ffprobe error: {}", error);
            return Err(anyhow::anyhow!("ffprobe failed: {}", error));
        }
        
        let meta_str = String::from_utf8_lossy(&output.stdout).to_string();
        println!("Metadata string: '{}'", meta_str.trim());
        
        let meta_parts: Vec<&str> = meta_str.trim().split(',').collect();
        println!("Parsed metadata parts: {:?}", meta_parts);
        
        if meta_parts.len() < 4 {
            return Err(anyhow::anyhow!("Insufficient metadata parts: expected 4, got {}", meta_parts.len()));
        }
        
        let width: u32 = meta_parts[0].parse()
            .context("Failed to parse width")?;
        let height: u32 = meta_parts[1].parse()
            .context("Failed to parse height")?;
        
        //  framerate (comes as fraction like "30000/1001")
        let fps_parts: Vec<&str> = meta_parts[2].split('/').collect();
        let fps = if fps_parts.len() == 2 {
            let num: f32 = fps_parts[0].parse()
                .context("Failed to parse fps numerator")?;
            let den: f32 = fps_parts[1].parse()
                .context("Failed to parse fps denominator")?;
            num / den
        } else {
            meta_parts[2].parse()
                .context("Failed to parse fps as direct value")?
        };
        
        let frame_count: usize = meta_parts[3].parse()
            .context("Failed to parse frame count")?;
            
        if frame_count == 0 {
            return Err(anyhow::anyhow!("Invalid frame count: 0"));
        }
        
        Ok((width, height, fps, frame_count))
    }
    
    /// Fallback metadata extraction using ffmpeg
    fn extract_metadata_fallback(path: &Path) -> Result<(u32, u32, f32, usize)> {
        // Use ffmpeg to get basic info
        let mut info_cmd = FfmpegCommand::new()
            .args(&["-hwaccel", "auto"]) // Try hardware acceleration
            .args(&["-i", path.to_str().unwrap()])
            .spawn()?;
            
        let mut info_iter = info_cmd.iter()?;
        let mut width = 640;
        let mut height = 480;
        let mut fps = 30.0;
        let mut duration_secs = 10.0;
        
        // Process log output to extract information
        while let Some(event) = info_iter.next() {
            if let FfmpegEvent::Log(_, log) = &event {
                // Look for dimensions like 1920x1080
                if log.contains("Video:") {
                    println!("Found video info: {}", log);
                    
                    // Extract dimensions
                    for part in log.split_whitespace() {
                        if part.contains("x") && part.chars().any(|c| c.is_digit(10)) {
                            let dims: Vec<&str> = part.split(['x', ',', ' ']).collect();
                            for (i, dim) in dims.iter().enumerate() {
                                if i == 0 && dim.chars().all(|c| c.is_digit(10)) {
                                    if let Ok(w) = dim.parse() {
                                        width = w;
                                    }
                                } else if i == 1 && dim.chars().all(|c| c.is_digit(10)) {
                                    if let Ok(h) = dim.parse() {
                                        height = h;
                                    }
                                }
                            }
                        }
                    }
                    
                    // Extract fps
                    if log.contains("fps") {
                        for (i, part) in log.split_whitespace().enumerate() {
                            if part.contains("fps") || (i > 0 && log.split_whitespace().nth(i-1).unwrap_or("").contains("fps")) {
                                let cleaned = part.replace("fps", "").replace(",", "").trim().to_string();
                                if let Ok(f) = cleaned.parse() {
                                    fps = f;
                                }
                            }
                        }
                    }
                }
                
                // Look for duration
                if log.contains("Duration:") {
                    println!("Found duration info: {}", log);
                    
                    // Extract duration in format HH:MM:SS.ms
                    if let Some(time_str) = log.split("Duration:").nth(1) {
                        let parts: Vec<&str> = time_str.split(',').collect();
                        if !parts.is_empty() {
                            let time_parts: Vec<&str> = parts[0].trim().split(':').collect();
                            if time_parts.len() >= 3 {
                                let hours: f32 = time_parts[0].trim().parse().unwrap_or(0.0);
                                let minutes: f32 = time_parts[1].trim().parse().unwrap_or(0.0);
                                let seconds: f32 = time_parts[2].trim().parse().unwrap_or(0.0);
                                
                                duration_secs = hours * 3600.0 + minutes * 60.0 + seconds;
                            }
                        }
                    }
                }
            }
        }
        
        // Calculate approximate frame count based on duration and fps
        let frame_count = (duration_secs * fps).round() as usize;
        
        if width == 0 || height == 0 || fps <= 0.0 || frame_count == 0 {
            return Err(anyhow::anyhow!("Failed to extract valid metadata from fallback method"));
        }
        
        println!("Extracted metadata (fallback): {}x{} @ {:.2} fps, {} frames ({}s)", 
                 width, height, fps, frame_count, duration_secs);
        
        Ok((width, height, fps, frame_count))
    }
    
    fn extract_first_frame(path: &Path, target_width: u32) -> Result<RgbaImage> {
        println!("Extracting first frame from: {:?}", path);
        
        let mut first_frame_cmd = FfmpegCommand::new()
            .args(&["-hwaccel", "auto"]) // hw acceleration 
            .input(path.to_str().unwrap())
            .args(&["-frames:v", "1"])  // Just the first frame
            .args(&["-vf", &format!("scale={}:-1", target_width)]) 
            .args(&["-pix_fmt", "rgb24"]) //  RGB24 format
            .rawvideo()
            .spawn()?;
            
        let mut first_frame_iter = first_frame_cmd.iter()?;
        
        while let Some(event) = first_frame_iter.next() {
            match event {
                FfmpegEvent::OutputFrame(frame) => {
                    println!("Found frame: {}x{}, {} bytes", frame.width, frame.height, frame.data.len());
                    
                    // Check if the data size is what we expect for RGB24
                    let expected_size = frame.width as usize * frame.height as usize * 3;
                    if frame.data.len() != expected_size {
                        println!("Data size mismatch: got {} bytes, expected {} bytes", 
                                 frame.data.len(), expected_size);
                        continue;
                    }
                    
                    // rgba
                    let mut rgba_data = Vec::with_capacity(frame.width as usize * frame.height as usize * 4);
                    for chunk in frame.data.chunks(3) {
                        rgba_data.extend_from_slice(chunk); // Add RGB
                        rgba_data.push(255);               //fully opaque
                    }
                    
                    if let Some(rgba_image) = RgbaImage::from_raw(
                        frame.width as u32,
                        frame.height as u32,
                        rgba_data
                    ) {
                        return Ok(rgba_image);
                    } else {
                        println!("Failed to create RgbaImage from converted frame data");
                    }
                },
                FfmpegEvent::Log(level, msg) => {
                    if level != ffmpeg_sidecar::event::LogLevel::Info {
                        println!("FFmpeg log [{:?}]: {}", level, msg);
                    }
                },
                FfmpegEvent::Error(err) => {
                    println!("FFmpeg error: {}", err);
                },
                _ => {}
            }
        }
        
        Err(anyhow::anyhow!("No usable frames were extracted from the video"))
    }
    
    fn start_batch_loading(&mut self, path: &Path) {
        if self.batch_loading_thread.is_some() {
            return;
        }
        
        let start_frame = self.next_batch_start;
        let end_frame = std::cmp::min(start_frame + self.batch_size, self.frame_count);
        
        if start_frame >= end_frame {
            return;
        }
        
        println!("Starting batch load for frames {}-{}", start_frame, end_frame);
        
        self.batch_ready.store(false, Ordering::SeqCst);
        
        let path_clone = path.to_path_buf();
        let batch_ready = self.batch_ready.clone();
        let new_frames = self.new_frames.clone();
        let fps = self.fps;
        let target_width = self.target_width;
        
        //  background thread for loading frames
        let handle = std::thread::spawn(move || {
            let seek_time = start_frame as f32 / fps;
            
            // Use a single FFmpeg command to extract multiple frames
            let mut cmd = FfmpegCommand::new()
                .args(&["-hwaccel", "auto"]) // Hardware acceleration
                .input(path_clone.to_str().unwrap())
                .args(&["-ss", &seek_time.to_string()])  // Seek to start time
                .args(&["-frames:v", &(end_frame - start_frame).to_string()]) // Get N frames
                .args(&["-vf", &format!("scale={}:-1", target_width)]) // Scale to target width
                .args(&["-pix_fmt", "rgb24"])  // Use RGB24 format
                .rawvideo()
                .spawn()
                .expect("Failed to spawn FFmpeg for batch loading");
                
            let mut frames = Vec::new();
            
            if let Ok(mut iter) = cmd.iter() {
                while let Some(event) = iter.next() {
                    if let FfmpegEvent::OutputFrame(frame) = event {
                        let expected_size = frame.width as usize * frame.height as usize * 3;
                        if frame.data.len() == expected_size {
                            let mut rgba_data = Vec::with_capacity(frame.width as usize * frame.height as usize * 4);
                            for chunk in frame.data.chunks(3) {
                                rgba_data.extend_from_slice(chunk);
                                rgba_data.push(255);
                            }
                            
                            if let Some(rgba_image) = RgbaImage::from_raw(
                                frame.width as u32,
                                frame.height as u32,
                                rgba_data
                            ) {
                                frames.push(rgba_image);
                            }
                        }
                    }
                }
            }
            
            println!("Batch loaded {} frames", frames.len());
            
            if let Ok(mut cache) = new_frames.lock() {
                *cache = frames;
            }
            
            batch_ready.store(true, Ordering::SeqCst);
        });
        
        self.batch_loading_thread = Some(handle);
        self.next_batch_start = end_frame;
    }
    
    fn check_batch_ready(&mut self) -> bool {
        // Check if the batch loading has completed
        if self.batch_ready.load(Ordering::SeqCst) {
            if let Some(handle) = self.batch_loading_thread.take() {
                // Wait for the thread to finish (should be instant since the flag is set)
                if handle.join().is_ok() {
                    // Get the new frames from the shared vector
                    if let Ok(mut new_frames) = self.new_frames.lock() {
                        // Add new frames to our cache
                        for frame in new_frames.drain(..) {
                            self.frame_cache.push_back(frame);
                            
                            if self.frame_cache.len() > self.max_cache_size {
                                self.frame_cache.pop_front();
                            }
                        }
                        
                        return true;
                    }
                }
            }
        }
        
        false
    }
    
    /// Seeks to a specific frame in the video
    pub fn seek_to_frame(
        &mut self, 
        queue: &wgpu::Queue,
        path: &Path, 
        frame_number: usize
    ) -> Result<()> {
        let seek_time = frame_number as f32 / self.fps;
        println!("Seeking to frame {} ({}s)", frame_number, seek_time);
        
        let mut seek_cmd = FfmpegCommand::new()
            .args(&["-hwaccel", "auto"]) // Hardware acceleration
            .input(path.to_str().unwrap())
            .args(&["-ss", &seek_time.to_string()]) // Seek position
            .args(&["-frames:v", "1"])              // Just one frame
            .args(&["-vf", &format!("scale={}:-1", self.target_width)]) // Scale to target width
            .args(&["-pix_fmt", "rgb24"])           // RGB24 format
            .rawvideo()
            .spawn()?;
            
        let mut frame_iter = seek_cmd.iter()?;
        
        while let Some(event) = frame_iter.next() {
            match event {
                FfmpegEvent::OutputFrame(frame) => {
                    println!("Found frame at {}s: {}x{}, {} bytes", 
                             seek_time, frame.width, frame.height, frame.data.len());
                    
                    // Check if the data size is what we expect for RGB24
                    let expected_size = frame.width as usize * frame.height as usize * 3; // RGB = 3 bytes per pixel
                    if frame.data.len() != expected_size {
                        println!("Data size mismatch: got {} bytes, expected {} bytes", 
                                 frame.data.len(), expected_size);
                        continue;
                    }
                    
                    // Convert from RGB to RGBA by adding alpha channel
                    let mut rgba_data = Vec::with_capacity(frame.width as usize * frame.height as usize * 4);
                    for chunk in frame.data.chunks(3) {
                        rgba_data.extend_from_slice(chunk); // Add RGB
                        rgba_data.push(255);               // Add alpha (fully opaque)
                    }
                    
                    if let Some(rgba_image) = RgbaImage::from_raw(
                        frame.width as u32,
                        frame.height as u32,
                        rgba_data
                    ) {
                        // Update current texture with new frame
                        self.texture_manager.update(queue, &rgba_image);
                        self.current_frame = frame_number;
                        return Ok(());
                    } else {
                        println!("Failed to create RgbaImage from converted frame data");
                    }
                },
                FfmpegEvent::Log(level, msg) => {
                    if level != ffmpeg_sidecar::event::LogLevel::Info {
                        println!("FFmpeg seek log [{:?}]: {}", level, msg);
                    }
                },
                FfmpegEvent::Error(err) => {
                    println!("FFmpeg seek error: {}", err);
                },
                _ => {}
            }
        }
        
        Err(anyhow::anyhow!("Could not find usable frame at {}s", seek_time))
    }
    
    /// Updates the video manager, advancing frames based on timing
    pub fn update(&mut self, queue: &wgpu::Queue, path: &Path) -> Result<bool> {
        if !self.is_playing {
            return Ok(false);
        }
    
    // Check if we have new frames from batch loading
    self.check_batch_ready();
    
    // Start a new batch load if needed and we're not already loading
    if self.batch_loading_thread.is_none() && self.next_batch_start < self.frame_count {
        self.start_batch_loading(path);
    }
    
    // Calculate frame timing
    let now = Instant::now();
    let frame_duration = Duration::from_secs_f32(1.0 / self.fps);
    let elapsed = now.duration_since(self.last_update_time);
    
    if elapsed < frame_duration {
        return Ok(false);
    }
    
    // Determine how many frames to advance
    let frames_to_advance = (elapsed.as_secs_f32() / frame_duration.as_secs_f32()).floor() as usize;
    if frames_to_advance == 0 {
        return Ok(false);
    }
    
    self.last_update_time = now;
    let new_frame = self.current_frame + frames_to_advance;
    
    // Handle end of video
    if new_frame >= self.frame_count {
        if self.loop_video {
            self.current_frame = new_frame % self.frame_count;
            println!("Video looped from frame {} to {}", new_frame, self.current_frame);
        } else {
            println!("Video reached end (frame {})", self.frame_count - 1);
            self.is_playing = false;
            self.current_frame = self.frame_count - 1;
            return Ok(false);
        }
    } else {
        self.current_frame = new_frame;
    }
    
    // Try to find the frame in our cache
    // Convert current_frame to a frame index in our cache
    if self.current_frame < self.frame_cache.len() {
        // Direct access if frame is in range
        if let Some(frame) = self.frame_cache.get(self.current_frame) {
            self.texture_manager.update(queue, frame);
            return Ok(true);
        }
    }
    
    // Frame wasn't in cache, try to seek to it directly
    match self.seek_to_frame(queue, path, self.current_frame) {
        Ok(_) => Ok(true),
        Err(e) => {
            println!("Error seeking to frame {}: {:?}", self.current_frame, e);
            // Continue playback despite error
            Ok(false)
        }
    }
}
    
    /// Gets the current playback position in seconds
    pub fn current_time(&self) -> f32 {
        self.current_frame as f32 / self.fps
    }
    
    /// Gets the duration of the video in seconds
    pub fn duration_seconds(&self) -> f32 {
        self.duration.as_secs_f32()
    }
    
    /// Sets the playback speed
    pub fn set_playback_speed(&mut self, speed: f32) {
        // Adjust frame timing based on speed
        self.fps = self.fps * speed;
        println!("Playback speed adjusted, new fps: {:.2}", self.fps);
    }
}
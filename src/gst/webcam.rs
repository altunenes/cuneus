use crate::gst::video::{AudioLevel, SpectrumData};
use crate::texture::TextureManager;
use anyhow::{anyhow, Result};
use gst::prelude::*;
use gstreamer as gst;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;
use log::{debug, info, warn};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use wgpu;

/// Manages a webcam texture that can be updated frame by frame. My approach is actually same for src/gst/video.rs
pub struct WebcamTextureManager {
    /// The underlying TextureManager that handles the WGPU resources
    texture_manager: TextureManager,
    /// The GStreamer pipeline for webcam capture
    pipeline: gst::Pipeline,
    /// The AppSink element that receives decoded frames
    appsink: gst_app::AppSink,
    /// Current webcam dimensions
    dimensions: (u32, u32),
    /// Whether the webcam is currently active
    is_active: Arc<Mutex<bool>>,
    /// Last frame update time
    last_update: Instant,
    /// Frame buffer for the most recently captured frame
    current_frame: Arc<Mutex<Option<image::RgbaImage>>>,
    /// Whether the webcam texture has been initialized
    texture_initialized: bool,
    /// Frame counter for debugging
    frame_count: usize,
    /// Webcam device name/index
    device_name: String,
    /// Whether the pipeline successfully attached a microphone branch.
    /// We only post-process spectrum/level when this is true.
    has_audio: bool,
    /// Spectrum analysis data from the mic input (same shape as video.rs).
    spectrum_data: Arc<Mutex<SpectrumData>>,
    /// Audio level (RMS/peak) from the mic input.
    audio_level: Arc<Mutex<AudioLevel>>,
}

impl WebcamTextureManager {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bind_group_layout: &wgpu::BindGroupLayout,
        device_index: Option<u32>,
    ) -> Result<Self> {
        // Create a default 1x1 texture initially, this will be replaced with first webcam frame
        let default_image = image::RgbaImage::new(1, 1);
        let texture_manager = TextureManager::new(device, queue, &default_image, bind_group_layout);

        let device_name = device_index
            .map(|i| format!("/dev/video{i}"))
            .unwrap_or_else(|| "0".to_string());

        info!("Creating webcam capture from device: {device_name}");

        let pipeline = gst::Pipeline::new();
        //MAC  :   https://gstreamer.freedesktop.org/documentation/applemedia/avfvideosrc.html?gi-language=c#avfvideosrc-page
        //Linux:   https://gstreamer.freedesktop.org/documentation/video4linux2/v4l2src.html?gi-language=c#v4l2src-page
        //Win  :   https://gstreamer.freedesktop.org/documentation/winks/index.html?gi-language=c#ksvideosrc-page
        // Will be traditional pipeline.
        // Create source element for webcam
        #[cfg(target_os = "linux")]
        let source = gst::ElementFactory::make("v4l2src")
            .name("webcam_source")
            .property("device", &device_name)
            .build()
            .map_err(|_| anyhow!("Failed to create v4l2src element"))?;

        #[cfg(target_os = "macos")]
        let source = gst::ElementFactory::make("avfvideosrc")
            .name("webcam_source")
            .property("device-index", device_index.unwrap_or(0) as i32)
            .build()
            .map_err(|_| anyhow!("Failed to create avfvideosrc element"))?;

        #[cfg(target_os = "windows")]
        let source = gst::ElementFactory::make("ksvideosrc")
            .name("webcam_source")
            .property("device-index", device_index.unwrap_or(0) as i32)
            .build()
            .map_err(|_| anyhow!("Failed to create ksvideosrc element"))?;

        // Create caps filter to set resolution and framerate
        let caps_filter = gst::ElementFactory::make("capsfilter")
            .name("caps")
            .build()
            .map_err(|_| anyhow!("Failed to create capsfilter element"))?;

        // Set preferred webcam format: EXPERIMENTAL
        let caps = gst::Caps::builder("video/x-raw")
            .field("width", 1280i32)
            .field("height", 720i32)
            .field("framerate", gst::Fraction::new(30, 1))
            .build();
        caps_filter.set_property("caps", &caps);

        // videorate element to stabilize frame timing
        let videorate = gst::ElementFactory::make("videorate")
            .name("rate")
            .build()
            .map_err(|_| anyhow!("Failed to create videorate element"))?;

        // Convert to proper format
        let videoconvert = gst::ElementFactory::make("videoconvert")
            .name("convert")
            .build()
            .map_err(|_| anyhow!("Failed to create videoconvert element"))?;

        // Output sink for video
        let appsink = gst::ElementFactory::make("appsink")
            .name("sink")
            .build()
            .map_err(|_| anyhow!("Failed to create appsink element"))?;

        let appsink = appsink
            .dynamic_cast::<gst_app::AppSink>()
            .map_err(|_| anyhow!("Failed to cast to AppSink"))?;

        // Configure appsink
        appsink.set_caps(Some(
            &gst::Caps::builder("video/x-raw")
                .field("format", gst_video::VideoFormat::Rgba.to_str())
                .build(),
        ));
        appsink.set_max_buffers(2);
        appsink.set_drop(true);
        appsink.set_sync(false);

        pipeline
            .add_many([
                &source,
                &caps_filter,
                &videorate,
                &videoconvert,
                appsink.upcast_ref(),
            ])
            .map_err(|_| anyhow!("Failed to add webcam elements to pipeline"))?;

        // Link elements
        gst::Element::link_many([
            &source,
            &caps_filter,
            &videorate,
            &videoconvert,
            appsink.upcast_ref(),
        ])
        .map_err(|_| anyhow!("Failed to link webcam elements"))?;

        // Audio branch: microphone → audioconvert → audioresample → 44.1kHz caps
        // → spectrum (FFT) → level (RMS/peak) → fakesink. Posts spectrum and
        // level messages on the bus identical to gst/video.rs so RenderKit's
        // audio spectrum path treats webcam mic and video audio uniformly.
        let has_audio = Self::try_attach_audio_branch(&pipeline);

        // Create shared state
        let current_frame = Arc::new(Mutex::new(None));
        let current_frame_clone = current_frame.clone();
        let is_active = Arc::new(Mutex::new(false));

        // Setup callbacks to receive frames
        appsink.set_callbacks(
            gst_app::AppSinkCallbacks::builder()
                .new_sample(move |sink| {
                    let sample = match sink.pull_sample() {
                        Ok(sample) => sample,
                        Err(_) => return Err(gst::FlowError::Eos),
                    };

                    let buffer = match sample.buffer() {
                        Some(buffer) => buffer,
                        _ => return Err(gst::FlowError::Error),
                    };

                    let caps = match sample.caps() {
                        Some(caps) => caps,
                        _ => return Err(gst::FlowError::Error),
                    };

                    let video_info = match gst_video::VideoInfo::from_caps(caps) {
                        Ok(info) => info,
                        Err(_) => return Err(gst::FlowError::Error),
                    };

                    let map = match buffer.map_readable() {
                        Ok(map) => map,
                        Err(_) => return Err(gst::FlowError::Error),
                    };

                    // Access the raw frame data
                    let frame_data = map.as_slice();
                    let width = video_info.width() as usize;
                    let height = video_info.height() as usize;

                    // Create an RgbaImage from the frame data
                    let mut rgba_image = image::RgbaImage::new(width as u32, height as u32);

                    // Stride might be larger than width * 4
                    let stride = video_info.stride()[0] as usize;

                    for y in 0..height {
                        let src_start = y * stride;
                        let src_end = src_start + width * 4;
                        let dst_start = y * width * 4;
                        let dst_end = dst_start + width * 4;

                        let dst_buffer = rgba_image.as_mut();
                        if src_end <= frame_data.len() && dst_end <= dst_buffer.len() {
                            dst_buffer[dst_start..dst_end]
                                .copy_from_slice(&frame_data[src_start..src_end]);
                        }
                    }

                    if let Ok(mut frame_lock) = current_frame_clone.lock() {
                        *frame_lock = Some(rgba_image);
                    }

                    Ok(gst::FlowSuccess::Ok)
                })
                .build(),
        );

        // init the obj
        let webcam_texture = Self {
            texture_manager,
            pipeline,
            appsink,
            dimensions: (1280, 720),
            is_active,
            last_update: Instant::now(),
            current_frame,
            texture_initialized: false,
            frame_count: 0,
            device_name,
            has_audio,
            spectrum_data: Arc::new(Mutex::new(SpectrumData::default())),
            audio_level: Arc::new(Mutex::new(AudioLevel::default())),
        };

        info!("Webcam texture manager created successfully");
        Ok(webcam_texture)
    }

    pub fn start(&mut self) -> Result<()> {
        info!("Starting webcam capture");
        match self.pipeline.set_state(gst::State::Playing) {
            Ok(_) => {
                *self.is_active.lock().unwrap() = true;

                // lets wait a moment for the pipeline to start
                std::thread::sleep(std::time::Duration::from_millis(100));

                // Try to get actual dimensions from the pipeline
                if let Some(pad) = self.appsink.static_pad("sink") {
                    if let Some(caps) = pad.current_caps() {
                        if let Some(s) = caps.structure(0) {
                            if let (Ok(width), Ok(height)) =
                                (s.get::<i32>("width"), s.get::<i32>("height"))
                            {
                                self.dimensions = (width as u32, height as u32);
                                info!("Webcam dimensions: {width}x{height}");
                            }
                        }
                    }
                }

                Ok(())
            }
            Err(e) => Err(anyhow!("Failed to start webcam: {:?}", e)),
        }
    }

    pub fn stop(&mut self) -> Result<()> {
        info!("Stopping webcam capture");
        match self.pipeline.set_state(gst::State::Null) {
            Ok(_) => {
                *self.is_active.lock().unwrap() = false;
                Ok(())
            }
            Err(e) => Err(anyhow!("Failed to stop webcam: {:?}", e)),
        }
    }

    /// I need this for wgpu
    pub fn texture_manager(&self) -> &TextureManager {
        &self.texture_manager
    }

    pub fn update_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Result<bool> {
        // No update needed if webcam is not active
        if !*self.is_active.lock().unwrap() {
            return Ok(false);
        }
        let _ = self.poll_audio_messages();

        // Check if we have a NEW frame to process
        let frame_to_process = {
            let mut frame_lock = self.current_frame.lock().unwrap();
            frame_lock.take()
        };

        // If we have a frame, update the texture
        if let Some(frame) = frame_to_process {
            self.frame_count += 1;

            let width = frame.width();
            let height = frame.height();

            // Log less frequently to reduce spam
            if self.frame_count % 60 == 0 {
                debug!(
                    "Processing webcam frame #{} (dimensions: {}x{})",
                    self.frame_count, width, height
                );
            }

            // ALWAYS recreate the texture for the first frame or if dimensions don't match
            let should_recreate = !self.texture_initialized
                || self.dimensions != (width, height)
                || self.dimensions.0 <= 1
                || self.dimensions.1 <= 1
                || self.frame_count <= 3;

            if should_recreate {
                info!("Creating new webcam texture with dimensions: {width}x{height}");

                // Create a completely new texture with the frame's dimensions
                let new_texture_manager =
                    TextureManager::new(device, queue, &frame, bind_group_layout);

                self.texture_manager = new_texture_manager;
                self.dimensions = (width, height);
                self.texture_initialized = true;
            } else {
                self.texture_manager.update(queue, &frame);
            }

            // Update the last update time
            self.last_update = Instant::now();

            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    pub fn is_active(&self) -> bool {
        *self.is_active.lock().unwrap()
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Whether the pipeline successfully attached the mic branch.
    pub fn has_audio(&self) -> bool {
        self.has_audio
    }

    /// Current spectrum data from the mic.
    pub fn spectrum_data(&self) -> SpectrumData {
        self.spectrum_data.lock().map(|d| d.clone()).unwrap_or_default()
    }

    /// Current RMS/peak from the mic.
    pub fn audio_level(&self) -> AudioLevel {
        self.audio_level.lock().map(|l| l.clone()).unwrap_or_default()
    }

    /// BPM is not meaningful for live mic input
    pub fn get_bpm(&self) -> f32 {
        0.0
    }

    /// mic → spectrum + level → fakesink
    /// if fail:
    /// (no mic, missing element, permission denied) leave video working
    /// and just skip audio rather than aborting can aboty
    fn try_attach_audio_branch(pipeline: &gst::Pipeline) -> bool {
        // Platform-specific microphone source.
        #[cfg(target_os = "linux")]
        let mic_factory = "pulsesrc";
        #[cfg(target_os = "macos")]
        let mic_factory = "osxaudiosrc";
        #[cfg(target_os = "windows")]
        let mic_factory = "wasapisrc";

        let mic = match gst::ElementFactory::make(mic_factory).name("mic_source").build() {
            Ok(e) => e,
            Err(_) => {
                warn!("Webcam: no microphone source available ({mic_factory}); audio disabled");
                return false;
            }
        };

        let audioconvert = match gst::ElementFactory::make("audioconvert").build() {
            Ok(e) => e,
            Err(_) => {
                warn!("Webcam: failed to create audioconvert; audio disabled");
                return false;
            }
        };
        let audioresample = match gst::ElementFactory::make("audioresample").build() {
            Ok(e) => e,
            Err(_) => {
                warn!("Webcam: failed to create audioresample; audio disabled");
                return false;
            }
        };
        let audio_caps_filter = match gst::ElementFactory::make("capsfilter").build() {
            Ok(e) => e,
            Err(_) => {
                warn!("Webcam: failed to create audio capsfilter; audio disabled");
                return false;
            }
        };
        let audio_caps = gst::Caps::builder("audio/x-raw")
            .field("rate", 44100i32)
            .field("channels", 1i32)
            .build();
        audio_caps_filter.set_property("caps", &audio_caps);

        let spectrum = match gst::ElementFactory::make("spectrum")
            .name("webcam_spectrum")
            .property("bands", 128u32)
            .property("threshold", -60i32)
            .property("post-messages", true)
            .property("message-magnitude", true)
            .property("message-phase", false)
            .property("interval", 50000000u64) // 50ms — matches video pipeline
            .build()
        {
            Ok(e) => e,
            Err(_) => {
                warn!("Webcam: failed to create spectrum element; audio disabled");
                return false;
            }
        };

        let level = match gst::ElementFactory::make("level")
            .name("webcam_level")
            .property("interval", 50000000u64)
            .property("message", true)
            .property("post-messages", true)
            .build()
        {
            Ok(e) => e,
            Err(_) => {
                warn!("Webcam: failed to create level element; audio disabled");
                return false;
            }
        };

        let audio_sink = match gst::ElementFactory::make("fakesink")
            .name("webcam_audio_sink")
            .property("sync", false)
            .property("async", false)
            .build()
        {
            Ok(e) => e,
            Err(_) => {
                warn!("Webcam: failed to create fakesink; audio disabled");
                return false;
            }
        };

        if pipeline
            .add_many([
                &mic,
                &audioconvert,
                &audioresample,
                &audio_caps_filter,
                &spectrum,
                &level,
                &audio_sink,
            ])
            .is_err()
        {
            warn!("Webcam: failed to add audio elements to pipeline; audio disabled");
            return false;
        }

        if gst::Element::link_many([
            &mic,
            &audioconvert,
            &audioresample,
            &audio_caps_filter,
            &spectrum,
            &level,
            &audio_sink,
        ])
        .is_err()
        {
            warn!("Webcam: failed to link audio chain; audio disabled");
            return false;
        }

        info!("Webcam: microphone audio branch attached ({mic_factory})");
        true
    }

    /// Poll the GStreamer bus for spectrum/level messages from the webcam's
    /// audio branch. Mirrors VideoTextureManager::poll_audio_messages but
    /// drops BPM handling (not meaningful for live mic).
    pub fn poll_audio_messages(&mut self) -> bool {
        if !self.has_audio {
            return false;
        }

        let mut updated = false;
        let Some(bus) = self.pipeline.bus() else { return false };

        while let Some(message) = bus.pop() {
            if let gst::MessageView::Element(element) = message.view() {
                let Some(structure) = element.structure() else { continue };

                if structure.name() == "spectrum" {
                    let mut magnitude_values: Vec<f32> = Vec::with_capacity(128);


                    let struct_str = structure.to_string();
                    if let Some(start_idx) = struct_str.find("magnitude=(float){") {
                        if let Some(end_idx) = struct_str[start_idx..].find('}') {
                            let magnitude_str = &struct_str
                                [start_idx + "magnitude=(float){".len()..start_idx + end_idx];
                            for value_str in magnitude_str.split(',') {
                                if let Ok(value) = value_str.trim().parse::<f32>() {
                                    magnitude_values.push(value);
                                }
                            }
                        }
                    }

                    // Fall back to indexed field access.
                    if magnitude_values.is_empty() {
                        for i in 0..128 {
                            let field_name = format!("magnitude[{i}]");
                            if let Ok(value) = structure.get::<f32>(&field_name) {
                                magnitude_values.push(value);
                            } else {
                                break;
                            }
                        }
                    }

                    if !magnitude_values.is_empty() {
                        let bands = magnitude_values.len();
                        if let Ok(mut data) = self.spectrum_data.lock() {
                            *data = SpectrumData {
                                bands,
                                magnitudes: magnitude_values,
                                phases: None,
                                timestamp: structure.get("timestamp").ok(),
                            };
                        }
                        updated = true;
                    }
                }

                if structure.name() == "level" {
                    let mut rms_db_val = None;
                    let mut peak_val = None;

                    if let Ok(rms_list) = structure.get::<gst::glib::ValueArray>("rms") {
                        for val in rms_list.iter() {
                            if let Ok(rms_db) = val.get::<f64>() {
                                rms_db_val = Some(rms_db);
                                break;
                            }
                        }
                    }
                    if let Ok(peak_list) = structure.get::<gst::glib::ValueArray>("peak") {
                        for val in peak_list.iter() {
                            if let Ok(peak_db) = val.get::<f64>() {
                                peak_val = Some(10.0_f64.powf(peak_db / 20.0));
                                break;
                            }
                        }
                    }
                    if let (Some(rms_db), Some(peak)) = (rms_db_val, peak_val) {
                        let rms_linear = 10.0_f64.powf(rms_db / 20.0);
                        if let Ok(mut lvl) = self.audio_level.lock() {
                            *lvl = AudioLevel {
                                rms: rms_linear,
                                rms_db,
                                peak,
                            };
                        }
                        updated = true;
                    }
                }
            }
        }
        updated
    }

    /// Get available webcam devices:
    /// DEVICE LISTS: I tested only macos: https://www.ffmpeg.org/ffmpeg-devices.html
    /// But according here, this should work anyway...
    pub fn list_devices() -> Vec<String> {
        let mut devices = Vec::new();

        #[cfg(target_os = "linux")]
        {
            // On Linux, check for /dev/video* devices
            for i in 0..10 {
                let device_path = format!("/dev/video{}", i);
                if std::path::Path::new(&device_path).exists() {
                    devices.push(device_path);
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            // On macOS, AVFoundation devices are usually indexed 0, 1, 2...
            for i in 0..5 {
                devices.push(format!("Camera {i}"));
            }
        }

        #[cfg(target_os = "windows")]
        {
            // On Windows, DirectShow devices are usually indexed 0, 1, 2...
            for i in 0..5 {
                devices.push(format!("Camera {}", i));
            }
        }

        if devices.is_empty() {
            devices.push("Default Camera".to_string());
        }

        devices
    }
}

impl Drop for WebcamTextureManager {
    fn drop(&mut self) {
        info!("Shutting down webcam pipeline");
        let _ = self.pipeline.set_state(gst::State::Null);
    }
}

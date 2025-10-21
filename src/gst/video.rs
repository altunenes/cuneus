use crate::texture::TextureManager;
use anyhow::{anyhow, Result};
use gst::glib::ControlFlow;
use gst::prelude::*;
use gstreamer as gst;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;
use log::{debug, error, info, warn};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use wgpu;

#[derive(Debug, Clone)]
pub struct SpectrumData {
    /// Number of frequency bands
    pub bands: usize,
    /// Magnitude values for each frequency band in dB
    pub magnitudes: Vec<f32>,
    /// Phase values for each frequency band
    pub phases: Option<Vec<f32>>,
    /// Timestamp of the spectrum data
    pub timestamp: Option<gst::ClockTime>,
}

impl Default for SpectrumData {
    fn default() -> Self {
        Self {
            bands: 0,
            magnitudes: Vec::new(),
            phases: None,
            timestamp: None,
        }
    }
}

/// Here I created a struct to organize the video text mang.
/// Manages a video texture that can be updated frame by frame
pub struct VideoTextureManager {
    /// The underlying TextureManager that handles the WGPU resources
    texture_manager: TextureManager,
    /// The GStreamer pipeline for video decoding
    pipeline: gst::Pipeline,
    /// The AppSink element that receives decoded frames
    appsink: gst_app::AppSink,
    /// Whether the video has an audio track
    has_audio: bool,
    /// Audio volume (0.0 to 1.0)
    volume: Arc<Mutex<f64>>,
    /// Whether audio is muted
    is_muted: Arc<Mutex<bool>>,
    /// Current video dimensions
    dimensions: (u32, u32),
    /// Video duration in nanoseconds (if available)
    duration: Option<gst::ClockTime>,
    /// Current position in the video
    position: Arc<Mutex<gst::ClockTime>>,
    /// Frame rate of the video
    framerate: Option<gst::Fraction>,
    /// Whether the video is currently playing
    is_playing: Arc<Mutex<bool>>,
    /// Whether to loop the video when it ends
    loop_playback: Arc<Mutex<bool>>,
    /// Last frame update time
    last_update: Instant,
    /// Frame buffer for the most recently decoded frame
    current_frame: Arc<Mutex<Option<image::RgbaImage>>>,
    /// Path to the video file
    video_path: String,
    /// Whether the video texture has been initialized
    texture_initialized: bool,
    /// Frame counter for debugging
    frame_count: usize,
    /// Spectrum analysis enabled
    spectrum_enabled: bool,
    /// Number of frequency bands for spectrum analysis
    spectrum_bands: usize,
    /// Threshold in dB for spectrum analysis
    spectrum_threshold: i32,
    /// Spectrum data from the most recent analysis
    spectrum_data: Arc<Mutex<SpectrumData>>,
    /// bpm
    bpm_value: Arc<Mutex<f32>>,
}

impl VideoTextureManager {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bind_group_layout: &wgpu::BindGroupLayout,
        video_path: impl AsRef<Path>,
    ) -> Result<Self> {
        // Create a default 1x1 texture initially, note that, this going to be replaced with first video frame
        let default_image = image::RgbaImage::new(1, 1);
        let texture_manager = TextureManager::new(device, queue, &default_image, bind_group_layout);

        let path_str = video_path
            .as_ref()
            .to_str()
            .ok_or_else(|| anyhow!("Invalid video path"))?
            .to_string();

        info!("Creating video texture from: {}", path_str);

        let pipeline = gst::Pipeline::new();

        // Source element - read from file
        let filesrc = gst::ElementFactory::make("filesrc")
            .name("source")
            .property("location", &path_str)
            .build()
            .map_err(|_| anyhow!("Failed to create filesrc element"))?;
        // Decoding element
        let decodebin = gst::ElementFactory::make("decodebin")
            .name("decoder")
            .build()
            .map_err(|_| anyhow!("Failed to create decodebin element"))?;

        // videorate element to enforce correct frame timing
        let videorate = gst::ElementFactory::make("videorate")
            .name("rate")
            .build()
            .map_err(|_| anyhow!("Failed to create videorate element"))?;
        // Convert to proper format
        let videoconvert = gst::ElementFactory::make("videoconvert")
            .name("convert")
            .build()
            .map_err(|_| anyhow!("Failed to create videoconvert element"))?;
        //caps filter to force framerate if needed
        let capsfilter = gst::ElementFactory::make("capsfilter")
            .name("capsfilter")
            .build()
            .map_err(|_| anyhow!("Failed to create capsfilter element"))?;

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
        // Drop old buffers when full
        appsink.set_drop(true);
        appsink.set_sync(true);

        // video elements goes to the pipeline
        pipeline
            .add_many(&[
                &filesrc,
                &decodebin,
                &videorate,
                &videoconvert,
                &capsfilter,
                &appsink.upcast_ref(),
            ])
            .map_err(|_| anyhow!("Failed to add video elements to pipeline"))?;

        // Link elements that can be linked statically
        gst::Element::link_many(&[
            &videorate,
            &videoconvert,
            &capsfilter,
            &appsink.upcast_ref(),
        ])
        .map_err(|_| anyhow!("Failed to link video elements"))?;

        gst::Element::link_many(&[&filesrc, &decodebin])
            .map_err(|_| anyhow!("Failed to link filesrc to decodebin"))?;

        // Set up pad-added signal for dynamic linking from decodebin -> videorate
        let videorate_weak = videorate.downgrade();
        let has_audio = Arc::new(Mutex::new(false));
        let has_audio_clone = has_audio.clone();

        // now audio elements reference holders
        let audioconvert_weak = Arc::new(Mutex::new(None));

        // Default spectrum configuration
        let spectrum_bands = 128;
        let spectrum_threshold = -60;
        let spectrum_enabled = true;
        let spectrum_data = Arc::new(Mutex::new(SpectrumData::default()));

        // bus watch for spectrum messages with debug
        let bus = pipeline.bus().expect("Pipeline has no bus");
        let spectrum_data_clone2 = spectrum_data.clone();
        let bpm_value_clone = Arc::new(Mutex::new(0.0));
        // message handler for spectrum messages (these are my sanity checks)
        let _ = bus.add_watch(move |_, message| {
            // Log ALL message types
            info!("Bus message received: type={:?}", message.type_());

            // Print source information
            if let Some(src) = message.src() {
                info!("Message source: {}", src.name());
            }

            match message.view() {
                gst::MessageView::Element(element) => {
                    if let Some(structure) = element.structure() {
                        info!("Element message structure name: '{}'", structure.name());

                        // Explicitly check for spectrum messages
                        if structure.name() == "spectrum" {
                            info!("SPECTRUM MESSAGE DETECTED ");
                            info!("Full structure: {}", structure.to_string());

                            // Try ALL possible ways to extract spectrum data
                            let mut magnitude_values = Vec::new();

                            // Method 1: Direct indexing
                            for i in 0..5 {  // Just try first 5 bands initially
                                let field_name = format!("magnitude[{}]", i);
                                match structure.get::<f32>(&field_name) {
                                    Ok(value) => {
                                        info!("✅ Method 1 - Band {}: {} dB", i, value);
                                        magnitude_values.push(value);
                                    },
                                    Err(e) => {
                                        info!("❌ Method 1 failed: {:?}", e);
                                        break;
                                    }
                                }
                            }

                            // Method 2: Try to access magnitude as array field
                            if structure.has_field("magnitude") {
                                info!("✅ Structure has 'magnitude' field");
                            } else {
                                info!("❌ Structure does NOT have 'magnitude' field");
                            }

                            // Method 3: Parse from structure string
                            let struct_str = structure.to_string();
                            info!("Structure string: {}", struct_str);

                            // If we found magnitude values through any method, process them
                            if !magnitude_values.is_empty() {
                                // Continue extracting all magnitude values
                                let mut i = magnitude_values.len();
                                loop {
                                    let field_name = format!("magnitude[{}]", i);
                                    if let Ok(value) = structure.get::<f32>(&field_name) {
                                        magnitude_values.push(value);
                                        i += 1;
                                    } else {
                                        break;
                                    }
                                }

                                // Log summary of extracted data
                                info!("Extracted {} magnitude values", magnitude_values.len());

                                // Calculate average magnitude
                                let avg_magnitude = magnitude_values.iter().sum::<f32>() / magnitude_values.len() as f32;
                                info!("Average magnitude: {:.2} dB", avg_magnitude);

                                // Find peak frequency
                                if let Some((peak_idx, &peak_val)) = magnitude_values.iter()
                                    .enumerate()
                                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                                {
                                    info!("Peak frequency: band {} at {:.2} dB", peak_idx, peak_val);
                                }

                                // Update spectrum data
                                if let Ok(mut data) = spectrum_data_clone2.lock() {
                                    *data = SpectrumData {
                                        bands: magnitude_values.len(),
                                        magnitudes: magnitude_values,
                                        phases: None,
                                        timestamp: structure.get("timestamp").ok(),
                                    };
                                }
                            } else {
                                warn!("⚠️ Failed to extract any magnitude values from spectrum message");
                            }
                        }

                        if structure.name() == "bpm" || structure.name().contains("bpm") {
                                    info!("BPM MESSAGE DETECTED");
                                    info!("Full BPM structure: {}", structure.to_string());
                                    // Try to extract BPM value from structure
                                    if let Ok(bpm_val) = structure.get::<f32>("bpm") {
                                        info!("BPM detected: {:.1}", bpm_val);
                                        // Update our stored BPM value
                                        if let Ok(mut bpm_lock) = bpm_value_clone.lock() {
                                            *bpm_lock = bpm_val as f32;
                                        }
                                    }
                                }
                            }
                        },
                        gst::MessageView::Tag(tag) => {
                            let tags = tag.tags();
                            // Check for BPM tag
                            if let Some(bpm) = tags.get::<gst::tags::BeatsPerMinute>() {
                                let bpm_val = bpm.get();
                                info!("BPM tag detected: {:.1}", bpm_val);

                                if bpm_val > 0.0 {
                                    if let Ok(mut bpm_lock) = bpm_value_clone.lock() {
                                        *bpm_lock = bpm_val as f32;
                                    }
                                }
                    }
                },
                gst::MessageView::Error(err) => {
                    error!("Pipeline error: {} ({})", err.error(), err.debug().unwrap_or_default());
                },
                _ => (),
            }
            ControlFlow::Continue
        }).map_err(|_| anyhow!("Failed to add bus watch"));

        decodebin.connect_pad_added(move |_, pad| {
            let caps = match pad.current_caps() {
                Some(caps) => caps,
                _ => return,
            };

            let structure = match caps.structure(0) {
                Some(s) => s,
                _ => return,
            };
            // Check if this is a video or audio stream. maybe there could be other way to handle this but I love simpicity
            if structure.name().starts_with("video/") {
                // Handle video path
                if let Some(videorate) = videorate_weak.upgrade() {
                    let sink_pad = match videorate.static_pad("sink") {
                        Some(pad) => pad,
                        _ => return,
                    };

                    if !sink_pad.is_linked() {
                        let _ = pad.link(&sink_pad);
                        info!("Linked decoder to videorate successfully");
                    }
                }
            } else if structure.name().starts_with("audio/") {
                // has_audio flag to true - we've detected an audio stream
                if let Ok(mut has_audio_lock) = has_audio_clone.lock() {
                    *has_audio_lock = true;
                    info!("Audio track detected in video");
                }

                // Now lets dynamically create the audio processing chain
                if let Ok(mut audioconvert_lock) = audioconvert_weak.lock() {
                    // Only create audio elements once when first audio pad is detected
                    if audioconvert_lock.is_none() {
                        // Create audio elements
                        let audioconvert = match gst::ElementFactory::make("audioconvert")
                            .name("audioconvert")
                            .build()
                        {
                            Ok(e) => e,
                            Err(_) => {
                                warn!("Failed to create audioconvert");
                                return;
                            }
                        };

                        let audioresample = match gst::ElementFactory::make("audioresample")
                            .name("audioresample")
                            .build()
                        {
                            Ok(e) => e,
                            Err(_) => {
                                warn!("Failed to create audioresample");
                                return;
                            }
                        };
                        let bpmdetect = match gst::ElementFactory::make("bpmdetect")
                            .name("bpmdetect")
                            .build()
                        {
                            Ok(e) => {
                                info!("Created BPM detector");
                                e
                            }
                            Err(_) => {
                                warn!("Failed to create BPM detector");
                                return;
                            }
                        };
                        let spectrum = match gst::ElementFactory::make("spectrum")
                            .name("spectrum")
                            .property("bands", spectrum_bands as u32)
                            .property("threshold", spectrum_threshold)
                            .property("post-messages", true)
                            .property("message-magnitude", true)
                            .property("message-phase", false)
                            .property("interval", 50000000u64)
                            .build()
                        {
                            Ok(e) => {
                                info!(
                                    "Created spectrum analyzer with {} bands and threshold {}dB",
                                    spectrum_bands, spectrum_threshold
                                );
                                e
                            }
                            Err(_) => {
                                warn!("Failed to create spectrum analyzer");
                                return;
                            }
                        };
                        let volume = match gst::ElementFactory::make("volume")
                            .name("volume")
                            .property("volume", 1.0)
                            .build()
                        {
                            Ok(e) => e,
                            Err(_) => {
                                warn!("Failed to create volume");
                                return;
                            }
                        };

                        // autoaudiosink should works on all platforms: https://gstreamer.freedesktop.org/documentation/autodetect/autoaudiosink.html?gi-language=c
                        let audio_sink = match gst::ElementFactory::make("autoaudiosink")
                            .name("audiosink")
                            .build()
                        {
                            Ok(e) => e,
                            Err(_) => {
                                warn!("Failed to create autoaudiosink");
                                return;
                            }
                        };

                        // Add elements to pipeline
                        if let Err(e) = pad
                            .parent_element()
                            .unwrap()
                            .parent()
                            .unwrap()
                            .downcast_ref::<gst::Pipeline>()
                            .unwrap()
                            .add_many(&[
                                &audioconvert,
                                &audioresample,
                                &bpmdetect,
                                &spectrum,
                                &volume,
                                &audio_sink,
                            ])
                        {
                            warn!("Failed to add audio elements: {:?}", e);
                            return;
                        }
                        // Link audio elements
                        if let Err(e) = gst::Element::link_many(&[
                            &audioconvert,
                            &audioresample,
                            &bpmdetect,
                            &spectrum,
                            &volume,
                            &audio_sink,
                        ]) {
                            warn!("Failed to link audio elements: {:?}", e);
                            return;
                        }

                        // Set elements to PAUSED state
                        let _ = audioconvert.sync_state_with_parent();
                        let _ = audioresample.sync_state_with_parent();
                        let _ = bpmdetect.sync_state_with_parent();
                        let _ = spectrum.sync_state_with_parent();
                        let _ = volume.sync_state_with_parent();
                        let _ = audio_sink.sync_state_with_parent();

                        *audioconvert_lock = Some(audioconvert.clone());
                    }

                    // Link decoder pad to audioconvert
                    if let Some(audioconvert) = &*audioconvert_lock {
                        let sink_pad = match audioconvert.static_pad("sink") {
                            Some(pad) => pad,
                            _ => return,
                        };

                        if !sink_pad.is_linked() {
                            match pad.link(&sink_pad) {
                                Ok(_) => {
                                    info!("Linked decoder to audioconvert successfully");
                                }
                                Err(err) => {
                                    warn!("Failed to link audio pad: {:?}", err);
                                }
                            }
                        }
                    }
                }
            }
        });

        // Create shared state
        let current_frame = Arc::new(Mutex::new(None));
        let current_frame_clone = current_frame.clone();
        let position = Arc::new(Mutex::new(gst::ClockTime::ZERO));
        let is_playing = Arc::new(Mutex::new(false));
        let volume_val = Arc::new(Mutex::new(1.0));
        let is_muted = Arc::new(Mutex::new(false));

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
                    // (We need to copy the data because buffer will be unmapped after this function)
                    let mut rgba_image = image::RgbaImage::new(width as u32, height as u32);

                    // Stride might be larger than width * 4
                    let stride = video_info.stride()[0] as usize;

                    for y in 0..height {
                        let src_start = y * stride;
                        let src_end = src_start + width * 4;
                        let dst_start = y * width * 4;
                        let dst_end = dst_start + width * 4;

                        // Copy row by row to handle stride correctly
                        let dst_buffer = rgba_image.as_mut();
                        if src_end <= frame_data.len() && dst_end <= dst_buffer.len() {
                            dst_buffer[dst_start..dst_end]
                                .copy_from_slice(&frame_data[src_start..src_end]);
                        }
                    }

                    // Store the frame
                    if let Ok(mut frame_lock) = current_frame_clone.lock() {
                        *frame_lock = Some(rgba_image);
                    }

                    Ok(gst::FlowSuccess::Ok)
                })
                .build(),
        );

        // init the object
        let mut video_texture = Self {
            texture_manager,
            pipeline,
            appsink,
            has_audio: false,
            volume: volume_val,
            is_muted,
            dimensions: (1, 1),
            duration: None,
            position,
            framerate: None,
            is_playing,
            loop_playback: Arc::new(Mutex::new(true)),
            last_update: Instant::now(),
            current_frame,
            video_path: path_str,
            texture_initialized: false,
            frame_count: 0,
            spectrum_enabled,
            spectrum_bands,
            spectrum_threshold,
            spectrum_data,
            bpm_value: Arc::new(Mutex::new(0.0)),
        };
        // Start pipeline in paused state to get video info
        if video_texture
            .pipeline
            .set_state(gst::State::Paused)
            .is_err()
        {
            return Err(anyhow!("Failed to set pipeline to PAUSED state"));
        }

        // Wait a bit for pipeline to settle
        std::thread::sleep(Duration::from_millis(200));

        let state_result = video_texture
            .pipeline
            .state(gst::ClockTime::from_seconds(1));
        if let (_, gst::State::Paused, _) = state_result {
            video_texture.query_video_info()?;
        } else {
            warn!("Pipeline not in PAUSED state, may not be able to query info");
        }
        // Set specific framerate in the caps filter if we detected one
        if let Some(framerate) = video_texture.framerate {
            if let Some(capsfilter_elem) = video_texture.pipeline.by_name("capsfilter") {
                let caps = gst::Caps::builder("video/x-raw")
                    .field("framerate", framerate)
                    .build();
                capsfilter_elem.set_property("caps", &caps);
                info!(
                    "Set capsfilter to force framerate {}/{}",
                    framerate.numer(),
                    framerate.denom()
                );
            }
        }
        video_texture.has_audio = *has_audio.lock().unwrap();
        info!("Video has audio: {}", video_texture.has_audio);

        info!("Video texture manager created successfully");
        Ok(video_texture)
    }

    /// Query video information (dimensions, duration, framerate)
    fn query_video_info(&mut self) -> Result<()> {
        // Query duration
        if let Some(duration) = self.pipeline.query_duration::<gst::ClockTime>() {
            self.duration = Some(duration);
            info!(
                "Video duration: {:?} ({:.2} seconds)",
                duration,
                duration.seconds() as f64
            );
        }

        // Now try to get video info
        if let Some(pad) = self.appsink.static_pad("sink") {
            if let Some(caps) = pad.current_caps() {
                if let Some(s) = caps.structure(0) {
                    //  dims
                    if let (Ok(width), Ok(height)) = (s.get::<i32>("width"), s.get::<i32>("height"))
                    {
                        self.dimensions = (width as u32, height as u32);
                        info!("Video dimensions: {}x{}", width, height);
                    }

                    // framerate
                    if let Ok(framerate) = s.get::<gst::Fraction>("framerate") {
                        self.framerate = Some(framerate);
                        info!(
                            "Video framerate: {}/{} (approx. {:.2} fps)",
                            framerate.numer(),
                            framerate.denom(),
                            framerate.numer() as f64 / framerate.denom() as f64
                        );
                    }
                }
            }
        }

        Ok(())
    }

    /// Get the texture manager (for binding to shaders)
    pub fn texture_manager(&self) -> &TextureManager {
        &self.texture_manager
    }

    /// Update the texture with the current video frame
    pub fn update_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Result<bool> {
        // No update needed if video is not playing
        if !*self.is_playing.lock().unwrap() {
            return Ok(false);
        }

        // Check if we have a NEW frame to process
        let frame_to_process = {
            let mut frame_lock = self.current_frame.lock().unwrap();
            frame_lock.take()
        };

        // If we have a frame, update the texture
        if let Some(frame) = frame_to_process {
            self.frame_count += 1;

            // Get frame dimensions
            let width = frame.width();
            let height = frame.height();

            // Log less frequently to reduce spam
            if self.frame_count % 30 == 0 {
                debug!(
                    "Processing video frame #{} (dimensions: {}x{})",
                    self.frame_count, width, height
                );
            }
            if self.has_audio && self.spectrum_enabled {
                if let Some(bus) = self.pipeline.bus() {
                    // Poll for pending messages
                    while let Some(message) = bus.pop() {
                        match message.view() {
                            gst::MessageView::Element(element) => {
                                if let Some(structure) = element.structure() {
                                    //spectrum data
                                    if structure.name() == "spectrum" {
                                        info!("🎵 Spectrum data received");

                                        // Extract magnitude values - two possible approaches
                                        let mut magnitude_values = Vec::with_capacity(128);

                                        // APPROACH 1: Parse from structure string
                                        let struct_str = structure.to_string();
                                        if struct_str.contains("magnitude=(float){") {
                                            // Extract magnitude values from string
                                            if let Some(start_idx) =
                                                struct_str.find("magnitude=(float){")
                                            {
                                                if let Some(end_idx) =
                                                    struct_str[start_idx..].find("}")
                                                {
                                                    let magnitude_str = &struct_str[start_idx
                                                        + "magnitude=(float){".len()
                                                        ..start_idx + end_idx];
                                                    let values: Vec<&str> =
                                                        magnitude_str.split(',').collect();

                                                    for value_str in values {
                                                        if let Ok(value) =
                                                            value_str.trim().parse::<f32>()
                                                        {
                                                            magnitude_values.push(value);
                                                        }
                                                    }

                                                    info!(
                                                        "Extracted {} magnitude values from string",
                                                        magnitude_values.len()
                                                    );
                                                }
                                            }
                                        }

                                        // APPROACH 2: Try to access directly by index if approach 1 fails
                                        if magnitude_values.is_empty() {
                                            for i in 0..128 {
                                                let field_name = format!("magnitude[{}]", i);
                                                if let Ok(value) = structure.get::<f32>(&field_name)
                                                {
                                                    magnitude_values.push(value);
                                                } else {
                                                    break;
                                                }
                                            }

                                            if !magnitude_values.is_empty() {
                                                info!(
                                                    "Extracted {} magnitude values by field access",
                                                    magnitude_values.len()
                                                );
                                            }
                                        }

                                        // Process spectrum data if we have it
                                        if !magnitude_values.is_empty() {
                                            // Calculate audio metrics
                                            let bands = magnitude_values.len();

                                            // Calculate average and normalize values
                                            // Values are in dB (typically negative, with higher/less negative being louder)
                                            let threshold = self.spectrum_threshold as f32;

                                            // Create normalized values from -60dB (silence) to 0dB (loudest)
                                            let normalized_values: Vec<f32> = magnitude_values
                                                .iter()
                                                .map(|&v| {
                                                    ((v - threshold) / -threshold).max(0.0).min(1.0)
                                                })
                                                .collect();

                                            // Calculate frequency band energy averages (useful for visualization)
                                            let bass_range = (bands as f32 * 0.1) as usize; // First 10% of frequencies
                                            let mid_range_start = bass_range;
                                            let mid_range_end = (bands as f32 * 0.5) as usize; // 10-50% of frequencies
                                            let high_range_start = mid_range_end;

                                            let bass_energy = if bass_range > 0 {
                                                normalized_values[0..bass_range].iter().sum::<f32>()
                                                    / bass_range as f32
                                            } else {
                                                0.0
                                            };

                                            let mid_energy = if mid_range_end > mid_range_start {
                                                normalized_values[mid_range_start..mid_range_end]
                                                    .iter()
                                                    .sum::<f32>()
                                                    / (mid_range_end - mid_range_start) as f32
                                            } else {
                                                0.0
                                            };

                                            let high_energy = if bands > high_range_start {
                                                normalized_values[high_range_start..]
                                                    .iter()
                                                    .sum::<f32>()
                                                    / (bands - high_range_start) as f32
                                            } else {
                                                0.0
                                            };

                                            // Find peak frequency band
                                            let peak_info = normalized_values
                                                .iter()
                                                .enumerate()
                                                .max_by(|(_, a), (_, b)| {
                                                    a.partial_cmp(b)
                                                        .unwrap_or(std::cmp::Ordering::Equal)
                                                });

                                            if let Some((peak_idx, &peak_val)) = peak_info {
                                                // Rough estimate of frequency based on band index
                                                let peak_freq =
                                                    (peak_idx as f32 / bands as f32) * 20000.0;
                                                info!(
                                                    "Peak freq: ~{:.0} Hz (band {}, val: {:.2})",
                                                    peak_freq, peak_idx, peak_val
                                                );
                                            }

                                            // Log energy metrics
                                            info!("Audio energy - Bass: {:.2}, Mid: {:.2}, High: {:.2}",
                                                 bass_energy, mid_energy, high_energy);

                                            // Update spectrum data
                                            if let Ok(mut data) = self.spectrum_data.lock() {
                                                *data = SpectrumData {
                                                    bands,
                                                    magnitudes: magnitude_values,
                                                    phases: None,
                                                    timestamp: structure.get("timestamp").ok(),
                                                };
                                            }
                                        }
                                    }

                                    // Check for BPM messages
                                    if structure.name() == "bpm" || structure.name().contains("bpm")
                                    {
                                        info!("🎵 BPM data received");
                                        info!("Full BPM structure: {}", structure.to_string());

                                        // Try to extract BPM value directly
                                        if let Ok(bpm_val) = structure.get::<f32>("bpm") {
                                            info!("BPM detected: {:.1}", bpm_val);
                                            if let Ok(mut bpm_lock) = self.bpm_value.lock() {
                                                // Apply musical heuristics to handle tempo octave ambiguity: https://www.ifs.tuwien.ac.at/~knees/publications/hoerschlaeger_etal_smc_2015.pdf
                                                let current_bpm = *bpm_lock;

                                                if current_bpm == 0.0 && bpm_val > 0.0 {
                                                    // First detection - apply preference for 70-150 BPM range
                                                    if bpm_val > 150.0 {
                                                        // If detected BPM is high, use half tempo
                                                        *bpm_lock = (bpm_val / 2.0) as f32;
                                                        info!(
                                                            "Initial BPM halved: {:.1} → {:.1}",
                                                            bpm_val, *bpm_lock
                                                        );
                                                    } else if bpm_val < 70.0 {
                                                        // If detected BPM is low, use double tempo
                                                        *bpm_lock = (bpm_val * 2.0) as f32;
                                                        info!(
                                                            "Initial BPM doubled: {:.1} → {:.1}",
                                                            bpm_val, *bpm_lock
                                                        );
                                                    } else {
                                                        // Within preferred range - use directly
                                                        *bpm_lock = bpm_val as f32;
                                                    }
                                                } else if current_bpm > 0.0 && bpm_val > 0.0 {
                                                    // Subsequent detection - check for tempo octave jumps
                                                    if bpm_val > current_bpm * 1.8
                                                        && bpm_val < current_bpm * 2.2
                                                    {
                                                        // Double tempo detected - stay in preferred range if possible
                                                        let target = if current_bpm >= 70.0
                                                            && current_bpm <= 150.0
                                                        {
                                                            current_bpm // Keep current if already in good range
                                                        } else if bpm_val >= 70.0
                                                            && bpm_val <= 150.0
                                                        {
                                                            bpm_val // Use new if it's in good range
                                                        } else {
                                                            // Neither in ideal range - prefer the lower value
                                                            current_bpm
                                                        };
                                                        *bpm_lock = target as f32;
                                                        info!("Tempo doubling corrected: {:.1} → {:.1}", bpm_val, *bpm_lock);
                                                    } else if bpm_val > current_bpm * 0.45
                                                        && bpm_val < current_bpm * 0.55
                                                    {
                                                        // Half tempo detected - stay in preferred range if possible
                                                        let target = if current_bpm >= 70.0
                                                            && current_bpm <= 150.0
                                                        {
                                                            current_bpm // Keep current if already in good range
                                                        } else if bpm_val >= 70.0
                                                            && bpm_val <= 150.0
                                                        {
                                                            bpm_val // Use new if it's in good range
                                                        } else {
                                                            // Neither in ideal range - prefer the higher value
                                                            current_bpm
                                                        };
                                                        *bpm_lock = target as f32;
                                                        info!("Tempo halving corrected: {:.1} → {:.1}", bpm_val, *bpm_lock);
                                                    } else {
                                                        // Apply light smoothing to avoid jumps
                                                        *bpm_lock = (current_bpm * 0.8
                                                            + bpm_val as f32 * 0.2)
                                                            as f32;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            // Also check for tag messages as bpmdetect might send tags
                            gst::MessageView::Tag(tag) => {
                                let tags = tag.tags();

                                // Check for BPM tag
                                if let Some(bpm) = tags.get::<gst::tags::BeatsPerMinute>() {
                                    let bpm_val = bpm.get();
                                    info!("BPM tag detected: {:.1}", bpm_val);

                                    if bpm_val > 0.0 {
                                        // Update stored BPM value with musical heuristics
                                        if let Ok(mut bpm_lock) = self.bpm_value.lock() {
                                            let current_bpm = *bpm_lock;
                                            let bpm_val_f32 = bpm_val as f32;

                                            // Tags are usually pre-processed, but I ll apply heuristics anyway
                                            if current_bpm == 0.0 {
                                                // First detection - apply preference for 70-150 BPM range
                                                if bpm_val_f32 > 150.0 {
                                                    *bpm_lock = bpm_val_f32 / 2.0;
                                                    info!(
                                                        "Tag BPM halved: {:.1} → {:.1}",
                                                        bpm_val, *bpm_lock as f64
                                                    );
                                                } else if bpm_val_f32 < 70.0 {
                                                    *bpm_lock = bpm_val_f32 * 2.0;
                                                    info!(
                                                        "Tag BPM doubled: {:.1} → {:.1}",
                                                        bpm_val, *bpm_lock as f64
                                                    );
                                                } else {
                                                    *bpm_lock = bpm_val_f32;
                                                }
                                            } else {
                                                // Check for octave relationship and prefer values in 70-150 range
                                                let double_current = current_bpm * 2.0;
                                                let half_current = current_bpm / 2.0;

                                                if (bpm_val_f32 > current_bpm * 1.8
                                                    && bpm_val_f32 < current_bpm * 2.2)
                                                    || (bpm_val_f32 > current_bpm * 0.45
                                                        && bpm_val_f32 < current_bpm * 0.55)
                                                {
                                                    // Octave relationship detected
                                                    // Choose value in preferred range
                                                    let candidates = [
                                                        bpm_val_f32,
                                                        current_bpm,
                                                        double_current,
                                                        half_current,
                                                    ];
                                                    let preferred = candidates
                                                        .iter()
                                                        .filter(|&&v| v >= 70.0 && v <= 150.0)
                                                        .min_by(|a, b| {
                                                            let a_dist = (**a - 110.0).abs();
                                                            let b_dist = (**b - 110.0).abs();
                                                            a_dist.partial_cmp(&b_dist).unwrap_or(
                                                                std::cmp::Ordering::Equal,
                                                            )
                                                        });

                                                    if let Some(&best_bpm) = preferred {
                                                        *bpm_lock = best_bpm;
                                                        info!("Tag BPM adjusted to preferred range: {:.1} → {:.1}", bpm_val, *bpm_lock as f64);
                                                    } else {
                                                        // No value in preferred range, use tag value
                                                        *bpm_lock = bpm_val_f32;
                                                    }
                                                } else {
                                                    // Not an octave relationship - tags are usually reliable
                                                    // Use 30% weighting for current value to avoid abrupt changes
                                                    *bpm_lock =
                                                        current_bpm * 0.3 + bpm_val_f32 * 0.7;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            _ => (),
                        }
                    }
                }
            }
            // ALWAYS recreate the texture for the first frame or if dimensions don't match
            let should_recreate = !self.texture_initialized
                || self.dimensions != (width, height)
                || self.dimensions.0 <= 1
                || self.dimensions.1 <= 1
                || self.frame_count <= 3;

            if should_recreate {
                info!("Creating new texture with dimensions: {}x{}", width, height);

                // Create a completely new texture with the frame's dimensions
                let new_texture_manager =
                    TextureManager::new(device, queue, &frame, bind_group_layout);

                self.texture_manager = new_texture_manager;
                self.dimensions = (width, height);
                self.texture_initialized = true;
            } else {
                self.texture_manager.update(queue, &frame);
            }

            // Get current position
            if let Some(position) = self.pipeline.query_position::<gst::ClockTime>() {
                *self.position.lock().unwrap() = position;

                // Check if we reached the end of the video
                if let Some(duration) = self.duration {
                    let near_end_threshold =
                        duration.saturating_sub(gst::ClockTime::from_mseconds(100));

                    if position >= near_end_threshold {
                        debug!(
                            "Near end of video (position: {:?}, duration: {:?})",
                            position, duration
                        );

                        if *self.loop_playback.lock().unwrap() {
                            debug!("Looping video");
                            self.seek(gst::ClockTime::ZERO)?;
                        } else {
                            debug!("Pausing at end of video");
                            self.pause()?;
                        }
                    }
                }
            }

            // Update the last update time
            self.last_update = Instant::now();

            Ok(true) // Texture was updated
        } else {
            Ok(false) // No update
        }
    }

    /// Start playing the video
    pub fn play(&mut self) -> Result<()> {
        info!("Playing video");
        match self.pipeline.set_state(gst::State::Playing) {
            Ok(_) => {
                *self.is_playing.lock().unwrap() = true;
                Ok(())
            }
            Err(e) => Err(anyhow!("Failed to start playback: {:?}", e)),
        }
    }

    /// Pause the video
    pub fn pause(&mut self) -> Result<()> {
        info!("Pausing video");
        match self.pipeline.set_state(gst::State::Paused) {
            Ok(_) => {
                *self.is_playing.lock().unwrap() = false;
                Ok(())
            }
            Err(e) => Err(anyhow!("Failed to pause playback: {:?}", e)),
        }
    }

    /// Seek to a specific position in the video
    pub fn seek(&mut self, position: gst::ClockTime) -> Result<()> {
        debug!("Seeking to position: {:?}", position);

        // are we sure the pipeline is in a state that can handle seeking??
        let current_state = self.pipeline.current_state();
        if current_state == gst::State::Null || current_state == gst::State::Ready {
            warn!("Cannot seek in current state: {:?}", current_state);
            return Ok(());
        }

        // exec the seek operation
        let seek_flags = gst::SeekFlags::FLUSH | gst::SeekFlags::KEY_UNIT;
        if self.pipeline.seek_simple(seek_flags, position).is_ok() {
            debug!("Seek successful");
            *self.position.lock().unwrap() = position;
            Ok(())
        } else {
            let err = anyhow!("Failed to seek to {:?}", position);
            error!("{}", err);
            Err(err)
        }
    }

    pub fn set_loop(&mut self, should_loop: bool) {
        *self.loop_playback.lock().unwrap() = should_loop;
        info!("Video loop set to: {}", should_loop);
    }

    /// audio volume (between 0.0 and 1.0)
    pub fn set_volume(&mut self, volume: f64) -> Result<()> {
        if !self.has_audio {
            debug!("Ignoring volume change request - video has no audio");
            return Ok(());
        }

        let clamped_volume = volume.max(0.0).min(1.0);
        *self.volume.lock().unwrap() = clamped_volume;

        if let Some(volume_elem) = self.pipeline.by_name("volume") {
            volume_elem.set_property("volume", clamped_volume);
            debug!("Set volume to {:.2}", clamped_volume);
            Ok(())
        } else {
            warn!("Volume element not found in pipeline");
            Ok(()) // Don't fail if element not found - video might still work
        }
    }

    /// Mutes or unmutes the audio
    pub fn set_mute(&mut self, muted: bool) -> Result<()> {
        if !self.has_audio {
            debug!("Ignoring mute request - video has no audio");
            return Ok(());
        }

        *self.is_muted.lock().unwrap() = muted;

        if let Some(volume_elem) = self.pipeline.by_name("volume") {
            volume_elem.set_property("mute", muted);
            debug!("Set mute to {}", muted);
            Ok(())
        } else {
            warn!("Volume element not found in pipeline");
            Ok(()) // Don't fail if element not found - video might still work
        }
    }

    pub fn toggle_mute(&mut self) -> Result<()> {
        let current_mute = *self.is_muted.lock().unwrap();
        self.set_mute(!current_mute)
    }

    /// Returns true if the video has an audio track
    pub fn has_audio(&self) -> bool {
        self.has_audio
    }

    /// Gets the current volume (0.0 to 1.0)
    pub fn volume(&self) -> f64 {
        *self.volume.lock().unwrap()
    }

    /// Returns true if audio is currently muted
    pub fn is_muted(&self) -> bool {
        *self.is_muted.lock().unwrap()
    }

    pub fn position(&self) -> gst::ClockTime {
        *self.position.lock().unwrap()
    }

    pub fn duration(&self) -> Option<gst::ClockTime> {
        self.duration
    }

    pub fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    pub fn framerate(&self) -> Option<(i32, i32)> {
        self.framerate.map(|f| (f.numer(), f.denom()))
    }

    pub fn is_playing(&self) -> bool {
        *self.is_playing.lock().unwrap()
    }

    pub fn is_looping(&self) -> bool {
        *self.loop_playback.lock().unwrap()
    }

    pub fn path(&self) -> &str {
        &self.video_path
    }

    /// Configure spectrum analysis parameters
    pub fn configure_spectrum(&mut self, bands: usize, threshold: i32) -> Result<()> {
        if !self.has_audio {
            debug!("Ignoring spectrum configuration - video has no audio");
            return Ok(());
        }

        self.spectrum_bands = bands;
        self.spectrum_threshold = threshold;

        if let Some(spectrum_elem) = self.pipeline.by_name("spectrum") {
            spectrum_elem.set_property("bands", bands as u32);
            spectrum_elem.set_property("threshold", threshold);
            info!(
                "Configured spectrum: {} bands, {}dB threshold",
                bands, threshold
            );
            Ok(())
        } else {
            warn!("Spectrum element not found in pipeline");
            Ok(()) // Don't fail if element not found
        }
    }

    /// Enable or disable spectrum analysis
    pub fn enable_spectrum(&mut self, enabled: bool) -> Result<()> {
        if !self.has_audio {
            debug!("Ignoring spectrum enable request - video has no audio");
            return Ok(());
        }

        self.spectrum_enabled = enabled;

        if let Some(spectrum_elem) = self.pipeline.by_name("spectrum") {
            spectrum_elem.set_property("post-messages", enabled);
            info!(
                "Spectrum analysis {} ",
                if enabled { "enabled" } else { "disabled" }
            );
            Ok(())
        } else {
            warn!("Spectrum element not found in pipeline");
            Ok(()) // Don't fail if element not found
        }
    }

    /// Get current spectrum data
    pub fn spectrum_data(&self) -> SpectrumData {
        match self.spectrum_data.lock() {
            Ok(data) => data.clone(),
            Err(_) => SpectrumData::default(),
        }
    }
    pub fn get_bpm(&self) -> f32 {
        if !self.has_audio {
            return 0.0;
        }

        match self.bpm_value.lock() {
            Ok(bpm) => *bpm,
            Err(_) => 0.0,
        }
    }
    /// Set interval between spectrum updates in milliseconds
    pub fn set_spectrum_interval(&mut self, interval_ms: u64) -> Result<()> {
        if !self.has_audio {
            debug!("Ignoring spectrum interval request - video has no audio");
            return Ok(());
        }

        if let Some(spectrum_elem) = self.pipeline.by_name("spectrum") {
            // Convert milliseconds to nanoseconds for GStreamer
            let interval_ns = interval_ms * 1_000_000;
            spectrum_elem.set_property("interval", interval_ns as u64);
            info!("Set spectrum interval to {}ms", interval_ms);
            Ok(())
        } else {
            warn!("Spectrum element not found in pipeline");
            Ok(()) // Don't fail if element not found
        }
    }
}

impl Drop for VideoTextureManager {
    fn drop(&mut self) {
        info!("Shutting down video pipeline");
        let _ = self.pipeline.set_state(gst::State::Null);
    }
}

// This file is part of the gstreamer, and its inits the spectrum analyzer and bpm.
// I also did some smoothing related to audio data for the spectrum analyzer.
#[cfg(feature = "media")]
use crate::gst::offline_audio::AudioSample;
#[cfg(feature = "media")]
use crate::gst::video::VideoTextureManager;
#[cfg(feature = "media")]
use crate::gst::webcam::WebcamTextureManager;
#[cfg(feature = "media")]
use crate::ResolutionUniform;
#[cfg(feature = "media")]
use crate::UniformBinding;
#[cfg(feature = "media")]
use log::info;

pub struct SpectrumAnalyzer {
    #[cfg(feature = "media")]
    prev_audio_data: [[f32; 4]; 32],
}

#[cfg(feature = "media")]
impl Default for SpectrumAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "media")]
impl SpectrumAnalyzer {
    pub fn new() -> Self {
        Self {
            prev_audio_data: [[0.0; 4]; 32],
        }
    }

    /// Reset per-band smoothing state. Called at export start/end so that
    /// transitions between live and offline data don't leak through the
    /// attack/decay filter.
    pub fn reset_smoothing(&mut self) {
        self.prev_audio_data = [[0.0; 4]; 32];
    }

    pub fn update_spectrum(
        &mut self,
        queue: &wgpu::Queue,
        resolution_uniform: &mut UniformBinding<ResolutionUniform>,
        video_texture_manager: &Option<VideoTextureManager>,
        using_video_texture: bool,
        webcam_texture_manager: &Option<WebcamTextureManager>,
        using_webcam_texture: bool,
    ) {
        // Initialize audio data arrays to zero
        for i in 0..32 {
            for j in 0..4 {
                resolution_uniform.data.audio_data[i][j] = 0.0;
            }
        }

        if using_video_texture {
            if let Some(video_manager) = video_texture_manager {
                if video_manager.has_audio() {
                    let spectrum_data = video_manager.spectrum_data();
                    let audio_level = video_manager.audio_level();
                    let bpm = video_manager.get_bpm();

                    if !spectrum_data.magnitudes.is_empty() {
                        self.process_audio_sample(
                            resolution_uniform,
                            &spectrum_data.magnitudes,
                            spectrum_data.bands,
                            audio_level.rms_db as f32,
                            bpm,
                            /* log_live = */ true,
                        );
                    }
                }
            }

            resolution_uniform.update(queue);
        } else if using_webcam_texture {
            // Webcam mic path: same but BPM stays at 0
            if let Some(webcam_manager) = webcam_texture_manager {
                if webcam_manager.has_audio() {
                    let spectrum_data = webcam_manager.spectrum_data();
                    let audio_level = webcam_manager.audio_level();
                    if !spectrum_data.magnitudes.is_empty() {
                        self.process_audio_sample(
                            resolution_uniform,
                            &spectrum_data.magnitudes,
                            spectrum_data.bands,
                            audio_level.rms_db as f32,
                            0.0,
                            /* log_live = */ true,
                        );
                    }
                }
            }

            resolution_uniform.update(queue);
        }
    }

    /// Offline path: feed a single timestamped sample collected by the offline
    /// analyzer into the same processing pipeline that live updates use, so the
    /// visual output is identical between preview and export.
    pub fn apply_offline_sample(
        &mut self,
        queue: &wgpu::Queue,
        resolution_uniform: &mut UniformBinding<ResolutionUniform>,
        sample: &AudioSample<'_>,
    ) {
        for i in 0..32 {
            for j in 0..4 {
                resolution_uniform.data.audio_data[i][j] = 0.0;
            }
        }

        self.process_audio_sample(
            resolution_uniform,
            sample.magnitudes,
            sample.bands,
            sample.rms_db as f32,
            sample.bpm,
            /* log_live = */ false,
        );

        resolution_uniform.update(queue);
    }

    /// Shared audio processing used by both the live and offline paths.
    /// Reads raw spectrum magnitudes (dB scale) and writes normalized,
    /// frequency-shaped, attack/decay-smoothed values into the resolution
    /// uniform, plus computed bass/mid/high/total energies and BPM.
    fn process_audio_sample(
        &mut self,
        resolution_uniform: &mut UniformBinding<ResolutionUniform>,
        magnitudes: &[f32],
        bands: usize,
        rms_db: f32,
        bpm: f32,
        log_live: bool,
    ) {
        resolution_uniform.data.bpm = bpm;

        // Highly sensitive threshold for detecting subtle high frequencies
        let threshold: f32 = -60.0;

        // Calculate adaptive gain based on RMS
        // Target RMS: -20dB (moderate loudness)
        let target_rms_db = -20.0;
        let adaptive_gain = if rms_db > -100.0 {
            let db_diff = target_rms_db - rms_db;
            let gain = 10.0_f32.powf(db_diff / 20.0);
            gain.max(0.3).min(3.0)
        } else {
            1.0
        };

        if log_live {
            info!(
                "Audio Level - RMS: {:.2}dB, Gain: {:.2}x",
                rms_db, adaptive_gain
            );
        }

        // Process only first 64 bands (we typically have 128 but its expensive)
        for i in 0..64 {
            let band_percent = i as f32 / 64.0;
            // Map to source index with slight emphasis on higher frequencies
            let source_idx = (band_percent * (bands as f32 / 2.0)) as usize;
            let width = 1;
            let end_idx = (source_idx + width).min(bands);

            if source_idx < bands {
                let mut peak: f32 = -120.0;
                for j in source_idx..end_idx {
                    if j < bands {
                        let val = magnitudes[j];
                        peak = peak.max(val);
                    }
                }
                // Map from dB scale to 0-1
                let mut normalized = ((peak - threshold) / -threshold).max(0.0).min(1.0);
                normalized = (normalized * adaptive_gain).min(1.0);
                // Frequency-specific processing
                let enhanced = if band_percent < 0.2 {
                    // Bass - slightly reduced
                    (normalized.powf(0.75) * 0.85).min(1.0)
                } else if band_percent < 0.4 {
                    // Low-mids - neutral
                    normalized.powf(0.7).min(1.0)
                } else if band_percent < 0.6 {
                    // Mids - slight boost
                    (normalized.powf(0.65) * 1.1).min(1.0)
                } else if band_percent < 0.8 {
                    // Upper-mids - moderate boost
                    (normalized.powf(0.55) * 1.6).min(1.0)
                } else {
                    // Highs - significant boost with lower power
                    (normalized.powf(0.4) * 3.0).min(1.0)
                };

                // Temporal smoothing with frequency-specific parameters
                let vec_idx = i / 4;
                let vec_component = i % 4;
                if vec_idx < 32 {
                    let prev_value = self.prev_audio_data[vec_idx][vec_component];
                    let attack = if band_percent < 0.6 { 0.6 } else { 0.7 };
                    let decay = if band_percent < 0.6 { 0.3 } else { 0.25 };

                    let smoothing_factor = if enhanced > prev_value { attack } else { decay };
                    let smoothed =
                        prev_value * (1.0 - smoothing_factor) + enhanced * smoothing_factor;
                    resolution_uniform.data.audio_data[vec_idx][vec_component] = smoothed;
                    self.prev_audio_data[vec_idx][vec_component] = smoothed;
                }
            }
        }

        // Compute audio energy for bass/mid/high ranges
        let mut bass_sum = 0.0f32;
        let mut mid_sum = 0.0f32;
        let mut high_sum = 0.0f32;

        for i in 0..64 {
            let vec_idx = i / 4;
            let component = i % 4;
            let value = resolution_uniform.data.audio_data[vec_idx][component];

            let freq = i as f32 / 64.0;
            if freq < 0.2 {
                bass_sum += value;
            } else if freq < 0.6 {
                mid_sum += value;
            } else {
                high_sum += value;
            }
        }

        let bass_energy = bass_sum / 13.0;
        let mid_energy = mid_sum / 26.0;
        let high_energy = high_sum / 25.0;
        let total_energy = (bass_energy * 1.5 + mid_energy + high_energy) / 3.5;

        resolution_uniform.data.bass_energy = bass_energy;
        resolution_uniform.data.mid_energy = mid_energy;
        resolution_uniform.data.high_energy = high_energy;
        resolution_uniform.data.total_energy = total_energy;

        // If we detect a beat, provide progressive boost to mid/high frequencies
        if bass_energy > 0.5 {
            let q1 = 16 / 4;
            let q2 = 16 / 2;
            let q3 = 3 * 16 / 4;

            for i in 0..16 {
                for j in 0..4 {
                    if i < q1 {
                        resolution_uniform.data.audio_data[i][j] *= 0.9;
                    } else if i < q2 {
                        resolution_uniform.data.audio_data[i][j] *= 1.1;
                    } else if i < q3 {
                        resolution_uniform.data.audio_data[i][j] *= 1.3;
                    } else {
                        resolution_uniform.data.audio_data[i][j] *= 1.7;
                    }
                }
            }
        }
    }
}

#[cfg(not(feature = "media"))]
impl SpectrumAnalyzer {
    pub fn new() -> Self {
        Self {}
    }
}

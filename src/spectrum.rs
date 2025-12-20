// This file is part of the gstreamer, and its inits the spectrum analyzer and bpm.
// I also did some smoothing related to audio data for the spectrum analyzer.
#[cfg(feature = "media")]
use crate::gst::video::VideoTextureManager;
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

    pub fn update_spectrum(
        &mut self,
        queue: &wgpu::Queue,
        resolution_uniform: &mut UniformBinding<ResolutionUniform>,
        video_texture_manager: &Option<VideoTextureManager>,
        using_video_texture: bool,
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
                    resolution_uniform.data.bpm = video_manager.get_bpm();

                    if !spectrum_data.magnitudes.is_empty() {
                        let bands = spectrum_data.bands;
                        // Highly sensitive threshold for detecting subtle high frequencies
                        let threshold: f32 = -60.0;

                        // Calculate adaptive gain based on RMS
                        // Target RMS: -20dB (moderate loudness)
                        // Loud songs (metal): RMS ~ -10dB → gain < 1.0 (reduce)
                        // Quiet songs: RMS ~ -30dB → gain > 1.0 (boost)
                        let target_rms_db = -20.0;
                        let current_rms_db = audio_level.rms_db as f32;
                        let adaptive_gain = if current_rms_db > -100.0 {
                            // Calculate gain to bring current RMS closer to target
                            let db_diff = target_rms_db - current_rms_db;
                            // Convert dB difference to linear gain (every 6dB ≈ 2x amplitude)
                            let gain = 10.0_f32.powf(db_diff / 20.0);
                            // Clamp to reasonable range (0.3 to 3.0)
                            gain.max(0.3).min(3.0)
                        } else {
                            1.0 // No adjustment if RMS is invalid
                        };

                        // sanity: to see RMS normalization values
                        info!(
                            "Audio Level - RMS: {:.2}dB, Peak: {:.3}, Gain: {:.2}x",
                            current_rms_db, audio_level.peak, adaptive_gain
                        );

                        // Process only first 64 bands (note that, we actually have 128 but its expensiive)
                        for i in 0..64 {
                            let band_percent = i as f32 / 64.0;
                            // Map to source index with slight emphasis on higher frequencies
                            let source_idx = (band_percent * (bands as f32 / 2.0)) as usize;
                            // Use narrow width for all frequencies for accuracy
                            let width = 1;
                            let end_idx = (source_idx + width).min(bands);

                            if source_idx < bands {
                                // Get peak value in this range
                                let mut peak: f32 = -120.0;
                                for j in source_idx..end_idx {
                                    if j < bands {
                                        let val = spectrum_data.magnitudes[j];
                                        peak = peak.max(val);
                                    }
                                }
                                // Map from dB scale to 0-1
                                let mut normalized =
                                    ((peak - threshold) / -threshold).max(0.0).min(1.0);
                                normalized = (normalized * adaptive_gain).min(1.0);
                                // Apply frequency-specific processing that's balanced
                                // Lower boost for bass, higher boost for treble
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
                                    // The critical adjustment for high frequency sensitivity
                                    (normalized.powf(0.4) * 3.0).min(1.0)
                                };

                                // No minimum thresholds - let silent frequencies be silent
                                // Temporal smoothing with frequency-specific parameters
                                let vec_idx = i / 4;
                                let vec_component = i % 4;
                                if vec_idx < 32 {
                                    let prev_value = self.prev_audio_data[vec_idx][vec_component];
                                    // Fast attack for all frequencies - slightly faster for highs
                                    let attack = if band_percent < 0.6 { 0.6 } else { 0.7 };
                                    let decay = if band_percent < 0.6 { 0.3 } else { 0.25 };

                                    // Apply smoothing
                                    let smoothing_factor = if enhanced > prev_value {
                                        attack // Rising
                                    } else {
                                        decay // Falling
                                    };
                                    // Calculate smoothed value
                                    let smoothed = prev_value * (1.0 - smoothing_factor)
                                        + enhanced * smoothing_factor;
                                    // Store the result
                                    resolution_uniform.data.audio_data[vec_idx][vec_component] =
                                        smoothed;
                                    // Store for next frame
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

                        // Normalize by band count
                        let bass_energy = bass_sum / 13.0; // bands 0-12
                        let mid_energy = mid_sum / 26.0;   // bands 13-38
                        let high_energy = high_sum / 25.0; // bands 39-63
                        let total_energy = (bass_energy * 1.5 + mid_energy + high_energy) / 3.5;

                        // Store in resolution uniform for shaders to access
                        resolution_uniform.data.bass_energy = bass_energy;
                        resolution_uniform.data.mid_energy = mid_energy;
                        resolution_uniform.data.high_energy = high_energy;
                        resolution_uniform.data.total_energy = total_energy;

                        // If we detect a beat, provide progressive boost to mid/high frequencies
                        if bass_energy > 0.5 {
                            // First quarter - bass
                            let q1 = 16 / 4;
                            // Second quarter - low-mids
                            let q2 = 16 / 2;
                            // Third quarter - upper-mids
                            let q3 = 3 * 16 / 4;

                            for i in 0..16 {
                                for j in 0..4 {
                                    if i < q1 {
                                        // No boost for bass (prevent dominance)
                                        // Actually reduce bass slightly on beats
                                        resolution_uniform.data.audio_data[i][j] *= 0.9;
                                    } else if i < q2 {
                                        // Small boost for low-mids
                                        resolution_uniform.data.audio_data[i][j] *= 1.1;
                                    } else if i < q3 {
                                        // Moderate boost for upper-mids
                                        resolution_uniform.data.audio_data[i][j] *= 1.3;
                                    } else {
                                        // Strong boost for highs during beats
                                        resolution_uniform.data.audio_data[i][j] *= 1.7;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            resolution_uniform.update(queue);
        }
    }
}

#[cfg(not(feature = "media"))]
impl SpectrumAnalyzer {
    pub fn new() -> Self {
        Self {}
    }
}

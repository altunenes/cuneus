// This file is part of the gstreamer, and its inits the spectrum analyzer and bpm.
// I also did some smoothing related to audio data for the spectrum analyzer.
#[cfg(feature = "media")]
use log::info;
#[cfg(feature = "media")]
use wgpu;

#[cfg(feature = "media")]
use crate::gst::video::VideoTextureManager;
#[cfg(feature = "media")]
use crate::ResolutionUniform;
#[cfg(feature = "media")]
use crate::UniformBinding;

pub struct SpectrumAnalyzer {
    #[cfg(feature = "media")]
    prev_audio_data: [[f32; 4]; 32],
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
                    resolution_uniform.data.bpm = video_manager.get_bpm();
                    info!("BPM: {}", resolution_uniform.data.bpm);

                    if !spectrum_data.magnitudes.is_empty() {
                        let bands = spectrum_data.bands;
                        // Highly sensitive threshold for detecting subtle high frequencies
                        let threshold: f32 = -60.0;

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
                                let normalized =
                                    ((peak - threshold) / -threshold).max(0.0).min(1.0);
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

                        // Beat detection with balanced boost across frequency spectrum
                        let mut bass_energy: f32 = 0.0;
                        let bass_bands = 64 / 16;
                        for i in 0..(bass_bands / 4) {
                            for j in 0..4 {
                                bass_energy += resolution_uniform.data.audio_data[i][j];
                            }
                        }
                        bass_energy /= bass_bands as f32;

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

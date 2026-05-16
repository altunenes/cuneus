// Offline audio analysis for export.
//
// During the export, frames render at a different rate than realtime audio playback,
// so the live spectrum/level/BPM data doesnt line up with the time of the
// frame being captured. This module runs a non real time gst pipeline
// (fakesink with sync=false) on the same src media to collect timestamped
// spectrum/level/BPM events, which can then be looked up at frame time during
// export and fed to the shader in place of the live data..

use anyhow::{anyhow, Result};
use gst::prelude::*;
use gstreamer as gst;
use log::{info, warn};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub struct SpectrumEvent {
    pub time_secs: f64,
    pub magnitudes: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct LevelEvent {
    pub time_secs: f64,
    pub rms_db: f64,
    pub peak: f64,
}

/// A complete offline analysis of an audio source: every spectrum/level event
/// emitted, plus the final BPM the analyzer settled on.
#[derive(Debug, Clone)]
pub struct OfflineAudioAnalysis {
    pub spectrum_events: Vec<SpectrumEvent>,
    pub level_events: Vec<LevelEvent>,
    pub bands: usize,
    pub duration_secs: f64,
    pub bpm: f32,
}

/// A point-in-time audio sample used by the spectrum analyzer's CPU-side
/// processing. Lifetime borrows magnitudes from the parent analysis.
pub struct AudioSample<'a> {
    pub magnitudes: &'a [f32],
    pub bands: usize,
    pub rms_db: f64,
    pub peak: f64,
    pub bpm: f32,
}

impl OfflineAudioAnalysis {
    /// Run a one shot gst pipeline at maximum speed to collect spectrum,
    /// level, and BPM events. Blocks until EOS or error.
    pub fn analyze(
        media_path: &str,
        bands: usize,
        threshold_db: i32,
        interval_ms: u64,
    ) -> Result<Self> {
        info!(
            "Offline analysis starting: {} (bands={}, threshold={}dB, interval={}ms)",
            media_path, bands, threshold_db, interval_ms
        );

        let pipeline = gst::Pipeline::new();

        let filesrc = gst::ElementFactory::make("filesrc")
            .name("source")
            .property("location", media_path)
            .build()
            .map_err(|_| anyhow!("Failed to create filesrc"))?;

        let decodebin = gst::ElementFactory::make("decodebin")
            .name("decoder")
            .build()
            .map_err(|_| anyhow!("Failed to create decodebin"))?;

        pipeline
            .add_many([&filesrc, &decodebin])
            .map_err(|_| anyhow!("Failed to add elements"))?;
        gst::Element::link_many([&filesrc, &decodebin])
            .map_err(|_| anyhow!("Failed to link filesrc to decodebin"))?;

        let interval_ns = interval_ms * 1_000_000;
        let pipeline_weak = pipeline.downgrade();
        let audio_chain_built = Arc::new(Mutex::new(false));
        let audio_chain_built_clone = audio_chain_built.clone();

        decodebin.connect_pad_added(move |_, pad| {
            let caps = match pad.current_caps() {
                Some(c) => c,
                None => return,
            };
            let structure = match caps.structure(0) {
                Some(s) => s,
                None => return,
            };
            if !structure.name().starts_with("audio/") {
                return;
            }

            let mut built = audio_chain_built_clone.lock().unwrap();
            if *built {
                return;
            }

            let pipeline = match pipeline_weak.upgrade() {
                Some(p) => p,
                None => return,
            };

            let audioconvert = gst::ElementFactory::make("audioconvert")
                .name("offline_audioconvert")
                .build()
                .expect("audioconvert");
            let audioresample = gst::ElementFactory::make("audioresample")
                .name("offline_audioresample")
                .build()
                .expect("audioresample");
            let level = gst::ElementFactory::make("level")
                .name("offline_level")
                .property("interval", interval_ns)
                .property("post-messages", true)
                .build()
                .expect("level");
            let spectrum = gst::ElementFactory::make("spectrum")
                .name("offline_spectrum")
                .property("bands", bands as u32)
                .property("threshold", threshold_db)
                .property("post-messages", true)
                .property("message-magnitude", true)
                .property("message-phase", false)
                .property("interval", interval_ns)
                .build()
                .expect("spectrum");
            let bpmdetect = gst::ElementFactory::make("bpmdetect")
                .name("offline_bpmdetect")
                .build()
                .expect("bpmdetect");
            let fakesink = gst::ElementFactory::make("fakesink")
                .name("offline_sink")
                .property("sync", false)
                .property("async", false)
                .build()
                .expect("fakesink");

            pipeline
                .add_many([
                    &audioconvert,
                    &audioresample,
                    &level,
                    &bpmdetect,
                    &spectrum,
                    &fakesink,
                ])
                .expect("add audio elements");

            gst::Element::link_many([
                &audioconvert,
                &audioresample,
                &level,
                &bpmdetect,
                &spectrum,
                &fakesink,
            ])
            .expect("link audio elements");

            let _ = audioconvert.sync_state_with_parent();
            let _ = audioresample.sync_state_with_parent();
            let _ = level.sync_state_with_parent();
            let _ = bpmdetect.sync_state_with_parent();
            let _ = spectrum.sync_state_with_parent();
            let _ = fakesink.sync_state_with_parent();

            let sink_pad = audioconvert
                .static_pad("sink")
                .expect("audioconvert sink pad");
            if !sink_pad.is_linked() {
                if let Err(e) = pad.link(&sink_pad) {
                    warn!("Failed to link decoder to audioconvert: {e:?}");
                    return;
                }
            }

            *built = true;
        });

        pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| anyhow!("Failed to start pipeline: {e:?}"))?;

        let bus = pipeline.bus().ok_or_else(|| anyhow!("Pipeline has no bus"))?;

        let mut spectrum_events: Vec<SpectrumEvent> = Vec::new();
        let mut level_events: Vec<LevelEvent> = Vec::new();
        let mut bpm: f32 = 0.0;
        let mut last_event_time: f64 = 0.0;

        for msg in bus.iter_timed(gst::ClockTime::from_seconds(60)) {
            match msg.view() {
                gst::MessageView::Eos(_) => {
                    info!("Offline analysis: EOS reached");
                    break;
                }
                gst::MessageView::Error(err) => {
                    let _ = pipeline.set_state(gst::State::Null);
                    return Err(anyhow!(
                        "Offline analysis pipeline error: {} ({})",
                        err.error(),
                        err.debug().unwrap_or_default()
                    ));
                }
                gst::MessageView::Element(element) => {
                    if let Some(structure) = element.structure() {
                        match structure.name().as_str() {
                            "spectrum" => {
                                let time_secs = structure
                                    .get::<gst::ClockTime>("timestamp")
                                    .map(|t| t.nseconds() as f64 / 1_000_000_000.0)
                                    .unwrap_or(last_event_time);
                                let mags = extract_magnitudes(structure);
                                if !mags.is_empty() {
                                    spectrum_events.push(SpectrumEvent {
                                        time_secs,
                                        magnitudes: mags,
                                    });
                                    last_event_time = time_secs;
                                }
                            }
                            "level" => {
                                let time_secs = structure
                                    .get::<gst::ClockTime>("timestamp")
                                    .map(|t| t.nseconds() as f64 / 1_000_000_000.0)
                                    .unwrap_or(last_event_time);
                                if let Some((rms_db, peak)) = extract_level(structure) {
                                    level_events.push(LevelEvent {
                                        time_secs,
                                        rms_db,
                                        peak,
                                    });
                                    last_event_time = time_secs;
                                }
                            }
                            "tempo" => {
                                if let Ok(val) = structure.get::<f64>("tempo") {
                                    if val > 0.0 {
                                        bpm = val as f32;
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                gst::MessageView::Tag(tag) => {
                    if let Some(b) = tag.tags().get::<gst::tags::BeatsPerMinute>() {
                        let v = b.get() as f32;
                        if v > 0.0 {
                            bpm = v;
                        }
                    }
                }
                _ => {}
            }
        }

        pipeline
            .set_state(gst::State::Null)
            .map_err(|e| anyhow!("Failed to stop pipeline: {e:?}"))?;

        let duration_secs = spectrum_events
            .last()
            .map(|e| e.time_secs)
            .unwrap_or(last_event_time);

        info!(
            "Offline analysis done: {} spectrum events, {} level events, duration {:.2}s, BPM={:.1}",
            spectrum_events.len(),
            level_events.len(),
            duration_secs,
            bpm
        );

        if spectrum_events.is_empty() {
            return Err(anyhow!(
                "Offline analysis collected no spectrum events; source may have no audio track"
            ));
        }

        Ok(Self {
            spectrum_events,
            level_events,
            bands,
            duration_secs,
            bpm,
        })
    }

    /// Returns the latest spectrum/level pair at or before `time_secs`. If
    /// `time_secs` is before any data, returns the first event.
    pub fn sample(&self, time_secs: f64) -> Option<AudioSample<'_>> {
        let spec = latest_at_or_before(&self.spectrum_events, time_secs, |e| e.time_secs)?;
        let lev = latest_at_or_before(&self.level_events, time_secs, |e| e.time_secs);
        Some(AudioSample {
            magnitudes: &spec.magnitudes,
            bands: self.bands,
            rms_db: lev.map(|l| l.rms_db).unwrap_or(-100.0),
            peak: lev.map(|l| l.peak).unwrap_or(0.0),
            bpm: self.bpm,
        })
    }
}

fn extract_magnitudes(structure: &gst::StructureRef) -> Vec<f32> {
    // The spectrum element exposes magnitude as an array typed field. the
    // existing live path falls back to string parsing because the typed
    // accessor returned NotFound on some platforms. We try both.
    let mut out: Vec<f32> = Vec::new();

    let s = structure.to_string();
    if let Some(start_idx) = s.find("magnitude=(float){") {
        if let Some(end_rel) = s[start_idx..].find('}') {
            let inner = &s[start_idx + "magnitude=(float){".len()..start_idx + end_rel];
            for tok in inner.split(',') {
                if let Ok(v) = tok.trim().parse::<f32>() {
                    out.push(v);
                }
            }
        }
    }

    if out.is_empty() {
        for i in 0..1024 {
            let field = format!("magnitude[{i}]");
            match structure.get::<f32>(&field) {
                Ok(v) => out.push(v),
                Err(_) => break,
            }
        }
    }

    out
}

fn extract_level(structure: &gst::StructureRef) -> Option<(f64, f64)> {
    let rms_list = structure.get::<gst::glib::ValueArray>("rms").ok()?;
    let peak_list = structure.get::<gst::glib::ValueArray>("peak").ok()?;
    let rms_db = rms_list.iter().next()?.get::<f64>().ok()?;
    let peak_db = peak_list.iter().next()?.get::<f64>().ok()?;
    let peak_linear = 10.0_f64.powf(peak_db / 20.0);
    Some((rms_db, peak_linear))
}

fn latest_at_or_before<T, F>(events: &[T], time_secs: f64, key: F) -> Option<&T>
where
    F: Fn(&T) -> f64,
{
    if events.is_empty() {
        return None;
    }
    let mut lo = 0usize;
    let mut hi = events.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if key(&events[mid]) <= time_secs {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if lo == 0 {
        events.first()
    } else {
        events.get(lo - 1)
    }
}


#[cfg(feature = "media")]
pub mod audio;
#[cfg(feature = "media")]
pub mod video;
#[cfg(feature = "media")]
pub mod webcam;
use log::info;

#[cfg(feature = "media")]
pub fn init() -> anyhow::Result<()> {
    // These are active untill I merge the PR
    //std::env::set_var("GST_DEBUG", "bpmdetect:5,pitch:5,soundtouch:5,bus:4,element:4");
    info!("Setting up GStreamer with enhanced logging");
    gstreamer::init()?;
    info!("GStreamer initialized successfully");
    Ok(())
}

#[cfg(not(feature = "media"))]
pub fn init() -> anyhow::Result<()> {
    info!("Media support disabled - skipping GStreamer initialization");
    Ok(())
}

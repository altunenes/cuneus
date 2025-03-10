pub mod video;

use log::info;

pub fn init() -> anyhow::Result<()> {
    // These are active untill I merge the PR
    std::env::set_var("GST_DEBUG", "bpmdetect:5,pitch:5,soundtouch:5,bus:4,element:4");
    info!("Setting up GStreamer with enhanced logging");
    gstreamer::init()?;
    info!("GStreamer initialized successfully");
    Ok(())
}
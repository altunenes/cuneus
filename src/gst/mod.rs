#[cfg(feature = "media")]
pub mod audio;
#[cfg(feature = "media")]
pub mod offline_audio;
#[cfg(feature = "media")]
pub mod video;
#[cfg(feature = "media")]
pub mod webcam;
use log::info;

#[cfg(feature = "media")]
pub fn init() -> anyhow::Result<()> {
    // These are active untill I merge the PR
    //std::env::set_var("GST_DEBUG", "bpmdetect:5,pitch:5,soundtouch:5,bus:4,element:4");
    use_bundled_gstreamer();
    gstreamer::init()?;
    info!("GStreamer initialized successfully");
    Ok(())
}

/// A release app ships its own GStreamer: next to the exe (Windows), in `gstreamer/` (Linux) or in the bundle's Resources (macOS)
#[cfg(feature = "media")]
fn use_bundled_gstreamer() {
    let Some(exe) = std::env::current_exe().ok() else { return };
    let Some(dir) = exe.parent() else { return };
    let Some(root) = [dir.join("gstreamer"), dir.join("../Resources/gstreamer"), dir.to_path_buf()]
        .into_iter()
        .find(|r| r.join("lib/gstreamer-1.0").is_dir())
    else {
        return;
    };
    let scanner = root.join("libexec/gstreamer-1.0").join(if cfg!(windows) { "gst-plugin-scanner.exe" } else { "gst-plugin-scanner" });
    let name = exe.file_stem().and_then(|s| s.to_str()).unwrap_or("cuneus");
    // called first in main, before any other thread exists
    unsafe {
        std::env::set_var("GST_PLUGIN_PATH", root.join("lib/gstreamer-1.0"));
        std::env::set_var("GST_PLUGIN_SYSTEM_PATH", "");
        std::env::set_var("GST_REGISTRY", std::env::temp_dir().join(format!("cuneus-{name}-registry.bin")));
        if scanner.is_file() {
            std::env::set_var("GST_PLUGIN_SCANNER", scanner);
        }
        std::env::remove_var("GSTREAMER_1_0_ROOT_MSVC_X86_64");
        std::env::remove_var("GSTREAMER_1_0_ROOT_MINGW_X86_64");
    }
    info!("Using bundled GStreamer at {}", root.display());
}

#[cfg(not(feature = "media"))]
pub fn init() -> anyhow::Result<()> {
    info!("Media support disabled - skipping GStreamer initialization");
    Ok(())
}

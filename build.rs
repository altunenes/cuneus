// Purpose: This build script configures the build environment for GStreamer integration.
//
// What this does:
// - Sets up necessary paths, environment variables and linker flags for GStreamer
// - Currently handles macOS-specific configuration for the GStreamer framework
//
// Customization for different environments:
// - macOS: If your GStreamer framework is installed in a non-standard location,
//   update the paths in the macOS section
// - Windows: You'll need to add a Windows-specific section similar to the macOS one,
//   typically pointing to your GStreamer installation directory (e.g.,
//   C:\gstreamer\1.0\msvc_x86_64\lib for MSVC builds)
// - Linux: For standard installations, pkg-config should find GStreamer without
//   any special configuration. For custom installations, add a Linux section
//   that sets PKG_CONFIG_PATH to your GStreamer lib/pkgconfig directory.
//
// For more information on build scripts, see:
// https://doc.rust-lang.org/cargo/reference/build-scripts.html
//
// Note that, installation of Gstreamer is not a hard task (please open issue if you have a trouble), I hope these explanations are not making you feel like it is a hard task. In
// windows for instance, just download the installer and click next, next, next, finish. That's all, it should automatically set the environment variables for you.
// And you will able to use Gstreamer in this project. Bellow is my own configuration for Gstreamer in my mac machine which I used via PKG_CONFIG_PATH.
// You can also use the same configuration in your mac machine. And I strongly recommend you to install it with PKG_CONFIG_PATH.
// Please see how I build the project in github actions, you can use it as a reference:
// github.com/altunenes/cuneus/blob/main/.github/workflows/release.yaml
use std::env;
fn main() {
    let target = env::var("CARGO_CFG_TARGET_OS");
    if target == Ok("macos".to_string()) {
        env::set_var(
            "PKG_CONFIG_PATH",
            "/Library/Frameworks/GStreamer.framework/Versions/Current/lib/pkgconfig",
        );
        let lib = "/Library/Frameworks/GStreamer.framework/Versions/Current/lib";
        env::set_var("GST_PLUGIN_PATH", lib);
        env::set_var("DYLD_FALLBACK_LIBRARY_PATH", lib);
        println!("cargo:rustc-link-search=framework=/Library/Frameworks");
        println!("cargo:rustc-link-arg=-Wl,-rpath,/Library/Frameworks/GStreamer.framework/Versions/Current/lib");
    }
}

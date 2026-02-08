// Purpose: This build script configures the build environment for GStreamer integration.
//
// What this does:
// - Sets up necessary paths, environment variables and linker flags for GStreamer
// - Handles macOS, Windows, and Linux configurations
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
use std::path::PathBuf;

fn main() {
    let target = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    match target.as_str() {
        "macos" => {
            let gst_base = "/Library/Frameworks/GStreamer.framework/Versions/Current";
            let lib = format!("{}/lib", gst_base);
            let pkgconfig = format!("{}/lib/pkgconfig", gst_base);

            env::set_var("PKG_CONFIG_PATH", &pkgconfig);
            env::set_var("GST_PLUGIN_PATH", &lib);
            env::set_var("DYLD_FALLBACK_LIBRARY_PATH", &lib);
            println!("cargo:rustc-link-search=framework=/Library/Frameworks");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib);
        }
        "windows" => {
            // Try GSTREAMER_1_0_ROOT_MSVC_X86_64 first
            let gst_dir = env::var("GSTREAMER_1_0_ROOT_MSVC_X86_64")
                .or_else(|_| env::var("GSTREAMER_1_0_ROOT_X86_64"))
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("C:\\gstreamer\\1.0\\msvc_x86_64"));

            let lib_dir = gst_dir.join("lib");
            let pkgconfig_dir = lib_dir.join("pkgconfig");

            if lib_dir.exists() {
                env::set_var("PKG_CONFIG_PATH", &pkgconfig_dir);
                println!("cargo:rustc-link-search=native={}", lib_dir.display());
            } else {
                println!(
                    "cargo:warning=GStreamer not found at {}. \
                     Install GStreamer or set GSTREAMER_1_0_ROOT_MSVC_X86_64.",
                    gst_dir.display()
                );
            }
        }
        "linux" => {
            // pkg-config for Linux without
            if env::var("PKG_CONFIG_PATH").is_err() {
                let common = [
                    "/usr/lib/x86_64-linux-gnu/pkgconfig",
                    "/usr/lib64/pkgconfig",
                    "/usr/local/lib/pkgconfig",
                ];
                for p in &common {
                    let path = PathBuf::from(p);
                    if path.exists() {
                        env::set_var("PKG_CONFIG_PATH", p);
                        break;
                    }
                }
            }
        }
        _ => {}
    }
}

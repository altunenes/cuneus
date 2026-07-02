# Build Instructions

## Prerequisites

1. Install Rust: https://rustup.rs/

2. Install GStreamer:
   - **Runtime package** (required)
   - **Development package** (required)
   - See: https://gstreamer.freedesktop.org/download/
## Build

```bash
# Clone and build
git clone https://github.com/altunenes/cuneus
cd cuneus
# Run a pure GPU shader
cargo run --release --example physarum

# Run a media shader (requires gstreamer library)
cargo run --release --example audiovis --features media
```

## Notes

- `build.rs` handles GStreamer library detection and linking when the `media` feature is enabled. You may need to adjust the `PKG_CONFIG_PATH` based on your GStreamer installation.
- Media shaders require GStreamer and `--features media`; non-media shaders build with the default feature set.

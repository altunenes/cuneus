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
# Run a specific shader
cargo run --release --example audiovis 
```

## Notes

- `build.rs` handles GStreamer library detection and linking. You may need to adjust the `PKG_CONFIG_PATH` based on your GStreamer installation.
- Media shaders require GStreamer, others can build with `--no-default-features`

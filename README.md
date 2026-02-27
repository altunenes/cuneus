[![Shader Binary Release](https://github.com/altunenes/cuneus/actions/workflows/release.yaml/badge.svg)](https://github.com/altunenes/cuneus/actions/workflows/release.yaml) [![crates.io](https://img.shields.io/crates/v/Cuneus.svg)](https://crates.io/crates/Cuneus)

<img src="https://github.com/user-attachments/assets/590dbd91-5eaa-4c04-b3f9-d579924fa4c3" alt="cuneus sdf" width="320" height="120" />


A tool for experimenting with WGSL shaders, it uses `wgpu` for rendering, `egui` for the UI and `winit` for windowing :-)

### Current Features

- Hot shader reloading
- Multi-pass, atomics etc
- 3DGS Rendering Inference (PLY Import, radix gpu sort)
- Interactive parameter adjustment, ez media imports through egui
- Easily use HDR textures, videos/webcam via UI
- Audio/Visual synchronization: Spectrum and BPM detection
- Real-time audio synthesis: Generate music directly from wgsl shaders
- Export HQ frames via egui

### Builder Pattern

Cuneus uses a declarative builder to configure your entire compute pipeline. You say *what* you need — the engine handles all bind group layouts, ping-pong buffers, and pipeline wiring:

```rust
// Define your multi-pass pipeline as a dependency graph:
let passes = vec![
    PassDescription::new("buffer_a", &[]),                       // no inputs
    PassDescription::new("buffer_b", &["buffer_a"]),             // reads buffer_a
    PassDescription::new("buffer_c", &["buffer_b", "buffer_c"]),  // reads buffer_b + own previous frame
    PassDescription::new("main_image", &["buffer_c"]),
];

let config = ComputeShader::builder()
    .with_multi_pass(&passes)           // the render graph above
    .with_custom_uniforms::<MyParams>() // UI-controllable parameters
    .with_mouse()                       // mouse input
    .build();
```

Dependencies are packed sequentially — `&["buffer_b", "buffer_c"]` becomes `input_texture0` and `input_texture1` in WGSL. Self-reference enables cross-frame feedback (ping-pong) automatically. One `.dispatch()` call runs the entire pipeline. See [usage.md](usage.md) for the full guide.

## Current look


  <a href="https://github.com/user-attachments/assets/25d47df4-45f5-4455-b2cf-ba673a8c081c">
    <img src="https://github.com/user-attachments/assets/25d47df4-45f5-4455-b2cf-ba673a8c081c" width="300" alt="Cuneus IDE Interface"/>
  </a>

## Keys

- `F` full screen/minimal screen, `H` hide egui

#### Usage

- If you want to try your own shaders, check out the [usage.md](usage.md) and see [BUILD.md](BUILD.md).
- **Optional Media Support**: GStreamer dependencies are optional - use `--no-default-features` for lightweight builds with pure GPU compute shaders.
- **When using cuneus as a dependency** (via `cargo add`):
  - Add `bytemuck = { version = "1", features = ["derive"] }` to dependencies (derive macros can't be re-exported)
  - Copy [build.rs](build.rs) to your project root to configure `GStreamer` paths (only needed for media features)
  - then simply use `use cuneus::prelude::*;`


#### Run examples

- `cargo run --release --example *file*`
- Or download on the [releases](https://github.com/altunenes/cuneus/releases)


# Gallery

| **Sinh3D** | **JFA** | **Volumetric Passage** |
|:---:|:---:|:---:|
| <a href="https://github.com/user-attachments/assets/0758e450-f0a7-4ab2-a063-b071ebedee99"><img src="https://github.com/user-attachments/assets/f77114f6-2937-4ca9-8438-1ee8303f447c" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/examples/sinh.rs) | <a href="https://github.com/user-attachments/assets/f07023a3-0d93-4740-a95c-49f16d815e29"><img src="https://github.com/user-attachments/assets/8c71ce99-58ff-4354-9c0a-0a0fd4e5032d" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/examples/jfa.rs) | <a href="https://github.com/user-attachments/assets/c19365ac-267f-4301-a9c8-42097d4b167a"><img src="https://github.com/user-attachments/assets/5ef301cd-cb11-4850-b013-13537939fd22" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/examples/volumepassage.rs)|

| **PathTracing Mandelbulb** | **CNN:EMNIST** | **Tame Impala** |
|:---:|:---:|:---:|
| <a href="https://github.com/user-attachments/assets/24083cae-7e96-4726-8509-fb3d5973308a"><img src="https://github.com/user-attachments/assets/e454b395-a1a0-4b91-a776-9afd1a789d23" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/examples/mandelbulb.rs) | <a href="https://github.com/user-attachments/assets/f692e325-a0d2-4ae6-9246-bcbddd85516f"><img src="https://github.com/user-attachments/assets/f692e325-a0d2-4ae6-9246-bcbddd85516f" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/examples/cnn.rs) | <a href="https://github.com/user-attachments/assets/b8b8ccf6-dee1-40d1-85c2-9b7d1ee9b6f5"><img src="https://github.com/user-attachments/assets/b8b8ccf6-dee1-40d1-85c2-9b7d1ee9b6f5" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/examples/tameimp.rs) |

| **Buddhabrot** | **FFT(Butterworth filter)** | **Clifford** |
|:---:|:---:|:---:|
| <a href="https://github.com/user-attachments/assets/93a17f27-695a-4249-9ff8-be2742926358"><img src="https://github.com/user-attachments/assets/93a17f27-695a-4249-9ff8-be2742926358" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/examples/buddhabrot.rs) | <a href="https://github.com/user-attachments/assets/5806af3b-a640-433c-b7ec-1ca051412300"><img src="https://github.com/user-attachments/assets/e1e7f7e9-5979-43fe-8bb0-ccda8e428fe5" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/examples/fft.rs) | <a href="https://github.com/user-attachments/assets/8b078f40-a989-4d07-bb2f-d19d8232cc9f"><img src="https://github.com/user-attachments/assets/8b078f40-a989-4d07-bb2f-d19d8232cc9f" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/examples/cliffordcompute.rs) |


| **Block Tower: 3D Game** | **System** | **2d Gaussian Splatting** |
|:---:|:---:|:---:|
| <a href="https://github.com/user-attachments/assets/9ce52cc1-31c0-4e50-88c7-2fb06d1a57b3"><img src="https://github.com/user-attachments/assets/9ce52cc1-31c0-4e50-88c7-2fb06d1a57b3" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/examples/blockgame.rs) | <a href="https://github.com/user-attachments/assets/8bd86b80-11fb-4757-93ca-703b8227bd4c"><img src="https://github.com/user-attachments/assets/c606c3bf-07de-40f4-abb4-f88de4031d69" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/examples/system.rs) | <a href="https://github.com/user-attachments/assets/f91c7ff7-f56e-43d5-9d60-97d9c037d700"><img src="https://github.com/user-attachments/assets/f91c7ff7-f56e-43d5-9d60-97d9c037d700" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/examples/gaussian.rs) |


| **SDneuron** | **path tracer** | **audio visualizer** |
|:---:|:---:|:---:|
| <a href="https://github.com/user-attachments/assets/bb5fc1c4-87bf-4eb9-8e0d-e54bcf32e0fb"><img src="https://github.com/user-attachments/assets/53efa317-8ec9-4435-988d-924d5efb6247" width="250" height ="200"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/examples/2dneuron.rs) | <a href="https://github.com/user-attachments/assets/45b8f532-f3fb-453c-b356-1d3c153d614a"><img src="https://github.com/user-attachments/assets/896228c3-7583-40de-9643-8b58aaec6050" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/examples/pathtracing.rs) | <a href="https://github.com/user-attachments/assets/3eda9c33-7961-4dd4-aad1-170ae32640e7"><img src="https://github.com/user-attachments/assets/3eda9c33-7961-4dd4-aad1-170ae32640e7" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/examples/audiovis.rs) |

| **sdvert** | **tree** | **rorschach** |
|:---:|:---:|:---:|
| <a href="https://github.com/user-attachments/assets/9306abfa-0516-4b7f-a80a-5674d0aa09bb"><img src="https://github.com/user-attachments/assets/5f463cce-492b-40ab-9b10-b855009c2c0a" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/examples/sdvert.rs) | <a href="https://github.com/user-attachments/assets/2f0bdc7c-d226-4091-bae7-b96561c1fb4f"><img src="https://github.com/user-attachments/assets/2f0bdc7c-d226-4091-bae7-b96561c1fb4f" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/examples/tree.rs) | <a href="https://github.com/user-attachments/assets/f6f9f7f6-121f-406b-b3fa-b049343ee6b3"><img src="https://github.com/user-attachments/assets/f6f9f7f6-121f-406b-b3fa-b049343ee6b3" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/examples/rorschach.rs) |

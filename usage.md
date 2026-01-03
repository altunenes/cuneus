# Cuneus Usage Guide

Cuneus is a GPU compute shader engine with a unified backend for single-pass, multi-pass, and atomic compute shaders. It features built-in UI controls, hot-reloading, media integration, and GPU-driven audio synthesis.

**Key Philosophy:** Declare what you need in the builder → get predictable bindings in WGSL. No manual binding management, no boilerplate. Add `.with_mouse()` in Rust, access `@group(2) mouse` in your shader. The **4-Group Binding Convention** guarantees where every resource lives: Group 0 (time), Group 1 (output/params), Group 2 (engine resources), Group 3 (user data/multi-pass). Everything flows from the builder.

## Core Concepts

### 1. The Unified Compute Pipeline

In Cuneus, almost everything is a compute shader. Instead of writing traditional vertex/fragment shaders, you write compute kernels that write directly to an output texture. The framework provides a simple renderer to blit this texture to the screen. This approach gives you maximum control and performance for GPU tasks.

### 2. The Builder Pattern (`ComputeShaderBuilder`)

The `ComputeShader::builder()` is the single entry point for configuring your shader. API allows you to specify exactly what resources your shader needs, and Cuneus handles all the complex WGPU boilerplate for you.

```rust
let config = ComputeShader::builder()
    .with_label("My Awesome Shader")
    .with_custom_uniforms::<MyParams>() // Custom parameters
    .with_mouse()                       // Enable mouse input
    .with_channels(1)                   // Enable one external texture (e.g., video)
    .build();
```

### 3. The 4-Group Binding Convention

Cuneus enforces a standard bind group layout to create a stable and predictable contract between your Rust code and your WGSL shader. This eliminates the need to manually track binding numbers.

| Group | Binding(s) | Description | Configuration |
| :--- | :--- | :--- | :--- |
| **0** | `@binding(0)` | **Per-Frame Data** (Time, frame count). | Engine-managed. Always available. |
| **1** | `@binding(0)`<br/>`@binding(1)`<br/>`@binding(2..)` | **Primary I/O & Params**. Output texture, your custom `UniformProvider`, and an optional input texture. | User-configured via builder (`.with_custom_uniforms()`, `.with_input_texture()`). |
| **2** | `@binding(0..N)` | **Global Engine Resources**. Mouse, fonts, audio buffer, atomics, and media channels. The binding order is fixed. | User-configured via builder (`.with_mouse()`, `.with_fonts()`, etc.). |
| **3** | `@binding(0..N)` | **User Data & Multi-Pass I/O**. User-defined storage buffers or textures for multi-pass feedback loops. | User-configured via builder (`.with_storage_buffer()` or `.with_multi_pass()`). |

### 4. Execution Models (Dispatching)

- **Automatic (`.dispatch()`):** This is the recommended method. It executes the entire pipeline you defined in the builder (including all multi-pass stages) and automatically increments the frame counter.
- **Manual (`.dispatch_stage()`):** This gives you fine-grained control to run specific compute kernels from your WGSL file. It is essential for advanced patterns like path tracing accumulation or conditional updates. **You must manually increment `compute_shader.current_frame` when using this method.**

### 5. Multi-Pass Models

The framework elegantly handles two types of multi-pass computation:

1. **Texture-Based (Ping-Pong):** Ideal for image processing and feedback effects. Intermediate results are stored in textures.
   - **Within-Frame**: The backend **automatically flips buffers between passes** during `.dispatch()`. Most multi-pass shaders use this.
   - **Across-Frame (Temporal)**: For effects that accumulate over time, call `.flip_buffers()` after `output.present()` to preserve state for the next frame.
   - *Examples with cross-frame feedback: `lich.rs`, `currents.rs`* - use flip_buffers()
   - *Examples with within-frame only: `kuwahara.rs`, `fluid.rs`, `jfa.rs`, `2dneuron.rs`* - no flip_buffers()

2. **Storage-Buffer-Based (Shared Memory):** Ideal for GPU algorithms like FFT or simulations like CNNs. All passes read from and write to the same large, user-defined storage buffers. This is enabled by using `.with_multi_pass()` *and* `.with_storage_buffer()`. No flip_buffers() needed.
   - *Examples: `fft.rs`, `cnn.rs`*

## Getting Started: Shader Structure

Every shader application follows a similar pattern implementing the `ShaderManager` trait.

```rust
use cuneus::prelude::*;
use cuneus::compute::*;

// 1. Define custom parameters for the UI
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MyParams {
    strength: f32,
    color: [f32; 3],
    _padding: f32,
}

impl UniformProvider for MyParams {
    fn as_bytes(&self) -> &[u8] { bytemuck::bytes_of(self) }
}

// 2. Define the main application struct
struct MyShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: MyParams,
}

// 3. Implement the ShaderManager trait
impl ShaderManager for MyShader {
    fn init(core: &Core) -> Self {
        // RenderKit handles the final blit to screen and UI (vertex/blit shaders built-in)
        let texture_bind_group_layout = RenderKit::create_standard_texture_layout(&core.device);
        let base = RenderKit::new(core, &texture_bind_group_layout, None);
        let initial_params = MyParams { /* ... */ };

        // --- To convert this to a Multi-Pass shader, make the following changes: ---
        
        // 1. (Multi-Pass) Define your passes and their dependencies.
        //    The string in `new()` is the WGSL entry point name.
        //    The slice `&[]` contains the names of the buffers this pass reads from.
        /*
        let passes = vec![
            PassDescription::new("buffer_a", &["buffer_a"]), // A pass that reads its own previous frame output
            PassDescription::new("main_image", &["buffer_a"]), // A pass that reads the final state of buffer_a
        ];
        */

        // Configure the compute shader using the builder
        let config = ComputeShader::builder()
            // For Single-Pass, use .with_entry_point():
            .with_entry_point("main")
            // 2. (Multi-Pass) Comment out .with_entry_point() and use .with_multi_pass() instead: (we define the passes above)
            // .with_multi_pass(&passes)
            .with_custom_uniforms::<MyParams>()
            .with_mouse()
            .with_label("My Shader")
            .build();

        // Create the compute shader instance
        let mut compute_shader = ComputeShader::from_builder(
            core,
            include_str!("shaders/my_shader.wgsl"),
            config,
        );

        // (Optional but recommended) Enable hot-reloading
        compute_shader.enable_hot_reload(/* ... */).unwrap();
        
        // Set initial parameters
        compute_shader.set_custom_params(initial_params, &core.queue);

        Self { base, compute_shader, current_params: initial_params }
    }

    fn update(&mut self, core: &Core) {
        // Update time uniform, check for hot-reloads, etc.
        let time = self.base.controls.get_time(&self.base.start_time);
        self.compute_shader.set_time(time, 1.0/60.0, &core.queue);
        self.compute_shader.check_hot_reload(&core.device);
    }

    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = core.device.create_command_encoder(/* ... */);

        // Execute the entire compute pipeline.
        // This works for both single-pass and multi-pass shaders automatically.
        self.compute_shader.dispatch(&mut encoder, core);

        // Display the final output texture and UI
        // ... rendering boilerplate ...
        core.queue.submit(Some(encoder.finish()));
        output.present();

        // 3. (Multi-Pass with Cross-Frame Feedback ONLY) If your effect needs to accumulate
        //    or preserve state across frames (like reaction-diffusion or temporal effects),
        //    call flip_buffers() here to save the current frame's output for the next frame.
        //    Most multi-pass shaders DON'T need this - the backend auto-flips within each frame.
        /*
        self.compute_shader.flip_buffers();  // Only for lich.rs, currents.rs style effects
        */

        Ok(())
    }
    
    fn handle_input(&mut self, _core: &Core, _event: &WindowEvent) -> bool {
        // Handle keyboard/mouse events
        false
    }
}
```

## Standard Bind Group Layout

Your WGSL shaders should follow this layout for predictable resource access.

```wgsl
// Group 0: Per-Frame Data (Engine-Managed)
struct TimeUniform { time: f32, delta: f32, frame: u32, /* ... */ };
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

// Group 1: Primary Pass I/O & Custom Parameters
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
// Optional: Your custom uniform struct
@group(1) @binding(1) var<uniform> params: MyParams; 
// Optional: Input texture for image processing
@group(1) @binding(2) var input_texture: texture_2d<f32>;
@group(1) @binding(3) var input_sampler: sampler;

// Group 2: Global Engine Resources
// IMPORTANT: Binding numbers are DYNAMIC based on what you enable in the builder.
// Resources are added in this order: mouse → fonts → audio → audio_spectrum → atomics → channels
// Example 1: Only .with_audio_spectrum() → audio_spectrum is @binding(0)
// Example 2: .with_audio_spectrum() + .with_atomic_buffer() → audio_spectrum @binding(0), atomic_buffer @binding(1)
// Example 3: .with_mouse() + .with_fonts() + .with_audio() → mouse @binding(0), fonts @binding(1-2), audio @binding(3)

// Mouse (if .with_mouse() is used) - takes 1 binding
@group(2) @binding(N) var<uniform> mouse: MouseUniform;
// Fonts (if .with_fonts() is used) - takes 2 bindings (uses textureLoad, no sampler needed)
@group(2) @binding(N) var<uniform> font_uniform: FontUniforms;
@group(2) @binding(N+1) var font_texture: texture_2d<f32>;
// Audio buffer (if .with_audio() is used) - takes 1 binding
@group(2) @binding(N) var<storage, read_write> audio_buffer: array<f32>;
// Audio spectrum (if .with_audio_spectrum() is used) - takes 1 binding
@group(2) @binding(N) var<storage, read> audio_spectrum: array<f32>;
// Atomic buffer (if .with_atomic_buffer() is used) - takes 1 binding
@group(2) @binding(N) var<storage, read_write> atomic_buffer: array<atomic<u32>>;
// Media channels (if .with_channels(2) is used) - takes 2 bindings per channel
@group(2) @binding(N) var channel0: texture_2d<f32>;
@group(2) @binding(N+1) var channel0_sampler: sampler;

// Group 3: User Data & Multi-Pass I/O
// User-defined storage buffers (if .with_storage_buffer() is used, this takes priority)
@group(3) @binding(0) var<storage, read_write> my_data: array<f32>;
// OR: Multi-pass input textures (if .with_multi_pass() is used without storage buffers)
@group(3) @binding(0) var input_texture0: texture_2d<f32>;
@group(3) @binding(1) var input_sampler0: sampler;
```

## Advanced Topics

### Workgroup Sizes

- **WGSL is the Source of Truth:** A workgroup size defined in your shader with `@workgroup_size(x, y, z)` will always be used to compile the pipeline.
- **Builder is a Fallback:** `.with_workgroup_size()` is only used if the WGSL entry point has no size decorator.
- **Per-Pass Specificity:** For multi-pass shaders, you can specify a unique workgroup size for each stage. This is critical for performance in algorithms like FFTs or CNNs.

```rust
// See cnn.rs for a practical example
let passes = vec![
    PassDescription::new("conv_layer1", &["canvas_update"])
        .with_workgroup_size([12, 12, 8]), // Custom size for this pass
    PassDescription::new("main_image", &["fully_connected"]), // Uses default or WGSL size
];
```

### Manual Dispatching

For effects like path tracing that require conditional accumulation, use `dispatch_stage()`. This prevents the frame counter from advancing automatically, allowing you to build up an image over multiple real frames that all correspond to a single logical `time_data.frame`.

```rust
// See mandelbulb.rs for a practical example
fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
    // ...
    // Set frame uniform manually for accumulation
    self.compute_shader.time_uniform.data.frame = self.frame_count;
    self.compute_shader.time_uniform.update(&core.queue);
    
    // Dispatch the single stage of the path tracer
    self.compute_shader.dispatch_stage(&mut encoder, core, 0);

    // Only increment the logical frame count when accumulation is active
    if self.current_params.accumulate > 0 {
        self.frame_count += 1;
    }
    // ...
}
```

## Media & Integration

### GPU Music Generation & Synthesis

Cuneus supports **bidirectional GPU-CPU audio workflows** using two complementary systems:

**1. Audio Visualization (`.with_audio_spectrum()`)** - Analyze loaded audio/video:
- **Flow**: Media file → GStreamer spectrum analyzer → CPU writes to buffer → GPU reads for visualization
- **Shader Access**: `@group(2) var<storage, read> audio_spectrum: array<f32>` (read-only)
- **Use Case**: Audio visualizers like `audiovis.rs`

**2. Audio Synthesis (`.with_audio()`)** - Generate music on GPU:
- **Flow**: GPU calculates frequencies/amplitudes → writes to buffer → CPU reads → GStreamer plays audio
- **Shader Access**: `@group(2) var<storage, read_write> audio_buffer: array<f32>` (read-write)
- **Use Case**: Music generators like `synth.rs`, `veridisquo.rs`

#### Composing Music on the GPU

You can write entire songs in your compute shader by calculating note sequences, melodies, and synthesis parameters:

```wgsl
// In WGSL: Compose music and write synthesis parameters
// This pattern is from veridisquo.wgsl - a complete GPU-composed song
if (global_id.x == 0u && global_id.y == 0u) {
    // Calculate melody notes based on time
    let beat = u32(u_time.time * tempo / 60.0);
    let melody_note = get_melody_for_beat(beat);
    let bass_note = get_bass_for_beat(beat);

    // Write to audio buffer for CPU playback
    audio_buffer[0] = melody_note.frequency;
    audio_buffer[1] = melody_note.amplitude;
    audio_buffer[2] = bass_note.frequency;
    audio_buffer[3] = bass_note.amplitude;
}
```

```rust
// In Rust: Read GPU-composed music and play it
// This pattern is from veridisquo.rs
if let Ok(data) = pollster::block_on(
    compute.read_audio_buffer(&core.device, &core.queue)
) {
    synth.set_voice(0, data[0], data[1], true);  // Melody
    synth.set_voice(1, data[2], data[3], true);  // Bass
}
```

**Examples:**

- `veridisquo.rs` - Complete GPU-composed song with melody and bassline
- `synth.rs` - Interactive polyphonic synthesizer with ADSR envelopes
- `debugscreen.rs` - Simple tone generation for testing

**Pro-tip - Generic Storage:** The `.with_audio()` buffer is just a `storage, read_write` array of floats. You don't have to use it for audio! Any shader can use it as generic persistent storage:

- `blockgame.rs` - Uses the "audio buffer" to store game state (score, block positions, camera) - no audio at all!
- The buffer persists across frames, making it stateful GPU applications beyond audio synthesis

### External Textures (`.with_channels()`)

The `.with_channels(N)` method exposes `N` texture/sampler pairs in Group 2, making them globally accessible to **all passes** of a multi-pass shader. This is the preferred way to pipe in video, webcam feeds, or static images into complex simulations.

- *Example: `fluid.rs` uses `.with_channels(1)` to feed a video into its simulation.*

### Audio Spectrum Analysis (`.with_audio_spectrum()`)

Use `.with_audio_spectrum(69)` to **visualize** audio from loaded media files. GStreamer's spectrum analyzer processes the audio stream and writes frequency data to a GPU buffer that your shader can read.

- **Buffer Layout**:
  - Indices 0-63: frequency band magnitudes (RMS-normalized)
  - Index 64: BPM value 
  - Index 65: bass energy (pre-computed, ~0-200Hz)
  - Index 66: mid energy (pre-computed, ~200-4000Hz)
  - Index 67: high energy (pre-computed, ~4000-20000Hz)
  - Index 68: total energy (weighted average)
- **Shader Access**: `@group(2) var<storage, read> audio_spectrum: array<f32>` (read-only)
- **Data Source**: Loaded audio/video files (mp3, wav, ogg, mp4, etc.)
- **Features**: RMS-normalized, real-time BPM detection, pre-computed energy bands
- **Example**: `audiovis.rs` - Spectrum visualizer with beat-synced animations

### Fonts

The `.with_fonts()` method provides texture (see `assets/fonts/fonttexture.png`) needed to render text directly inside your shader

- *Examples: `debugscreen.rs` uses this for its UI, and `cnn.rs` uses it to label its output bars.*

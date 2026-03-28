# Cuneus Usage Guide

Cuneus is a GPU compute shader engine with a unified backend for single-pass, multi-pass, and atomic compute shaders. It features built-in UI controls, hot-reloading, media integration, and GPU-driven audio synthesis.

**Key Philosophy:** Declare what you need in the builder → get predictable bindings in WGSL. No manual binding management, no boilerplate. Add `.with_mouse()` in Rust, access `@group(2) mouse` in your shader. The **4-Group Binding Convention** guarantees where every resource lives: Group 0 (time), Group 1 (output/params), Group 2 (engine resources), Group 3 (user data/multi-pass). Everything flows from the builder.

## Core Concepts

### 1. The Unified Compute Pipeline

In Cuneus, almost everything is a compute shader. Instead of writing traditional vertex/fragment shaders, you write compute kernels that write directly to an output texture. The framework provides a simple renderer to blit this texture to the screen. This approach gives you maximum control and performance for GPU tasks.

### 2. The Builder Pattern (`ComputeShaderBuilder`)

The `ComputeShader::builder()` is the single entry point for configuring your shader. It specifies exactly what resources your shader needs, and Cuneus handles all the complex WGPU boilerplate — including hot reload.

```rust
let config = ComputeShader::builder()
    .with_label("My Awesome Shader")
    .with_custom_uniforms::<MyParams>() // Custom parameters
    .with_mouse()                       // Enable mouse input
    .with_channels(1)                   // Enable one external texture (e.g., video)
    .build();

// The compute_shader! macro embeds the shader source AND enables hot reload automatically.
let compute_shader = cuneus::compute_shader!(core, "shaders/my_shader.wgsl", config);
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

1. **Texture-Based (Ping-Pong):** Ideal for image processing and feedback effects. Intermediate results are stored in textures. Each buffer independently tracks its write state, so any pass can read from any previous pass's output — and cross-frame feedback (self-referencing passes) works automatically.
   - *Examples with cross-frame feedback: `lich.rs`, `currents.rs`, `rorschach.rs`, `jfa.rs`*
   - *Examples with within-frame only: `kuwahara.rs`, `fluid.rs`, `2dneuron.rs`*

2. **Storage-Buffer-Based (Shared Memory):** Ideal for GPU algorithms like FFT or simulations like CNNs. All passes read from and write to the same large, user-defined storage buffers. This is enabled by using `.with_multi_pass()` *and* `.with_storage_buffer()`.
   - *Examples: `fft.rs`, `cnn.rs`*

## Getting Started: Shader Structure

Every shader application follows a similar pattern implementing the `ShaderManager` trait.

```rust
use cuneus::prelude::*;
use cuneus::compute::*;

// 1. Define custom parameters for the UI using the uniform_params! macro
// This adds #[repr(C)], derives, UniformProvider impl, and a compile-time
// assert that the struct size is a multiple of 16 bytes (catches padding errors).
cuneus::uniform_params! {
    struct MyParams {
        strength: f32,
        color: [f32; 3],
    }
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
        let base = RenderKit::new(core);
        let initial_params = MyParams { /* ... */ };

        // --- To convert this to a Multi-Pass shader, make the following changes: ---
        
        // 1. (Multi-Pass) Define your passes and their dependencies.
        //    The string in `new()` is the WGSL entry point name.
        //    The slice `&[]` lists buffers to bind as input_texture0, input_texture1, etc.
        //    Self-reference (e.g., "feedback" in its own inputs) enables cross-frame feedback.
        /*
        let passes = vec![
            PassDescription::new("compute_field", &[]),                  // No inputs
            PassDescription::new("feedback", &["compute_field"]),        // input_texture0 = compute_field
            PassDescription::new("main_image", &["feedback"]),
        ];
        // For cross-frame feedback (temporal effects), add self to inputs:
        // PassDescription::new("feedback", &["compute_field", "feedback"])
        // Then input_texture1 = feedback's PREVIOUS frame output (automatic)
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

        // Create the compute shader with automatic hot reload.
        let compute_shader = cuneus::compute_shader!(core, "shaders/my_shader.wgsl", config);

        // Set initial parameters
        compute_shader.set_custom_params(initial_params, &core.queue);

        Self { base, compute_shader, current_params: initial_params }
    }

    fn update(&mut self, _core: &Core) {
        // Hot-reload is checked automatically by dispatch()/dispatch_stage().
        // FPS tracking is updated automatically by end_frame().
    }

    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        // begin_frame() bundles surface texture + view + encoder into a FrameContext
        let mut frame = self.base.begin_frame(core)?;

        // get_ui_request() returns a ControlsRequest with time, window size, and FPS auto-populated
        let mut controls_request = self.base.controls
            .get_ui_request(&self.base.start_time, &core.size, self.base.fps_tracker.fps());

        // Build the UI (apply_default_style sets the standard theme)
        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                RenderKit::apply_default_style(ctx);
                // ... egui windows here ...
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        // Apply UI control requests after the UI closure:
        // - Non-media examples: use apply_control_request (handles time reset + param updates)
        //   self.base.apply_control_request(controls_request.clone());
        // - Media examples (video/webcam/hdri): use apply_media_requests (bundles
        //   apply_control_request + handle_video/webcam/hdri_requests in one call)
        //   self.base.apply_media_requests(core, &controls_request);

        // Execute the entire compute pipeline.
        // This works for both single-pass and multi-pass shaders automatically.
        self.compute_shader.dispatch(&mut frame.encoder, core);

        // Blit the compute shader's output texture to the screen
        self.base.renderer.render_to_view(&mut frame.encoder, &frame.view, &self.compute_shader);

        // end_frame() handles UI overlay + submit + present in one call
        self.base.end_frame(core, frame, full_output);

        // Cross-frame feedback (self-referencing passes) works automatically

        Ok(())
    }
    
    fn resize(&mut self, core: &Core) {
        // default_resize updates resolution uniform + resizes compute shader
        self.base.default_resize(core, &mut self.compute_shader);
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        // default_handle_input handles egui events + keyboard shortcuts (H to toggle UI, etc.)
        // For DroppedFile support, check the event after this call.
        // For mouse input, add: self.base.handle_mouse_input(core, event, false)
        self.base.default_handle_input(core, event)
    }
}
```

## Standard Bind Group Layout

Your WGSL shaders should follow this layout for predictable resource access.

```wgsl
// Group 0: Per-Frame Data (Engine-Managed)
struct TimeUniform { time: f32, delta: f32, frame: u32, _padding: u32 };
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

### Multi-Pass Texture Dependencies

When using `.with_multi_pass()`, the framework uses **ping-pong double-buffering** with per-buffer write tracking. Each buffer independently remembers which side was last written, so **any pass can read from any previous pass's output** — no adjacency restrictions.

**How dependencies map to input textures:**

The `&["dep1", "dep2"]` array in `PassDescription::new()` maps directly by position:

- `deps[0]` → `input_texture0` in WGSL (`@group(3) @binding(0)`)
- `deps[1]` → `input_texture1` in WGSL (`@group(3) @binding(2)`)
- `deps[N]` → `input_textureN` in WGSL (`@group(3) @binding(2*N)`)

There is **no hard limit** on the number of dependencies per pass. The Group 3 layout is automatically sized to fit the maximum dependency count across all passes. Passes with fewer dependencies repeat the first dependency for the remaining slots.

```rust
// Each pass reads from any previous pass — order doesn't matter
PassDescription::new("structure_tensor", &[]),
PassDescription::new("tensor_field", &["structure_tensor"]),     // input_texture0 = structure_tensor
PassDescription::new("kuwahara", &["tensor_field"]),             // input_texture0 = tensor_field
// Reading from non-adjacent passes is fine:
PassDescription::new("lic_edges", &["tensor_field", "kuwahara"]), // input_texture0 = tensor_field, 1 = kuwahara
PassDescription::new("main_image", &["lic_edges"]),
```

### Iterative Solvers via Duplicate Passes

Repeat the same entry point name to run iterative algorithms (e.g., Jacobi pressure) within a single `dispatch()` call:

```rust
let passes = vec![
    PassDescription::new("compute_field", &["compute_field"]),
    PassDescription::new("pressure", &["compute_field", "pressure"]),  // iteration 1
    PassDescription::new("pressure", &["compute_field", "pressure"]),  // iteration 2
    PassDescription::new("pressure", &["compute_field", "pressure"]),  // ...repeat N times
    PassDescription::new("project", &["compute_field", "pressure"]),
    PassDescription::new("main_image", &["project"]),
];
```

note that cuneus creates one buffer pair per unique name. Each dispatch flips the write side automatically, so iters ping/pong correctly: iter 1 writes `.0` → iter 2 reads `.0`, writes `.1` → iter 3 reads `.1`, writes `.0`, etc. Non-iterated passes stay fixed throughout. *Example: `fluid.rs` uses 12 Jacobi pressure iterations this way.*

### `dispatch()` vs `dispatch_stage()`

- **`dispatch()`** — Runs all passes with correct per-pass ping-pong bind groups. Auto-increments frame counter. Use for **texture-based multipass** (most shaders).
- **`dispatch_stage(encoder, core, index)`** — Runs one pass using **global** bind groups (no ping-pong awareness). Does not increment frame counter. Use for **storage-buffer multipass** (`fluidsim.rs`) or **path tracing accumulation** (`mandelbulb.rs`).

**Important:** `dispatch_stage()` cannot select correct ping-pong sides for texture-based multipass. For iterative texture-based solving, use duplicate passes with `dispatch()` instead.

### Per-Buffer Resolution

Each buffer can have its own resolution, independent of the screen size. This enables half-res blur passes, 1D lookup tables, fixed-size accumulation buffers, and more.

```rust
let passes = vec![
    PassDescription::new("scene", &[]),                              // Full screen resolution
    PassDescription::new("blur", &["scene"])
        .with_resolution_scale(0.5),                                 // Half-res (recomputed on resize)
    PassDescription::new("lut", &["scene"])
        .with_resolution(256, 1),                                    // Fixed 256x1 lookup table
    PassDescription::new("main_image", &["scene", "blur", "lut"]),   // Reads from all three
];
```

**How it works:**

- `.with_resolution(width, height)` — Fixed pixel size, never changes on resize
- `.with_resolution_scale(0.5)` — Relative to screen, recomputed on window resize
- No override — matches screen resolution (default behavior)
- Workgroup dispatch count is **automatically computed** from the buffer's actual dimensions
- Shaders query `textureDimensions()` at runtime, so they're already dimension-agnostic
- When reading from a different-sized buffer, use `textureDimensions(input_texture0)` to scale coordinates

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

### Mid-Frame Buffer Updates (`flush_encoder`)

When doing ping-pong buffer simulations, you may need buffer updates to take effect before the next dispatch. wgpu batches all `write_buffer` calls before any dispatches in the same submit, so use `core.flush_encoder()` to force changes through:

```rust
// Update params, submit, get new encoder
self.params.ping = 1 - self.params.ping;
self.compute_shader.set_custom_params(self.params, &core.queue);
frame.encoder = core.flush_encoder(frame.encoder);

// Now the next dispatch sees the updated ping value
self.compute_shader.dispatch_stage(&mut frame.encoder, core, NEXT_PASS);
```

*See `fluidsim.rs` for a full example with 20+ pressure iterations per frame.*

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

> **Note:** Audio synthesis examples use two different patterns depending on *who decides what to play*:
> - **GPU-driven** (`veridisquo.rs`): The shader composes the music (frequencies, amplitudes) and writes to the audio buffer. The CPU reads it back with `pollster::block_on(read_audio_buffer(...))` and feeds it to `SynthesisManager`. The GPU is the "composer".
> - **CPU-driven** (`synth.rs`): The user presses keys on the keyboard and the CPU calls `synth.set_voice(...)` directly. No GPU→CPU readback is needed — the GPU only handles visualization.

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

### External Textures

Two methods for external texture input:

**`.with_input_texture()`** - Single input in **Group 1** (bindings 2-3).

```wgsl
@group(1) @binding(2) var input_texture: texture_2d<f32>;
@group(1) @binding(3) var input_sampler: sampler;
```

```rust
compute_shader.update_input_texture(&tm.view, &tm.sampler, &core.device);
```

**Important for multi-pass:** When using `.dispatch()`, `input_texture` is only available in `main_image` pass. Intermediate passes do not receive it. To access `input_texture` from all passes, use `dispatch_stage()` instead. See `fft.rs` and `computecolors.rs` for this pattern.

**`.with_channels(N)`** - N texture/sampler pairs in **Group 2**. Accessible from **all passes** with both `.dispatch()` and `dispatch_stage()`.

```wgsl
@group(2) @binding(0) var channel0: texture_2d<f32>;
@group(2) @binding(1) var channel0_sampler: sampler;
@group(2) @binding(2) var channel1: texture_2d<f32>;
@group(2) @binding(3) var channel1_sampler: sampler;
```

```rust
compute_shader.update_channel_texture(0, &tm.view, &tm.sampler, &core.device, &core.queue);
compute_shader.update_channel_texture(1, &tm2.view, &tm2.sampler, &core.device, &core.queue);
```

*See `kuwahara.wgsl` where `channel0` is sampled from multiple passes via a helper function.*

**Summary:**

| Method                  | Single-pass | Multi-pass `.dispatch()` | Multi-pass `dispatch_stage()` |
|-------------------------|-------------|--------------------------|-------------------------------|
| `.with_input_texture()` | All passes  | `main_image` only        | All stages                    |
| `.with_channels()`      | All passes  | All passes               | All stages                    |

### Loading Textures From Code

Textures don't require egui or drag-and-drop. Call `base.load_media(core, path)` in `init()` to embed assets — it auto-detects format (PNG, JPG, HDR, EXR, MP4, etc.). You can still override via drag-and-drop at runtime.

```rust
// Single embedded texture — works with .with_input_texture() or .with_channels(1)
let mut base = RenderKit::new(core);
base.load_media(core, "assets/sky.hdr").ok(); // HDRI, image, or video

// Multiple independent channels — each bound separately
let mut cs = cuneus::compute_shader!(core, "shaders/my.wgsl", config); // config has .with_channels(2)
let img = image::open("assets/albedo.png").unwrap().into_rgba8();
let tex = TextureManager::new(&core.device, &core.queue, &img, &base.texture_bind_group_layout);
cs.update_channel_texture(0, &tex.view, &tex.sampler, &core.device, &core.queue); // channel0
// channel1 left empty (1x1 magenta fallback) or loaded the same way
```

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

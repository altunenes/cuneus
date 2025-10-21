# Cuneus Usage Guide

Cuneus is a Rust-based GPU shader engine with a unified backend that handles single-pass, multi-pass, and atomic compute shaders with built-in UI controls, hot-reloading, and media integration.


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

1.  **Texture-Based (Ping-Pong):** Ideal for image processing and feedback effects. Intermediate results are stored in textures that are automatically swapped between passes. This is enabled with `.with_multi_pass()` and does not require manual storage buffers.
    *   *Examples: `lich.rs`, `jfa.rs`, `currents.rs`*
2.  **Storage-Buffer-Based (Shared Memory):** Ideal for GPU algorithms like FFT or simulations like CNNs. All passes read from and write to the same large, user-defined storage buffers. This is enabled by using `.with_multi_pass()` *and* `.with_storage_buffer()`.
    *   *Examples: `fft.rs`, `cnn.rs`*

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
    render_kit: RenderKit,
    compute_shader: ComputeShader,
    current_params: MyParams,
}

// 3. Implement the ShaderManager trait
impl ShaderManager for MyShader {
    fn init(core: &Core) -> Self {
        // RenderKit handles the final blit to screen and UI (vertex/blit shaders built-in)
        let render_kit = RenderKit::new(core, &[/* bind group layouts */], None);
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
            .with_label("My Shader")
            // For Single-Pass, use .with_entry_point():
            .with_entry_point("main")
            // 2. (Multi-Pass) Comment out .with_entry_point() and use .with_multi_pass() instead: (we define the passes above)
            // .with_multi_pass(&passes)
            .with_custom_uniforms::<MyParams>()
            .with_mouse()
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

        Self { render_kit, compute_shader, current_params: initial_params }
    }

    fn update(&mut self, core: &Core) {
        // Update time uniform, check for hot-reloads, etc.
        let time = self.render_kit.controls.get_time(&self.render_kit.start_time);
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

        // 3. (Multi-Pass) For texture-based feedback (ping-pong), you must flip the buffers
        //    at the end of the frame so the next frame reads from the correct texture.
        /*
        self.compute_shader.flip_buffers();
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

// Group 2: Global Engine Resources (Order is fixed if multiple are enabled)
// The binding index for each resource depends on which resources before it were enabled.
// Example: If only mouse and atomics are enabled, mouse is @binding(0) and atomics is @binding(1).

// @binding(0): Mouse data (if .with_mouse() is used)
@group(2) @binding(0) var<uniform> mouse: MouseUniform; 
// @binding(1-3): Font data (if .with_fonts() is used)
@group(2) @binding(1) var<uniform> font_uniform: FontUniforms;
@group(2) @binding(2) var font_texture: texture_2d<f32>;
@group(2) @binding(3) var font_sampler: sampler;
// @binding(N): Audio buffer (if .with_audio() is used)
@group(2) @binding(4) var<storage, read_write> audio_buffer: array<f32>;
// @binding(N+1): Atomic buffer (if .with_atomic_buffer() is used)
@group(2) @binding(5) var<storage, read_write> atomic_buffer: array<atomic<u32>>;
// @binding(N+2..): Media channels (if .with_channels() is used)
@group(2) @binding(6) var channel0: texture_2d<f32>;
@group(2) @binding(7) var channel0_sampler: sampler;

// Group 3: User Data & Multi-Pass I/O
// User-defined storage buffers (if .with_storage_buffer() is used, this takes priority)
@group(3) @binding(0) var<storage, read_write> my_data: array<f32>;
// OR: Multi-pass input textures (if .with_multi_pass() is used without storage buffers)
@group(3) @binding(0) var input_texture0: texture_2d<f32>;
@group(3) @binding(1) var input_sampler0: sampler;
```

## Advanced Topics

### Workgroup Sizes

-   **WGSL is the Source of Truth:** A workgroup size defined in your shader with `@workgroup_size(x, y, z)` will always be used to compile the pipeline.
-   **Builder is a Fallback:** `.with_workgroup_size()` is only used if the WGSL entry point has no size decorator.
-   **Per-Pass Specificity:** For multi-pass shaders, you can specify a unique workgroup size for each stage. This is critical for performance in algorithms like FFTs or CNNs.
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

### GPU Audio Generation
You can generate audio synthesis parameters directly on the GPU and read them back on the CPU for playback. The `.with_audio(size)` builder method provides a `storage` buffer in Group 2.

```wgsl
// In WGSL: Write to the audio buffer
// This pattern is used by veridisquo.wgsl
if (global_id.x == 0u && global_id.y == 0u) {
    audio_buffer[0] = final_melody_freq;
    audio_buffer[1] = melody_amplitude;
    audio_buffer[2] = f32(waveform_type);
    audio_buffer[3] = final_bass_freq;
    audio_buffer[4] = bass_amplitude;
}
```
```rust
// In Rust: Read the buffer and send to the audio engine
// This pattern is used by veridisquo.rs
if let Ok(data) = pollster::block_on(compute.read_audio_buffer(&core.device, &core.queue)) {
    // data[0] is frequency, data[1] is amplitude, etc.
    synth.set_voice(0, data[0], data[1], true);
}
```
**Pro-tip:** The audio buffer is just a generic `array<f32>`. You can use it to store any `f32` data, like the game state in `blockgame.wgsl`.

### External Textures (`.with_channels()`)
The `.with_channels(N)` method exposes `N` texture/sampler pairs in Group 2, making them globally accessible to **all passes** of a multi-pass shader. This is the preferred way to pipe in video, webcam feeds, or static images into complex simulations.
*   *Example: `fluid.rs` uses `.with_channels(1)` to feed a video into its simulation.*

### Audio Spectrum Analysis (`.with_audio_spectrum()`)
Use `.with_audio_spectrum(64)` to access real-time frequency spectrum data from loaded audio/video files as a read-only storage buffer in Group 2.

### Fonts
The `.with_fonts()` method provides everything needed to render text directly inside your compute shader. This is perfect for debug overlays or creative typography effects.
*   *Examples: `debugscreen.rs` uses this for its UI, and `cnn.rs` uses it to label its output bars.*
# Cuneus Usage Guide

Cuneus is a GPU compute shader engine with a unified backend for single/multi pass, and atomic compute shaders. It features built in UI controls, hot-reloading, media integration, and GPU driven audio synthesis.

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
- **Manual (`.dispatch_stage()`):** Runs one pass at a time for fine grained control storage buffer multi pass (FFT, CNN), staged atomic pipelines, and iterative accumulation where you advance the frame counter yourself. **You must manually increment `compute_shader.current_frame` when using this method.**

### 5. Multi-Pass Models

The framework elegantly handles two types of multi-pass computation:

1. **Texture-Based (Ping-Pong):** Ideal for image processing and feedback effects. Intermediate results are stored in textures. Each buffer independently tracks its write state, so any pass can read from any previous pass's output — and cross-frame feedback (self-referencing passes) works automatically.
   - *Examples with cross-frame feedback: `lich.rs`, `rorschach.rs`, `jfa.rs`*
   - *Examples with within-frame only: `kuwahara.rs`, `fluid.rs`*

2. **Storage-Buffer-Based (Shared Memory):** Ideal for GPU algorithms like FFT or simulations like CNNs. All passes read from and write to the same large, user-defined storage buffers. This is enabled by using `.with_multi_pass()` *and* `.with_storage_buffer()`.
   - *Examples: `fft.rs`, `cnn.rs`*

**Group 3 holds one mode per shader.** Combining `.with_storage_buffer()` with `.with_multi_pass()` disables the ping-pong texture inputs (`input_textureN`/`input_samplerN`) — passes communicate through the storage buffer instead.

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
        self.base.renderer.render_to_view(&mut frame.encoder, &frame.view, &self.compute_shader.get_output_texture().bind_group);

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

Built in keys: H toggles the UI and F fullscreen. Rebind or free them with `self.base.key_handler.ui_key` / `fullscreen_key`, e.g. `= Some(Key::Named(NamedKey::F11))` or `= None`.

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
// `.with_atomic_buffer(channels)` sizes it width*height*channels — per-pixel, scales with
// the window. Use `.with_atomic_buffer_bytes(n)` when the data isn't per-pixel: the buffer
// stays a fixed size regardless of window size.
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
- **`dispatch_stage(encoder, core, index)`** — Runs one pass using **global** bind groups (no ping-pong awareness). Does not increment the frame counter. Use for **storage-buffer multipass** (`fft.rs`, `cnn.rs`) and **manually-staged / accumulating pipelines** (`buddhabrot.rs`).

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

### Manual Stage Dispatch

`dispatch_stage()` runs one pass at a time, so you decide what runs each frame and advance the frame counter yourself. This suits iterative accumulation (run the heavy pass only until it converges, keep compositing every frame) and any pipeline needing conditional or per-stage control. `dispatch_stage()` does not auto-increment — bump `current_frame` yourself.

```rust
// See buddhabrot.rs
fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
    // ...
    // Run each pass explicitly, by index, in order
    self.compute_shader.dispatch_stage(&mut frame.encoder, core, 0); // compute / accumulate
    self.compute_shader.dispatch_stage(&mut frame.encoder, core, 1); // composite / main_image

    // dispatch_stage does not auto increment advance the frame counter yourself
    self.compute_shader.current_frame += 1;
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

## 3D Models (glTF)

`MeshScene` draws `.glb`/`.gltf` models with your own WGSL materials into an HDR texture that a compute pass reads as a channel. `examples/meshlab.rs` uses every feature below.

| Piece | What it is | API |
| :--- | :--- | :--- |
| **Mesh** | Geometry, textures, skeleton and clips of a file; drawn any number of times | `add_mesh`, `replace_mesh` |
| **Material** | A WGSL file + its parameters + options; variants share the shader | `mesh_material!`, `material_variant`, `set_params`, `set_options` |
| **Object** | Mesh + material + one or many instances | `spawn`, `spawn_instances`, `set_transform`, `instances_mut`, `despawn` |

```rust
let mut scene = MeshScene::new(core);
let model = scene.add_mesh(core, &MeshData::from_gltf("model.glb")?);
// load a material (hot reloads on save)
let glow = cuneus::mesh_material!(scene, core, "shaders/glow.wgsl", GlowParams);
let obj = scene.spawn(model, glow, Mat4::IDENTITY);
// the post shader (built with .with_channels(1)) reads the colour as channel0
scene.attach(core, &mut post, 0);

// every frame
scene.set_params(&core.queue, glow, &params);
scene.render(&mut frame.encoder, core, &mut post, &MeshView::orbit(&camera, camera.pose, aspect));
post.dispatch(&mut frame.encoder, core);

// in update(): export draws the scene at export size and time
scene.handle_export(core, &mut post, &mut base, |scene, t, aspect| MeshView::orbit(&camera, camera.turntable(t), aspect));
```

The post shader needs `.with_channels(1)` in its builder (`2` with `attach_gbuffer`); `attach` keeps the scene's textures at its size through resizes and exports. `OrbitCamera` handles orbit/pan/fly input (`handle_event`, `update`); `MeshData::normalize_transform()` centres a model at unit size.

### Materials

A material is a `fragment` function and optionally a `vertex` function. [`src/mesh/prelude.wgsl`](src/mesh/prelude.wgsl) is prepended to it:

```wgsl
struct GlowParams { color: vec3<f32>, strength: f32 };
@group(1) @binding(0) var<uniform> material: GlowParams;

@fragment
fn fragment(mesh: VertexOutput, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    let base = gltf_base_color(mesh);
    let n = gltf_normal(mesh, is_front);
    let v = normalize(view.world_position - mesh.world_position.xyz);
    var col = pbr_direct(n, v, sun.direction, sun.color, base.rgb, 0.0, 0.5)
            * directional_shadow(mesh.world_position.xyz, mesh.world_normal);
    col += material.color * pow(1.0 - max(dot(n, v), 0.0), 3.0) * material.strength;
    return vec4<f32>(col, base.a);
}

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    // the default vertex output, then move it
    var out = mesh_vertex(vertex);
    let p = out.world_position.xyz + vec3<f32>(0.0, sin(out.world_position.x * 4.0 + globals.time) * 0.05, 0.0);
    out.world_position = vec4<f32>(p, 1.0);
    out.position = position_world_to_clip(p);
    return out;
}
```

| Available | What |
| :--- | :--- |
| `globals`, `view` | time, delta, frame; camera matrices, `world_position`, `viewport` |
| `VertexOutput` | `world_position`, `world_normal`, `uv`, `uv_b`, `world_tangent`, `color`, `instance_index` |
| `mesh_vertex(vertex)` | the default vertex output, to modify in your own `vertex` |
| `get_world_from_local`, `mesh_position_local_to_world`, `mesh_normal_local_to_world`, `mesh_tangent_local_to_world`, `position_world_to_clip` | instance transforms |
| `gltf_base_color`, `gltf_normal`, `gltf_metallic_roughness`, `gltf_emissive`, `gltf_occlusion`, `gltf_uv` | the model's own material and textures (missing textures read as neutral) |
| `material_texture0..3`, `material_sampler`, `equirect_uv(dir)` | the material's own textures |
| `get_instance_data(instance_index)` | 4 floats per instance |
| `sun`, `directional_shadow(pos, normal)` | the scene's sun and its shadow (1 lit, 0 shadowed) |
| `light_count()`, `light_sample(i, pos, normal)`, `light_shadow(i, pos, normal)` | point / spot lights: direction and arriving light (falloff, cone, shadow applied) |
| `pbr_direct`, `pbr_ambient` | GGX lighting helpers |

Animated models are posed before `vertex` runs, so one material works for static and animated models.

`MaterialOptions` (`scene.set_options`):
- `blend`: `Auto` (from the file), `Opaque`, `Alpha`, `Additive`
- `cull`: `Auto` (from the file's double-sided flag), `None`, `Back`, `Front`
- `shadows`: cast shadows
- `prepass`: depth-test opaque objects first so `fragment` runs once per visible pixel; set `false` when `fragment` uses `discard`

Material textures: `set_image_file(core, material, slot, path, srgb)` (PNG/JPG; `.hdr`/`.exr` stay HDR, with mipmaps), `set_image` (in memory), `set_texture` (any filterable texture view, e.g. a compute shader's output).

### Scene

- **Instances:** `spawn_instances(mesh, material, instances)` draws all copies in one draw call. Copies outside the camera are skipped (`scene.culling`; raise `scene.cull_margin` if a vertex shader pushes geometry outward).
- **GPU-driven instances:** `set_gpu_instances(core, obj, Some(GpuInstances { buffer, count, bounds }))` reads copies from a storage buffer, e.g. one a compute shader writes every frame: `array<struct { transform: mat4x4<f32>, data: vec4<f32>, anim: vec4<f32>, anim_b: vec4<f32> }>` (`GPU_INSTANCE_SIZE` = 112 bytes). They are culled on the GPU and drawn indirectly when the device supports `INDIRECT_FIRST_INSTANCE`; shadows see every copy. `bounds` is a world sphere around all of them.
- **Sun:** `scene.sun`: `direction` (or `set_angles(yaw, height)`), `color`, `shadows`, `shadow_size`, `softness`, `normal_bias`, `bounds` (`None` fits the shadow casters).
- **Lights:** `scene.lights.push(Light::point(pos, color, range))` or `Light::spot(pos, dir, color, range, angle)`, `.with_shadows()` (a spot uses 1 shadow map, a point 6). `scene.light_shadow_size` sets their resolution.
- **Picking:** `scene.pick(x, y)` in physical pixels; a frame or two later `scene.take_pick()` returns `Some(Some(Pick { object, instance, position }))`, or `Some(None)` for nothing.
- **Post channels:** `attach(core, &mut post, 0)` gives the colour (alpha 0 where empty, edges premultiplied: composite with `bg * (1 - a) + rgb`). `attach_gbuffer(core, &mut post, 1)` gives world normal (xyz) and view depth (w), coverage weighted like the colour.

### Animation

Files with skeletons, node animation or blend shapes keep them:

```rust
// names, durations
let clips = &scene.skinning(model).unwrap().animations;
// one clip, loops
scene.animate(obj, Some(walk), t);
// body + blend-shape layer
scene.animate_layers(obj, &[(walk, t, 1.0), (blink, t, 1.0)]);
// crossfade
scene.animate_layers(obj, &[(idle, t, 1.0), (attack, t2, fade)]);
// your own Pose
scene.set_pose(obj, pose);
```

Layers apply in order, each blending what it animates by its weight. `MeshAnimation::morph_only()` finds blend-shape-only clips. Pass `t.min(duration)` to play a clip once and hold its last frame. All instances of an object share its pose, unless it is a crowd.

**Crowds:** `scene.set_crowd(obj, true)` gives every instance its own clip: `Instance { anim: InstanceAnim::clip(walk, t).fade_to(run, t2, fade), .. }`, or `anim = (clip, time, next clip, next time)` and `anim_b.x = fade` in a GPU instance buffer. Clips are baked into bone matrices and posed in the vertex shader; blend shapes stay at rest.

**Loader notes:** a material's uv set, `KHR_texture_transform` and sampler come from its base colour texture and apply to all its textures. Not supported: tangent blend-shape deltas, non-uniform joint scale, Draco/meshopt-compressed files.

## Media & Integration

### GPU Music Generation & Synthesis

Cuneus supports **bidirectional GPU-CPU audio workflows** using two complementary systems:

**1. Audio Visualization (`.with_audio_spectrum()`)** - Analyze loaded audio/video:
- **Flow**: Media file → GStreamer spectrum analyzer → CPU writes to buffer → GPU reads for visualization
- **Shader Access**: `@group(2) var<storage, read> audio_spectrum: array<f32>` (read-only)
- **Use Case**: Audio visualizers like `audiovis.rs`

**Raw PCM access (host-side):** both `WebcamTextureManager` and `VideoTextureManager` also expose the time-domain waveform via `pop_pcm_samples(n)` — mono `f32` at 44.1kHz in a bounded ring buffer. Use it when you need raw samples instead of FFT magnitudes. Samples stay host-side unless you upload them yourself. Check `has_pcm()` before draining.

**2. Audio Synthesis (`.with_audio()`)** - Generate music on GPU:
- **Flow**: GPU computes raw PCM samples per-frame → CPU reads back → `PcmStreamManager` streams to audio output
- **Shader Access**: `@group(2) var<storage, read_write> audio_buffer: array<f32>` (read-write)
- **Use Case**: Music generators like `synth.rs`, `veridisquo.rs`

The shader writes interleaved stereo f32 samples (left, right, left, right...) to the audio buffer. The CPU reads them back each frame and pushes to GStreamer via `PcmStreamManager`. This is per-sample synthesis. you have full control over harmonics, effects, envelopes, anything.

#### Per-Sample GPU Audio Synthesis

Write a `mainSound(time) → vec2<f32>` function in WGSL. Thread (0,0) loops over samples and fills the buffer:

```wgsl
fn mainSound(t: f32) -> vec2<f32> {
    // Your synthesis here — harmonics, filters, delay lines, drums, anything
    let melody = sin(6.283 * 440.0 * t) * 0.3;
    return vec2<f32>(melody, melody); // stereo
}

// Thread (0,0) generates all samples for this frame
if (g.x == 0u && g.y == 0u) {
    for (var i = 0u; i < params.samples_to_generate; i++) {
        let t = f32(params.sample_offset + i) / params.sample_rate;
        let stereo = mainSound(t);
        audio_buffer[i * 2u] = stereo.x;
        audio_buffer[i * 2u + 1u] = stereo.y;
    }
}
```

```rust
// In Rust: read back PCM and push to audio output
let mut pcm = PcmStreamManager::new(Some(44100))?;
pcm.start()?;

// In update() each frame:
let prev = self.last_samples_generated;
if prev > 0 {
    if let Ok(data) = pollster::block_on(compute.read_audio_buffer(&device, &queue)) {
        pcm.push_samples(&data[..(prev * 2) as usize])?;
    }
}
// Time-sync: generate exactly the samples real-time demands
let needed = ((elapsed * 44100.0) as u64 - pcm.samples_written()).min(1024) as u32;
params.sample_offset = pcm.samples_written() as u32;
params.samples_to_generate = needed;
```

**Examples:**

- `veridisquo.rs` - Full song: drawbar organ, Moog bass, chord pads, kick drums, delay lines, sidechain
- `synth.rs` - Interactive keyboard synth with per-sample ADSR, filter, distortion, chorus, reverb
- `debugscreen.rs` - Simple tone generation using `SynthesisManager` (oscillator-based, not PCM)

**Pro-tip - Persistent GPU state:** A `storage, read_write` array of floats persists across frames, so you can keep arbitrary GPU-side state in it. Use `.with_storage_buffer(StorageBufferSpec::new("name", bytes))` for a dedicated state buffer (bound in group 3); the `.with_audio()` buffer is the same kind of array and can also be repurposed as scratch storage when you don't need sound.

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

**Multi-pass note:** `.with_input_texture()` fits single-pass shaders and storage-buffer multi-pass (`fft.rs`, `computecolors.rs`, run via `dispatch_stage()`). Don't use it with ping-pong (texture) multi-pass — it panics at startup with a Group 1 bind-group mismatch. For an HDRI/environment map, or any texture you sample from an intermediate pass, use `.with_channels()` instead.

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
| `.with_input_texture()` | All passes  | `main_image` only †      | All stages                    |
| `.with_channels()`      | All passes  | All passes               | All stages                    |

† Storage-buffer multi-pass only. Ping-pong multi-pass panics at startup — use `.with_channels()` (and for HDRIs, always).

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

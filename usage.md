
# How To Use Cuneus

In fact you can simply copy a rust file in the “bin” folder and just go to the wgsl stage. But to set the parameters in egui you only need to change the parameters.

## Quick Start

1. Copy one of the template files from `src/bin/` that best matches your needs:
   - `mandelbrot.rs`: Minimal single-pass shader without GUI controls
   - `spiral.rs`: Simple single-pass shader with texture support
   - `feedback.rs`: Basic two-pass shader
   - `fluid.rs`: Multi-pass shader with texture support
   - `attractor.rs`: Three-pass rendering example
   - `xmas.rs`: Single pass with extensive parameter controls
  
if you want 4 passes or more the logic is exactly the same. 

2. Rename and modify the copied file to create your shader
3. Focus on writing your WGSL shader code :-)

## Template Structure

### Basic Single Pass Shader (No GUI)

The simplest way to start is with a basic shader like `mandelbrot.rs`. This template includes:

1. Core imports and setup
2. Minimal shader parameters
3. Basic render pipeline
4. WGSL shader code

I created this by completely copying and pasting xmas.rs, and I could only focus on my shader.

```rust
// 1. Required imports
use cuneus::{Core, ShaderManager, BaseShader /* ... */};

// 2. Optional parameters if needed
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ShaderParams {
    // Your parameters here
}

// 3. Main shader structure
struct MyShader {
    base: BaseShader,
    // Add any additional fields needed
}

// 4. Implement required traits
impl ShaderManager for MyShader {
    fn init(core: &Core) -> Self { /* ... */ }
    fn update(&mut self, core: &Core) { /* ... */ }
    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> { /* ... */ }
    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool { /* ... */ }
}
```

### Adding GUI Controls (optimal)

To add parameter controls through egui:

1. Define your parameters struct
2. Add UI controls in the render function

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ShaderParams {
    rotation_speed: f32,
    intensity: f32,
    // Add more parameters as needed
}

// In render function:
let full_output = if self.base.key_handler.show_ui {
    self.base.render_ui(core, |ctx| {
        egui::Window::new("Settings").show(ctx, |ui| {
            changed |= ui.add(egui::Slider::new(&mut params.rotation_speed, 0.0..=5.0)
                .text("Rotation Speed")).changed();
            // Add more controls
        });
    })
};
```

## WGSL Shader Patterns

### Standard Vertex Shader
All shaders use this common vertex shader (vertex.wgsl):
```wgsl
struct VertexOutput {
    @location(0) tex_coords: vec2<f32>,
    @builtin(position) out_pos: vec4<f32>,
};

@vertex
fn vs_main(@location(0) pos: vec2<f32>) -> VertexOutput {
    let tex_coords = vec2<f32>(pos.x * 0.5 + 0.5, 1.0 - (pos.y * 0.5 + 0.5));
    return VertexOutput(tex_coords, vec4<f32>(pos, 0.0, 1.0));
}
```

### Single Pass Fragment Shader
Basic structure for a fragment shader:
```wgsl
// Time uniform
@group(0) @binding(0)
var<uniform> u_time: TimeUniform;

// Optional EGUI parameters
@group(1) @binding(0)
var<uniform> params: Params;

@fragment
fn fs_main(@builtin(position) FragCoord: vec4<f32>, 
           @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    // Your shader code here
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
```

### Multi-Pass Shader
For effects requiring multiple passes:
```wgsl
@group(0) @binding(0) var prev_frame: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;

@fragment
fn fs_pass1(...) -> @location(0) vec4<f32> {
    // First pass processing
}

@fragment
fn fs_pass2(...) -> @location(0) vec4<f32> {
    // Second pass processing
}
```


### Hot Reloading
cuneus supports hot reloading of shaders. Simply modify your WGSL files and they will automatically reload.

### Export Support
Built-in support for exporting frames as images. Access through the UI when enabled. "Start time" is not working correctly currently.

### Texture Support
Load and use textures in your shaders:
```rust
if let Some(ref texture_manager) = self.base.texture_manager {
    render_pass.set_bind_group(0, &texture_manager.bind_group, &[]);
}
```

## Resolution Handling

cuneus handles both logical and physical resolution:

1. Initial window size is set in logical pixels:

```rust
   let (app, event_loop) = ShaderApp::new("My Shader", 800, 600);
 ```
2.  On high-DPI displays (like Retina), the physical resolution is automatically scaled:
    e.g., 800x600 logical becomes 1600x1200 physical on a 2x scaling display
    Your shader's UV coordinates (0.0 to 1.0) automatically adapt to any resolution
    Export resolution can be set independently through the UI

Your WGSL shaders can access actual dimensions when needed:
```wgsl
let dimensions = vec2<f32>(textureDimensions(my_texture));
```

### Adding Interactive Controls
1. Start with a template that includes GUI (e.g., `xmas.rs`)
2. Define your parameters in the ShaderParams struct
3. Add UI controls in the render function
4. Connect parameters to your shader

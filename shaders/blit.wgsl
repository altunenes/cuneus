struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

@group(0) @binding(0) var texture: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;

@vertex
fn vs_main(@location(0) position: vec2<f32>) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(position, 0.0, 1.0);
    out.tex_coords = vec2<f32>(position.x * 0.5 + 0.5, 1.0 - (position.y * 0.5 + 0.5));
    return out;
}

@fragment
fn fs_main(@location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    var color = textureSample(texture, tex_sampler, tex_coords);
    color.a = 1.0;
    return color;
}
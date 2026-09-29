// Shown while a material fails to compile
@fragment
fn fragment(mesh: VertexOutput, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    let n = gltf_normal(mesh, is_front);
    return vec4<f32>(gltf_base_color(mesh).rgb * (0.2 + 0.8 * max(dot(n, normalize(vec3<f32>(0.4, 1.0, 0.6))), 0.0)), 1.0);
}

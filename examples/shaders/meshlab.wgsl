// meshlab material: PBR sun + lights + HDRI, rim glow and bands, wobble deform, pick highlight.
// Output is linear HDR; values above 1 glow in the bloom pass.

struct MeshParams {
    ambient: f32,
    gloss: f32,
    rim: f32,
    rim_power: f32,
    rim_r: f32,
    rim_g: f32,
    rim_b: f32,
    pattern: f32,
    pattern_scale: f32,
    pattern_speed: f32,
    wobble: f32,
    wobble_speed: f32,
    emissive: f32,
    opacity: f32,
    hue_spread: f32,
    shadow: f32,
    env: f32,
    _q0: f32,
    _q1: f32,
    _q2: f32,
};
@group(1) @binding(0) var<uniform> material: MeshParams;

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    let world_from_local = get_world_from_local(vertex.instance_index);
    var p = mesh_position_local_to_world(world_from_local, vec4<f32>(vertex.position, 1.0)).xyz;
    let n = mesh_normal_local_to_world(vertex.normal, vertex.instance_index);
    // wobble: a wave travelling up the model, along the normal; each copy has its own phase
    let phase = get_instance_data(vertex.instance_index).x * 6.2832;
    p += n * material.wobble * 0.04 * sin(p.y * 9.0 - globals.time * material.wobble_speed + phase);
    out.world_position = vec4<f32>(p, 1.0);
    out.position = position_world_to_clip(p);
    out.world_normal = n;
    out.uv = vertex.uv;
    out.world_tangent = mesh_tangent_local_to_world(world_from_local, vertex.tangent, vertex.instance_index);
    out.color = vertex.color;
    out.instance_index = vertex.instance_index;
    out.uv_b = vertex.uv_b;
    return out;
}

fn hue_rotate(c: vec3<f32>, a: f32) -> vec3<f32> {
    let k = vec3<f32>(0.57735);
    return c * cos(a) + cross(k, c) * sin(a) + k * dot(k, c) * (1.0 - cos(a));
}

@fragment
fn fragment(mesh: VertexOutput, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    let base = gltf_base_color(mesh);
    let n = gltf_normal(mesh, is_front);
    let mr = gltf_metallic_roughness(mesh);
    let rough = clamp(mr.y * material.gloss, 0.04, 1.0);
    let p = mesh.world_position.xyz;
    let v = normalize(view.world_position - p);

    // the scene's sun with its shadow + sky/ground ambient
    let lit = mix(1.0, directional_shadow(p, mesh.world_normal), material.shadow);
    var col = pbr_direct(n, v, sun.direction, sun.color, base.rgb, mr.x, rough) * lit;
    let sky = vec3<f32>(0.35, 0.4, 0.5) * material.ambient;
    let ground = vec3<f32>(0.25, 0.22, 0.2) * material.ambient;
    col += pbr_ambient(n, v, base.rgb, mr.x, rough, sky, ground) * gltf_occlusion(mesh);

    // point / spot lights of the scene
    for (var i = 0u; i < light_count(); i++) {
        let l = light_sample(i, p, mesh.world_normal);
        col += pbr_direct(n, v, l.direction, l.radiance, base.rgb, mr.x, rough);
    }

    // HDRI in material_texture0: blurred reflection + diffuse light (white 1x1 until one is loaded)
    let f0 = mix(vec3<f32>(0.04), base.rgb, mr.x);
    let refl = textureSampleLevel(material_texture0, material_sampler, equirect_uv(reflect(-v, n)), rough * 9.0).rgb;
    let irr = textureSampleLevel(material_texture0, material_sampler, equirect_uv(n), 9.0).rgb;
    col += (refl * f0 + irr * base.rgb * (1.0 - mr.x)) * material.env * gltf_occlusion(mesh);

    // fresnel rim glow, hue shifted per copy
    let hue = get_instance_data(mesh.instance_index).x * material.hue_spread * 6.2832;
    let rim_col = max(hue_rotate(vec3<f32>(material.rim_r, material.rim_g, material.rim_b), hue), vec3<f32>(0.0));
    col += rim_col * pow(1.0 - max(dot(n, v), 0.0), material.rim_power) * material.rim;

    // energy bands sweeping up the surface
    let band = pow(max(sin(p.y * material.pattern_scale - globals.time * material.pattern_speed), 0.0), 24.0);
    col += rim_col * band * material.pattern * 3.0;

    // the model's own emission
    col += gltf_emissive(mesh) * material.emissive;

    // picked copy (instance data .y)
    col += vec3<f32>(1.0, 0.75, 0.3) * get_instance_data(mesh.instance_index).y * (0.3 + pow(1.0 - max(dot(n, v), 0.0), 2.0));

    return vec4<f32>(col, base.a * material.opacity);
}

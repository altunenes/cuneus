// poses one vertex: blend shapes, then a blend of 4 palette matrices; vertices are 5 vec4s (see MeshVertex)
// morph: first delta pair, target count, first weight
struct SkinVertex { joints: vec4<u32>, weights: vec4<f32>, morph: vec4<u32> };
@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> skin: array<SkinVertex>;
@group(0) @binding(2) var<storage, read> palette: array<mat4x4<f32>>;
@group(0) @binding(3) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> deltas: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> morph_weights: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = id.x + id.y * groups.x * 64u;
    if i >= arrayLength(&skin) { return; }
    let s = skin[i];
    let m = palette[s.joints.x] * s.weights.x + palette[s.joints.y] * s.weights.y
          + palette[s.joints.z] * s.weights.z + palette[s.joints.w] * s.weights.w;
    let a = src[i * 5u];
    let b = src[i * 5u + 1u];
    let t = src[i * 5u + 2u];
    var pos = a.xyz;
    var nrm = vec3<f32>(a.w, b.x, b.y);
    for (var k = 0u; k < s.morph.y; k++) {
        let w = morph_weights[s.morph.z + k];
        let d = (s.morph.x + k) * 2u;
        pos += deltas[d].xyz * w;
        nrm += deltas[d + 1u].xyz * w;
    }
    let r = mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz);
    let n0 = r * nrm;
    let n = select(vec3<f32>(0.0), normalize(n0), dot(n0, n0) > 0.0);
    let t0 = r * t.xyz;
    let tt = select(vec3<f32>(0.0), normalize(t0), dot(t0, t0) > 0.0);
    dst[i * 5u] = vec4<f32>((m * vec4<f32>(pos, 1.0)).xyz, n.x);
    dst[i * 5u + 1u] = vec4<f32>(n.y, n.z, b.z, b.w);
    dst[i * 5u + 2u] = vec4<f32>(tt, t.w);
    dst[i * 5u + 3u] = src[i * 5u + 3u];
    dst[i * 5u + 4u] = src[i * 5u + 4u];
}

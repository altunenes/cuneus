// GPU copies: add normal matrix (cofactors) + ids, write all to the full region (shadows), pack
// the visible ones into the culled region; `finalize` writes the packed count into the draw args

struct In { transform: mat4x4<f32>, data: vec4<f32>, anim: vec4<f32>, anim_b: vec4<f32> };
struct Out {
    world_from_local: mat4x4<f32>,
    normal_from_local: mat4x4<f32>,
    data: vec4<f32>,
    anim: vec4<f32>,
    anim_b: vec4<f32>,
    ids: vec4<u32>,
};
struct Params {
    // first slot of the packed (visible) copies
    culled: u32,
    count: u32,
    // first slot of all copies, 0xffffffff when no shadows need them
    full: u32,
    // object index + 1, for picking
    object: u32,
    // mesh bounds, local
    center: vec3<f32>,
    // bounds radius times the cull margin
    radius: f32,
    // 0: keep every copy
    cull: u32,
    prims: u32,
    _p0: u32,
    _p1: u32,
    planes: array<vec4<f32>, 6>,
};

@group(0) @binding(0) var<storage, read> src: array<In>;
@group(0) @binding(1) var<storage, read_write> dst: array<Out>;
@group(0) @binding(2) var<uniform> params: Params;
// [0] visible counter, then per primitive: index count, instance count, first index, base vertex, first instance
@group(0) @binding(3) var<storage, read_write> args: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = id.x + id.y * groups.x * 64u;
    if i >= params.count || i >= arrayLength(&src) { return; }
    let s = src[i];
    let m = s.transform;
    let a = m[0].xyz;
    let b = m[1].xyz;
    let c = m[2].xyz;
    let det = dot(a, cross(b, c));
    let k = select(0.0, 1.0 / det, abs(det) > 1e-20);
    let n = mat4x4<f32>(vec4<f32>(cross(b, c) * k, 0.0), vec4<f32>(cross(c, a) * k, 0.0), vec4<f32>(cross(a, b) * k, 0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0));
    let out = Out(m, n, s.data, s.anim, s.anim_b, vec4<u32>(params.object, i, 0u, 0u));
    if params.full != 0xffffffffu { dst[params.full + i] = out; }

    if params.cull != 0u {
        let p = (m * vec4<f32>(params.center, 1.0)).xyz;
        let r = params.radius * max(length(a), max(length(b), length(c)));
        for (var j = 0; j < 6; j++) {
            if dot(params.planes[j].xyz, p) + params.planes[j].w < -r { return; }
        }
    }
    let slot = atomicAdd(&args[0], 1u);
    dst[params.culled + slot] = out;
}

@compute @workgroup_size(1)
fn finalize() {
    let n = atomicLoad(&args[0]);
    for (var p = 0u; p < params.prims; p++) {
        atomicStore(&args[4u + p * 5u + 1u], n);
    }
}

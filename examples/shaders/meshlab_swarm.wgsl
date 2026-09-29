// meshlab swarm: writes the GpuInstances buffer (transform, data, anim, anim_b; 112 bytes each),
// including each copy's clip and timing for crowds

struct TimeUniform { time: f32, delta: f32, frame: u32, _padding: u32 };
@group(0) @binding(0) var<uniform> u_time: TimeUniform;

struct Swarm {
    count: u32,
    spread: f32,
    scale: f32,
    // body clips to spread over the copies (0 = rest pose)
    clips: u32,
    center: vec3<f32>,
    _q: f32,
};
@group(1) @binding(1) var<uniform> swarm: Swarm;

struct Instance { transform: mat4x4<f32>, data: vec4<f32>, anim: vec4<f32>, anim_b: vec4<f32> };
@group(3) @binding(0) var<storage, read_write> instances: array<Instance>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= swarm.count { return; }
    let f = f32(i);
    let h = fract(sin(f * 12.9898) * 43758.545);
    // golden-angle disc, each copy orbiting at its own speed, bobbing up and down
    let a = f * 2.39996 + u_time.time * (0.15 + h * 0.35);
    let r = swarm.spread * sqrt((f + 0.5) / f32(swarm.count));
    let pos = vec3<f32>(cos(a) * r, sin(f * 0.37 + u_time.time * 2.0) * swarm.spread * 0.12, sin(a) * r);
    // translate(pos) * rotate_y(facing) * scale * translate(-center)
    let c = cos(-a);
    let s = sin(-a);
    let x = vec3<f32>(c, 0.0, -s) * swarm.scale;
    let y = vec3<f32>(0.0, 1.0, 0.0) * swarm.scale;
    let z = vec3<f32>(s, 0.0, c) * swarm.scale;
    let t = pos - (x * swarm.center.x + y * swarm.center.y + z * swarm.center.z);
    instances[i].transform = mat4x4<f32>(vec4<f32>(x, 0.0), vec4<f32>(y, 0.0), vec4<f32>(z, 0.0), vec4<f32>(t, 1.0));
    instances[i].data = vec4<f32>(h, 0.0, 0.0, 0.0);
    // clip, time, next clip, next time; each copy its own clip, speed and phase
    let clip = select(-1.0, f32(i % max(swarm.clips, 1u)), swarm.clips > 0u);
    instances[i].anim = vec4<f32>(clip, u_time.time * (0.8 + h * 0.4) + h * 10.0, -1.0, 0.0);
    instances[i].anim_b = vec4<f32>(0.0);
}

// Live CNN training 47 chrs
// Enes Altun, 2025 MIT License
// inspired by kishimisu, 2023: https://www.shadertoy.com/view/msVXWD

struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

struct CNNParams {
    brush_size: f32, clear_canvas: i32, drawing: i32, mode: u32,
    steps: u32, batch: u32, step_base: u32, reset: u32,
    lr: f32, teach: i32, teach_slot: u32, rep_n: u32,
    fibers: f32, back: f32, aug: f32, dream: f32,
    yaw: f32, elev: f32, zoom: f32, cam: f32,
    bloom: f32, expo: f32, height: f32, kernels: f32,
    conf: f32, seed: u32, _p0: f32, _p1: f32,
}
// Group 1: Primary Pass I/O & Parameters
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: CNNParams;

// Group 2: Engine Resources (Mouse & Fonts)
// note that, WebGPU only supports a maximum of 4 groups (0-3). But you can use more bindings :)
@group(2) @binding(0) var<uniform> mouse: MouseUniform;
@group(2) @binding(1) var<uniform> font_texture_uniform: FontUniforms;
@group(2) @binding(2) var t_font_texture_atlas: texture_2d<f32>;

// weights, optimiser, batch, activations, gradients, history, stats, the drawing pad and its inference
@group(3) @binding(0) var<storage, read_write> net: array<f32>;
// the 3D view in HDR, its quarter-res bloom ping-pong, fiber glow (half res, fixed-point rgb)
@group(3) @binding(1) var<storage, read_write> hdr: array<vec2<u32>>;
@group(3) @binding(2) var<storage, read_write> b1: array<vec2<u32>>;
@group(3) @binding(3) var<storage, read_write> b2: array<vec2<u32>>;
@group(3) @binding(4) var<storage, read_write> fib: array<atomic<u32>>;

struct FontUniforms {
    atlas_size: vec2<f32>,
    char_size: vec2<f32>,
    screen_size: vec2<f32>,
    grid_size: vec2<f32>,
};

struct MouseUniform {
    position: vec2<f32>,
    click_position: vec2<f32>,
    wheel: vec2<f32>,
    buttons: vec2<u32>,
};          

const INPUT_SIZE: u32 = 28u;
const CONV1_SIZE: u32 = 12u;  
const CONV2_SIZE: u32 = 4u;   
const FEATURE_MAPS_1: u32 = 16u;
const FEATURE_MAPS_2: u32 = 32u;
const NUM_CLASSES: u32 = 47u;

fn normalize_input(value: f32) -> f32 {
    return (value - NMEAN) / NSTD;
}

fn canvas_index(x: i32, y: i32) -> u32 {
    return u32(y * i32(INPUT_SIZE) + x);
}

fn conv1_index(x: i32, y: i32, fmap: i32) -> u32 {
    return u32(fmap * i32(CONV1_SIZE * CONV1_SIZE) + y * i32(CONV1_SIZE) + x);
}

fn conv2_index(x: i32, y: i32, fmap: i32) -> u32 {
    return u32(fmap * i32(CONV2_SIZE * CONV2_SIZE) + y * i32(CONV2_SIZE) + x);
}

fn sample_canvas(pos: vec2<i32>) -> f32 {
    if (pos.x < 0 || pos.x >= i32(INPUT_SIZE) || pos.y < 0 || pos.y >= i32(INPUT_SIZE)) {
        return 0.0;
    }
    return net[O_CAN + u32(canvas_index(pos.x, pos.y))];
}

fn sample_conv1(pos: vec2<i32>, fmap: i32) -> f32 {
    if (pos.x < 0 || pos.x >= i32(CONV1_SIZE) || pos.y < 0 || pos.y >= i32(CONV1_SIZE) || fmap < 0 || fmap >= i32(FEATURE_MAPS_1)) {
        return 0.0;
    }
    return net[O_C1 + u32(conv1_index(pos.x, pos.y, fmap))];
}

fn sample_conv2(pos: vec2<i32>, fmap: i32) -> f32 {
    if (pos.x < 0 || pos.x >= i32(CONV2_SIZE) || pos.y < 0 || pos.y >= i32(CONV2_SIZE) || fmap < 0 || fmap >= i32(FEATURE_MAPS_2)) {
        return 0.0;
    }
    return net[O_C2 + u32(conv2_index(pos.x, pos.y, fmap))];
}

// ---- live training on distorted glyphs from the font atlas
const W1: u32 = 0u;
const B1: u32 = 400u;
const W2: u32 = 416u;
const B2: u32 = 13216u;
const WF: u32 = 13248u;
const BF: u32 = 14752u;
const NP: u32 = 14799u;
const PS: u32 = 14800u;
const O_W: u32 = 0u;
const O_M: u32 = 14800u;
const O_V: u32 = 29600u;
const O_G: u32 = 44400u;
const O_GV: u32 = 59200u;
const O_BATCH: u32 = 74000u;
const O_LAB: u32 = 124176u;
const O_A1: u32 = 124240u;
const O_R1: u32 = 271696u;
const O_A2: u32 = 419152u;
const O_R2: u32 = 451920u;
const O_GAP: u32 = 484688u;
const O_DL: u32 = 486736u;
const O_DG: u32 = 489744u;
const O_DA1: u32 = 491792u;
// dreams: images, their input gradient, conv2 pool, d(gap), p(class), gradient rms
const O_DRX: u32 = 639248u;
const O_DRG: u32 = 676096u;
const O_DRA2: u32 = 712944u;
const O_DRR2: u32 = 737008u;
const O_DRDG: u32 = 761072u;
const O_DRP: u32 = 762576u;
const O_DRN: u32 = 762624u;
const O_CONF: u32 = 778512u;
const O_ST: u32 = 780816u;
const O_HIST: u32 = 782864u;
const O_REP: u32 = 783888u;
const O_REPL: u32 = 834064u;
const O_CAN: u32 = 834128u;
const O_C1: u32 = 834912u;
const O_C2: u32 = 837216u;
const O_FC: u32 = 837728u;
const O_DRA1: u32 = 837856u;
const O_DRR1: u32 = 946144u;
const O_DRD1: u32 = 1054432u;
// stats slots; S_GM: mean |grad| of conv1, conv2, fc, gap; S_BATCH: label, pred, loss, correct per sample
const S_STEP: u32 = 0u;
const S_LOSS: u32 = 1u;
const S_ACC: u32 = 2u;
const S_SHOW: u32 = 7u;
const S_GM: u32 = 8u;
const S_DG: u32 = 16u;
const S_BATCH: u32 = 400u;
const NMEAN: f32 = 0.175;
const NSTD: f32 = 0.33;

fn st(i: u32) -> f32 { return net[O_ST + i]; }
fn get_conv1_weight(fmap: i32, idx: i32) -> f32 { return net[O_W + W1 + u32(fmap * 25 + idx)]; }
fn get_conv1_bias(fmap: i32) -> f32 { return net[O_W + B1 + u32(fmap)]; }
fn get_conv2_weight(o: i32, i: i32, idx: i32) -> f32 { return net[O_W + W2 + u32((o * 16 + i) * 25 + idx)]; }
fn get_conv2_bias(o: i32) -> f32 { return net[O_W + B2 + u32(o)]; }
fn get_fc_weight(c: i32, k: i32) -> f32 { return net[O_W + WF + u32(c * 32 + k)]; }
fn get_fc_bias(c: i32) -> f32 { return net[O_W + BF + u32(c)]; }

fn thash(a0: u32) -> u32 { var a = a0; a ^= a >> 16u; a *= 0x7feb352du; a ^= a >> 15u; a *= 0x846ca68bu; a ^= a >> 16u; return a; }
fn trnd(s: ptr<function, u32>) -> f32 { *s = thash(*s); return f32(*s >> 8u) / 16777216.0; }
fn gauss(s: ptr<function, u32>) -> f32 { let u = max(trnd(s), 1e-7); return sqrt(-2.0 * log(u)) * cos(6.2832 * trnd(s)); }

@compute @workgroup_size(64, 1, 1)
fn train_init(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (params.reset == 0u) { return; }
    let i = gid.x;
    if (i < PS) {
        var s = thash(i * 747796405u + params.seed * 2891336453u + 1u);
        var v = 0.0;
        // He initialisation
        if (i < B1) { v = gauss(&s) * sqrt(2.0 / 25.0); }
        else if (i >= W2 && i < B2) { v = gauss(&s) * sqrt(2.0 / 400.0); }
        else if (i >= WF && i < BF) { v = gauss(&s) * sqrt(1.0 / 32.0); }
        net[O_W + i] = v;
        net[O_M + i] = 0.0;
        net[O_V + i] = 0.0;
        net[O_G + i] = 0.0;
        net[O_GV + i] = 0.0;
    }
    if (i < 36848u) { var s = thash(i * 2654435761u + params.seed); net[O_DRX + i] = 0.15 * trnd(&s); }
    if (i < 2304u) { net[O_CONF + i] = 0.0; }
    if (i < 2048u) { net[O_ST + i] = 0.0; }
    if (i < 1024u) { net[O_HIST + i] = 0.0; }
}

@compute @workgroup_size(64, 1, 1)
fn teach(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (params.teach < 0 || i >= 784u) { return; }
    let x = i % 28u;
    let y = i / 28u;
    net[O_REP + params.teach_slot * 784u + i] = net[O_CAN + (27u - y) * 28u + x];
    if (i == 0u) { net[O_REPL + params.teach_slot] = f32(params.teach); }
}

fn atlas(code: u32, uv: v2) -> f32 {
    if (any(uv < v2(0.0)) || any(uv >= v2(1.0))) { return 0.0; }
    let a = (v2(f32(code % 16u), f32(code / 16u)) + 0.05 + uv * 0.9) / 16.0 * font_texture_uniform.atlas_size - 0.5;
    let i = clamp(vec2<i32>(floor(a)), vec2<i32>(0), vec2<i32>(font_texture_uniform.atlas_size) - 2);
    let f = fract(a);
    let s00 = textureLoad(t_font_texture_atlas, i, 0).r;
    let s10 = textureLoad(t_font_texture_atlas, i + vec2<i32>(1, 0), 0).r;
    let s01 = textureLoad(t_font_texture_atlas, i + vec2<i32>(0, 1), 0).r;
    let s11 = textureLoad(t_font_texture_atlas, i + vec2<i32>(1, 1), 0).r;
    return mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y);
}

// one distorted glyph (or a taught drawing) per batch slot, network orientation (y down)
fn gen(gid: u32, r: u32) {
    let b = gid / 784u;
    let px = gid % 784u;
    if (r >= params.steps || b >= params.batch) { return; }
    let x = f32(px % 28u);
    let y = f32(px / 28u);
    var s = thash((params.step_base + r) * 9781u + b * 6271u + params.seed * 7919u + 17u);
    var v = 0.0;
    var label = 0u;
    if (params.rep_n > 0u && trnd(&s) < 0.35) {
        let slot = min(u32(trnd(&s) * f32(params.rep_n)), params.rep_n - 1u);
        label = u32(net[O_REPL + slot]);
        let d = vec2<i32>(i32(trnd(&s) * 3.0) - 1, i32(trnd(&s) * 3.0) - 1);
        let q = vec2<i32>(i32(x), i32(y)) - d;
        if (all(q >= vec2<i32>(0)) && all(q < vec2<i32>(28))) { v = net[O_REP + slot * 784u + u32(q.y * 28 + q.x)]; }
    } else {
        label = min(u32(trnd(&s) * 47.0), 46u);
        let a = params.aug;
        let rot = (trnd(&s) - 0.5) * 0.5 * a;
        let shear = (trnd(&s) - 0.5) * 0.6 * a;
        let sc = 1.0 + (trnd(&s) - 0.5) * 0.3 * a;
        let t = (v2(trnd(&s), trnd(&s)) - 0.5) * 0.14 * a;
        let thick = 0.2 + 0.4 * trnd(&s);
        let ph = v2(trnd(&s), trnd(&s)) * 6.2832;
        let el = 0.04 * a * trnd(&s);
        let code = class_char(i32(label));
            for (var k = 0; k < 4; k++) {
            var q = (v2(x, y) + 0.5 + (v2(f32(k & 1), f32(k >> 1u)) - 0.5) * 0.66) / 28.0 - 0.5;
            q += el * v2(sin(q.y * 9.0 + ph.x), sin(q.x * 9.0 + ph.y));
            q -= t;
            q = v2(cos(rot) * q.x + sin(rot) * q.y, -sin(rot) * q.x + cos(rot) * q.y);
            q.x -= shear * q.y;
            q /= sc;
            v += smoothstep(thick - 0.15, thick + 0.15, atlas(code, q + 0.5)) * 0.25;
        }
    }
    net[O_BATCH + b * 784u + px] = v;
    if (px == 0u) { net[O_LAB + b] = f32(label); }
}

fn inval(b: u32, x: u32, y: u32) -> f32 { return (net[O_BATCH + b * 784u + y * 28u + x] - NMEAN) / NSTD; }

// conv1 + ReLU + 2x2 max pool over a raw 28x28 image at src; (value, which of the four won)
fn pool1(src: u32, i: u32, x: u32, y: u32) -> v2 {
    var mx = -1.0;
    var arg = 0u;
    for (var d = 0u; d < 4u; d++) {
        let cx = 2u * x + (d & 1u);
        let cy = 2u * y + (d >> 1u);
        var sum = net[O_W + B1 + i];
        for (var k = 0u; k < 25u; k++) { sum += net[O_W + W1 + i * 25u + k] * (net[src + (cy + k / 5u) * 28u + cx + k % 5u] - NMEAN) / NSTD; }
        let a = max(sum, 0.0);
        if (a > mx) { mx = a; arg = d; }
    }
    return v2(mx, f32(arg));
}
// conv2 over the 16 pooled maps at src
fn pool2(src: u32, j: u32, x: u32, y: u32) -> v2 {
    var mx = -1.0;
    var arg = 0u;
    for (var d = 0u; d < 4u; d++) {
        let cx = 2u * x + (d & 1u);
        let cy = 2u * y + (d >> 1u);
        var sum = net[O_W + B2 + j];
        for (var i = 0u; i < 16u; i++) {
            let wb = O_W + W2 + (j * 16u + i) * 25u;
            let ab = src + i * 144u;
            for (var k = 0u; k < 25u; k++) { sum += net[wb + k] * net[ab + (cy + k / 5u) * 12u + cx + k % 5u]; }
        }
        let a = max(sum, 0.0);
        if (a > mx) { mx = a; arg = d; }
    }
    return v2(mx, f32(arg));
}

fn fwd1(gid: u32, r: u32) {
    let b = gid / 2304u;
    if (r >= params.steps || b >= params.batch) { return; }
    let rem = gid % 2304u;
    let o = pool1(O_BATCH + b * 784u, rem / 144u, rem % 12u, (rem % 144u) / 12u);
    net[O_A1 + gid] = o.x;
    net[O_R1 + gid] = o.y;
}

fn fwd2(gid: u32, r: u32) {
    let b = gid / 512u;
    if (r >= params.steps || b >= params.batch) { return; }
    let rem = gid % 512u;
    let o = pool2(O_A1 + b * 2304u, rem / 16u, rem % 4u, (rem % 16u) / 4u);
    net[O_A2 + gid] = o.x;
    net[O_R2 + gid] = o.y;
}

// pool, logits, softmax cross-entropy, dL/dlogits
fn head(gid: u32, r: u32) {
    let b = gid;
    if (r >= params.steps || b >= params.batch) { return; }
    var g: array<f32, 32>;
    for (var k = 0u; k < 32u; k++) {
        var s = 0.0;
        for (var q = 0u; q < 16u; q++) { s += net[O_A2 + b * 512u + k * 16u + q]; }
        g[k] = s / 16.0;
        net[O_GAP + b * 32u + k] = g[k];
    }
    var lg: array<f32, 47>;
    var mx = -1e9;
    var pred = 0u;
    for (var c = 0u; c < 47u; c++) {
        var s = net[O_W + BF + c];
        for (var k = 0u; k < 32u; k++) { s += net[O_W + WF + c * 32u + k] * g[k]; }
        lg[c] = s;
        if (s > mx) { mx = s; pred = c; }
    }
    var sum = 0.0;
    for (var c = 0u; c < 47u; c++) { sum += exp(lg[c] - mx); }
    let label = u32(net[O_LAB + b]);
    for (var c = 0u; c < 47u; c++) {
        let pr = exp(lg[c] - mx) / sum;
        net[O_DL + b * 47u + c] = (pr - select(0.0, 1.0, c == label)) / f32(params.batch);
    }
    let loss = -(lg[label] - mx - log(sum));
    net[O_ST + S_BATCH + b * 4u] = f32(label);
    net[O_ST + S_BATCH + b * 4u + 1u] = f32(pred);
    net[O_ST + S_BATCH + b * 4u + 2u] = loss;
    net[O_ST + S_BATCH + b * 4u + 3u] = select(0.0, 1.0, pred == label);
}

// fully connected weights and biases
fn bfc(gid: u32, r: u32) {
    if (r >= params.steps || gid >= 1551u) { return; }
    var g = 0.0;
    if (gid < 1504u) {
        let c = gid / 32u;
        let k = gid % 32u;
        for (var b = 0u; b < params.batch; b++) { g += net[O_DL + b * 47u + c] * net[O_GAP + b * 32u + k]; }
    } else {
        let c = gid - 1504u;
        for (var b = 0u; b < params.batch; b++) { g += net[O_DL + b * 47u + c]; }
    }
    net[O_G + WF + gid] = g;
}

// dL/d(average pooled feature)
fn bgap(gid: u32, r: u32) {
    let b = gid / 32u;
    if (r >= params.steps || b >= params.batch) { return; }
    let k = gid % 32u;
    var g = 0.0;
    for (var c = 0u; c < 47u; c++) { g += net[O_W + WF + c * 32u + k] * net[O_DL + b * 47u + c]; }
    net[O_DG + gid] = g;
}

// dL/d(conv2 pre-activation) at a pool unit's winner; a2, dg: one sample's pooled conv2 and d(gap)
fn dpre(a2: u32, dg: u32, j: u32, q: u32) -> f32 { return select(0.0, net[dg + j] / 16.0, net[a2 + j * 16u + q] > 0.0); }
fn dpre2(b: u32, j: u32, q: u32) -> f32 { return dpre(O_A2 + b * 512u, O_DG + b * 32u, j, q); }
// dL/d(conv1 pooled output) of map i at row u, column v
fn dpool1(a2: u32, r2: u32, dg: u32, i: u32, u: i32, v: i32) -> f32 {
    var g = 0.0;
    for (var j = 0u; j < 32u; j++) {
        for (var q = 0u; q < 16u; q++) {
            let dp = dpre(a2, dg, j, q);
            if (dp == 0.0) { continue; }
            let arg = u32(net[r2 + j * 16u + q]);
            let ky = u - i32(2u * (q / 4u) + (arg >> 1u));
            let kx = v - i32(2u * (q % 4u) + (arg & 1u));
            if (ky < 0 || ky > 4 || kx < 0 || kx > 4) { continue; }
            g += dp * net[O_W + W2 + (j * 16u + i) * 25u + u32(ky * 5 + kx)];
        }
    }
    return g;
}

fn bw2(gid: u32, r: u32) {
    if (r >= params.steps || gid >= 12832u) { return; }
    var g = 0.0;
    if (gid < 12800u) {
        let j = gid / 400u;
        let i = (gid % 400u) / 25u;
        let kk = gid % 25u;
        for (var b = 0u; b < params.batch; b++) {
            for (var q = 0u; q < 16u; q++) {
                let dp = dpre2(b, j, q);
                if (dp == 0.0) { continue; }
                let arg = u32(net[O_R2 + b * 512u + j * 16u + q]);
                let cx = 2u * (q % 4u) + (arg & 1u) + kk % 5u;
                let cy = 2u * (q / 4u) + (arg >> 1u) + kk / 5u;
                g += dp * net[O_A1 + b * 2304u + i * 144u + cy * 12u + cx];
            }
        }
        net[O_G + W2 + gid] = g;
    } else {
        let j = gid - 12800u;
        for (var b = 0u; b < params.batch; b++) { for (var q = 0u; q < 16u; q++) { g += dpre2(b, j, q); } }
        net[O_G + B2 + j] = g;
    }
}

fn bda1(gid: u32, r: u32) {
    let b = gid / 2304u;
    if (r >= params.steps || b >= params.batch) { return; }
    let rem = gid % 2304u;
    net[O_DA1 + gid] = dpool1(O_A2 + b * 512u, O_R2 + b * 512u, O_DG + b * 32u, rem / 144u, i32((rem % 144u) / 12u), i32(rem % 12u));
}

fn bw1(gid: u32, r: u32) {
    if (r >= params.steps || gid >= 416u) { return; }
    var g = 0.0;
    let i = select(gid / 25u, gid - 400u, gid >= 400u);
    let kk = gid % 25u;
    for (var b = 0u; b < params.batch; b++) {
        for (var q = 0u; q < 144u; q++) {
            let o = b * 2304u + i * 144u + q;
            if (net[O_A1 + o] <= 0.0) { continue; }
            let dp = net[O_DA1 + o];
            if (gid >= 400u) { g += dp; continue; }
            let arg = u32(net[O_R1 + o]);
            let cx = 2u * (q % 12u) + (arg & 1u) + kk % 5u;
            let cy = 2u * (q / 12u) + (arg >> 1u) + kk / 5u;
            g += dp * inval(b, cx, cy);
        }
    }
    net[O_G + W1 + gid] = g;
}

// Adam; |grad| averaged for the fibers
fn adam(gid: u32, r: u32) {
    if (r >= params.steps || gid >= NP) { return; }
    let g = clamp(net[O_G + gid], -5.0, 5.0);
    let t = f32(params.step_base + r + 1u);
    let m = 0.9 * net[O_M + gid] + 0.1 * g;
    let v = 0.999 * net[O_V + gid] + 0.001 * g * g;
    net[O_M + gid] = m;
    net[O_V + gid] = v;
    let mh = m / (1.0 - pow(0.9, t));
    let vh = v / (1.0 - pow(0.999, t));
    net[O_W + gid] -= params.lr * mh / (sqrt(vh) + 1e-8);
    net[O_GV + gid] = mix(net[O_GV + gid], abs(g), 0.05);
}

// confusion rows, one per thread; thread 0: loss, accuracy, history
fn tstat(gid: u32, r: u32) {
    if (r >= params.steps || gid >= 47u) { return; }
    let row = gid;
    for (var c = 0u; c < 47u; c++) { net[O_CONF + row * 47u + c] *= 0.997; }
    var ls = 0.0;
    var ac = 0.0;
    for (var b = 0u; b < params.batch; b++) {
        let o = O_ST + S_BATCH + b * 4u;
        if (u32(net[o]) == row) { net[O_CONF + row * 47u + u32(net[o + 1u])] += 1.0; }
        ls += net[o + 2u];
        ac += net[o + 3u];
    }
    if (row != 0u) { return; }
    let step = params.step_base + r;
    let nb = f32(params.batch);
    let k = select(0.02, 1.0, step == 0u);
    net[O_ST + S_LOSS] = mix(st(S_LOSS), ls / nb, k);
    net[O_ST + S_ACC] = mix(st(S_ACC), ac / nb, k);
    net[O_ST + S_STEP] = f32(step + 1u);
    if (step % 16u == 0u) {
        let h = (step / 16u) % 256u;
        net[O_HIST + h * 2u] = st(S_LOSS);
        net[O_HIST + h * 2u + 1u] = st(S_ACC);
    }
}

@compute @workgroup_size(1, 1, 1)
fn gstat(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (params.steps == 0u) { return; }
    var s1 = 0.0;
    for (var i = 0u; i < 400u; i++) { s1 += net[O_GV + W1 + i]; }
    var s2 = 0.0;
    for (var i = 0u; i < 12800u; i++) { s2 += net[O_GV + W2 + i]; }
    var s3 = 0.0;
    for (var i = 0u; i < 1504u; i++) { s3 += net[O_GV + WF + i]; }
    net[O_ST + S_GM] = s1 / 400.0;
    net[O_ST + S_GM + 1u] = s2 / 12800.0;
    net[O_ST + S_GM + 2u] = s3 / 1504.0;
    var sg = 0.0;
    for (var k = 0u; k < 32u; k++) {
        var s = 0.0;
        for (var b = 0u; b < params.batch; b++) { s += abs(net[O_DG + b * 32u + k]); }
        net[O_ST + S_DG + k] = mix(st(S_DG + k), s / f32(params.batch), 0.1);
        sg += st(S_DG + k);
    }
    net[O_ST + S_GM + 3u] = sg / 32.0;
}

// ---- dreams: each class's image climbs its own log-probability through the live weights (activation maximisation)
// shifted a little every frame so it can't exploit exact pixel positions
fn djit(c: u32) -> vec2<i32> { let h = thash(time_data.frame * 47u + c + 991u); return vec2<i32>(i32(h % 5u), i32((h >> 8u) % 5u)) - 2; }

@compute @workgroup_size(64, 1, 1)
fn d_in(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x / 784u;
    if (params.dream <= 0.0 || c >= 47u) { return; }
    let q = vec2<i32>(i32(gid.x % 28u), i32((gid.x % 784u) / 28u)) + djit(c);
    var v = 0.0;
    if (all(q >= vec2<i32>(0)) && all(q < vec2<i32>(28))) { v = net[O_DRX + c * 784u + u32(q.y * 28 + q.x)]; }
    net[O_DRG + gid.x] = v;
}
@compute @workgroup_size(64, 1, 1)
fn d_fwd1(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x / 2304u;
    if (params.dream <= 0.0 || c >= 47u) { return; }
    let rem = gid.x % 2304u;
    let o = pool1(O_DRG + c * 784u, rem / 144u, rem % 12u, (rem % 144u) / 12u);
    net[O_DRA1 + gid.x] = o.x;
    net[O_DRR1 + gid.x] = o.y;
}
@compute @workgroup_size(64, 1, 1)
fn d_fwd2(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x / 512u;
    if (params.dream <= 0.0 || c >= 47u) { return; }
    let rem = gid.x % 512u;
    let o = pool2(O_DRA1 + c * 2304u, rem / 16u, rem % 4u, (rem % 16u) / 4u);
    net[O_DRA2 + gid.x] = o.x;
    net[O_DRR2 + gid.x] = o.y;
}
// d log p(c) / d gap = WF[c] - sum_k p_k WF[k]
@compute @workgroup_size(64, 1, 1)
fn d_head(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (params.dream <= 0.0 || c >= 47u) { return; }
    var g: array<f32, 32>;
    for (var k = 0u; k < 32u; k++) {
        var s = 0.0;
        for (var q = 0u; q < 16u; q++) { s += net[O_DRA2 + c * 512u + k * 16u + q]; }
        g[k] = s / 16.0;
    }
    var lg: array<f32, 47>;
    var mx = -1e9;
    for (var k = 0u; k < 47u; k++) {
        var s = net[O_W + BF + k];
        for (var m = 0u; m < 32u; m++) { s += net[O_W + WF + k * 32u + m] * g[m]; }
        lg[k] = s;
        mx = max(mx, s);
    }
    var sum = 0.0;
    for (var k = 0u; k < 47u; k++) { lg[k] = exp(lg[k] - mx); sum += lg[k]; }
    for (var m = 0u; m < 32u; m++) {
        var d = net[O_W + WF + c * 32u + m];
        for (var k = 0u; k < 47u; k++) { d -= lg[k] / sum * net[O_W + WF + k * 32u + m]; }
        net[O_DRDG + c * 32u + m] = d;
    }
    net[O_DRP + c] = lg[c] / sum;
}
@compute @workgroup_size(64, 1, 1)
fn d_bda1(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x / 2304u;
    if (params.dream <= 0.0 || c >= 47u) { return; }
    let rem = gid.x % 2304u;
    net[O_DRD1 + gid.x] = dpool1(O_DRA2 + c * 512u, O_DRR2 + c * 512u, O_DRDG + c * 32u, rem / 144u, i32((rem % 144u) / 12u), i32(rem % 12u));
}
// down to the input pixel through each winning conv1 position that covers it
@compute @workgroup_size(64, 1, 1)
fn d_bin(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x / 784u;
    if (params.dream <= 0.0 || c >= 47u) { return; }
    let x = i32(gid.x % 28u);
    let y = i32((gid.x % 784u) / 28u);
    var g = 0.0;
    for (var i = 0u; i < 16u; i++) {
        for (var d = 0; d < 9; d++) {
            let pu = y / 2 - d / 3;
            let pv = x / 2 - d % 3;
            if (pu < 0 || pv < 0 || pu > 11 || pv > 11) { continue; }
            let o = c * 2304u + i * 144u + u32(pu * 12 + pv);
            if (net[O_DRA1 + o] <= 0.0) { continue; }
            let arg = u32(net[O_DRR1 + o]);
            let ky = y - (2 * pu + i32(arg >> 1u));
            let kx = x - (2 * pv + i32(arg & 1u));
            if (ky < 0 || ky > 4 || kx < 0 || kx > 4) { continue; }
            g += net[O_DRD1 + o] * net[O_W + W1 + i * 25u + u32(ky * 5 + kx)];
        }
    }
    net[O_DRG + gid.x] = g / NSTD;
}
@compute @workgroup_size(64, 1, 1)
fn d_norm(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (params.dream <= 0.0 || c >= 47u) { return; }
    var s = 0.0;
    for (var i = 0u; i < 784u; i++) { let g = net[O_DRG + c * 784u + i]; s += g * g; }
    net[O_DRN + c] = sqrt(s / 784.0) + 1e-8;
}
// normalised ascent step, staged in the now free d(conv1 pool) slot
@compute @workgroup_size(64, 1, 1)
fn d_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x / 784u;
    if (params.dream <= 0.0 || c >= 47u) { return; }
    let p = vec2<i32>(i32(gid.x % 28u), i32((gid.x % 784u) / 28u)) - djit(c);
    var g = 0.0;
    if (all(p >= vec2<i32>(0)) && all(p < vec2<i32>(28))) { g = net[O_DRG + c * 784u + u32(p.y * 28 + p.x)]; }
    net[O_DRD1 + gid.x] = net[O_DRX + gid.x] + 0.05 * g / net[O_DRN + c];
}
// blur and decay keep it a drawing instead of adversarial noise
@compute @workgroup_size(64, 1, 1)
fn d_blur(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x / 784u;
    if (params.dream <= 0.0 || c >= 47u) { return; }
    let x = i32(gid.x % 28u);
    let y = i32((gid.x % 784u) / 28u);
    var b = 0.0;
    var w = 0.0;
    for (var k = 0; k < 9; k++) {
        let q = vec2<i32>(x + k % 3 - 1, y + k / 3 - 1);
        if (any(q < vec2<i32>(0)) || any(q > vec2<i32>(27))) { continue; }
        let kw = select(1.0, 2.0, k % 2 == 1) * select(1.0, 4.0, k == 4);
        b += net[O_DRD1 + c * 784u + u32(q.y * 28 + q.x)] * kw;
        w += kw;
    }
    net[O_DRX + gid.x] = clamp(mix(net[O_DRD1 + gid.x], b / w, 0.15) * 0.992, 0.0, 1.0);
}

// up to four training steps a frame
@compute @workgroup_size(64, 1, 1) fn gen_0(@builtin(global_invocation_id) g: vec3<u32>) { gen(g.x, 0u); }
@compute @workgroup_size(64, 1, 1) fn fwd1_0(@builtin(global_invocation_id) g: vec3<u32>) { fwd1(g.x, 0u); }
@compute @workgroup_size(64, 1, 1) fn fwd2_0(@builtin(global_invocation_id) g: vec3<u32>) { fwd2(g.x, 0u); }
@compute @workgroup_size(64, 1, 1) fn head_0(@builtin(global_invocation_id) g: vec3<u32>) { head(g.x, 0u); }
@compute @workgroup_size(64, 1, 1) fn bfc_0(@builtin(global_invocation_id) g: vec3<u32>) { bfc(g.x, 0u); }
@compute @workgroup_size(64, 1, 1) fn bgap_0(@builtin(global_invocation_id) g: vec3<u32>) { bgap(g.x, 0u); }
@compute @workgroup_size(64, 1, 1) fn bw2_0(@builtin(global_invocation_id) g: vec3<u32>) { bw2(g.x, 0u); }
@compute @workgroup_size(64, 1, 1) fn bda1_0(@builtin(global_invocation_id) g: vec3<u32>) { bda1(g.x, 0u); }
@compute @workgroup_size(64, 1, 1) fn bw1_0(@builtin(global_invocation_id) g: vec3<u32>) { bw1(g.x, 0u); }
@compute @workgroup_size(64, 1, 1) fn adam_0(@builtin(global_invocation_id) g: vec3<u32>) { adam(g.x, 0u); }
@compute @workgroup_size(64, 1, 1) fn tstat_0(@builtin(global_invocation_id) g: vec3<u32>) { tstat(g.x, 0u); }
@compute @workgroup_size(64, 1, 1) fn gen_1(@builtin(global_invocation_id) g: vec3<u32>) { gen(g.x, 1u); }
@compute @workgroup_size(64, 1, 1) fn fwd1_1(@builtin(global_invocation_id) g: vec3<u32>) { fwd1(g.x, 1u); }
@compute @workgroup_size(64, 1, 1) fn fwd2_1(@builtin(global_invocation_id) g: vec3<u32>) { fwd2(g.x, 1u); }
@compute @workgroup_size(64, 1, 1) fn head_1(@builtin(global_invocation_id) g: vec3<u32>) { head(g.x, 1u); }
@compute @workgroup_size(64, 1, 1) fn bfc_1(@builtin(global_invocation_id) g: vec3<u32>) { bfc(g.x, 1u); }
@compute @workgroup_size(64, 1, 1) fn bgap_1(@builtin(global_invocation_id) g: vec3<u32>) { bgap(g.x, 1u); }
@compute @workgroup_size(64, 1, 1) fn bw2_1(@builtin(global_invocation_id) g: vec3<u32>) { bw2(g.x, 1u); }
@compute @workgroup_size(64, 1, 1) fn bda1_1(@builtin(global_invocation_id) g: vec3<u32>) { bda1(g.x, 1u); }
@compute @workgroup_size(64, 1, 1) fn bw1_1(@builtin(global_invocation_id) g: vec3<u32>) { bw1(g.x, 1u); }
@compute @workgroup_size(64, 1, 1) fn adam_1(@builtin(global_invocation_id) g: vec3<u32>) { adam(g.x, 1u); }
@compute @workgroup_size(64, 1, 1) fn tstat_1(@builtin(global_invocation_id) g: vec3<u32>) { tstat(g.x, 1u); }
@compute @workgroup_size(64, 1, 1) fn gen_2(@builtin(global_invocation_id) g: vec3<u32>) { gen(g.x, 2u); }
@compute @workgroup_size(64, 1, 1) fn fwd1_2(@builtin(global_invocation_id) g: vec3<u32>) { fwd1(g.x, 2u); }
@compute @workgroup_size(64, 1, 1) fn fwd2_2(@builtin(global_invocation_id) g: vec3<u32>) { fwd2(g.x, 2u); }
@compute @workgroup_size(64, 1, 1) fn head_2(@builtin(global_invocation_id) g: vec3<u32>) { head(g.x, 2u); }
@compute @workgroup_size(64, 1, 1) fn bfc_2(@builtin(global_invocation_id) g: vec3<u32>) { bfc(g.x, 2u); }
@compute @workgroup_size(64, 1, 1) fn bgap_2(@builtin(global_invocation_id) g: vec3<u32>) { bgap(g.x, 2u); }
@compute @workgroup_size(64, 1, 1) fn bw2_2(@builtin(global_invocation_id) g: vec3<u32>) { bw2(g.x, 2u); }
@compute @workgroup_size(64, 1, 1) fn bda1_2(@builtin(global_invocation_id) g: vec3<u32>) { bda1(g.x, 2u); }
@compute @workgroup_size(64, 1, 1) fn bw1_2(@builtin(global_invocation_id) g: vec3<u32>) { bw1(g.x, 2u); }
@compute @workgroup_size(64, 1, 1) fn adam_2(@builtin(global_invocation_id) g: vec3<u32>) { adam(g.x, 2u); }
@compute @workgroup_size(64, 1, 1) fn tstat_2(@builtin(global_invocation_id) g: vec3<u32>) { tstat(g.x, 2u); }
@compute @workgroup_size(64, 1, 1) fn gen_3(@builtin(global_invocation_id) g: vec3<u32>) { gen(g.x, 3u); }
@compute @workgroup_size(64, 1, 1) fn fwd1_3(@builtin(global_invocation_id) g: vec3<u32>) { fwd1(g.x, 3u); }
@compute @workgroup_size(64, 1, 1) fn fwd2_3(@builtin(global_invocation_id) g: vec3<u32>) { fwd2(g.x, 3u); }
@compute @workgroup_size(64, 1, 1) fn head_3(@builtin(global_invocation_id) g: vec3<u32>) { head(g.x, 3u); }
@compute @workgroup_size(64, 1, 1) fn bfc_3(@builtin(global_invocation_id) g: vec3<u32>) { bfc(g.x, 3u); }
@compute @workgroup_size(64, 1, 1) fn bgap_3(@builtin(global_invocation_id) g: vec3<u32>) { bgap(g.x, 3u); }
@compute @workgroup_size(64, 1, 1) fn bw2_3(@builtin(global_invocation_id) g: vec3<u32>) { bw2(g.x, 3u); }
@compute @workgroup_size(64, 1, 1) fn bda1_3(@builtin(global_invocation_id) g: vec3<u32>) { bda1(g.x, 3u); }
@compute @workgroup_size(64, 1, 1) fn bw1_3(@builtin(global_invocation_id) g: vec3<u32>) { bw1(g.x, 3u); }
@compute @workgroup_size(64, 1, 1) fn adam_3(@builtin(global_invocation_id) g: vec3<u32>) { adam(g.x, 3u); }
@compute @workgroup_size(64, 1, 1) fn tstat_3(@builtin(global_invocation_id) g: vec3<u32>) { tstat(g.x, 3u); }

// right button clears
@compute @workgroup_size(1, 1, 1)
fn canvas_update(@builtin(global_invocation_id) id: vec3<u32>) {
    let pos = vec2<i32>(id.xy);
    if any(pos >= vec2(i32(INPUT_SIZE))) { return; }
    let idx = canvas_index(pos.x, pos.y);
    let btns = mouse.buttons.x;
    if params.mode == 0u {
        if params.steps > 0u {
            net[O_CAN + idx] = net[O_BATCH + u32(27 - pos.y) * 28u + u32(pos.x)];
            if all(pos == vec2(0)) { net[O_ST + S_SHOW] = net[O_LAB]; }
        }
        return;
    }
    if params.clear_canvas == 1 || (btns & 2u) != 0u {
        net[O_CAN + u32(idx)] = 0.;
        return;
    }
    if (btns & 1u) != 0u && params.drawing == 1 {
        let res = vec2<f32>(textureDimensions(output));
        let mp = mouse_pad(res);
        if mp.x >= 0.0 {
            let intensity = 1. - smoothstep(0., brush_r(), length(mp - (vec2<f32>(pos) + 0.5)));
            if intensity > 0. { net[O_CAN + u32(idx)] = min(1., net[O_CAN + u32(idx)] + intensity * 0.3); }
        }
    }
}

@compute @workgroup_size(1, 1, 1)
fn conv_layer1(@builtin(global_invocation_id) id: vec3<u32>) {
    let pos = vec3<i32>(id);
    if any(pos.xy >= vec2(i32(CONV1_SIZE))) || pos.z >= i32(FEATURE_MAPS_1) { return; }
    
    let pool_xy = pos.xy * 2;
    var max_val = -1000.;
    
    for (var d = 0; d < 4; d++) {
        let conv_xy = pool_xy + vec2(d & 1, d >> 1);
        var sum = get_conv1_bias(pos.z);
        
        for (var k = 0; k < 25; k++) {
            let kxy = vec2(k % 5, k / 5);
            let input_xy = conv_xy + kxy;
            
            let flipped_y = i32(INPUT_SIZE) - 1 - input_xy.y;
            let flipped_xy = vec2(input_xy.x, flipped_y);
            
            let input_val = normalize_input(sample_canvas(flipped_xy));
            sum += input_val * get_conv1_weight(pos.z, k);
        }
        
        max_val = max(max_val, max(0., sum));
    }
    
    net[O_C1 + u32(conv1_index(pos.x, pos.y, pos.z))] = max_val;
}

@compute @workgroup_size(1, 1, 1)
fn conv_layer2(@builtin(global_invocation_id) id: vec3<u32>) {
    let pos = vec3<i32>(id);
    if any(pos.xy >= vec2(i32(CONV2_SIZE))) || pos.z >= i32(FEATURE_MAPS_2) { return; }
    
    let pool_xy = pos.xy * 2;
    var max_val = -1000.;
    
    for (var d = 0; d < 4; d++) {
        let conv_xy = pool_xy + vec2(d & 1, d >> 1);
        var sum = get_conv2_bias(pos.z);
        
        for (var fm = 0; fm < i32(FEATURE_MAPS_1); fm++) {
            for (var k = 0; k < 25; k++) {
                let kxy = vec2(k % 5, k / 5);
                let input_xy = conv_xy + kxy;
                let input_val = sample_conv1(input_xy, fm);
                sum += input_val * get_conv2_weight(pos.z, fm, k);
            }
        }
        
        max_val = max(max_val, max(0., sum));
    }
    
    net[O_C2 + u32(conv2_index(pos.x, pos.y, pos.z))] = max_val;
}

@compute @workgroup_size(1, 1, 1)
fn fully_connected(@builtin(global_invocation_id) id: vec3<u32>) {
    let class_idx = i32(id.x);
    
    if class_idx >= 47 { return; }
    var sum = get_fc_bias(class_idx);
    
    
    for (var fmap = 0; fmap < 32; fmap++) {
        
        // Global Average Pooling
        var map_sum = 0.0;
        for (var y = 0; y < 4; y++) {
            for (var x = 0; x < 4; x++) {
                map_sum += net[O_C2 + u32(conv2_index(x, y, fmap))];
            }
        }
        let map_avg = map_sum / 16.0;
        
        // Note: get_fc_weight now takes the feature map index, not a pixel index
        sum += map_avg * get_fc_weight(class_idx, fmap);
    }
    
    net[O_FC + u32(class_idx)] = sum;
}
alias v2 = vec2<f32>; alias v3 = vec3<f32>; alias v4 = vec4<f32>;
const MAXPIX: u32 = 3840u * 2400u;
const FHP: u32 = MAXPIX / 4u + 4096u;
const FIX: f32 = 1024.0;
// fc_data after the 47 logits
const F_TOP: u32 = 48u;
const F_PROB: u32 = 51u;
const F_MAX: u32 = 54u;
const F_SUM: u32 = 55u;
const F_PICK: u32 = 56u;
const F_MOUSE: u32 = 60u;
const F_INK: u32 = 62u;
const F_CAM: u32 = 64u;
// fiber bundles: input -> L1, L1 -> L2, L2 -> averages, averages -> classes
const FIB1: u32 = 2304u;
const FIB2: u32 = 10496u;
const FIB3: u32 = 11008u;
const FIBN: u32 = 12512u;

// tiles of m x m neurons, one empty slot between tiles
struct Layer { org: v2, cell: f32, m: i32, cols: i32, rows: i32, n: i32, hs: f32 };
fn layer(l: i32) -> Layer {
    switch (l) {
        case 0: { return Layer(v2(0.0, -14.0), 1.0, 28, 1, 1, 1, 0.25); }
        case 1: { return Layer(v2(40.0, -14.0), 0.55, 12, 4, 4, 16, 3.0); }
        case 2: { return Layer(v2(80.0, -14.6), 0.75, 4, 4, 8, 32, 4.0); }
        case 3: { return Layer(v2(106.0, -12.0), 1.6, 1, 4, 8, 32, 6.0); }
        default: { return Layer(v2(129.0, -13.5), 1.8, 1, 6, 8, 47, 10.0); }
    }
}
fn layer_col(l: i32) -> v3 {
    var c = array<v3, 5>(v3(0.9, 0.95, 1.0), v3(1.0, 0.55, 0.15), v3(0.2, 0.8, 1.0), v3(0.75, 0.4, 1.0), v3(1.0, 0.8, 0.3));
    return c[clamp(l, 0, 4)];
}
fn slots(L: Layer) -> vec2<i32> { return vec2<i32>(L.cols, L.rows) * (L.m + 1) - 1; }
fn top_h(L: Layer) -> f32 { return L.cell * 0.3 + 1.5 * L.hs * params.height + 0.01; }

// (tile, x, y), tile < 0 in a gap
fn neuron(l: i32, s: vec2<i32>) -> vec3<i32> {
    let L = layer(l);
    let u = s % (L.m + 1);
    let t = s / (L.m + 1);
    let tile = t.y * L.cols + t.x;
    if (u.x == L.m || u.y == L.m || tile >= L.n) { return vec3<i32>(-1); }
    return vec3<i32>(tile, u.x, L.m - 1 - u.y);
}
fn prob(c: i32) -> f32 { return exp(net[O_FC + u32(c)] - net[O_FC + u32(F_MAX)]) / max(net[O_FC + u32(F_SUM)], 1e-6); }
fn gap_avg(k: i32) -> f32 {
    var s = 0.0;
    for (var i = 0; i < 16; i++) { s += net[O_C2 + u32(u32(k * 16 + i))]; }
    return s / 16.0;
}
// roughly 0..1
fn act(l: i32, nr: vec3<i32>) -> f32 {
    switch (l) {
        case 0: { return sample_canvas(vec2<i32>(nr.y, 27 - nr.z)); }
        case 1: { return sample_conv1(nr.yz, nr.x) / 3.0; }
        case 2: { return sample_conv2(nr.yz, nr.x) / 4.0; }
        case 3: { return gap_avg(nr.x) / 2.5; }
        default: { return prob(nr.x); }
    }
}
fn box_h(l: i32, a: f32) -> f32 { let L = layer(l); return L.cell * 0.3 + clamp(a, 0.0, 1.5) * L.hs * params.height; }
fn anchor(l: i32, t: i32, c: v2) -> v3 {
    let L = layer(l);
    let s = v2(vec2<i32>(t % L.cols, t / L.cols) * (L.m + 1)) + v2(c.x, f32(L.m) - c.y);
    let q = clamp(vec2<i32>(floor(c)), vec2<i32>(0), vec2<i32>(L.m - 1));
    return v3(L.org + s * L.cell, box_h(l, act(l, vec3<i32>(t, q))));
}

struct NHit { t: f32, n: v3, l: i32, s: vec2<i32> };

fn box_hit(ro: v3, rd: v3, lo: v3, hi: v3) -> v4 {
    let inv = 1.0 / select(rd, v3(1e-8), abs(rd) < v3(1e-8));
    let t1 = (lo - ro) * inv;
    let t2 = (hi - ro) * inv;
    let tn = min(t1, t2);
    let tf = max(t1, t2);
    let t0 = max(max(tn.x, tn.y), tn.z);
    let t3 = min(min(tf.x, tf.y), tf.z);
    if (t3 < t0 || t0 <= 0.0) { return v4(-1.0); }
    var n = v3(0.0, 0.0, -sign(rd.z));
    if (tn.x >= tn.y && tn.x >= tn.z) { n = v3(-sign(rd.x), 0.0, 0.0); } else if (tn.y >= tn.z) { n = v3(0.0, -sign(rd.y), 0.0); }
    return v4(t0, n);
}

// towers stay inside their slots, so the first hit is the nearest
fn trace_layer(ro: v3, rd: v3, l: i32, best: ptr<function, NHit>) {
    let L = layer(l);
    let ns = slots(L);
    let inv = 1.0 / select(rd, v3(1e-8), abs(rd) < v3(1e-8));
    let lo = v3(L.org, 0.0);
    let hi = v3(L.org + v2(ns) * L.cell, top_h(L));
    let tn = min((lo - ro) * inv, (hi - ro) * inv);
    let tf = max((lo - ro) * inv, (hi - ro) * inv);
    let te = max(max(max(tn.x, tn.y), tn.z), 0.0);
    let tx = min(min(tf.x, tf.y), tf.z);
    if (tx <= te || te > (*best).t) { return; }
    let q = ro.xy + rd.xy * (te + 1e-3);
    var c = clamp(vec2<i32>(floor((q - L.org) / L.cell)), vec2<i32>(0), ns - 1);
    let stp = select(vec2<i32>(-1), vec2<i32>(1), rd.xy > v2(0.0));
    var tm = (L.org + (v2(c) + select(v2(0.0), v2(1.0), rd.xy > v2(0.0))) * L.cell - ro.xy) * inv.xy;
    let dt = abs(L.cell * inv.xy);
    for (var i = 0; i < 160; i++) {
        if (any(c < vec2<i32>(0)) || any(c >= ns)) { break; }
        let nr = neuron(l, c);
        if (nr.x >= 0) {
            let blo = L.org + (v2(c) + 0.12) * L.cell;
            let b = box_hit(ro, rd, v3(blo, 0.0), v3(blo + 0.76 * L.cell, box_h(l, act(l, nr))));
            if (b.x > 0.0) {
                if (b.x < (*best).t) { *best = NHit(b.x, b.yzw, l, c); }
                return;
            }
        }
        if (min(tm.x, tm.y) > tx) { break; }
        if (tm.x < tm.y) { tm.x += dt.x; c.x += stp.x; } else { tm.y += dt.y; c.y += stp.y; }
    }
}
fn trace_net(ro: v3, rd: v3) -> NHit {
    var best = NHit(1e30, v3(0.0, 0.0, 1.0), -1, vec2<i32>(0));
    for (var l = 0; l < 5; l++) { trace_layer(ro, rd, l, &best); }
    return best;
}

struct Cam { ro: v3, f: v3, rt: v3, up: v3 };
fn cam() -> Cam {
    let aim = v3(74.0, 0.0, 4.0);
    let el = radians(clamp(params.elev, 5.0, 85.0));
    let yw = radians(params.yaw);
    let ro = aim + 150.0 / max(params.zoom, 0.1) * v3(sin(yw) * cos(el), -cos(yw) * cos(el), sin(el));
    let f = normalize(aim - ro);
    let rt = normalize(cross(f, v3(0.0, 0.0, 1.0)));
    return Cam(ro, f, rt, cross(rt, f));
}
fn cam_ray(c: Cam, px: v2, D: v2) -> v3 {
    let sp = (px - 0.5 * D) / D.y * 0.8;
    return normalize(c.f + sp.x * c.rt - sp.y * c.up);
}
// distance < 0 behind the camera
fn project(c: Cam, p: v3, D: v2) -> v3 {
    let d = p - c.ro;
    let z = dot(d, c.f);
    if (z <= 0.1) { return v3(-1.0); }
    return v3(v2(dot(d, c.rt), -dot(d, c.up)) / z * D.y / 0.8 + 0.5 * D, length(d));
}
// negative off the pad
fn pad_hit(px: v2, D: v2) -> v2 {
    let c = cam();
    let rd = cam_ray(c, px, D);
    let L = layer(0);
    if (rd.z >= 0.0) { return v2(-1.0); }
    let q = (c.ro.xy + rd.xy * ((L.cell * 0.3 - c.ro.z) / rd.z) - L.org) / L.cell;
    if (any(q < v2(0.0)) || any(q >= v2(28.0))) { return v2(-1.0); }
    return q;
}
// xy top-left, z side
fn pad2d_rect(D: v2) -> v3 { let s = floor(0.3 * D.y); return v3(16.0, D.y - s - 16.0, s); }
// row 0 at the bottom like the canvas; negative off the pad
fn pad2d(px: v2, D: v2) -> v2 {
    let r = pad2d_rect(D);
    let q = (px - r.xy) / r.z * 28.0;
    if (any(q < v2(0.0)) || any(q >= v2(28.0))) { return v2(-1.0); }
    return v2(q.x, 28.0 - q.y);
}
fn mouse_pad(D: v2) -> v2 {
    let m = pad2d(mouse.position * D, D);
    if (m.x >= 0.0) { return m; }
    return pad_hit(mouse.position * D, D);
}
fn brush_r() -> f32 { return params.brush_size * 280.0; }

// the last conv cells see 16 x 16 input pixels, stride 4
fn cam_at(p: v2) -> f32 {
    let g = clamp((p + 0.5 - 7.5) / 4.0, v2(0.0), v2(3.0));
    let i = vec2<i32>(min(floor(g), v2(2.0)));
    let f = g - v2(i);
    let a = net[O_FC + u32(F_CAM + u32(i.y * 4 + i.x))];
    let b = net[O_FC + u32(F_CAM + u32(i.y * 4 + i.x + 1))];
    let c = net[O_FC + u32(F_CAM + u32((i.y + 1) * 4 + i.x))];
    let d = net[O_FC + u32(F_CAM + u32((i.y + 1) * 4 + i.x + 1))];
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
fn heat(x: f32) -> v3 { return clamp(v3(1.5 * x, 1.5 * x - 0.5, 3.0 * x - 2.0), v3(0.0), v3(1.0)) + v3(0.0, 0.0, 0.25 * (1.0 - x)); }

// xy min, zw max; empty when none
fn pick_rf() -> v4 {
    let l = i32(net[O_FC + u32(F_PICK)]);
    let nr = neuron(max(l, 0), vec2<i32>(i32(net[O_FC + u32(F_PICK + 1u)]), i32(net[O_FC + u32(F_PICK + 2u)])));
    if (l == 1) { let o = v2(nr.yz) * 2.0; return v4(o, o + 5.0); }
    if (l == 2) { let o = v2(nr.yz) * 4.0; return v4(o, o + 15.0); }
    if (l >= 3) { return v4(0.0, 0.0, 27.0, 27.0); }
    return v4(1.0, 1.0, -1.0, -1.0);
}
fn in_rect(p: v2, r: v4) -> bool { return all(p >= r.xy) && all(p <= r.zw); }

fn class_char(idx: i32) -> u32 {
    if (idx <= 9) { return 48u + u32(idx); }
    if (idx <= 35) { return 65u + u32(idx - 10); }
    var lower = array<u32, 11>(97u, 98u, 100u, 101u, 102u, 103u, 104u, 110u, 113u, 114u, 116u);
    return lower[clamp(idx - 36, 0, 10)];
}
// bilinear, uv y down
fn glyph(code: u32, uv: v2) -> f32 {
    if (any(uv < v2(0.0)) || any(uv >= v2(1.0))) { return 0.0; }
    let a = (v2(f32(code % 16u), f32(code / 16u)) + 0.05 + uv * 0.9) / 16.0 * font_texture_uniform.atlas_size - 0.5;
    let i = clamp(vec2<i32>(floor(a)), vec2<i32>(0), vec2<i32>(font_texture_uniform.atlas_size) - 2);
    let f = fract(a);
    let s00 = textureLoad(t_font_texture_atlas, i, 0).r;
    let s10 = textureLoad(t_font_texture_atlas, i + vec2<i32>(1, 0), 0).r;
    let s01 = textureLoad(t_font_texture_atlas, i + vec2<i32>(0, 1), 0).r;
    let s11 = textureLoad(t_font_texture_atlas, i + vec2<i32>(1, 1), 0).r;
    return smoothstep(0.1, 0.9, mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y) * 0.8);
}
// asp = box width / height
fn percent(uv: v2, pr: f32, asp: f32) -> f32 {
    let pc = u32(round(clamp(pr, 0.0, 1.0) * 100.0));
    var d = array<u32, 4>(pc / 100u, (pc / 10u) % 10u, pc % 10u, 37u);
    let first = select(select(2u, 1u, pc >= 10u), 0u, pc >= 100u);
    let n = 4u - first;
    let x = uv.x * asp - (asp - f32(n) * 0.55) * 0.5;
    if (x < 0.0 || any(uv < v2(0.0)) || uv.y >= 1.0) { return 0.0; }
    let k = u32(floor(x / 0.55));
    if (k >= n) { return 0.0; }
    var code = 48u + d[first + k];
    if (first + k == 3u) { code = 37u; }
    return glyph(code, v2(fract(x / 0.55), uv.y));
}

@compute @workgroup_size(1, 1, 1)
fn analyze(@builtin(global_invocation_id) id: vec3<u32>) {
    var mx = -1e9;
    for (var i = 0; i < 47; i++) { mx = max(mx, net[O_FC + u32(i)]); }
    var sum = 0.0;
    for (var i = 0; i < 47; i++) { sum += exp(net[O_FC + u32(i)] - mx); }
    net[O_FC + u32(F_MAX)] = mx;
    net[O_FC + u32(F_SUM)] = sum;
    var top = vec3<i32>(0);
    var tp = v3(-1.0);
    for (var i = 0; i < 47; i++) {
        let pr = exp(net[O_FC + u32(i)] - mx) / sum;
        if (pr > tp.x) { tp = v3(pr, tp.xy); top = vec3<i32>(i, top.xy); }
        else if (pr > tp.y) { tp = v3(tp.x, pr, tp.y); top = vec3<i32>(top.x, i, top.y); }
        else if (pr > tp.z) { tp.z = pr; top.z = i; }
    }
    for (var k = 0u; k < 3u; k++) { net[O_FC + u32(F_TOP + k)] = f32(top[k]); net[O_FC + u32(F_PROB + k)] = tp[k]; }
    // CAM (Zhou et al. 2016), exact under global average pooling
    var cm: array<f32, 16>;
    var cmax = 1e-4;
    for (var c = 0; c < 16; c++) {
        var s = 0.0;
        for (var k = 0; k < 32; k++) { s += get_fc_weight(top.x, k) * net[O_C2 + u32(u32(k * 16 + c))]; }
        cm[c] = max(s, 0.0);
        cmax = max(cmax, cm[c]);
    }
    for (var c = 0; c < 16; c++) { net[O_FC + u32(F_CAM + u32(c))] = cm[c] / cmax; }
    var ink = 0.0;
    for (var i = 0u; i < 784u; i++) { ink += net[O_CAN + u32(i)]; }
    net[O_FC + u32(F_INK)] = f32(ink > 2.0);
    let res = v2(textureDimensions(output));
    let mp = mouse_pad(res);
    net[O_FC + u32(F_MOUSE)] = mp.x;
    net[O_FC + u32(F_MOUSE + 1u)] = mp.y;
    net[O_FC + u32(F_PICK)] = -1.0;
    if (mp.x < 0.0) {
        let c = cam();
        let h = trace_net(c.ro, cam_ray(c, mouse.position * res, res));
        if (h.l > 0) {
            net[O_FC + u32(F_PICK)] = f32(h.l);
            net[O_FC + u32(F_PICK + 1u)] = f32(h.s.x);
            net[O_FC + u32(F_PICK + 2u)] = f32(h.s.y);
        }
    }
}

fn ray_seg(ro: v3, rd: v3, a: v3, b: v3) -> v2 {
    let ba = b - a;
    let oa = ro - a;
    let bb = dot(ba, ba);
    let rb = dot(rd, ba);
    let den = bb - rb * rb;
    var s = clamp((dot(oa, ba) - rb * dot(oa, rd)) / max(den, 1e-6), 0.0, 1.0);
    let t = max(dot(a + ba * s - ro, rd), 0.0);
    return v2(length(ro + rd * t - a - ba * s), t);
}

fn hologram(c: Cam, rd: v3, tmax: f32) -> v3 {
    if (net[O_FC + u32(F_INK)] < 0.5) { return v3(0.0); }
    let ctr = v3(138.9, 6.0, 30.0 * clamp(params.height, 0.5, 2.0));
    let sz = 18.0;
    var col = v3(0.0);
    let win = i32(net[O_FC + u32(F_TOP)]);
    let w = anchor(4, win, v2(0.5));
    let bm = ray_seg(c.ro, rd, w, ctr - c.up * sz * 0.52);
    if (bm.y < tmax) { col += v3(1.0, 0.85, 0.5) * exp(-bm.x * bm.x / 0.04) * 3.0 + v3(1.0, 0.7, 0.3) * exp(-bm.x * bm.x / 1.2) * 0.25; }
    let th = dot(ctr - c.ro, c.f) / dot(rd, c.f);
    if (th <= 0.0 || th > tmax) { return col; }
    let q = c.ro + rd * th - ctr;
    let uv = v2(dot(q, c.rt), -dot(q, c.up)) / sz + 0.5;
    if (any(uv < v2(0.0)) || any(uv > v2(1.0))) { return col; }
    var gold = v3(1.0, 0.8, 0.35);
    let truth = i32(st(S_SHOW));
    if (params.mode == 0u && truth != win) { gold = v3(1.0, 0.3, 0.25); }
    let scan = 0.8 + 0.2 * sin(uv.y * 260.0 - time_data.time * 3.0);
    var a = glyph(class_char(win), (uv - v2(0.18, 0.0)) / 0.64) * 3.2;
    a += percent((uv - v2(0.3, 0.66)) / v2(0.4, 0.12), net[O_FC + u32(F_PROB)], 3.33) * 1.6;
    for (var k = 1u; k < 3u; k++) {
        let o = v2(select(0.12, 0.66, k == 2u), 0.83);
        a += glyph(class_char(i32(net[O_FC + u32(F_TOP + k)])), (uv - o) / 0.11) * 0.7;
        a += percent((uv - o - v2(0.11, 0.02)) / v2(0.2, 0.07), net[O_FC + u32(F_PROB + k)], 2.86) * 0.5;
    }
    if (params.mode == 0u) { a += glyph(class_char(truth), (uv - v2(0.8, 0.04)) / 0.15) * 0.9; }
    let e = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    col += gold * (a * scan + smoothstep(0.012, 0.0, e) * 0.6 + 0.025);
    return col;
}

// conv1 kernels above their maps: orange positive, blue negative
fn kernel_tile(ro: v3, rd: v3, tmax: f32) -> v4 {
    let L = layer(1);
    let zk = top_h(L) + 3.0;
    if (params.kernels <= 0.0 || abs(rd.z) < 1e-5) { return v4(0.0); }
    let t = (zk - ro.z) / rd.z;
    if (t <= 0.0 || t > tmax) { return v4(0.0); }
    let q = ((ro + rd * t).xy - L.org) / L.cell;
    let tile = vec2<i32>(floor(q / 13.0));
    if (any(tile < vec2<i32>(0)) || any(tile > vec2<i32>(3))) { return v4(0.0); }
    let u = (q - v2(tile) * 13.0) / 12.0;
    if (any(u < v2(0.0)) || any(u >= v2(1.0))) { return v4(0.0); }
    let m = u32(tile.y * 4 + tile.x);
    let cell = vec2<u32>(u * 5.0);
    let w = net[O_W + W1 + m * 25u + (4u - cell.y) * 5u + cell.x];
    var mx = 1e-6;
    for (var k = 0u; k < 25u; k++) { mx = max(mx, abs(net[O_W + W1 + m * 25u + k])); }
    let a = w / mx;
    let edge = fract(u * 5.0);
    let line = smoothstep(0.08, 0.0, min(min(edge.x, 1.0 - edge.x), min(edge.y, 1.0 - edge.y)));
    let col = select(v3(0.2, 0.55, 1.0), v3(1.0, 0.55, 0.15), a > 0.0) * abs(a) * 2.0 + v3(0.25) * line;
    return v4(col * params.kernels, 0.55);
}
// rows true class, columns predicted, rows normalised
const CONF_ORG = v2(129.0, 17.0);
const CONF_SIZE: f32 = 20.0;
fn confusion(xy: v2) -> v3 {
    let q = (xy - CONF_ORG) / CONF_SIZE;
    if (params.conf <= 0.0 || any(q < v2(0.0)) || any(q >= v2(1.0))) { return v3(-1.0); }
    let c = vec2<u32>(q * 47.0);
    let row = 46u - c.y;
    var sum = 0.0;
    for (var k = 0u; k < 47u; k++) { sum += net[O_CONF + row * 47u + k]; }
    let v = net[O_CONF + row * 47u + c.x] / max(sum, 1.0);
    let g = fract(q * 47.0);
    let line = smoothstep(0.1, 0.0, min(min(g.x, 1.0 - g.x), min(g.y, 1.0 - g.y))) * 0.03;
    let hue = select(v3(1.0, 0.25, 0.45), v3(1.0, 0.82, 0.35), row == c.x);
    return (hue * pow(v, 0.6) * 2.5 + v3(line)) * params.conf + v3(0.01);
}

// dream gallery, 8 x 6 tiles turned toward the camera about the vertical
const DREAM_C = v3(74.0, 4.0, 48.0);
const DP: f32 = 6.2;
const DS: f32 = 5.5;
fn dream_n(c: Cam) -> v3 { return normalize(v3(c.ro.xy - DREAM_C.xy, 0.0) + v3(0.0, -1e-4, 0.0)); }
fn dream_tile(c: Cam, k: u32) -> v3 {
    let n = dream_n(c);
    return DREAM_C + v3(-n.y, n.x, 0.0) * (f32(k % 8u) - 3.5) * DP + v3(0.0, 0.0, (2.5 - f32(k / 8u)) * DP);
}
fn dream_px(k: u32, p: v2) -> f32 {
    let x = clamp(p, v2(0.0), v2(27.0));
    let i = vec2<u32>(min(floor(x), v2(26.0)));
    let f = x - v2(i);
    let o = O_DRX + k * 784u + i.y * 28u + i.x;
    return mix(mix(net[o], net[o + 1u], f.x), mix(net[o + 28u], net[o + 29u], f.x), f.y);
}
struct DHit { col: v3, a: f32, t: f32 };
fn dream_wall(c: Cam, rd: v3, tmax: f32) -> DHit {
    let none = DHit(v3(0.0), 0.0, tmax);
    let n = dream_n(c);
    let den = dot(rd, n);
    if (params.dream <= 0.0 || abs(den) < 1e-5) { return none; }
    let t = dot(DREAM_C - c.ro, n) / den;
    if (t <= 0.0 || t > tmax) { return none; }
    let q = c.ro + rd * t - DREAM_C;
    let g = v2(dot(q, v3(-n.y, n.x, 0.0)) + 4.0 * DP, 3.0 * DP - q.z) / DP;
    let cell = vec2<i32>(floor(g));
    if (any(cell < vec2<i32>(0)) || cell.x > 7 || cell.y > 5 || cell.y * 8 + cell.x > 46) { return none; }
    let k = u32(cell.y * 8 + cell.x);
    let f = (fract(g) * DP - 0.5 * (DP - DS)) / DS;
    if (any(f < v2(0.0)) || any(f >= v2(1.0))) { return none; }
    let v = dream_px(k, f * 28.0 - 0.5);
    let p = net[O_DRP + k];
    var col = mix(v3(1.0, 0.45, 0.12), v3(1.0, 0.9, 0.7), v) * v * v * (0.4 + 2.6 * p);
    let e = min(min(f.x, 1.0 - f.x), min(f.y, 1.0 - f.y));
    var frame = mix(v3(0.04, 0.05, 0.07), v3(1.0, 0.8, 0.3) * 0.6, p);
    if (net[O_FC + u32(F_INK)] > 0.5 && i32(k) == i32(net[O_FC + u32(F_TOP)])) { frame = v3(0.35, 0.6, 1.0) * 1.2; }
    col += frame * smoothstep(0.015, 0.0, e);
    // p(class | its own dream)
    if (f.y > 0.965 && f.x < p) { col += v3(1.0, 0.8, 0.3) * 0.7; }
    col += v3(0.5, 0.45, 0.4) * glyph(class_char(i32(k)), (f - 0.04) / 0.16) * 0.3;
    return DHit(col * params.dream + v3(0.008, 0.01, 0.016), 0.6, t);
}

fn net_view(px: v2, D: v2) -> v4 {
    let c = cam();
    let ro = c.ro;
    let rd = cam_ray(c, px, D);
    let h = trace_net(ro, rd);
    let l = normalize(v3(-0.4, -0.6, 0.7));
    let sky = mix(v3(0.01, 0.012, 0.02), v3(0.03, 0.035, 0.06), clamp(rd.z * 2.0 + 0.5, 0.0, 1.0));
    let mp = v2(net[O_FC + u32(F_MOUSE)], net[O_FC + u32(F_MOUSE + 1u)]);
    var col: v3;
    var depth = 6e4;
    if (h.l < 0 && rd.z >= 0.0) {
        col = sky;
    } else if (h.l < 0) {
        let tf = -ro.z / rd.z;
        depth = tf;
        let fp = ro + rd * tf;
        col = v3(0.006);
        for (var k = 0; k < 5; k++) {
            let L = layer(k);
            let q = (fp.xy - L.org) / (v2(slots(L)) * L.cell);
            if (all(q > v2(-0.03)) && all(q < v2(1.03))) {
                let e = min(min(q.x + 0.03, 1.03 - q.x), min(q.y + 0.03, 1.03 - q.y)) * 30.0;
                col = v3(0.018) + layer_col(k) * select(0.06, 0.18, k == 0) * smoothstep(0.4, 0.0, e);
            }
        }
        let cf = confusion(fp.xy);
        if (cf.x >= 0.0) { col = cf; }
        col = mix(col, sky, clamp(tf / 500.0, 0.0, 1.0));
    } else {
        let pos = ro + rd * h.t;
        depth = h.t;
        let nr = neuron(h.l, h.s);
        let a = act(h.l, nr);
        var base = layer_col(h.l);
        var glow = pow(clamp(a, 0.0, 1.5), 1.3) * 2.5;
        let top = h.n.z > 0.5;
        let L = layer(h.l);
        if (h.l == 0) {
            let cm = cam_at(v2(nr.yz));
            base = mix(v3(0.9), heat(cm), clamp(params.cam, 0.0, 1.0) * smoothstep(0.05, 0.3, cm) * smoothstep(0.0, 0.2, a));
            glow = a * 1.1 + 0.03;
        }
        if (h.l == 4) {
            glow = sqrt(a) * 4.0;
            if (nr.x == i32(net[O_FC + u32(F_TOP)])) { base = v3(1.0, 0.95, 0.8); }
        }
        let pl = i32(net[O_FC + u32(F_PICK)]);
        var hi = 0.0;
        if (pl == h.l && all(vec2<i32>(i32(net[O_FC + u32(F_PICK + 1u)]), i32(net[O_FC + u32(F_PICK + 2u)])) == h.s)) { hi = 1.0; }
        if (h.l == 0 && in_rect(v2(nr.yz), pick_rf())) { hi = 0.5; }
        if (h.l == 1 && pl == 2) {
            let o = v2(neuron(2, vec2<i32>(i32(net[O_FC + u32(F_PICK + 1u)]), i32(net[O_FC + u32(F_PICK + 2u)]))).yz) * 2.0;
            if (in_rect(v2(nr.yz), v4(o, o + 5.0))) { hi = 0.5; }
        }
        let ndl = max(dot(h.n, l), 0.0);
        col = base * (0.04 + 0.12 * ndl) + base * glow * select(0.35, 1.0, top);
        col += v3(0.35, 0.6, 1.0) * hi * select(0.4, 1.2, top);
        let lq = fract((pos.xy - L.org) / L.cell) - 0.5;
        col += base * 0.25 * smoothstep(0.33, 0.38, max(abs(lq.x), abs(lq.y))) * f32(top);
        if (h.l == 4 && top) {
            let g = glyph(class_char(nr.x), v2(lq.x, -lq.y) / 0.76 + 0.5);
            col = mix(col, select(v3(0.6), v3(0.02), glow > 1.0), g * 0.85);
        }
    }
    if (mp.x >= 0.0 && depth < 6e4) {
        let L = layer(0);
        let q = ((ro + rd * depth).xy - L.org) / L.cell;
        col += v3(0.4, 0.7, 1.0) * smoothstep(0.18, 0.0, abs(length(q - mp) - brush_r())) * 1.5;
    }
    let kt = kernel_tile(ro, rd, depth);
    col = mix(col, kt.rgb, kt.a);
    let dw = dream_wall(c, rd, depth);
    col = col * (1.0 - dw.a) + dw.col;
    depth = dw.t;
    return v4(col + hologram(c, rd, depth), depth);
}

fn pack_px(c: v3, d: f32) -> vec2<u32> { return vec2<u32>(pack2x16float(c.rg), pack2x16float(v2(c.b, d))); }
fn unpack_px(q: vec2<u32>) -> v3 { return v3(unpack2x16float(q.x), unpack2x16float(q.y).x); }
fn aces(x: v3) -> v3 { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), v3(0.0), v3(1.0)); }
fn qdims(d: vec2<u32>) -> vec2<u32> { return (d + 3u) / 4u; }
fn hdims(d: vec2<u32>) -> vec2<u32> { return (d + 1u) / 2u; }

@compute @workgroup_size(16, 16, 1)
fn scene3d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output);
    if (gid.x >= dims.x || gid.y >= dims.y || dims.x * dims.y > MAXPIX) { return; }
    var c = net_view(v2(gid.xy) + 0.5, v2(dims));
    c = select(c, v4(0.0, 0.0, 0.0, 6e4), c != c);
    hdr[gid.y * dims.x + gid.x] = pack_px(clamp(c.rgb, v3(0.0), v3(1000.0)), min(c.w, 6e4));
}

@compute @workgroup_size(16, 16, 1)
fn fiber_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let hd = hdims(textureDimensions(output));
    if (gid.x >= hd.x || gid.y >= hd.y) { return; }
    let i = gid.y * hd.x + gid.x;
    for (var k = 0u; k < 3u; k++) { atomicStore(&fib[k * FHP + i], 0u); }
}

// one fiber per thread, depth-tested, splatted at half res; the last 47 lead to the dreams
@compute @workgroup_size(64, 1, 1)
fn fibers(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    let dims = textureDimensions(output);
    if (dims.x * dims.y > MAXPIX) { return; }
    if (id >= FIBN + 47u) { return; }
    let dt = id >= FIBN;
    if (!dt && params.fibers <= 0.0 && params.back <= 0.0) { return; }
    var p0: v3;
    // backward: |dL/dw| relative to its layer, log-scaled, times the loss
    var gb = 0.0;
    var p1: v3;
    var b = 0.0;
    var l0 = 0;
    if (id < FIB1) {
        let k = i32(id / 144u);
        let x = i32(id % 144u) % 12;
        let y = i32(id % 144u) / 12;
        // only where there's ink
        let ink = sample_canvas(vec2<i32>(2 * x + 2, 25 - 2 * y)) + sample_canvas(vec2<i32>(2 * x + 3, 24 - 2 * y));
        b = act(1, vec3<i32>(k, x, y)) * clamp(ink, 0.0, 1.0) * 0.12;
        var gs = 0.0;
        for (var q = 0u; q < 25u; q++) { gs += net[O_GV + W1 + u32(k) * 25u + q]; }
        gb = grad_light(gs / max(25.0 * st(S_GM), 1e-12)) * clamp(ink, 0.0, 1.0) * 0.1;
        p0 = anchor(0, 0, v2(f32(2 * x) + 3.0, f32(2 * y) + 3.0));
        p1 = anchor(1, k, v2(f32(x), f32(y)) + 0.5);
    } else if (id < FIB2) {
        let g = i32(id - FIB1);
        let n = g / 16;
        let k = g % 16;
        let j = n / 16;
        let x = (n % 16) % 4;
        let y = (n % 16) / 4;
        b = act(2, vec3<i32>(j, x, y)) * clamp(act(1, vec3<i32>(k, 2 * x + 2, 2 * y + 2)) * 1.5, 0.0, 1.0) * 0.06;
        var gs = 0.0;
        for (var q = 0u; q < 25u; q++) { gs += net[O_GV + W2 + u32(j * 16 + k) * 25u + q]; }
        gb = grad_light(gs / max(25.0 * st(S_GM + 1u), 1e-12)) * 0.05;
        p0 = anchor(1, k, v2(f32(2 * x) + 3.0, f32(2 * y) + 3.0));
        p1 = anchor(2, j, v2(f32(x), f32(y)) + 0.5);
        l0 = 1;
    } else if (id < FIB3) {
        let g = i32(id - FIB2);
        let j = g / 16;
        let x = (g % 16) % 4;
        let y = (g % 16) / 4;
        b = act(2, vec3<i32>(j, x, y)) * 0.35;
        gb = grad_light(st(S_DG + u32(j)) / max(st(S_GM + 3u), 1e-12)) * 0.2;
        p0 = anchor(2, j, v2(f32(x), f32(y)) + 0.5);
        p1 = anchor(3, j, v2(0.5));
        l0 = 2;
    } else if (dt) {
        // output neuron up to its dream, as bright as the dream is convincing
        let k = id - FIBN;
        b = (0.06 + 1.5 * net[O_DRP + k]) * 0.3;
        p0 = anchor(4, i32(k), v2(0.5));
        p1 = dream_tile(cam(), k) - v3(0.0, 0.0, 0.5 * DS);
        l0 = 4;
    } else {
        let g = i32(id - FIB3);
        let k = g / 47;
        let cl = g % 47;
        b = max(get_fc_weight(cl, k) * gap_avg(k), 0.0) * 0.25 * (0.05 + 3.0 * prob(cl));
        gb = grad_light(net[O_GV + WF + u32(cl * 32 + k)] / max(st(S_GM + 2u), 1e-12)) * 0.12;
        p0 = anchor(3, k, v2(0.5));
        p1 = anchor(4, cl, v2(0.5));
        l0 = 3;
    }
    b = min(b, 1.5) * select(params.fibers, params.dream, dt);
    gb *= params.back * clamp(st(S_LOSS) / log(47.0), 0.03, 1.0);
    if (b + gb < 0.004) { return; }
    let c = cam();
    let D = v2(dims);
    let hd = hdims(dims);
    let lift = v3(0.0, 0.0, 0.3 * length(p1 - p0) + 2.0);
    var c0 = p0 + lift;
    var c1 = p1 + lift;
    // straight up out of the tower, into the tile from below
    if (dt) { c0 = p0 + v3(0.0, 0.0, 0.6 * (p1.z - p0.z)); c1 = p1 - v3(0.0, 0.0, 0.4 * (p1.z - p0.z)); }
    // about one sample per half-res pixel
    let s0 = project(c, p0, D);
    let sm = project(c, 0.125 * (p0 + p1) + 0.375 * (c0 + c1), D);
    let s1 = project(c, p1, D);
    let n = u32(clamp((length(sm.xy - s0.xy) + length(s1.xy - sm.xy)) * 0.6, 8.0, 240.0));
    let ph = f32(hu(id) >> 8u) / 16777216.0 * 6.2832;
    var ca = layer_col(l0);
    var cb = layer_col(l0 + 1);
    if (dt) { ca = v3(0.2, 1.0, 0.55); cb = v3(0.35, 0.9, 1.0); }
    for (var i = 0u; i < n; i++) {
        let s = (f32(i) + 0.5) / f32(n);
        let u = 1.0 - s;
        let p = u * u * u * p0 + 3.0 * u * u * s * c0 + 3.0 * u * s * s * c1 + s * s * s * p1;
        let pr = project(c, p, D);
        if (pr.z < 0.0 || any(pr.xy < v2(0.0)) || any(pr.xy >= D - 1.0)) { continue; }
        let fp = vec2<u32>(pr.xy);
        if (pr.z > unpack2x16float(hdr[fp.y * dims.x + fp.x].y).y + 0.5) { continue; }
        let pulse = 0.15 + 0.85 * pow(0.5 + 0.5 * sin(6.2832 * (s * 2.0 - time_data.time * 0.7) + ph), 8.0);
        let pb = 0.15 + 0.85 * pow(0.5 + 0.5 * sin(6.2832 * (s * 2.0 + time_data.time * 0.9) + ph * 1.7), 8.0);
        let v = (mix(ca, cb, s) * b * pulse + v3(0.35, 0.75, 1.0) * gb * pb) * 0.35 * FIX;
        let hp = pr.xy * 0.5 - 0.5;
        let hi = vec2<u32>(max(floor(hp), v2(0.0)));
        let f = fract(hp);
        var w = array<f32, 4>((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
        for (var k = 0u; k < 4u; k++) {
            let q = min(hi + vec2<u32>(k & 1u, k >> 1u), hd - 1u);
            let o = q.y * hd.x + q.x;
            atomicAdd(&fib[o], u32(v.r * w[k]));
            atomicAdd(&fib[FHP + o], u32(v.g * w[k]));
            atomicAdd(&fib[2u * FHP + o], u32(v.b * w[k]));
        }
    }
}
fn hu(a0: u32) -> u32 { var a = a0; a ^= a >> 16u; a *= 0x7feb352du; a ^= a >> 15u; a *= 0x846ca68bu; a ^= a >> 16u; return a; }
fn grad_light(ratio: f32) -> f32 { return clamp(log2(1.0 + max(ratio, 0.0)) / 2.0, 0.0, 1.5); }
fn scene_px(px: vec2<u32>, dims: vec2<u32>) -> v3 {
    let hd = hdims(dims);
    let x = clamp((v2(px) + 0.5) * 0.5 - 0.5, v2(0.0), v2(hd) - 1.0);
    let i = vec2<u32>(floor(x));
    let f = fract(x);
    let j = min(i + 1u, hd - 1u);
    var fc = v3(0.0);
    for (var k = 0u; k < 4u; k++) {
        let q = vec2<u32>(select(i.x, j.x, (k & 1u) == 1u), select(i.y, j.y, k >= 2u));
        let o = q.y * hd.x + q.x;
        let w = select(1.0 - f.x, f.x, (k & 1u) == 1u) * select(1.0 - f.y, f.y, k >= 2u);
        fc += v3(f32(atomicLoad(&fib[o])), f32(atomicLoad(&fib[FHP + o])), f32(atomicLoad(&fib[2u * FHP + o]))) * w;
    }
    return unpack_px(hdr[px.y * dims.x + px.x]) + fc / FIX;
}

// bloom: Karis 4x4 above white, then a separable gaussian at quarter res
@compute @workgroup_size(16, 16, 1)
fn bloom_down(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output);
    let q = qdims(dims);
    if (gid.x >= q.x || gid.y >= q.y || dims.x * dims.y > MAXPIX) { return; }
    var c = v3(0.0);
    var ws = 0.0;
    for (var k = 0u; k < 16u; k++) {
        let s = scene_px(min(gid.xy * 4u + vec2<u32>(k & 3u, k >> 2u), dims - 1u), dims);
        let w = 1.0 / (1.0 + max(s.r, max(s.g, s.b)));
        c += s * w;
        ws += w;
    }
    c /= ws;
    b1[gid.y * q.x + gid.x] = pack_px(c * smoothstep(0.6, 1.3, max(c.r, max(c.g, c.b)) * params.expo), 0.0);
}
fn blur(src_b1: bool, gid: vec2<u32>, dir: vec2<i32>) {
    let dims = textureDimensions(output);
    let q = qdims(dims);
    if (gid.x >= q.x || gid.y >= q.y || dims.x * dims.y > MAXPIX) { return; }
    var c = v3(0.0);
    var ws = 0.0;
    for (var k = -12; k <= 12; k++) {
        let s = clamp(vec2<i32>(gid) + dir * k, vec2<i32>(0), vec2<i32>(q) - 1);
        let i = u32(s.y) * q.x + u32(s.x);
        let w = exp(-f32(k * k) / 50.0);
        c += unpack_px(select(b2[i], b1[i], src_b1)) * w;
        ws += w;
    }
    let o = gid.y * q.x + gid.x;
    if (src_b1) { b2[o] = pack_px(c / ws, 0.0); } else { b1[o] = pack_px(c / ws, 0.0); }
}
@compute @workgroup_size(16, 16, 1) fn bloom_h(@builtin(global_invocation_id) gid: vec3<u32>) { blur(true, gid.xy, vec2<i32>(1, 0)); }
@compute @workgroup_size(16, 16, 1) fn bloom_v(@builtin(global_invocation_id) gid: vec3<u32>) { blur(false, gid.xy, vec2<i32>(0, 1)); }
fn bloom_at(qp: v2, dims: vec2<u32>) -> v3 {
    let q = qdims(dims);
    let x = clamp(qp - 0.5, v2(0.0), v2(q) - 1.0);
    let i = vec2<u32>(floor(x));
    let f = fract(x);
    let j = min(i + 1u, q - 1u);
    return mix(mix(unpack_px(b1[i.y * q.x + i.x]), unpack_px(b1[i.y * q.x + j.x]), f.x),
               mix(unpack_px(b1[j.y * q.x + i.x]), unpack_px(b1[j.y * q.x + j.x]), f.x), f.y);
}

@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let res = textureDimensions(output);
    if any(id.xy >= res) { return; }
    var c: v3;
    if (res.x * res.y > MAXPIX) { c = net_view(v2(id.xy) + 0.5, v2(res)).rgb; } else {
        c = scene_px(id.xy, res) + bloom_at((v2(id.xy) + 0.5) / 4.0, res) * params.bloom;
    }
    var col = aces(c * params.expo);
    let D = v2(res);
    let px = v2(id.xy) + 0.5;
    let r = pad2d_rect(D);
    let lq = (px - r.xy) / r.z;
    if (all(lq >= v2(-0.012)) && all(lq <= v2(1.012))) {
        col = v3(0.35);
        let q = pad2d(px, D);
        if (q.x >= 0.0) {
            let cc = vec2<i32>(q);
            let v = sample_canvas(cc);
            let np = v2(f32(cc.x), f32(27 - cc.y));
            let cm = cam_at(np);
            col = mix(v3(0.03, 0.035, 0.05), v3(0.95), v);
            col = mix(col, heat(cm) * (0.35 + 0.65 * v), clamp(params.cam, 0.0, 1.0) * 0.7 * smoothstep(0.05, 0.3, cm));
            let rf = pick_rf();
            if (in_rect(np, rf) && !in_rect(np, rf + v4(1.0, 1.0, -1.0, -1.0))) { col = mix(col, v3(0.35, 0.6, 1.0), 0.8); }
            let mp = v2(net[O_FC + u32(F_MOUSE)], net[O_FC + u32(F_MOUSE + 1u)]);
            if (mp.x >= 0.0) { col = mix(col, v3(0.4, 0.7, 1.0), smoothstep(0.15, 0.0, abs(length(q - mp) - brush_r()))); }
        }
    }
    col = hud(px, D, col);
    textureStore(output, vec2<i32>(id.xy), v4(col, 1.0));
}

// right-aligned digits
fn number(px: v2, p: v2, h: f32, n0: u32) -> f32 {
    var n = n0;
    var a = 0.0;
    for (var k = 0; k < 8; k++) {
        a = max(a, glyph(48u + n % 10u, (px - p + v2(f32(k) * h * 0.55, 0.0)) / h));
        n /= 10u;
        if (n == 0u) { break; }
    }
    return a;
}
fn hud(px: v2, D: v2, c0: v3) -> v3 {
    let o = v2(D.x - 380.0, D.y - 170.0);
    let q = px - o;
    if (any(q < v2(0.0)) || any(q >= v2(364.0, 154.0))) { return c0; }
    var col = mix(c0, v3(0.02, 0.025, 0.04), 0.75);
    let steps = u32(st(S_STEP));
    let cnt = min(steps / 16u + 1u, 256u);
    let first = (steps / 16u + 1u) - cnt;
    let pa = q - v2(10.0, 34.0);
    if (all(pa >= v2(0.0)) && all(pa < v2(344.0, 110.0)) && steps > 0u) {
        let e = min(u32(pa.x / 344.0 * f32(cnt)), cnt - 1u);
        let h = (first + e) % 256u;
        let ly = (1.0 - clamp(net[O_HIST + h * 2u] / log(47.0), 0.0, 1.0)) * 110.0;
        let ay = (1.0 - clamp(net[O_HIST + h * 2u + 1u], 0.0, 1.0)) * 110.0;
        col += v3(0.3, 0.75, 1.0) * smoothstep(2.0, 0.5, abs(pa.y - (110.0 - ly)));
        col += v3(1.0, 0.8, 0.3) * smoothstep(2.0, 0.5, abs(pa.y - ay));
        if (fract(pa.y / 27.5) < 0.02) { col += v3(0.03); }
    }
    var t = glyph(97u, (q - v2(10.0, 6.0)) / 20.0) + glyph(99u, (q - v2(21.0, 6.0)) / 20.0) + glyph(99u, (q - v2(32.0, 6.0)) / 20.0);
    t += percent((q - v2(50.0, 6.0)) / v2(70.0, 20.0), st(S_ACC), 3.5);
    col = mix(col, v3(1.0, 0.8, 0.3), clamp(t, 0.0, 1.0));
    col = mix(col, v3(0.6, 0.65, 0.7), number(q, v2(345.0, 8.0), 16.0, steps));
    return col;
}

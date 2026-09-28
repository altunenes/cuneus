// Enes Altun, 2025; MIT License
// 2D Gaussian Splatting with Real-time Training
// https://shader-slang.org/blog/2025/04/04/neural-gfx-in-an-afternoon/
// https://shader-slang.org/blog/2025/04/30/neural-graphics-first-principles-performance/

alias v2 = vec2<f32>;
alias v3 = vec3<f32>;
alias v4 = vec4<f32>;
alias m2 = mat2x2<f32>;

const PI = 3.14159265;
const MAX_G = 40000u;
const G_PER_TILE = 2048u;
const GAUSS_TILE = 2048u;
const WX = 8u;
const WY = 8u;
const WG = WX * WY;
const GRADS = 12u;
const CELL = 64u;
const MAXC = 4096u;
const CAP = 2048u;
const BIGC = 16u;
const PALB = 200000u;

struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
@group(0) @binding(0) var<uniform> u_time: TimeUniform;

@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;

struct GaussianParams {
    num_gaussians: u32,
    learning_rate: f32,
    color_learning_rate: f32,
    reset_training: u32,
    show_target: u32,
    show_error: u32,
    opacity_learning_rate: f32,
    error_scale: f32,
    min_sigma: f32,
    max_sigma: f32,
    freq_max: f32,
    random_seed: u32,
    iteration: u32,
    sigma_learning_rate: f32,
    draw_progress: f32,
    draw_grow: f32,
    oil_enable: u32,
    hardness: f32,
    bristle_amt: f32,
    canvas_amt: f32,
    edge_rag: f32,
    impasto: f32,
    curve_amt: f32,
    mode: u32,
    draw_prepare: u32,
    lr_decay_rate: f32,
    l1_mix: f32,
    dens: f32,
    _pd: u32,
    par: f32,
    pal_k: f32,
    pal_amt: f32,
    _pe: f32,
    _pf: f32,
    _pg: f32,
    _ph: f32,
};
@group(1) @binding(1) var<uniform> p: GaussianParams;

@group(2) @binding(0) var t_target: texture_2d<f32>;
@group(2) @binding(1) var s_target: sampler;

struct GaussianData {
    center: v2,
    sigma_xx: f32,
    sigma_xy: f32,
    sigma_yy: f32,
    _padding: f32,
    gpad0: f32,
    gpad1: f32,
    color: v3,
    opacity: f32,
};
@group(3) @binding(0) var<storage, read_write> g_data: array<GaussianData>;

@group(3) @binding(1) var<storage, read_write> g_grad: array<atomic<u32>>;
@group(3) @binding(2) var<storage, read_write> adam_m: array<f32>;
@group(3) @binding(3) var<storage, read_write> adam_v: array<f32>;
@group(3) @binding(4) var<storage, read_write> draw_rank: array<u32>;
// per 8x8 tile error, x1000, written by the renderer, read by respawn
@group(3) @binding(5) var<storage, read_write> err_grid: array<atomic<u32>>;
@group(3) @binding(6) var<storage, read_write> bin_cnt: array<atomic<u32>>;
@group(3) @binding(7) var<storage, read_write> bin_idx: array<u32>;

//shared memory

var<workgroup> b_cnt_atom: atomic<u32>;
var<workgroup> b_cnt: u32;
var<workgroup> b_idx: array<u32, G_PER_TILE>;
// I use this buffer to sum gradients within the workgroup first.
// This drastically reduces atomic contention since only thread 0 writes to global memory.
// v3 packed: one reduction handles 3 gradient components at once
var<workgroup> red_buf: array<v3, WG>;

// helpers

fn hash4(p:v4)->v4 {
    var q = fract(p * v4(.1031, .1030, .0973, .1099));
    q += dot(q, q.wzxy + 33.33);
    return fract((q.xxyz + q.yzzw) * q.zywx);
}

fn inv_m2(m:m2)->m2 {
    let d = 1. / determinant(m);
    return m2(v2(m[1][1]*d, -m[0][1]*d), v2(-m[1][0]*d, m[0][0]*d));
}

// OBB (Oriented Bounding Box) for Gaussian culling
struct OBB { c:v2, r:m2, s:v2 };

fn obb_hit(a:OBB, b:OBB)->bool {
    let c_pts = array<v2,4>(v2(-1.), v2(1.,-1.), v2(1.), v2(-1.,1.));
    let ira = transpose(a.r); let irb = transpose(b.r);
    var pa: array<v2,4>; var pb: array<v2,4>;
    
    for(var i=0u; i<4u; i++){
        pa[i] = a.c + ira * (c_pts[i] * a.s);
        pb[i] = b.c + irb * (c_pts[i] * b.s);
    }
    return !(sep(pa,pb,a.r) || sep(pa,pb,b.r));
}

fn sep(pa:array<v2,4>, pb:array<v2,4>, ax:m2)->bool {
    for(var i=0u; i<2u; i++){
        let a = ax[i];
        var min_a = dot(pa[0],a); var max_a = min_a;
        var min_b = dot(pb[0],a); var max_b = min_b;
        for(var j=1u; j<4u; j++){
            let da = dot(pa[j],a); let db = dot(pb[j],a);
            min_a = min(min_a, da); max_a = max(max_a, da);
            min_b = min(min_b, db); max_b = max(max_b, db);
        }
        if(max_a < min_b || max_b < min_a) { return true; }
    }
    return false;
}
// Cheap conservative broad-phase: a 3-sigma bounding circle vs the axis-aligned tile.
// Returns true if they definitely DON'T overlap, so we can skip the expensive OBB SAT
// (and the cos/sin in get_bounds) for the vast majority of Gaussians far from the tile.
fn aabb_miss(center:v2, rad:f32, tl:v2, th:v2)->bool {
    return (center.x + rad < tl.x) || (center.x - rad > th.x) ||
           (center.y + rad < tl.y) || (center.y - rad > th.y);
}

// radius (in sigmas) where opacity*falloff drops below 1/255
fn vis_k(op:f32)->f32 { return clamp(sqrt(max(2. * log(max(op, 1e-4) * 255.), 0.)), 0., 3.); }

// gauss bounds
fn get_bounds(g:GaussianData)->OBB {
    let s = v2(g.sigma_xx, g.sigma_yy) * vis_k(g.opacity);
    let c = cos(g.sigma_xy); let sn = sin(g.sigma_xy);
    let rot = m2(v2(c, sn), v2(-sn, c));
    return OBB(g.center, rot, s);
}

fn eval_g(g:GaussianData, uv:v2)->v4 {
    let d_raw = uv - g.center;
    // Rotation logic
    let c = cos(g.sigma_xy); let s = sin(g.sigma_xy);
    let d = v2(d_raw.x*c + d_raw.y*s, d_raw.y*c - d_raw.x*s);
    
    let sx = max(g.sigma_xx, .001);
    let sy = max(g.sigma_yy, .001);
    let dsq = (d.x*d.x)/(sx*sx) + (d.y*d.y)/(sy*sy);
    let w = min(.99, g.opacity * exp(-.5 * dsq));
    return v4(g.color, w);
}

fn h21(p: v2) -> f32 {
    var q = fract(p * v2(.1031, .1173));
    q += dot(q, q.yx + 33.33);
    return fract((q.x + q.y) * q.x);
}
fn vn2(p: v2) -> f32 {
    let i = floor(p); let f = fract(p); let u = f * f * (3. - 2. * f);
    let a = h21(i); let b = h21(i + v2(1., 0.));
    let c = h21(i + v2(0., 1.)); let d = h21(i + v2(1., 1.));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// loss gradient: blend of squared error and smooth L1
fn loss_grad(e:v3, n:f32)->v3 { return mix(2. * e, e / sqrt(e * e + 1e-4), p.l1_mix) / n; }

// respawn spot: the worst of 8 random tiles, or random when dens is 0
fn err_spawn(h:v4, seed:f32)->v2 {
    var best = clamp(h.xy, v2(.05), v2(.95));
    if (p.dens <= 0.) { return best; }
    let dim = textureDimensions(output);let tw = (dim.x + WX - 1u) / WX;let th = (dim.y + WY - 1u) / WY;
    var be = -1.;
    for (var k = 0u; k < 8u; k++) {
        let q = hash4(v4(seed, f32(k) * 7.1, h.z * 13., h.w * 17.));
        let c = clamp(q.xy, v2(.02), v2(.98));
        let t = min(vec2<u32>(c * v2(f32(tw), f32(th))), vec2<u32>(tw - 1u, th - 1u));
        let e = f32(atomicLoad(&err_grid[min(t.y * tw + t.x, PALB - 1u)])) * mix(1., q.z, 1. - p.dens);
        if (e > be) { be = e; best = c; }
    }
    return best;
}

// palette colour k
fn palc(k:u32)->v3 {
    let a = PALB + k * 4u;
    return v3(bitcast<f32>(atomicLoad(&err_grid[a])), bitcast<f32>(atomicLoad(&err_grid[a+1u])), bitcast<f32>(atomicLoad(&err_grid[a+2u])));
}
// gabor keeps its sizes in sigma_xx/sigma_xy
fn szn(g:GaussianData)->f32 { return clamp(select(max(g.sigma_xx, g.sigma_xy), max(g.sigma_xx, g.sigma_yy), p.mode == 0u) / max(p.max_sigma, 1e-4), 0., 1.); }
// display colour: blended toward its palette colour
fn dcol(g:GaussianData)->v3 {
    if (p.pal_amt <= 0. || p.pal_k < 2.) { return g.color; }
    // palette slot: _padding in gaussian mode, the unused opacity in gabor mode
    return mix(g.color, palc(min(u32(select(g.opacity, g._padding, p.mode == 0u)), 15u)), p.pal_amt);
}
// parallax: small strokes sit in front and sway more
fn poff(g:GaussianData)->v2 {
    if (p.par <= 0.) { return v2(0.); }
    let t = u_time.time;
    return (.5 - szn(g)) * p.par * .03 * v2(sin(t * .6), sin(t * .43) * .6);
}

fn painter_key(center: v2, size: f32) -> f32 {
    let size_key = clamp(1. - size / max(p.max_sigma, 1e-4), 0., 1.);
    let sweep_key = clamp((center.x + center.y) * 0.5 + (h21(center * 91.7) - .5) * 0.14, 0., 1.);
    return 0.55 * size_key + 0.45 * sweep_key;
}

fn reveal_alpha(idx: u32) -> f32 {
    if (p.draw_progress < 0.) { return 1.; }
    let nr = f32(draw_rank[idx]) / max(f32(p.num_gaussians), 1.);
    let band = mix(0.06, 0.012, p.draw_grow);
    return smoothstep(nr, nr + band, p.draw_progress);
}

fn draw_mult(idx: u32, along: f32, seed: f32) -> f32 {
    if (p.draw_progress < 0.) { return 1.; }
    let lr = reveal_alpha(idx);
    let dir = select(1., -1., fract(seed * .0137) > .5);
    let t = clamp(along * dir * 0.42 + 0.5, 0., 1.);
    let along_mask = smoothstep(lr + 0.05, lr - 0.03, t);
    return lr * mix(1.0, along_mask, p.draw_grow);
}

fn eval_oil(g: GaussianData, uv: v2, idx: u32) -> v4 {
    let d_raw = uv - g.center;
    let c = cos(g.sigma_xy); let s = sin(g.sigma_xy);
    let d = v2(d_raw.x * c + d_raw.y * s, d_raw.y * c - d_raw.x * s);
    let sx = max(g.sigma_xx, .0005); let sy = max(g.sigma_yy, .0005);
    let seed = h21(v2(f32(idx) * .137 + 3.1, f32(idx) * .091 + 7.7)) * 40.;
    let bend = p.curve_amt * (fract(seed * .137) - .5) * 2.;
    let nx = d.x / sx;
    let ny = d.y / sy - bend * min(nx * nx, 4.);
    let pdf0 = .5 * (nx * nx + ny * ny);

    let streak = vn2(v2(nx * .6, ny * 7.) + seed) - .5;
    let esf = select(streak * .35, streak, streak < 0.);
    let pdf = pdf0 * max(1. + p.edge_rag * 2. * esf, 0.4);
    let hard = 1. + (p.hardness - 1.) * clamp(max(sx, sy) / .02, 0., 1.);
    let a = g.opacity * exp(-pow(pdf, hard));

    var col = g.color;
    col *= (1. - p.bristle_amt * .55 * max(0., -streak));
    col += g.color * p.bristle_amt * .25 * max(0.,  streak);
    let relief = clamp(ny, -1., 1.);
    col *= (1. - p.impasto * .5 * max(0., relief));
    col += g.color * p.impasto * .5 * max(0., -relief);
    let a2 = a * draw_mult(idx, nx, seed);
    return v4(clamp(col, v3(0.), v3(2.)), min(.999, a2));
}

fn gabor_bounds(g: GaussianData) -> OBB {
    let s = v2(g.sigma_xx, g.sigma_xy) * 3.;
    let c = cos(g.sigma_yy); let sn = sin(g.sigma_yy);
    let rot = m2(v2(c, sn), v2(-sn, c));
    return OBB(g.center, rot, s);
}

fn gabor_eval(g: GaussianData, uv: v2) -> v3 {
    let dr = uv - g.center;
    let c = cos(g.sigma_yy); let s = sin(g.sigma_yy);
    let dlx = dr.x * c + dr.y * s;
    let dly = dr.y * c - dr.x * s;
    let sx = max(g.sigma_xx, .001); let sy = max(g.sigma_xy, .001);
    let env = exp(-.5 * ((dlx * dlx) / (sx * sx) + (dly * dly) / (sy * sy)));
    let carrier = .5 + .5 * cos(g._padding * dlx + g.gpad0);   // freq, phase
    return g.color * (g.gpad1 * env * carrier);                // amplitude
}

fn gabor_oil_eval(g: GaussianData, uv: v2, idx: u32) -> v3 {
    let dr = uv - g.center;
    let c = cos(g.sigma_yy); let s = sin(g.sigma_yy);
    let dlx = dr.x * c + dr.y * s;
    let dly = dr.y * c - dr.x * s;
    let sx = max(g.sigma_xx, .001); let sy = max(g.sigma_xy, .001);
    let seed = h21(v2(f32(idx) * .137 + 3.1, f32(idx) * .091 + 7.7)) * 40.;
    let bend = p.curve_amt * (fract(seed * .137) - .5) * 2.;
    let nx = dlx / sx;
    let ny = dly / sy - bend * min(nx * nx, 4.);
    let pdf0 = .5 * (nx * nx + ny * ny);
    let streak = vn2(v2(nx * .6, ny * 7.) + seed) - .5;
    let esf = select(streak * .35, streak, streak < 0.);
    let pdf = pdf0 * max(1. + p.edge_rag * 2. * esf, 0.4);   // floor: ragged but never a runaway scratch
    let hard = 1. + (p.hardness - 1.) * clamp(max(sx, sy) / .02, 0., 1.);
    let env = exp(-pow(pdf, hard));
    let carrier = .5 + .5 * cos(g._padding * dlx + g.gpad0);
    var col = g.color;
    col *= (1. - p.bristle_amt * .55 * max(0., -streak));
    col += g.color * p.bristle_amt * .25 * max(0., streak);
    let relief = clamp(ny, -1., 1.);
    col *= (1. - p.impasto * .5 * max(0., relief));
    col += g.color * p.impasto * .5 * max(0., -relief);
    let dm = draw_mult(idx, nx, seed);
    return clamp(col, v3(0.), v3(2.)) * (g.gpad1 * env * carrier * dm);
}

struct GaborGrads { cx:f32, cy:f32, sxx:f32, syy:f32, ang:f32, fr:f32, ph:f32, cr:f32, cg:f32, cb:f32, amp:f32 };

fn gabor_calc_grads(g: GaussianData, uv: v2, go: v3) -> GaborGrads {
    var r: GaborGrads;
    let dr = uv - g.center;
    let c = cos(g.sigma_yy); let s = sin(g.sigma_yy);
    let dlx = dr.x * c + dr.y * s;
    let dly = dr.y * c - dr.x * s;
    let sx = max(g.sigma_xx, .001); let sy = max(g.sigma_xy, .001);
    let vx = sx * sx; let vy = sy * sy;
    let env = exp(-.5 * ((dlx * dlx) / vx + (dly * dly) / vy));
    let psi = g._padding * dlx + g.gpad0;
    let carr = .5 + .5 * cos(psi);
    let dcarr = -.5 * sin(psi);
    let A = g.gpad1 * env * carr;
    let gA = dot(go, g.color);
    let gv = gA * g.gpad1;
    let gdx = gv * (carr * env * (-dlx / vx) + env * dcarr * g._padding);
    let gdy = gv * (carr * env * (-dly / vy));
    r.cx = -(gdx * c - gdy * s);
    r.cy = -(gdx * s + gdy * c);
    r.ang = gdx * dly - gdy * dlx;
    r.sxx = gv * carr * env * dlx * dlx / (vx * sx);
    r.syy = gv * carr * env * dly * dly / (vy * sy);
    r.fr = gv * (env * dcarr * dlx);
    r.ph = gv * (env * dcarr);
    r.cr = go.r * A; r.cg = go.g * A; r.cb = go.b * A;
    r.amp = gA * env * carr;
    return r;
}

// Bitonic sort for sorting Gaussian indices within a tile

// sorts the first n entries (n = power of two)
fn sort(lid:u32, n:u32) {
    workgroupBarrier();
    var k=2u;
    while(k<=n){
        var j=k/2u;
        while(j>0u){
            var i=lid;
            while(i<n){
                let l=i^j;
                if(l>i){
                    let swp = (((i&k)==0u) && (b_idx[i]>b_idx[l])) || 
                              (((i&k)!=0u) && (b_idx[i]<b_idx[l]));
                    if(swp){
                        let t=b_idx[i]; b_idx[i]=b_idx[l]; b_idx[l]=t;
                    }
                }
                i+=WX*WY;
            }
            workgroupBarrier(); j/=2u;
        }
        k*=2u;
    }
}

// Gradient helpers
// Atomic gradient accumulation

fn add_grad(idx:u32, v:f32) {
    if(abs(v)<1e-12){return;}
    loop {
        let old_b = atomicLoad(&g_grad[idx]);
        let new_b = bitcast<u32>(bitcast<f32>(old_b) + v);
        if(atomicCompareExchangeWeak(&g_grad[idx], old_b, new_b).exchanged){ break; }
    }
}

fn reduce_grad3(lid:u32, base:u32, off:u32, val:v3) {
    red_buf[lid] = clamp(val, v3(-100.), v3(100.));
    workgroupBarrier();
    // Parallel reduction over the WG threads (log2(WG) steps)
    var s = WG / 2u;
    while(s>0u){
        if(lid<s){ red_buf[lid] += red_buf[lid+s]; }
        workgroupBarrier(); s/=2u;
    }
    if(lid==0u){
        let r = red_buf[0];
        add_grad(base+off+0u, r.x);
        add_grad(base+off+1u, r.y);
        add_grad(base+off+2u, r.z);
    }
    workgroupBarrier();
}

struct Grads { cx:f32, cy:f32, sxx:f32, sxy:f32, syy:f32, cr:f32, cg:f32, cb:f32, op:f32 };
// This is the core of the training. I manually apply the chain rule here to 
// calculate how much each Gaussian parameter (pos, size, rotation, color) 
// contributed to the pixel error. It includes the tricky rotation derivatives.
fn calc_grads(g:GaussianData, uv:v2, go:v4)->Grads {
    var r: Grads;
    let dr = uv - g.center;
    let c = cos(g.sigma_xy); let s = sin(g.sigma_xy);
    let dl_x = dr.x*c + dr.y*s;
    let dl_y = dr.y*c - dr.x*s;
    
    let sx = max(g.sigma_xx, .001); let sy = max(g.sigma_yy, .001);
    let vx = sx*sx; let vy = sy*sy;
    let dsq = (dl_x*dl_x)/vx + (dl_y*dl_y)/vy;
    let w = exp(-.5*dsq);
    let a = min(.99, g.opacity * w);

    let gc = go.rgb; let ga = go.a;
    var gw = 0.; var g_op = 0.;
    if(a<.99){ gw = ga * g.opacity; g_op = ga * w; }
    
    let gdsq = gw * w * -.5;
    let gdx = gdsq * 2. * dl_x / vx;
    let gdy = gdsq * 2. * dl_y / vy;
    
    // Angular gradient from rotation derivative
    let g_ang = gdx * dl_y + gdy * (-dl_x);
    
    r.cx = -(gdx*c - gdy*s);
    r.cy = -(gdx*s + gdy*c);
    r.sxx = gdsq * (-2. * dl_x * dl_x) / (vx * sx);
    r.syy = gdsq * (-2. * dl_y * dl_y) / (vy * sy);
    r.sxy = g_ang;
    
    r.cr = gc.r; r.cg = gc.g; r.cb = gc.b; r.op = g_op;
    return r;
}

// --- Kernels ---

// Initializes the Gaussians. I scatter them randomly across the screen, 
// sample their initial color from the target image to give them a head start,
// and randomize their sizes.
@compute @workgroup_size(256, 1, 1)
fn init_gaussians(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    // Initialize the whole capacity
    if (i >= MAX_G) { return; }
    if (p.reset_training == 0u && p.iteration > 1u) { return; }

    for (var k = 0u; k < GRADS; k++) { adam_m[i*GRADS+k] = 0.; adam_v[i*GRADS+k] = 0.; }

    let s = f32(p.random_seed);
    let h1 = hash4(v4(f32(i)*.123, s*.456, f32(i)*.789, s*.012));
    let h2 = hash4(v4(s*.345, f32(i)*.678, s*.901, f32(i)*.234));
    let h3 = hash4(v4(f32(i)*.567, s*.234, f32(i)*.890, s*.567));

    if (p.mode != 0u) {
        var gg: GaussianData;
        gg.center = clamp(h1.xy, v2(.001), v2(.999));
        let tcg = textureSampleLevel(t_target, s_target, gg.center, 0.).rgb;
        gg.color = clamp(tcg + (h2.rgb - .5) * .1, v3(0.), v3(1.));
        let across = mix(p.min_sigma, p.max_sigma, h1.z * h1.z);
        gg.sigma_xx = across;                                       // sx
        gg.sigma_xy = mix(p.min_sigma, p.max_sigma, h1.w * h1.w);   // sy
        gg.sigma_yy = (h2.w - .5) * 2. * PI;                        // angle
        gg._padding = min(p.freq_max, (PI / across) * mix(.5, 1., h3.y)); // freq
        gg.gpad0 = h3.z * 2. * PI;                                  // phase
        gg.gpad1 = mix(.02, .10, h3.x);                            // amplitude
        gg.opacity = 0.;
        g_data[i] = gg;
        return;
    }

    var g: GaussianData;
    g.center = clamp(h1.xy, v2(.001), v2(.999));
    
    let tc = textureSampleLevel(t_target, s_target, g.center, 0.).rgb;
    g.color = clamp(tc + (h2.rgb-.5)*.1, v3(0.), v3(1.));
    
    g.sigma_xx = mix(p.min_sigma, p.max_sigma, h1.z*h1.z);
    g.sigma_yy = mix(p.min_sigma, p.max_sigma, h1.w*h1.w);
    g.sigma_xy = (h2.w-.5) * 2. * PI;
    g.opacity = mix(.1, .5, h3.x);
    
    g_data[i] = g;
}

fn render_gabor(gid: vec3<u32>, lid: vec3<u32>, wid: vec3<u32>) {
    let dim = textureDimensions(output);
    let valid = (gid.x < dim.x && gid.y < dim.y);
    let uv = (v2(f32(gid.x), f32(gid.y)) + .5) / v2(f32(dim.x), f32(dim.y));
    let li = lid.x + lid.y * WX;

    if (li == 0u) { b_cnt_atom = 0u; b_cnt = 0u; }
    workgroupBarrier();

    let tl = v2(f32(wid.x*WX), f32(wid.y*WY)) / v2(f32(dim.x), f32(dim.y));
    let th = v2(f32((wid.x+1u)*WX), f32((wid.y+1u)*WY)) / v2(f32(dim.x), f32(dim.y));
    let tb = OBB((tl+th)*.5, m2(v2(1.,0.),v2(0.,1.)), (th-tl)*.5 + .001);

    var i = li;
    while (i < p.num_gaussians) {
        let gg = g_data[i];
        if (reveal_alpha(i) >= 0.004) {
            let rad = 3. * max(gg.sigma_xx, gg.sigma_xy) + p.par * .03;
            if (!aabb_miss(gg.center, rad, tl, th) && obb_hit(gabor_bounds(gg), tb)) {
                let idx = atomicAdd(&b_cnt_atom, 1u);
                if (idx < G_PER_TILE) { b_idx[idx] = i; }
            }
        }
        i += WG;
    }
    workgroupBarrier();
    if (li == 0u) { b_cnt = min(atomicLoad(&b_cnt_atom), G_PER_TILE); }
    workgroupBarrier();

    // display: oil, palette and parallax, never touching training
    let need_disp = p.oil_enable != 0u || p.par > 0. || (p.pal_amt > 0. && p.pal_k >= 2.);
    var acc = v3(0.);
    var acc_oil = v3(0.);
    for (var j = 0u; j < b_cnt; j++) {
        let g = g_data[b_idx[j]];
        acc += gabor_eval(g, uv);
        if (need_disp) {
            var gd = g; gd.color = dcol(g);
            let du = uv - poff(g);
            if (p.oil_enable != 0u) { acc_oil += gabor_oil_eval(gd, du, b_idx[j]); } else { acc_oil += gabor_eval(gd, du); }
        }
    }
    let fin_native = clamp(acc, v3(0.), v3(1.));

    if (p.show_error == 0u && p.draw_progress < 0.) {
        var go = v3(0.);
        if (valid) {
            let tgt = textureSampleLevel(t_target, s_target, uv, 0.).rgb;
            go = loss_grad(fin_native - tgt, f32(dim.x * dim.y));
        }
        for (var j = 0u; j < b_cnt; j++) {
            let gi = b_idx[j];
            let g = g_data[gi];
            var gs = GaborGrads(0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.);
            if (valid) { gs = gabor_calc_grads(g, uv, go); }
            let base = gi * GRADS;
            reduce_grad3(li, base, 0u, v3(gs.cx,  gs.cy,  gs.sxx));
            reduce_grad3(li, base, 3u, v3(gs.syy, gs.ang, gs.fr));
            reduce_grad3(li, base, 6u, v3(gs.ph,  gs.cr,  gs.cg));
            reduce_grad3(li, base, 9u, v3(gs.cb,  gs.amp, 0.));
        }
    }

    // tile error for error-guided respawn
    if (p.show_error == 0u && p.draw_progress < 0.) {
        var te = 0.;
        if (valid) { te = dot(abs(fin_native - textureSampleLevel(t_target, s_target, uv, 0.).rgb), v3(1.)); }
        red_buf[li] = v3(te, 0., 0.);
        workgroupBarrier();
        var sr = WG / 2u;
        while (sr > 0u) { if (li < sr) { red_buf[li] += red_buf[li + sr]; } workgroupBarrier(); sr /= 2u; }
        let ti = wid.y * ((dim.x + WX - 1u) / WX) + wid.x;
        if (li == 0u && ti < PALB) { atomicStore(&err_grid[ti], u32(red_buf[0].x / f32(WG) * 1000.)); }
        workgroupBarrier();
    }

    var out_rgb = fin_native;
    if (need_disp) {
        out_rgb = clamp(acc_oil, v3(0.), v3(1.));
        if (p.oil_enable != 0u && p.canvas_amt > 0.) {
            let cuv = uv * v2(f32(dim.x), f32(dim.y));
            let weave = (sin(cuv.x * 1.3) * .5 + .5) * (sin(cuv.y * 1.3) * .5 + .5);
            let fib = vn2(cuv * .5) - .5;
            let canvas = 1. + p.canvas_amt * (.28 * (weave - .5) + .18 * fib);
            out_rgb = clamp(out_rgb * canvas, v3(0.), v3(1.));
        }
    }
    var finstore = v4(out_rgb, 1.);
    if (p.show_error != 0u && valid) {
        let tgt = textureSampleLevel(t_target, s_target, uv, 0.).rgb;
        finstore = v4(abs(fin_native - tgt) * p.error_scale, 1.);
    }
    if (valid) { textureStore(output, gid.xy, finstore); }
}

// The main engine. It performs tile-based culling and sorting for performance.
// It runs the forward pass to get the pixel color, calculates the error against the target,
// and then immediately runs the backward pass to compute gradients.
@compute @workgroup_size(8, 8, 1)
fn render_display(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let dim = textureDimensions(output);
    let valid = (gid.x < dim.x && gid.y < dim.y);
    let uv = (v2(f32(gid.x), f32(gid.y)) + .5) / v2(f32(dim.x), f32(dim.y));
    let li = lid.x + lid.y * WX;

    if (p.show_target != 0u) {
        if(valid){ textureStore(output, gid.xy, textureSampleLevel(t_target, s_target, uv, 0.)); }
        return;
    }

    if (p.mode != 0u) { render_gabor(gid, lid, wid); return; }

    // Tile Setup
    if (li == 0u) { b_cnt_atom = 0u; b_cnt = 0u; }
    workgroupBarrier();

    let tl = v2(f32(wid.x*WX), f32(wid.y*WY)) / v2(f32(dim.x), f32(dim.y));
    let th = v2(f32((wid.x+1u)*WX), f32((wid.y+1u)*WY)) / v2(f32(dim.x), f32(dim.y));
    let tb = OBB((tl+th)*.5, m2(v2(1.,0.),v2(0.,1.)), (th-tl)*.5 + .001);

    // Culling: the shared big list, then this tile's 64 px cell
    let cw = (dim.x + CELL - 1u) / CELL; let chh = (dim.y + CELL - 1u) / CELL;
    let fall = cw * chh > MAXC;
    let nbig = min(atomicLoad(&bin_cnt[MAXC]), MAX_G);
    let cell = (wid.y * WY / CELL) * cw + wid.x * WX / CELL;
    var ncel = 0u;
    if (!fall) { ncel = min(atomicLoad(&bin_cnt[cell]), CAP); }
    var i = li;
    while (i < nbig + ncel) {
        var gi = 0u;
        if (i < nbig) { gi = bin_idx[MAXC * CAP + i]; } else { gi = bin_idx[cell * CAP + i - nbig]; }
        let gg = g_data[gi];
        let rad = vis_k(gg.opacity) * max(gg.sigma_xx, gg.sigma_yy) + p.par * .03;
        if(!aabb_miss(gg.center, rad, tl, th) && obb_hit(get_bounds(gg), tb)){
            let idx = atomicAdd(&b_cnt_atom, 1u);
            if(idx < GAUSS_TILE){ b_idx[idx] = gi; }
        }
        i += WG;
    }
    workgroupBarrier();

    if (li == 0u) { b_cnt = min(atomicLoad(&b_cnt_atom), GAUSS_TILE); }
    let cnt = workgroupUniformLoad(&b_cnt);
    var n2 = 2u;
    while (n2 < cnt) { n2 *= 2u; }

    // Padding & Sort
    i = li;
    while(i < n2){
        if(i >= cnt){ b_idx[i] = 0xFFFFFFFFu; }
        i += WX*WY;
    }
    workgroupBarrier();
    sort(li, n2);

    var col = v4(0.,0.,0.,1.);
    for(var j=0u; j<b_cnt; j++){
        if(b_idx[j] >= p.num_gaussians){ break; }
        let c = eval_g(g_data[b_idx[j]], uv);
        let rgb = c.rgb * c.a; let T = col.w;
        col = v4(col.rgb + rgb*T, T*(1.-c.a));
        if(col.w < .001){ break; }
    }
    var fin = v4(clamp(col.rgb, v3(0.), v3(1.)), 1.);

    // display composite: oil, palette and parallax never touch training
    let need_disp = p.oil_enable != 0u || p.par > 0. || (p.pal_amt > 0. && p.pal_k >= 2.);
    var oil_rgb = fin.rgb;
    if (need_disp && p.show_error == 0u && p.show_target == 0u) {
        var oil = v4(0., 0., 0., 1.);
        for (var j = 0u; j < b_cnt; j++) {
            if (b_idx[j] >= p.num_gaussians) { break; }
            let gs0 = g_data[b_idx[j]];
            var gs = gs0; gs.color = dcol(gs0);
            let du = uv - poff(gs0);
            var co = eval_g(gs, du);
            if (p.oil_enable != 0u) { co = eval_oil(gs, du, b_idx[j]); }
            let ca = co.a;
            let rgb = co.rgb * ca; let T = oil.w;
            oil = v4(oil.rgb + rgb * T, T * (1. - ca));
            if (oil.w < .001) { break; }
        }
        oil_rgb = clamp(oil.rgb, v3(0.), v3(1.));

        if (p.oil_enable != 0u && p.canvas_amt > 0.) {
            let cuv = uv * v2(f32(dim.x), f32(dim.y));
            let weave = (sin(cuv.x * 1.3) * .5 + .5) * (sin(cuv.y * 1.3) * .5 + .5);
            let fib = vn2(cuv * .5) - .5;
            let canvas = 1. + p.canvas_amt * (.28 * (weave - .5) + .18 * fib);
            oil_rgb = clamp(oil_rgb * canvas, v3(0.), v3(1.));
        }
    }

    if (p.show_target == 0u && p.show_error == 0u && p.draw_progress < 0.) {
        var go = v3(0.);
        if (valid) {
            let tgt = textureSampleLevel(t_target, s_target, uv, 0.).rgb;
            go = loss_grad(fin.rgb - tgt, f32(dim.x*dim.y));
        }

        // Backward pass over front to back alpha compositing
        var T = 1.;
        var acc = v3(0.);
        let C = col.rgb;
        for(var j=0u; j<b_cnt; j++){
            let gi = b_idx[j];
            if(gi >= p.num_gaussians){ continue; }

            let g = g_data[gi];
            let c = eval_g(g, uv);
            let Ti = T;
            acc += c.rgb * c.a * Ti;
            let suffix = C - acc;
            var gs = Grads(0.,0.,0.,0.,0.,0.,0.,0.,0.);

            if(valid && c.a > .0001 && Ti > .001){
                let gc = go * c.a * Ti;
                let oma = max(1. - c.a, 1e-3);
                let ga = dot(go, c.rgb * Ti) - dot(go, suffix) / oma;
                gs = calc_grads(g, uv, v4(gc, ga));
            }
            T = Ti * (1. - c.a);

            let base = gi * 9u;
            reduce_grad3(li, base, 0u, v3(gs.cx,  gs.cy,  gs.sxx));
            reduce_grad3(li, base, 3u, v3(gs.sxy, gs.syy, gs.cr));
            reduce_grad3(li, base, 6u, v3(gs.cg,  gs.cb,  gs.op));
        }
    }

    // tile error for error-guided respawn
    if (p.show_error == 0u && p.draw_progress < 0.) {
        var te = 0.;
        if (valid) { te = dot(abs(fin.rgb - textureSampleLevel(t_target, s_target, uv, 0.).rgb), v3(1.)); }
        red_buf[li] = v3(te, 0., 0.);
        workgroupBarrier();
        var sr = WG / 2u;
        while (sr > 0u) { if (li < sr) { red_buf[li] += red_buf[li + sr]; } workgroupBarrier(); sr /= 2u; }
        let ti = wid.y * ((dim.x + WX - 1u) / WX) + wid.x;
        if (li == 0u && ti < PALB) { atomicStore(&err_grid[ti], u32(red_buf[0].x / f32(WG) * 1000.)); }
        workgroupBarrier();
    }

    // Viz
    if (p.show_error != 0u && valid) {
        let tgt = textureSampleLevel(t_target, s_target, uv, 0.).rgb;
        fin = v4(abs(fin.rgb - tgt) * p.error_scale, 1.);
    } else if (need_disp && p.show_target == 0u && valid) {
        fin = v4(oil_rgb, 1.);
    }

    if (valid) { textureStore(output, gid.xy, fin); }
}

fn update_gabor(i: u32) {
    let b1 = .9; let b2 = .999; let eps = 1e-8;
    let t = f32(p.iteration) + 1.;
    let b1c = 1. - pow(b1, t);
    let b2c = 1. - pow(b2, t);
    let lr_decay = max(.15, 1. / (1. + f32(p.iteration) * p.lr_decay_rate));
    let pos_lr = p.learning_rate * lr_decay;      // center, angle, phase
    let sig_lr = p.sigma_learning_rate * lr_decay; // size, frequency
    let col_lr = p.color_learning_rate * lr_decay;
    let amp_lr = p.opacity_learning_rate * lr_decay;

    var g = g_data[i];
    let bi = i * GRADS;
    let g_cx = bitcast<f32>(atomicLoad(&g_grad[bi+0u]));
    let g_cy = bitcast<f32>(atomicLoad(&g_grad[bi+1u]));
    let g_sx = bitcast<f32>(atomicLoad(&g_grad[bi+2u]));
    let g_sy = bitcast<f32>(atomicLoad(&g_grad[bi+3u]));
    let g_an = bitcast<f32>(atomicLoad(&g_grad[bi+4u]));
    let g_fr = bitcast<f32>(atomicLoad(&g_grad[bi+5u]));
    let g_ph = bitcast<f32>(atomicLoad(&g_grad[bi+6u]));
    let g_r  = bitcast<f32>(atomicLoad(&g_grad[bi+7u]));
    let g_g  = bitcast<f32>(atomicLoad(&g_grad[bi+8u]));
    let g_b  = bitcast<f32>(atomicLoad(&g_grad[bi+9u]));
    let g_am = bitcast<f32>(atomicLoad(&g_grad[bi+10u]));

    var m=adam_m[bi]; var v=adam_v[bi];
    m=b1*m+(1.-b1)*g_cx; v=b2*v+(1.-b2)*g_cx*g_cx; adam_m[bi]=m; adam_v[bi]=v;
    g.center.x -= pos_lr/(sqrt(v/b2c)+eps)*(m/b1c);

    m=adam_m[bi+1u]; v=adam_v[bi+1u];
    m=b1*m+(1.-b1)*g_cy; v=b2*v+(1.-b2)*g_cy*g_cy; adam_m[bi+1u]=m; adam_v[bi+1u]=v;
    g.center.y -= pos_lr/(sqrt(v/b2c)+eps)*(m/b1c);

    m=adam_m[bi+2u]; v=adam_v[bi+2u];
    m=b1*m+(1.-b1)*g_sx; v=b2*v+(1.-b2)*g_sx*g_sx; adam_m[bi+2u]=m; adam_v[bi+2u]=v;
    g.sigma_xx -= sig_lr/(sqrt(v/b2c)+eps)*(m/b1c);        // sx

    m=adam_m[bi+3u]; v=adam_v[bi+3u];
    m=b1*m+(1.-b1)*g_sy; v=b2*v+(1.-b2)*g_sy*g_sy; adam_m[bi+3u]=m; adam_v[bi+3u]=v;
    g.sigma_xy -= sig_lr/(sqrt(v/b2c)+eps)*(m/b1c);        // sy

    m=adam_m[bi+4u]; v=adam_v[bi+4u];
    m=b1*m+(1.-b1)*g_an; v=b2*v+(1.-b2)*g_an*g_an; adam_m[bi+4u]=m; adam_v[bi+4u]=v;
    g.sigma_yy -= pos_lr/(sqrt(v/b2c)+eps)*(m/b1c);        // angle

    m=adam_m[bi+5u]; v=adam_v[bi+5u];
    m=b1*m+(1.-b1)*g_fr; v=b2*v+(1.-b2)*g_fr*g_fr; adam_m[bi+5u]=m; adam_v[bi+5u]=v;
    g._padding -= sig_lr/(sqrt(v/b2c)+eps)*(m/b1c);        // freq

    m=adam_m[bi+6u]; v=adam_v[bi+6u];
    m=b1*m+(1.-b1)*g_ph; v=b2*v+(1.-b2)*g_ph*g_ph; adam_m[bi+6u]=m; adam_v[bi+6u]=v;
    g.gpad0 -= pos_lr/(sqrt(v/b2c)+eps)*(m/b1c);           // phase

    m=adam_m[bi+7u]; v=adam_v[bi+7u];
    m=b1*m+(1.-b1)*g_r; v=b2*v+(1.-b2)*g_r*g_r; adam_m[bi+7u]=m; adam_v[bi+7u]=v;
    g.color.r -= col_lr/(sqrt(v/b2c)+eps)*(m/b1c);

    m=adam_m[bi+8u]; v=adam_v[bi+8u];
    m=b1*m+(1.-b1)*g_g; v=b2*v+(1.-b2)*g_g*g_g; adam_m[bi+8u]=m; adam_v[bi+8u]=v;
    g.color.g -= col_lr/(sqrt(v/b2c)+eps)*(m/b1c);

    m=adam_m[bi+9u]; v=adam_v[bi+9u];
    m=b1*m+(1.-b1)*g_b; v=b2*v+(1.-b2)*g_b*g_b; adam_m[bi+9u]=m; adam_v[bi+9u]=v;
    g.color.b -= col_lr/(sqrt(v/b2c)+eps)*(m/b1c);

    m=adam_m[bi+10u]; v=adam_v[bi+10u];
    m=b1*m+(1.-b1)*g_am; v=b2*v+(1.-b2)*g_am*g_am; adam_m[bi+10u]=m; adam_v[bi+10u]=v;
    g.gpad1 -= amp_lr/(sqrt(v/b2c)+eps)*(m/b1c);           // amplitude

    g.center = clamp(g.center, v2(0.), v2(1.));
    g.sigma_xx = clamp(g.sigma_xx, p.min_sigma, p.max_sigma);
    g.sigma_xy = clamp(g.sigma_xy, p.min_sigma, p.max_sigma);
    g._padding = clamp(g._padding, 0., min(p.freq_max, PI / max(g.sigma_xx, .001)));
    g.color = clamp(g.color, v3(0.), v3(1.));
    g.gpad1 = clamp(g.gpad1, 0., 2.);

    let dead = g.gpad1 <= .02;
    let check = ((p.iteration + i) % 97u == 0u) && (p.reset_training == 0u) && (p.iteration > 60u);
    if (dead && check) {
        let h = hash4(v4(f32(p.iteration), f32(i), g.center.x, g.center.y));
        g.center = err_spawn(h, f32(p.iteration) * .37 + f32(i) * .013);
        let across = mix(p.min_sigma, p.max_sigma, h.x * h.x);
        g.sigma_xx = across;
        g.sigma_xy = mix(p.min_sigma, p.max_sigma, h.y * h.y);
        g.sigma_yy = (h.z - .5) * 2. * PI;
        g._padding = min(p.freq_max, (PI / across) * mix(.5, 1., h.w));
        g.gpad0 = h.x * 2. * PI;
        g.color = textureSampleLevel(t_target, s_target, g.center, 0.).rgb;
        g.gpad1 = .03;
        for (var k=0u; k<GRADS; k++) { adam_m[bi+k]=0.; adam_v[bi+k]=0.; }
    }
    g_data[i] = g;
}

// 3. Update (Adam)
@compute @workgroup_size(256, 1, 1)
fn update_gaussians(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.num_gaussians) { return; }
    if (p.draw_progress >= 0.) { return; }

    if (p.mode != 0u) { update_gabor(i); return; }

    let b1 = .9; let b2 = .999; let eps = 1e-8;
    let t = f32(p.iteration) + 1.;
    let b1c = 1. - pow(b1, t);
    let b2c = 1. - pow(b2, t);
    let lr_decay = max(.04, 1. / (1. + f32(p.iteration) * p.lr_decay_rate));
    let pos_lr = p.learning_rate * lr_decay;
    let sig_lr = p.sigma_learning_rate * lr_decay;
    let col_lr = p.color_learning_rate * lr_decay;
    let op_lr  = p.opacity_learning_rate * lr_decay;

    var g = g_data[i];

    let bi = i * 9u; 
    let g_cx = bitcast<f32>(atomicLoad(&g_grad[bi+0u]));
    let g_cy = bitcast<f32>(atomicLoad(&g_grad[bi+1u]));
    let g_sx = bitcast<f32>(atomicLoad(&g_grad[bi+2u]));
    let g_sa = bitcast<f32>(atomicLoad(&g_grad[bi+3u]));
    let g_sy = bitcast<f32>(atomicLoad(&g_grad[bi+4u]));
    let g_r = bitcast<f32>(atomicLoad(&g_grad[bi+5u]));
    let g_g = bitcast<f32>(atomicLoad(&g_grad[bi+6u]));
    let g_b = bitcast<f32>(atomicLoad(&g_grad[bi+7u]));
    let g_op = bitcast<f32>(atomicLoad(&g_grad[bi+8u]));

    // Adam Step - Unrolled for per-param update
    // Center X
    var m=adam_m[bi]; var v=adam_v[bi];
    m = b1*m + (1.-b1)*g_cx; v = b2*v + (1.-b2)*g_cx*g_cx;
    adam_m[bi]=m; adam_v[bi]=v;
    g.center.x -= pos_lr/(sqrt(v/b2c)+eps)*(m/b1c);

    // Center Y
    m=adam_m[bi+1u]; v=adam_v[bi+1u];
    m = b1*m + (1.-b1)*g_cy; v = b2*v + (1.-b2)*g_cy*g_cy;
    adam_m[bi+1u]=m; adam_v[bi+1u]=v;
    g.center.y -= pos_lr/(sqrt(v/b2c)+eps)*(m/b1c);

    // Sigma XX
    m=adam_m[bi+2u]; v=adam_v[bi+2u];
    m = b1*m + (1.-b1)*g_sx; v = b2*v + (1.-b2)*g_sx*g_sx;
    adam_m[bi+2u]=m; adam_v[bi+2u]=v;
    g.sigma_xx -= sig_lr/(sqrt(v/b2c)+eps)*(m/b1c);

    // Sigma XY (Angle)
    m=adam_m[bi+3u]; v=adam_v[bi+3u];
    m = b1*m + (1.-b1)*g_sa; v = b2*v + (1.-b2)*g_sa*g_sa;
    adam_m[bi+3u]=m; adam_v[bi+3u]=v;
    g.sigma_xy -= pos_lr/(sqrt(v/b2c)+eps)*(m/b1c);

    // Sigma YY
    m=adam_m[bi+4u]; v=adam_v[bi+4u];
    m = b1*m + (1.-b1)*g_sy; v = b2*v + (1.-b2)*g_sy*g_sy;
    adam_m[bi+4u]=m; adam_v[bi+4u]=v;
    g.sigma_yy -= sig_lr/(sqrt(v/b2c)+eps)*(m/b1c);

    // Colors
    m=adam_m[bi+5u]; v=adam_v[bi+5u]; // R
    m=b1*m+(1.-b1)*g_r; v=b2*v+(1.-b2)*g_r*g_r;
    adam_m[bi+5u]=m; adam_v[bi+5u]=v;
    g.color.r -= col_lr/(sqrt(v/b2c)+eps)*(m/b1c);

    m=adam_m[bi+6u]; v=adam_v[bi+6u]; // G
    m=b1*m+(1.-b1)*g_g; v=b2*v+(1.-b2)*g_g*g_g;
    adam_m[bi+6u]=m; adam_v[bi+6u]=v;
    g.color.g -= col_lr/(sqrt(v/b2c)+eps)*(m/b1c);

    m=adam_m[bi+7u]; v=adam_v[bi+7u]; // B
    m=b1*m+(1.-b1)*g_b; v=b2*v+(1.-b2)*g_b*g_b;
    adam_m[bi+7u]=m; adam_v[bi+7u]=v;
    g.color.b -= col_lr/(sqrt(v/b2c)+eps)*(m/b1c);

    // Opacity
    m=adam_m[bi+8u]; v=adam_v[bi+8u];
    m = b1*m + (1.-b1)*g_op; v = b2*v + (1.-b2)*g_op*g_op;
    adam_m[bi+8u]=m; adam_v[bi+8u]=v;
    g.opacity -= op_lr/(sqrt(v/b2c)+eps)*(m/b1c);

    // Constraints
    g.center = clamp(g.center, v2(0.), v2(1.));
    g.sigma_xx = clamp(g.sigma_xx, p.min_sigma, p.max_sigma);
    g.sigma_yy = clamp(g.sigma_yy, p.min_sigma, p.max_sigma);
    g.color = clamp(g.color, v3(0.), v3(1.));
    g.opacity = clamp(g.opacity, .01, .99);

    // Densification (Teleport logic)
    // If invisible or huge lazy blob, kill it and respawn
    // error-guided: weak splats are recycled too, and more often
    let eg = p.dens > 0.;
    let dead = g.opacity <= select(.02, .05, eg);
    let huge = (g.sigma_xx >= p.max_sigma * .98 || g.sigma_yy >= p.max_sigma * .98) && g.opacity < .1;
    let check = ((p.iteration + i) % select(149u, 61u, eg) == 0u) && (p.reset_training == 0u) && (p.iteration > 60u);

    if ((dead || huge) && check) {
        let h = hash4(v4(f32(p.iteration), f32(i), g.center.x, g.center.y));
        g.center = err_spawn(h, f32(p.iteration) * .37 + f32(i) * .013);
        let rs = select(p.min_sigma * 1.5, max(p.min_sigma * 1.5, .004), eg);
        g.sigma_xx = rs;
        g.sigma_yy = rs;
        g.sigma_xy = (h.z-.5) * 2. * PI;
        g.color = textureSampleLevel(t_target, s_target, g.center, 0.).rgb;
        g.opacity = .06;
        
        // Reset momentum or it flies away
        for(var k=0u; k<9u; k++){
            adam_m[bi+k] = 0.; adam_v[bi+k] = 0.;
        }
    }
    g_data[i] = g;
}

// 4. Clear
@compute @workgroup_size(256, 1, 1)
fn clear_gradients(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.num_gaussians * GRADS) { return; }   // GRADS=12 covers both modes (gaussian uses 9)
    atomicStore(&g_grad[i], 0u);
}

// binning: each splat registers in the 64 px cells it covers, or in the shared big list
@compute @workgroup_size(256, 1, 1)
fn bin_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x <= MAXC) { atomicStore(&bin_cnt[gid.x], 0u); }
}

@compute @workgroup_size(256, 1, 1)
fn bin_splats(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.num_gaussians || p.mode != 0u) { return; }
    let g = g_data[i];
    let vk = vis_k(g.opacity);
    if (vk <= 0. || reveal_alpha(i) < .004) { return; }
    let dim = textureDimensions(output); let df = v2(f32(dim.x), f32(dim.y));
    let cw = (dim.x + CELL - 1u) / CELL; let chh = (dim.y + CELL - 1u) / CELL;
    let rad = vk * max(g.sigma_xx, g.sigma_yy) + p.par * .03;
    let c0 = vec2<u32>(clamp((g.center - rad) * df / f32(CELL), v2(0.), v2(f32(cw - 1u), f32(chh - 1u))));
    let c1 = vec2<u32>(clamp((g.center + rad) * df / f32(CELL), v2(0.), v2(f32(cw - 1u), f32(chh - 1u))));
    let n = (c1.x - c0.x + 1u) * (c1.y - c0.y + 1u);
    if (n > BIGC || cw * chh > MAXC) {
        let k = atomicAdd(&bin_cnt[MAXC], 1u);
        if (k < MAX_G) { bin_idx[MAXC * CAP + k] = i; }
        return;
    }
    for (var y = c0.y; y <= c1.y; y++) { for (var x = c0.x; x <= c1.x; x++) {
        let c = y * cw + x;
        let k = atomicAdd(&bin_cnt[c], 1u);
        if (k < CAP) { bin_idx[c * CAP + k] = i; }
    }}
}

// palette: online k-means over splat colours (display only)
@compute @workgroup_size(256, 1, 1)
fn pal_assign(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.num_gaussians || p.pal_k < 2. || p.pal_amt <= 0.) { return; }
    let K = u32(clamp(p.pal_k, 2., 16.));
    let c = g_data[i].color;
    var bk = 0u; var bd = 1e9;
    for (var k = 0u; k < K; k++) { let e = c - palc(k); let dd = dot(e, e); if (dd < bd) { bd = dd; bk = k; } }
    if (p.mode == 0u) { g_data[i]._padding = f32(bk); } else { g_data[i].opacity = f32(bk); }
    let a = PALB + 64u + bk * 4u;
    atomicAdd(&err_grid[a], u32(c.r * 1024.)); atomicAdd(&err_grid[a+1u], u32(c.g * 1024.));
    atomicAdd(&err_grid[a+2u], u32(c.b * 1024.)); atomicAdd(&err_grid[a+3u], 1u);
}

@compute @workgroup_size(16, 1, 1)
fn pal_update(@builtin(local_invocation_id) lid: vec3<u32>) {
    let k = lid.x;
    if (p.pal_k < 2. || p.pal_amt <= 0.) { return; }
    let a = PALB + 64u + k * 4u;
    let n = atomicExchange(&err_grid[a+3u], 0u);
    let sm = v3(f32(atomicExchange(&err_grid[a], 0u)), f32(atomicExchange(&err_grid[a+1u], 0u)), f32(atomicExchange(&err_grid[a+2u], 0u))) / 1024.;
    var c = palc(k);
    // empty or fresh: reseed from a splat
    if (n == 0u || p.iteration <= 2u) { c = g_data[(k * 2654435761u + p.iteration * 97u) % max(p.num_gaussians, 1u)].color; }
    else { c = sm / f32(n); }
    let b = PALB + k * 4u;
    atomicStore(&err_grid[b], bitcast<u32>(c.r)); atomicStore(&err_grid[b+1u], bitcast<u32>(c.g)); atomicStore(&err_grid[b+2u], bitcast<u32>(c.b));
}

@compute @workgroup_size(256, 1, 1)
fn compute_draw_rank(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.num_gaussians) { return; }
    if (p.draw_prepare == 0u) { return; }
    let gi = g_data[i];
    let si = select(max(gi.sigma_xx, gi.sigma_yy), max(gi.sigma_xx, gi.sigma_xy), p.mode != 0u);
    let ki = painter_key(gi.center, si);
    var rank = 0u;
    for (var j = 0u; j < p.num_gaussians; j++) {
        let gj = g_data[j];
        let sj = select(max(gj.sigma_xx, gj.sigma_yy), max(gj.sigma_xx, gj.sigma_xy), p.mode != 0u);
        let kj = painter_key(gj.center, sj);
        if (kj < ki || (kj == ki && j < i)) { rank += 1u; }
    }
    draw_rank[i] = rank;
}
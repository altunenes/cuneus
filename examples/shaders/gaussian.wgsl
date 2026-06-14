// Enes Altun, 2025; MIT License
// 2D Gaussian Splatting with Real-time Training
// https://shader-slang.org/blog/2025/04/04/neural-gfx-in-an-afternoon/
// https://shader-slang.org/blog/2025/04/30/neural-graphics-first-principles-performance/

alias v2 = vec2<f32>;
alias v3 = vec3<f32>;
alias v4 = vec4<f32>;
alias m2 = mat2x2<f32>;

const PI = 3.14159265;
const MAX_G = 20000u;
const G_PER_TILE = 512u;
const WX = 8u;
const WY = 8u;
const WG = WX * WY;


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
    _reserved1: f32,
    random_seed: u32,
    iteration: u32,
    sigma_learning_rate: f32,
    _padding0: u32,
    _padding1: u32,
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
    color: v3,
    opacity: f32,
};
@group(3) @binding(0) var<storage, read_write> g_data: array<GaussianData>;

@group(3) @binding(1) var<storage, read_write> g_grad: array<atomic<u32>>; 
@group(3) @binding(2) var<storage, read_write> adam_m: array<f32>;
@group(3) @binding(3) var<storage, read_write> adam_v: array<f32>;

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

// gauss bounds
fn get_bounds(g:GaussianData)->OBB {
    let s = v2(g.sigma_xx, g.sigma_yy) * 3.; // 3 std devs
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

// Bitonic sort for sorting Gaussian indices within a tile

fn sort(lid:u32) {
    workgroupBarrier();
    var k=2u;
    while(k<=G_PER_TILE){
        var j=k/2u;
        while(j>0u){
            var i=lid;
            while(i<G_PER_TILE){
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

    let s = f32(p.random_seed);
    let h1 = hash4(v4(f32(i)*.123, s*.456, f32(i)*.789, s*.012));
    let h2 = hash4(v4(s*.345, f32(i)*.678, s*.901, f32(i)*.234));
    let h3 = hash4(v4(f32(i)*.567, s*.234, f32(i)*.890, s*.567));

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
    let uv = v2(f32(gid.x), f32(gid.y)) / v2(f32(dim.x), f32(dim.y));
    let li = lid.x + lid.y * WX;

    if (p.show_target != 0u) {
        if(valid){ textureStore(output, gid.xy, textureSampleLevel(t_target, s_target, uv, 0.)); }
        return;
    }

    // Tile Setup
    if (li == 0u) { b_cnt_atom = 0u; b_cnt = 0u; }
    workgroupBarrier();

    let tl = v2(f32(wid.x*WX), f32(wid.y*WY)) / v2(f32(dim.x), f32(dim.y));
    let th = v2(f32((wid.x+1u)*WX), f32((wid.y+1u)*WY)) / v2(f32(dim.x), f32(dim.y));
    let tb = OBB((tl+th)*.5, m2(v2(1.,0.),v2(0.,1.)), (th-tl)*.5 + .001);

    // Culling
    var i = li;
    while(i < p.num_gaussians){
        let gg = g_data[i];
        let rad = 3. * max(gg.sigma_xx, gg.sigma_yy);
        if(!aabb_miss(gg.center, rad, tl, th) && obb_hit(get_bounds(gg), tb)){
            let idx = atomicAdd(&b_cnt_atom, 1u);
            if(idx < G_PER_TILE){ b_idx[idx] = i; }
        }
        i += WG;
    }
    workgroupBarrier();
    
    if (li == 0u) { b_cnt = min(atomicLoad(&b_cnt_atom), G_PER_TILE); }
    workgroupBarrier();

    // Padding & Sort
    i = li;
    while(i < G_PER_TILE){
        if(i >= b_cnt){ b_idx[i] = 0xFFFFFFFFu; }
        i += WX*WY;
    }
    workgroupBarrier();
    sort(li);

    var col = v4(0.,0.,0.,1.);
    for(var j=0u; j<b_cnt; j++){
        if(b_idx[j] >= p.num_gaussians){ break; }
        let c = eval_g(g_data[b_idx[j]], uv);
        let rgb = c.rgb * c.a; let T = col.w;
        col = v4(col.rgb + rgb*T, T*(1.-c.a));
        if(col.w < .001){ break; }
    }
    var fin = v4(clamp(col.rgb, v3(0.), v3(1.)), 1.);

    if (p.show_target == 0u && p.show_error == 0u) {
        var go = v3(0.);
        if (valid) {
            let tgt = textureSampleLevel(t_target, s_target, uv, 0.).rgb;
            go = 2. * (fin.rgb - tgt) / f32(dim.x*dim.y);
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

    // Viz
    if (p.show_error != 0u && valid) {
        let tgt = textureSampleLevel(t_target, s_target, uv, 0.).rgb;
        fin = v4(abs(fin.rgb - tgt) * p.error_scale, 1.);
    }

    if (valid) { textureStore(output, gid.xy, fin); }
}

// 3. Update (Adam)
@compute @workgroup_size(256, 1, 1)
fn update_gaussians(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.num_gaussians) { return; }

    let b1 = .9; let b2 = .999; let eps = 1e-8;
    let t = f32(p.iteration) + 1.;
    let b1c = 1. - pow(b1, t);
    let b2c = 1. - pow(b2, t);
    let lr_decay = max(.15, 1. / (1. + f32(p.iteration) * .0008));
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
    let dead = g.opacity <= .02;
    let huge = (g.sigma_xx >= p.max_sigma * .98 || g.sigma_yy >= p.max_sigma * .98) && g.opacity < .1;
    let check = (p.iteration % 30u == 0u) && (p.reset_training == 0u) && (p.iteration > 60u);

    if ((dead || huge) && check) {
        let h = hash4(v4(f32(p.iteration), f32(i), g.center.x, g.center.y));
        g.center = clamp(h.xy, v2(.05), v2(.95));
        g.sigma_xx = p.min_sigma * 1.5;
        g.sigma_yy = p.min_sigma * 1.5;
        g.sigma_xy = (h.z-.5) * 2. * PI;
        g.color = textureSampleLevel(t_target, s_target, g.center, 0.).rgb;
        g.opacity = .15;
        
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
    if (i >= p.num_gaussians * 9u) { return; }
    atomicStore(&g_grad[i], 0u);
}
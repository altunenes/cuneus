// Block Game, Enes Altun, 2025, MIT License

struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
@group(0) @binding(0) var<uniform> u_time: TimeUniform;

// Group 1: Output texture + game/audio uniform
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;

struct GameUniform {
    camera_height: f32,
    camera_angle: f32,
    camera_scale: f32,
    volume: f32,
    sample_offset: u32,
    samples_to_generate: u32,
    sample_rate: f32,
    _pad: f32,
};
@group(1) @binding(1) var<uniform> u_game: GameUniform;

const TAU: f32 = 6.2831853;

// Group 2: Engine Resources (mouse, fonts, storage)
struct MouseUniform {
    position: vec2<f32>,         
    click_position: vec2<f32>,   
    wheel: vec2<f32>,            
    buttons: vec2<u32>,          
};
@group(2) @binding(0) var<uniform> u_mouse: MouseUniform;

// Group 2: Engine Resources continued (fonts + game storage)
struct FontUniforms {
    atlas_size: vec2<f32>,
    char_size: vec2<f32>,
    screen_size: vec2<f32>,
    grid_size: vec2<f32>,
};
@group(2) @binding(1) var<uniform> font_texture_uniform: FontUniforms;
@group(2) @binding(2) var t_font_texture_atlas: texture_2d<f32>;
// Group 2 binding 3: real PCM audio sample buffer (interleaved stereo f32).
@group(2) @binding(3) var<storage, read_write> audio_buffer: array<f32>;
// Group 3: game state storage (blocks, score, camera-follow, event timestamps).
@group(3) @binding(0) var<storage, read_write> game_data: array<f32>;

const FONT_SPACING: f32 = 2.0;

// render single character
fn ch(pp: vec2<f32>, pos: vec2<f32>, code: u32, size: f32) -> f32 {
    let char_size_pixels = vec2<f32>(size, size);
    let relative_pos = pp - pos;

    // Check bounds
    if (relative_pos.x < 0.0 || relative_pos.x >= char_size_pixels.x ||
        relative_pos.y < 0.0 || relative_pos.y >= char_size_pixels.y) {
        return 0.0;
    }

    // Calculate UV coordinates within the character cell
    let local_uv = relative_pos / char_size_pixels;

    // calc char pos in atlas grid (16x16)
    let grid_x = code % 16u;
    let grid_y = code / 16u;


    let padding = 0.05;
    let padded_uv = local_uv * (1.0 - 2.0 * padding) + vec2<f32>(padding);

    // atlas UV coords
    let cell_size_uv = vec2<f32>(1.0 / 16.0, 1.0 / 16.0);
    let cell_offset = vec2<f32>(f32(grid_x), f32(grid_y)) * cell_size_uv;
    let final_uv = cell_offset + padded_uv * cell_size_uv;

    // sample font atlas with textureLoad
    let atlas_coord = vec2<i32>(
        i32(final_uv.x * font_texture_uniform.atlas_size.x),
        i32(final_uv.y * font_texture_uniform.atlas_size.y)
    );
    let sample = textureLoad(t_font_texture_atlas, atlas_coord, 0);

    // red channel font data + anti-alias
    let font_alpha = sample.r * 0.8;
    return smoothstep(0.1, 0.9, font_alpha);
}

// char spacing
fn adv(size: f32) -> f32 {
    return size * (1.0 / FONT_SPACING);
}

// render number
fn num(pp: vec2<f32>, pos: vec2<f32>, number: u32, size: f32) -> f32 {
    let char_advance = adv(size);
    var alpha = 0.0;
    var temp_num = number;
    var digit_count = 0u;

    // Count digits
    if (temp_num == 0u) {
        digit_count = 1u;
    } else {
        var count_temp = temp_num;
        while (count_temp > 0u) {
            count_temp = count_temp / 10u;
            digit_count++;
        }
    }

    // Render digits from right to left
    temp_num = number;
    for (var i = 0u; i < digit_count; i++) {
        let digit = temp_num % 10u;
        let digit_char_code = 48u + digit;
        let digit_pos = pos + vec2<f32>(f32(digit_count - 1u - i) * char_advance, 0.0);
        let char_alpha = ch(pp, digit_pos, digit_char_code, size);
        alpha = max(alpha, char_alpha);
        temp_num = temp_num / 10u;
    }

    return alpha;
}

// a word of up to 16 ASCII codes
fn word(pp: vec2<f32>, pos: vec2<f32>, c: array<u32, 16>, n: u32, size: f32) -> f32 {
    var a = 0.0;
    for (var i = 0u; i < n; i++) {
        a = max(a, ch(pp, pos + vec2<f32>(f32(i) * adv(size), 0.0), c[i], size));
    }
    return a;
}

// the same word centred on x; glyphs sit in the middle of their cells
fn centered(pp: vec2<f32>, x: f32, y: f32, c: array<u32, 16>, n: u32, size: f32) -> f32 {
    return word(pp, vec2<f32>(x - (f32(n - 1u) * adv(size) + size) * 0.5, y), c, n, size);
}

fn digits(n: u32) -> u32 {
    var c = 1u;
    var v = n / 10u;
    while (v > 0u) { v /= 10u; c++; }
    return c;
}

// BLOCK TOWER, CLICK TO START, PERFECT MATCH =, MORE POINTS, SCORE:, GAME OVER, CLICK TO RESTART, PERFECT!
const W_TITLE = array<u32, 16>(66u, 76u, 79u, 67u, 75u, 32u, 84u, 79u, 87u, 69u, 82u, 32u, 32u, 32u, 32u, 32u);
const W_START = array<u32, 16>(67u, 76u, 73u, 67u, 75u, 32u, 84u, 79u, 32u, 83u, 84u, 65u, 82u, 84u, 32u, 32u);
const W_MATCH = array<u32, 16>(80u, 69u, 82u, 70u, 69u, 67u, 84u, 32u, 77u, 65u, 84u, 67u, 72u, 32u, 61u, 32u);
const W_POINTS = array<u32, 16>(77u, 79u, 82u, 69u, 32u, 80u, 79u, 73u, 78u, 84u, 83u, 32u, 32u, 32u, 32u, 32u);
const W_SCORE = array<u32, 16>(83u, 67u, 79u, 82u, 69u, 58u, 32u, 32u, 32u, 32u, 32u, 32u, 32u, 32u, 32u, 32u);
const W_OVER = array<u32, 16>(71u, 65u, 77u, 69u, 32u, 79u, 86u, 69u, 82u, 32u, 32u, 32u, 32u, 32u, 32u, 32u);
const W_RESTART = array<u32, 16>(67u, 76u, 73u, 67u, 75u, 32u, 84u, 79u, 32u, 82u, 69u, 83u, 84u, 65u, 82u, 84u);
const W_PERFECT = array<u32, 16>(80u, 69u, 82u, 70u, 69u, 67u, 84u, 33u, 32u, 32u, 32u, 32u, 32u, 32u, 32u, 32u);

// one unit of screen size: 1 at the 600x800 reference window, following the smaller side
fn ui_scale(ss: vec2<f32>) -> f32 { return min(ss.x / 600.0, ss.y / 800.0); }

// game indices
const O = array<u32,9>(0,1,2,3,4,5,6,7,8); // state,score,block,click,cam_y,cam_h,cam_a,cam_s,perf_time
const BD = 100u; // block data start
const BS = 10u;  // block size

// stuff
struct Block { p: vec3<f32>, s: vec3<f32>, c: vec3<f32>, perf: f32, };
struct Mat { alb: vec3<f32>, r: f32, m: f32, f: f32, }; // material
struct Light { p: vec3<f32>, c: vec3<f32>, i: f32, }; // light


// get block material
fn mat(id: u32) -> Mat {
    let h = fract(f32(id) * .618034);
    var alb: vec3<f32>;
    
    if (h < .33) { alb = vec3(.8, .2 + h * 1.8, .1); }
    else if (h < .66) { alb = vec3(.1 + (.66 - h) * 2.1, .8, .2); }
    else { alb = vec3(.2, .1 + (h - .66) * 2.1, .9); }
    
    return Mat(alb, .1 + h * .7, select(.1, .8, id % 3u == 0u), .04);
}

// ggx stuff:
// note that, ggx Trowbridge and Reitz specular model approximation inspired by: https://www.shadertoy.com/view/dltGWl,  Poisson, 2023: "subsurface lighting model"
// But also see: for pretty lightings: https://www.shadertoy.com/view/cl3GWr, Poisson, 2023
fn dggx(nh: f32, r: f32) -> f32 {
    let a2 = r * r * r * r;
    let d = nh * nh * (a2 - 1.) + 1.;
    return a2 / (3.14159 * d * d);
}
 // geometry smith
fn gsmith(nv: f32, nl: f32, r: f32) -> f32 {
    let k = (r + 1.) * (r + 1.) *.125;
    return (nl / (nl * (1. - k) + k)) * (nv / (nv * (1. - k) + k));
}

// fresnel
fn fschlick(ct: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3(1.) - f0) * pow(clamp(1. - ct, 0., 1.), 5.);
}

// cook torrance lighting
fn ct(ld: vec3<f32>, vd: vec3<f32>, n: vec3<f32>, m: Mat, lc: vec3<f32>, li: f32) -> vec3<f32> {
    let h = normalize(ld + vd);
    let nl = max(dot(n, ld), 0.);
    let nv = max(dot(n, vd), 0.);
    let nh = max(dot(n, h), 0.);
    let hv = max(dot(h, vd), 0.);
    
    let d = dggx(nh, m.r);
    let g = gsmith(nv, nl, m.r);
    let f0 = mix(vec3(m.f), m.alb, m.m);
    let f = fschlick(hv, f0);
    
    let spec = d * g * f / (4. * nv * nl + .0001);
    let kd = (vec3(1.) - f) * (1. - m.m);
    let diff = kd * m.alb / 3.14159;
    
    return (diff + spec) * lc * li * nl;
}

// ambient calc
fn amb(m: Mat, ao: f32) -> vec3<f32> {
    return m.alb * vec3(.1, .15, .25) * ao;
}

// game state getters/setters
fn gs() -> u32 { return u32(game_data[O[0]]); } // get state
fn ss(s: u32) { game_data[O[0]] = f32(s); } // set state
fn gsc() -> u32 { return u32(game_data[O[1]]); } // get score  
fn ssc(s: u32) { game_data[O[1]] = f32(s); } // set score
fn gcb() -> u32 { return u32(game_data[O[2]]); } // get current block
fn scb(b: u32) { game_data[O[2]] = f32(b); } // set current block
fn gct() -> bool { return game_data[O[3]] > .5; } // get click triggered
fn sct(t: bool) { game_data[O[3]] = select(0., 1., t); } // set click triggered
fn gcy() -> f32 { return game_data[O[4]]; } // get camera y
fn scy(y: f32) { game_data[O[4]] = y; } // set camera y
fn gch() -> f32 { return u_game.camera_height; } // camera
fn gca() -> f32 { return u_game.camera_angle; }
fn gcs() -> f32 { return u_game.camera_scale; }
// events store their start sample as float bits (+EV0 keeps them normal floats), so ages stay sample-exact
const EV0: u32 = 16777216u;
fn atime() -> f32 { return bitcast<f32>(u_game.sample_offset + EV0); }
fn gpt() -> f32 { return game_data[O[8]]; } // get perfect time
fn spt(t: f32) { game_data[O[8]] = t; } // set perfect time

// camera follows the tower in world units, once it is a few blocks tall
fn updcam() {
    let cb = gcb();
    if (cb > 0u) {
        scy(mix(gcy(), max(f32(cb) * .6 - 2.4, 0.), .1));
    }
}

// get stored block
fn gb(id: u32) -> Block {
    if (id >= 50u) { return Block(vec3(0.), vec3(0.), vec3(0.), 0.); }
    
    let i = BD + id * BS;
    return Block(
        vec3(game_data[i], game_data[i+1u], game_data[i+2u]),
        vec3(game_data[i+3u], game_data[i+4u], game_data[i+5u]), 
        vec3(game_data[i+6u], game_data[i+7u], game_data[i+8u]),
        game_data[i+9u]
    );
}

// set stored block  
fn sb(id: u32, b: Block) {
    if (id < 50u) {
        let i = BD + id * BS;
        game_data[i] = b.p.x;   game_data[i+1u] = b.p.y; game_data[i+2u] = b.p.z;
        game_data[i+3u] = b.s.x; game_data[i+4u] = b.s.y; game_data[i+5u] = b.s.z;
        game_data[i+6u] = b.c.x; game_data[i+7u] = b.c.y; game_data[i+8u] = b.c.z;
        game_data[i+9u] = b.perf;
    }
}

// world to isometric
fn w2i(wp: vec3<f32>) -> vec2<f32> {
    let ap = wp - vec3(0., gch(), 0.);
    let a = gca();
    let rp = vec3(ap.x * cos(a) - ap.z * sin(a), ap.y, ap.x * sin(a) + ap.z * cos(a));
    return vec2((rp.x - rp.z) * .866, (rp.x + rp.z) * .5 - rp.y);
}

fn cross2(a: vec2<f32>, b: vec2<f32>) -> f32 { return a.x * b.y - a.y * b.x; }

// point in quad
fn piq(p: vec2<f32>, v0: vec2<f32>, v1: vec2<f32>, v2: vec2<f32>, v3: vec2<f32>) -> bool {
    let d = vec4(cross2(v1-v0, p-v0), cross2(v2-v1, p-v1), cross2(v3-v2, p-v2), cross2(v0-v3, p-v3));
    return all(d >= vec4(0.)) || all(d <= vec4(0.));
}

fn lights() -> array<Light, 3> {
    return array<Light, 3>(
        Light(normalize(vec3(2., 3., 1.5)), vec3(1., .95, .8), 2.5),
        Light(normalize(vec3(-1.5, 2., -1.)), vec3(.4, .6, .9), 1.2),
        Light(normalize(vec3(0., 1., -2.)), vec3(.9, .7, .3), .8)
    );
}

// simple ao
fn ao(wp: vec3<f32>, n: vec3<f32>) -> f32 {
    return .3 + .7 * (clamp(wp.y / 10., 0., 1.) + clamp(dot(n, vec3(0., 1., 0.)), 0., 1.) * .5);
}

// render block with lighting  
fn rbl(pp: vec2<f32>, b: Block, ss: vec2<f32>, id: u32) -> vec3<f32> {
    if (any(b.s <= vec3(0.))) { return vec3(0.); }
    
    let m = mat(id);
    var fm = m;
    if (b.perf > .5) { 
        let pulse = sin(u_time.time * 6.) * .3 + .7;
        fm = Mat(m.alb + vec3(.3, .2, .1) * pulse, m.r * .5, m.m, m.f + .2); 
    }
    
    let scale = gcs() * ui_scale(ss);
    let co = vec2(ss.x * .5, ss.y * .7 + gcy() * scale);
     let cs = w2i(b.p + vec3(0., b.s.y * .5, 0.)) * scale + co;
    let rad = (b.s.x + b.s.z + b.s.y + 1.) * scale;
    if (distance(pp, cs) > rad) { return vec3(0.); }

    let hw = b.s.x * .5;
    let hd = b.s.z * .5;
    
    // bottom/top corners
    let bc = array<vec3<f32>, 4>(
        b.p + vec3(-hw, 0., -hd), b.p + vec3(hw, 0., -hd),
        b.p + vec3(hw, 0., hd), b.p + vec3(-hw, 0., hd)
    );
    let tc = array<vec3<f32>, 4>(
        b.p + vec3(-hw, b.s.y, -hd), b.p + vec3(hw, b.s.y, -hd),
        b.p + vec3(hw, b.s.y, hd), b.p + vec3(-hw, b.s.y, hd)
    );
    
    // to screen space
    var bs = array<vec2<f32>, 4>();
    var ts = array<vec2<f32>, 4>();
    for (var i = 0u; i < 4u; i++) {
        bs[i] = w2i(bc[i]) * scale + co;
        ts[i] = w2i(tc[i]) * scale + co;
    }
    
    let ls = lights();
    var fc = vec3(0.);
    var hit = false;
    let vd = normalize(vec3(0., 0., -1.));
    
    // check faces - top
    if (piq(pp, ts[0], ts[1], ts[2], ts[3])) {
        let n = vec3(0., 1., 0.);
        let wsp = b.p + vec3(0., b.s.y, 0.);
        var lc = amb(fm, ao(wsp, n));
        for (var i = 0; i < 3; i++) { lc += ct(ls[i].p, vd, n, fm, ls[i].c, ls[i].i); }
        fc = lc; hit = true;
    }
    
    // left face
    if (!hit && piq(pp, bs[0], ts[0], ts[3], bs[3])) {
        let n = vec3(-1., 0., 0.);
        let wsp = b.p + vec3(-hw, b.s.y * .5, 0.);
        var lc = amb(m, ao(wsp, n));
        for (var i = 0; i < 3; i++) { lc += ct(ls[i].p, vd, n, m, ls[i].c, ls[i].i) * .8; }
        fc = lc; hit = true;
    }
    
    // right face  
    if (!hit && piq(pp, bs[1], bs[2], ts[2], ts[1])) {
        let n = vec3(1., 0., 0.);
        let wsp = b.p + vec3(hw, b.s.y * .5, 0.);
        var lc = amb(m, ao(wsp, n));
        for (var i = 0; i < 3; i++) { lc += ct(ls[i].p, vd, n, m, ls[i].c, ls[i].i) * .7; }
        fc = lc; hit = true;
    }
    
    // front face
    if (!hit && piq(pp, bs[0], bs[1], ts[1], ts[0])) {
        let n = vec3(0., 0., 1.);
        let wsp = b.p + vec3(0., b.s.y * .5, hd);
        var lc = amb(m, ao(wsp, n));
        for (var i = 0; i < 3; i++) { lc += ct(ls[i].p, vd, n, m, ls[i].c, ls[i].i) * .9; }
        fc = lc; hit = true;
    }
    
    // back face
    if (!hit && piq(pp, bs[2], bs[3], ts[3], ts[2])) {
        let n = vec3(0., 0., -1.);
        let wsp = b.p + vec3(0., b.s.y * .5, -hd);
        var lc = amb(m, ao(wsp, n));
        for (var i = 0; i < 3; i++) { lc += ct(ls[i].p, vd, n, m, ls[i].c, ls[i].i) * .6; }
        fc = lc; hit = true;
    }
    
    return fc;
}

// get moving block pos
fn gmbp() -> vec3<f32> {
    let cb = gcb();
    if (gs() != 1u || cb == 0u) { return vec3(0., -100., 0.); }
    
    let th = f32(cb - 1u) * .6;
    let osc = sin(u_time.time * 4.) * 2.5;
    return vec3(osc, th + .6, 0.);
}

// init game
fn init() {
    if (u_time.frame == 1u) {
        // foundation
        sb(0u, Block(vec3(0., 0., 0.), vec3(4., .6, 4.), vec3(.8, .6, .4), 0.));
        
        ss(0u); ssc(0u); scb(1u); sct(false); scy(0.); spt(-999.);
    }
}

// update game logic
fn upd() {
    let mc = (u_mouse.buttons.x & 1u) != 0u;
    let state = gs();
    updcam();
    
    // click detection
    let wc = gct();
    if (mc && !wc) {
        sct(true);
        
        if (state == 0u) {
            // start
            ss(1u); ssc(0u); scb(1u); scy(0.);
        }
        else if (state == 1u) {
            // drop block
            let cb = gcb();
            if (cb < 30u) {
                let mp = gmbp();
                let pb = gb(cb - 1u);
                
                var nb = Block(vec3(mp.x, f32(cb) * .6, mp.z), vec3(0., .6, pb.s.z), vec3(0.), 0.);
                
                // trimming
                let ox = abs(mp.x - pb.p.x);
                nb.s.x = max(pb.s.x - ox, .2);
                
                // perfect match check
                nb.perf = select(0., 1., ox < .05);
                 // trigger perfect effect (visual + audio chime)
                if (nb.perf > .5) { spt(u_time.time); game_data[10] = atime(); }
                
                // adjust pos
                nb.p.x = select(pb.p.x - (pb.s.x - nb.s.x) * .5, 
                               pb.p.x + (pb.s.x - nb.s.x) * .5, mp.x > pb.p.x);
                nb.p.z = pb.p.z;
                
                // material color
                let m = mat(cb);
                nb.c = m.alb;
                
                if (nb.s.x < .5) { ss(2u); game_data[11] = atime(); } // game over tone
                else { sb(cb, nb); scb(cb + 1u); ssc(gsc() + select(10u, 20u, nb.perf > .5)); game_data[9] = atime(); game_data[12] = f32(cb); game_data[13] = clamp(mp.x / 2.5, -1., 1.); }
            }
        }
        else if (state == 2u) {
            // reset
            ss(0u); ssc(0u); scb(1u); scy(0.);
            for (var i = 1u; i < 30u; i++) {
                sb(i, Block(vec3(0.), vec3(0.), vec3(0.), 0.));
            }
        }
    } else if (!mc) { sct(false); }
}

// all text is sized and placed in ui_scale units, centred on the window
fn txt(pp: vec2<f32>, ss: vec2<f32>) -> vec3<f32> {
    let state = gs();
    let u = ui_scale(ss);
    let cx = ss.x * .5;
    var tc = vec3(0.);

    // perfect placement feedback
    let pt = gpt();
    let dt = u_time.time - pt;
    if (dt < 2. && dt > 0. && pt > 0. && state == 1u) {
        let fade = 1. - dt / 2.;
        let size = 80. * u * (1. + sin(dt * 8.) * .2 * fade);
        if (centered(pp, cx, ss.y * .3 - size * .5, W_PERFECT, 8u, size) > 0.01) { tc = vec3(0.1, 0.05, 0.0) * fade; }
    }

    if (state == 0u) {
        if (centered(pp, cx, ss.y * .12, W_TITLE, 11u, 64. * u) > 0.01) { tc = vec3(1., 1., 0.); }
        if (centered(pp, cx, ss.y * .25, W_START, 14u, 32. * u) > 0.01) { tc = vec3(0.8, 0.1, 0.0); }
        if (centered(pp, cx, ss.y * .335, W_MATCH, 15u, 24. * u) > 0.01) { tc = vec3(0.1, 0.05, 0.0); }
        if (centered(pp, cx, ss.y * .37, W_POINTS, 11u, 24. * u) > 0.01) { tc = vec3(0.1, 0.05, 0.0); }
    } else if (state == 1u) {
        let size = 48. * u;
        let m = 40. * u;
        if (word(pp, vec2(m, m), W_SCORE, 6u, size) > 0.01) { tc = vec3(1.); }
        if (num(pp, vec2(m + 6. * adv(size), m), gsc(), size) > 0.01) { tc = vec3(.01, .01, .01); }
    } else if (state == 2u) {
        if (centered(pp, cx, ss.y * .4, W_OVER, 9u, 60. * u) > 0.01) { tc = vec3(1., .2, .2); }
        // final score, label and number centred together
        let size = 40. * u;
        let x0 = cx - (f32(5u + digits(gsc())) * adv(size) + size) * .5;
        if (word(pp, vec2(x0, ss.y * .5), W_SCORE, 6u, size) > 0.01) { tc = vec3(1.); }
        if (num(pp, vec2(x0 + 6. * adv(size), ss.y * .5), gsc(), size) > 0.01) { tc = vec3(1.); }
        if (centered(pp, cx, ss.y * .6, W_RESTART, 16u, 32. * u) > 0.01) { tc = vec3(.1); }
    }

    return tc;
}

// audio
fn hsh(x: u32) -> u32 {
    var v = x;
    v ^= v >> 16u; v *= 0x7feb352du;
    v ^= v >> 15u; v *= 0x846ca68bu;
    v ^= v >> 16u;
    return v;
}
fn wn(n: u32, sd: u32) -> f32 { return f32(hsh(n * 0x9e3779b9u ^ sd) >> 8u) / 8388608.0 - 1.0; }
fn vnz(n: u32, m: u32, sd: u32) -> f32 {
    let i = n / m;
    var f = f32(n % m) / f32(m);
    f = f * f * (3.0 - 2.0 * f);
    return mix(wn(i, sd), wn(i + 1u, sd), f);
}
fn eage(k: u32, n: u32) -> f32 {
    let e = bitcast<u32>(game_data[k]);
    if (game_data[k] <= 0. || n + EV0 < e) { return -1.; }
    return f32(n + EV0 - e) / u_game.sample_rate;
}
fn pan2(p: f32) -> vec2<f32> {
    let a = (p + 1.) * .785398;
    return vec2(cos(a), sin(a)) * 1.414;
}
fn lp12(f: f32, fc: f32, res: f32) -> f32 {
    let w = f / fc;
    let q = 0.5 + res * 2.0;
    let d = 1.0 - w * w;
    return inverseSqrt(d * d + w * w / (q * q));
}

// band-limited saw at phase `ph` (cycles) through a 12 dB low-pass
fn saw(ph: f32, f: f32, fc: f32) -> f32 {
    let th = TAU * fract(ph);
    let c2 = 2.0 * cos(th);
    var s0 = 0.0;
    var s1 = sin(th);
    var s = 0.0;
    for (var h = 1; h <= 24; h++) {
        let hz = f * f32(h);
        if (hz > 12000.0) { break; }
        s += s1 / f32(h) * lp12(hz, fc, 0.4);
        let s2 = c2 * s1 - s0;
        s0 = s1;
        s1 = s2;
    }
    return s;
}

// FM bell: a 1:1 body and a short 14:1 tine
fn bell(a: f32, f: f32, dec: f32) -> f32 {
    if (a < 0.) { return 0.; }
    let w = TAU * fract(f * a);
    let m = 1.2 * exp(-a / .3) * sin(w) + .5 * exp(-a / .025) * sin(TAU * fract(f * 14. * a));
    return sin(w + m) * exp(-a * dec) * smoothstep(0., .002, a);
}

// sound effects `dl` seconds late, for the echo
fn sfx(n: u32, dl: f32) -> vec2<f32> {
    var s = vec2(0.);

    // place: a wooden thock plus a bell climbing the pentatonic scale with the tower, panned where the block landed
    let dp = eage(9u, n) - dl;
    if (dp >= 0. && dp < 1.2) {
        let k = u32(max(game_data[12], 0.));
        var deg = array<f32, 5>(0., 2., 4., 7., 9.);
        let f = 261.63 * exp2((deg[k % 5u] + 12. * f32(min(k / 5u, 2u))) / 12.);
        let thock = sin(TAU * fract(90. * dp + 90. * .02 * (1. - exp(-dp / .02)))) * exp(-dp / .06) + wn(n, 3u) * exp(-dp / .004) * .3;
        s += pan2(game_data[13] * .6) * (bell(dp, f, 3.5) * .22 + thock * .25 * smoothstep(0., .001, dp));
    }

    // perfect: a bright bell arpeggio bouncing left and right, with a shimmer on top
    let dq = eage(10u, n) - dl;
    if (dq >= 0. && dq < 1.4) {
        var notes = array<f32, 4>(523.25, 659.25, 783.99, 1046.5);
        for (var j = 0u; j < 4u; j++) {
            let side = select(vec2(1., .55), vec2(.55, 1.), j % 2u == 1u);
            s += side * bell(dq - f32(j) * .06, notes[j], 4.) * .16;
        }
        s += vec2(wn(n, 5u) - wn(n - 1u, 5u), wn(n, 7u) - wn(n - 1u, 7u)) * .04 * exp(-dq / .25) * smoothstep(0., .01, dq);
    }

    // game over: a detuned saw falling in three notes, the last one sagging, over a low rumble
    let dg = eage(11u, n) - dl;
    if (dg >= 0. && dg < 1.8) {
        var gn = array<f32, 3>(392., 311.13, 261.63);
        for (var j = 0u; j < 3u; j++) {
            let a = dg - f32(j) * .16;
            if (a >= 0.) {
                let f = gn[j];
                let ph = select(f * a, f * (a - .1 * a * a), j == 2u);
                let fc = f * (3. + 4. * exp(-a / .2));
                s += vec2(saw(ph * .996, f, fc), saw(ph * 1.004 + .3, f, fc)) * exp(-a * select(5., 2.5, j == 2u)) * smoothstep(0., .004, a) * .14;
            }
        }
        s += vec2(vnz(n, 90u, 9u)) * exp(-dg / .5) * .25 * smoothstep(0., .02, dg);
    }
    return s;
}

// dry effects plus a ping-pong echo
fn game_sound(n: u32) -> vec2<f32> {
    let s = sfx(n, 0.) + sfx(n, .18) * vec2(.25, .08) + sfx(n, .36) * vec2(.05, .12);
    return tanh(s);
}

@compute @workgroup_size(8, 8, 1)
fn sim(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (all(gid.xy == vec2(0u))) {
        init();
        upd();
    }
}

@compute @workgroup_size(8, 8, 1)
fn main_image(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    // audio: one thread per sample, after the sim pass has moved the game on
    let ai = (wid.y * nw.x + wid.x) * 64u + li;
    if (ai < u_game.samples_to_generate) {
        let v = game_sound(u_game.sample_offset + ai) * u_game.volume;
        audio_buffer[ai * 2u] = v.x;
        audio_buffer[ai * 2u + 1u] = v.y;
    }

    let ss = vec2<f32>(textureDimensions(output));
    let pp = vec2<f32>(gid.xy);
    
    if any(pp >= ss) { return; }
    
    // background
    let ny = pp.y / ss.y;
    let nx = pp.x / ss.x;
    
    let sky = vec3(.4, .7, 1.);
    let hor = vec3(.9, .6, .3);
    let gnd = vec3(.2, .3, .4);
    
    let noise = sin(pp.x * .01 + u_time.time * .5) * sin(pp.y * .01) * .1;
    
    var col = select(mix(gnd, hor, ny * 1.67), mix(hor, sky, (ny - .6) * 2.5), ny > .6);
    col += noise * vec3(.02, .02, .04);
    
    // vignette
    let vig = smoothstep(0., .3, min(nx, min(1. - nx, min(ny, 1. - ny))));
    col = mix(col * .7, col, vig);
    
    let state = gs();
    
    if (state == 1u) {
        // playing - render blocks
        let cb = gcb();
        
        // placed blocks
        for (var i = 0u; i < cb && i < 30u; i++) {
            let b = gb(i);
            if (b.s.x > 0.) {
                let bc = rbl(pp, b, ss, i);
                if (length(bc) > 0.) { col = bc; }
            }
        }
        
        // moving block
        let mp = gmbp();
        if (mp.y > -50.) {
            let pb = gb(cb - 1u);
            let m = mat(cb);
            let pulse = sin(u_time.time * 8.) * .3 + .7;
            let mb = Block(mp, vec3(pb.s.x, .6, pb.s.z), m.alb * pulse, 0.);
            
            let mbc = rbl(pp, mb, ss, cb);
            if (length(mbc) > 0.) { col = mbc; }
        }
    } else {
        // menu/gameover - foundation
        let f = gb(0u);
        let fc = rbl(pp, f, ss, 0u);
        if (length(fc) > 0.) { col = fc; }
        
        if (state == 0u) { col *= sin(u_time.time * 3.) * .2 + .8; }
        else if (state == 2u) { col = mix(col, vec3(1., .3, .3), .3); }
    }
    
    // text overlay
    let tc = txt(pp, ss);
    if (length(tc) > 0.) { col = tc; }
    
    // perfect placement flash effect
    let pt = gpt();
    let dt = u_time.time - pt;
    if (dt < .5 && dt > 0. && pt > 0. && state == 1u) {
        let flash_intensity = (1. - dt / .5) * .3;
        col = mix(col, vec3(1., 1., .7), flash_intensity * sin(dt * 20.) * .5 + flash_intensity * .5);
    }
    
    // post processing: tone map vig, gamma etc etc
    col *= 1.2; // exposure
    col = (col * (2.51 * col + .03)) / (col * (2.43 * col + .59) + .14);
    col = pow(col, vec3(1. / 2.2)); // gamma
    col = col * .8 + .2 * col * col * (3. - 2. * col); 
    col *= vec3(1.05, 1., 1.02); 
    
    let uv = pp / ss;
    let vf = 1. - .15 * smoothstep(.3, 1., distance(uv, vec2(.5)));
    col = clamp(col * vf, vec3(0.), vec3(1.));
    
    textureStore(output, vec2<i32>(gid.xy), vec4(col, 1.));
}
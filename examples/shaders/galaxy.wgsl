// Enes Altun, 2026;
// This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 Unported License.
// Spectral galaxy — IFS chaos game + spectral splat, TAA, bokeh.
// Inspiration: https://compute.toys/view/68, wrigher: Flame fractal (Chaos Game) 
struct TimeUniform { time: f32, delta: f32, frame: u32, _padding: u32 };
@group(0) @binding(0) var<uniform> u_t: TimeUniform;

struct Params {
    tw: f32, rot: f32, cor: f32, spr: f32,
    coh: f32, fb: f32, an: f32, it: f32,
    hue: f32, spc: f32, dst: f32, cv: f32,
    dof: f32, foc: f32, blm: f32, vig: f32,
    br: f32, expo: f32, gam: f32, sat: f32,
    trv: f32, orb: f32, zm: f32, arm: f32,
    bke: f32, bkb: f32, bkf: f32, taw: f32,
    // pers: share of last frame's light the grid keeps
    shp: f32, tlt: f32, dg: f32, q0: f32,
    // logd: log-density tone mapping amount
    logd: f32, pers: f32, q2: f32, q3: f32,
};
@group(1) @binding(0) var out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> p: Params;
// trajectory points kept between frames (w = 1 once valid)
@group(2) @binding(0) var<storage, read_write> trj: array<v4>;
@group(2) @binding(1) var<storage, read_write> atm: array<atomic<u32>>;
@group(3) @binding(0) var t0: texture_2d<f32>; @group(3) @binding(1) var s0: sampler;
@group(3) @binding(2) var t1: texture_2d<f32>; @group(3) @binding(3) var s1: sampler;

alias v2 = vec2<f32>; alias v3 = vec3<f32>; alias v4 = vec4<f32>;
alias m2 = mat2x2<f32>; alias m3 = mat3x3<f32>; alias u3 = vec3<u32>;
const pi = 3.14159265; const tau = 6.28318531;
// trajectories/frame (= 8192*256, must match Splat workgroup)
const NTH = 2097152u;
const GAIN = 800.0;

const spectrum = array<v3, 45>(
    v3(0.002362,0.000253,0.010482), v3(0.019110,0.002004,0.086011), v3(0.084736,0.008756,0.389366),
    v3(0.204492,0.021391,0.972542), v3(0.314679,0.038676,1.553480), v3(0.383734,0.062077,1.967280),
    v3(0.370702,0.089456,1.994800), v3(0.302273,0.128201,1.745370), v3(0.195618,0.185190,1.317560),
    v3(0.080507,0.253589,0.772125), v3(0.016172,0.339133,0.415254), v3(0.003816,0.460777,0.218502),
    v3(0.037465,0.606741,0.112044), v3(0.117749,0.761757,0.060709), v3(0.236491,0.875211,0.030451),
    v3(0.376772,0.961988,0.013676), v3(0.529826,0.991761,0.003988), v3(0.705224,0.997340,0.000000),
    v3(0.878655,0.955552,0.000000), v3(1.014160,0.868934,0.000000), v3(1.118520,0.777405,0.000000),
    v3(1.123990,0.658341,0.000000), v3(1.030480,0.527963,0.000000), v3(0.856297,0.398057,0.000000),
    v3(0.647467,0.283493,0.000000), v3(0.431567,0.179828,0.000000), v3(0.268329,0.107633,0.000000),
    v3(0.152568,0.060281,0.000000), v3(0.081261,0.031800,0.000000), v3(0.040851,0.015905,0.000000),
    v3(0.019941,0.007749,0.000000), v3(0.009577,0.003718,0.000000), v3(0.004553,0.001768,0.000000),
    v3(0.002175,0.000846,0.000000), v3(0.001045,0.000407,0.000000), v3(0.000508,0.000199,0.000000),
    v3(0.000251,0.000098,0.000000), v3(0.000126,0.000050,0.000000), v3(0.000065,0.000025,0.000000),
    v3(0.000033,0.000013,0.000000), v3(0.000018,0.000007,0.000000), v3(0.000009,0.000004,0.000000),
    v3(0.000005,0.000002,0.000000), v3(0.000003,0.000001,0.000000), v3(0.000002,0.000001,0.000000)
);
const xyz_rgb = m3(3.2404542,-0.9692660,0.0556434, -1.5371385,1.8760108,-0.2040259, -0.4985314,0.0415560,1.0572252);
fn w2x(wl: f32) -> v3 { let x=(wl-390.)*0.1; let i=u32(clamp(x,0.,43.)); return mix(spectrum[i], spectrum[i+1u], fract(x)); }

var<private> R: v2;
var<private> sd: u32;
fn hu(a0: u32) -> u32 { var a=a0; a^=a>>16u; a*=0x7feb352du; a^=a>>15u; a*=0x846ca68bu; a^=a>>16u; return a; }
fn hf() -> f32 { var s=hu(sd); sd=s; return f32(s)/f32(0xffffffffu); }
fn hv3() -> v3 { return v3(hf(),hf(),hf()); }


fn h3(q0: v3) -> f32 { var q=fract(q0*0.1031); q+=dot(q,q.zyx+31.32); return fract((q.x+q.y)*q.z); }
fn vn3(x: v3) -> f32 {
    let i=floor(x); let f=fract(x); let u=f*f*(3.-2.*f);
    return mix(mix(mix(h3(i+v3(0.,0.,0.)),h3(i+v3(1.,0.,0.)),u.x), mix(h3(i+v3(0.,1.,0.)),h3(i+v3(1.,1.,0.)),u.x),u.y),
               mix(mix(h3(i+v3(0.,0.,1.)),h3(i+v3(1.,0.,1.)),u.x), mix(h3(i+v3(0.,1.,1.)),h3(i+v3(1.,1.,1.)),u.x),u.y), u.z);
}

fn rot(a: f32) -> m2 { return m2(cos(a),-sin(a),sin(a),cos(a)); }
fn rX(a: f32) -> m3 { let r=rot(a); return m3(1.,0.,0., 0.,r[0][0],r[0][1], 0.,r[1][0],r[1][1]); }
fn rY(a: f32) -> m3 { let r=rot(a); return m3(r[0][0],0.,r[0][1], 0.,1.,0., r[1][0],0.,r[1][1]); }
fn rZ(a: f32) -> m3 { let r=rot(a); return m3(r[0][0],r[0][1],0., r[1][0],r[1][1],0., 0.,0.,1.); }

// YCoCg (TAA clamp) — adapted from gelami/mrange (CC0: https://www.shadertoy.com/view/fXSGR1).
fn r2y(c: v3) -> v3 { return m3(0.25,0.5,-0.25, 0.5,0.,0.5, 0.25,-0.5,-0.25)*c; }
fn y2r(c: v3) -> v3 { return m3(1.,1.,1., 1.,0.,-1., -1.,1.,-1.)*c; }
fn aces(c: v3) -> v3 {
    let n1=m3(0.59719,0.07600,0.02840, 0.35458,0.90834,0.13383, 0.04823,0.01566,0.83777);
    let n2=m3(1.60475,-0.10208,-0.00327, -0.53108,1.10813,-0.07276, -0.07367,-0.00605,1.07602);
    var v=n1*c; let a=v*(v+0.0245786)-0.000090537; let b=v*(0.983729*v+0.4329510)+0.238081;
    return n2*(a/b);
}

// one bounds checked pixel, stochastically rounded so faint light isn't dropped
fn adp(ix: i32, iy: i32, Ru: vec2<u32>, ly: u32, c: v3) {
    if(ix<0||iy<0||ix>=i32(Ru.x)||iy>=i32(Ru.y)){return;}
    let i=u32(ix)+Ru.x*u32(iy);
    let d=hf();
    atomicAdd(&atm[i], u32(max(0.,c.x)+d));
    atomicAdd(&atm[i+ly], u32(max(0.,c.y)+d));
    atomicAdd(&atm[i+2u*ly], u32(max(0.,c.z)+d));
}
fn rbf(ix: i32, iy: i32, w: u32, h: u32) -> v3 {
    if(ix<0||iy<0||ix>=i32(w)||iy>=i32(h)){return v3(0.);}
    let i=u32(ix)+w*u32(iy); let o=w*h;
    return v3(f32(atomicLoad(&atm[i])), f32(atomicLoad(&atm[i+o])), f32(atomicLoad(&atm[i+2u*o])));
}

// P1: clear the grid (own dispatch so bloom won't race the zeroing...)
@compute @workgroup_size(16,16,1)
fn Clear(@builtin(global_invocation_id) id: u3) {
    let d=vec2<u32>(textureDimensions(out)); if(id.x>=d.x||id.y>=d.y){return;}
    let i=id.x+d.x*id.y; let o=d.x*d.y;
    // fade instead of clearing, so particles blend over frames
    let r=f32(hu(i*1664525u+u_t.frame*1013904223u))/4294967295.;
    for(var k=0u;k<4u;k++){
        let j=i+k*o;
        atomicStore(&atm[j], u32(f32(atomicLoad(&atm[j]))*p.pers+r));
    }
}

// P2: simple chaos game box (see chaos game tutorial if you looking somewhere to start: https://compute.toys/view/120 by Slerpy)
// or: https://compute.toys/view/68 by wrighter)
@compute @workgroup_size(256,1,1)
fn Splat(@builtin(global_invocation_id) id: u3) {
    if(id.x>=NTH){return;}
    let Ru=vec2<u32>(textureDimensions(out)); R=v2(Ru); let ly=Ru.x*Ru.y;
    // noise/frame so TAA averages
    sd=hu((id.x+u_t.frame*NTH)*747796405u+2891336453u); sd=hu(sd);
    let t=u_t.time*0.1*p.an;

    let wl=clamp(400.+p.hue*240.+hf()*p.spc*280., 390., 700.);
    let wx=(wl-600.)/200.;
    // continue from last frame's point: already on the attractor, so little warm-up
    let prev=trj[id.x];
    let carried=prev.w>0.5 && u_t.frame>2u;
    var pt=select((hv3()-0.5)*4., prev.xyz, carried);
    let warm=select(12, 2, carried);

    let ra=rX(1.8+sin(t*0.5)*0.1);
    let rb=rZ(0.6+cos(t*0.3)*0.1);
    let rc=rY(p.rot*3.);
    let drift=sin(4.1)*0.2;
    let its=i32(clamp(p.it,20.,120.));
    // spherical fold dominance
    let th=mix(0.35,0.85,p.fb); 
    // we are creating the shape in bellow loop. most of them experimentally found
    for(var i=0; i<its; i++){
        let r=hf()+p.coh*wx+0.15*f32(i)*drift;
        if(r<0.08){
            pt=abs(pt)-v3(1.2,0.2,1.1); pt=pt*ra; pt*=(1.1+p.tw*0.2);
        } else if(r<0.4){ 
            let rl=length(pt.xy)+0.01; var a=atan2(pt.y,pt.x);
            a+=log(rl+2.2)*(0.5+p.tw*2.5); a+=0.6*t;
            let n=1.+floor(p.arm*4.99); a+=floor(hf()*n)*(tau/n);
            pt.x=rl*cos(a); pt.y=rl*sin(a); pt.z*=0.5; pt=pt*(0.4+0.08*rl);
        } else if(r<th){
            let dd=dot(pt,pt); pt/=max(0.5,dd*(1.+p.spr)); pt+=v3(0.5,0.1,0.8)*p.cor; pt=pt*rb;
        } else if(r<th+0.18){
            let r2=length(pt.xy)*(1.5+p.spr); let s=sin(r2); let c=cos(r2);
            pt=v3(pt.x*c-pt.y*s, pt.x*s+pt.y*c, pt.z); pt*=0.82;
        } else {
            pt+=v3(sin(t),cos(t*0.7),0.)*0.1; pt*=0.3; pt=pt*rc;
        }

        if(i<warm){ continue; }

        let ct=30.*p.trv*0.1;
        var q=pt;
        q=q*rY(0.2+ct+p.orb*u_t.time*0.05);
        q=q*rX(0.2+sin(ct*0.7)*0.3);
        q=q*rZ(cos(ct*0.5)*0.2);
        q=q*rX(p.tlt);
        q.z+=2.5; let z=q.z; if(z<0.1){ continue; }
        let dp=clamp((z-1.5)/3., 0., 1.);   // 0 near .. 1 far
        var sp=(q.xy/z)*p.zm;

        // dof
        if(p.dof>0.){
            let coc=abs(z-(2.5+p.foc))*p.dof*0.01;
            if(coc>1e-4){
                let a=hf()*tau; let r01=hf(); var poly=1.;
                if(p.bkb>=3.){ let sg=tau/p.bkb; let aa=a-sg*(floor(a/sg)+0.5); poly=cos(sg*0.5)/cos(aa); }
                let rd=pow(r01, mix(0.5,0.07,clamp(p.bke,0.,1.)))*coc*poly;
                let bo=v2(cos(a),sin(a))*rd; sp+=bo+bo*wx*p.bkf;
            }
        }

        sp.x*=R.y/R.x;
        let uv=sp*0.5+0.5; if(uv.x<=0.||uv.x>=1.||uv.y<=0.||uv.y>=1.){ continue; }
        let fx=uv.x*R.x; let fy=uv.y*R.y;

        // colours
        let cr=length(pt.xy); let rc2=length(pt);
        var wf=wl+(cr-1.)*p.cv*40.; wf+=exp(-rc2*0.9)*60.; wf-=smoothstep(1.2,3.2,rc2)*30.;
        wf-=dp*p.dg*30.;
        let cx=w2x(clamp(wf,390.,700.));
        let il=1.+4.5/(0.25+rc2*rc2*1.3); 
        let l=min(2., 0.16/(1.+z*0.12));
        let dg=max(0.1, 1.+p.dg*(0.5-dp)*2.2);
        // one jittered pixel: a box splat over frames, 1/4 of the atomics
        let jo=v2(hf(),hf())-0.5;
        adp(i32(floor(fx+jo.x)), i32(floor(fy+jo.y)), Ru, ly, cx*l*il*dg*p.br*GAIN);

        // some dusts (skipped when off, the noise is costly)
        if(p.dst>0.){
            let dn=vn3(pt*2.6+v3(0.,0.,t*0.15)+vn3(pt*1.1)*0.6);
            let dm=smoothstep(0.52,0.80,dn);
            if(dm>0.){
                let dx=i32(floor(fx+jo.x)); let dy=i32(floor(fy+jo.y));
                if(dx>=0&&dy>=0&&dx<i32(Ru.x)&&dy<i32(Ru.y)){ atomicAdd(&atm[u32(dx)+Ru.x*u32(dy)+3u*ly], u32(dm*l*500.+hf())); }
            }
        }
    }
    // keep the point; a runaway one restarts
    trj[id.x]=select(v4(0.), v4(pt,1.), length(pt)<1e3);
}

// P3: resolve atomic -> linear (emission + dust extinction, D65)
@compute @workgroup_size(16,16,1)
fn resolve_raw(@builtin(global_invocation_id) id: u3) {
    let d=vec2<u32>(textureDimensions(out)); if(id.x>=d.x||id.y>=d.y){return;}
    let ss=f32(d.x*d.y);
    var xyz=rbf(i32(id.x),i32(id.y),d.x,d.y); xyz*=v3(0.95,1.,1.08);
    // the faded sum holds 1/(1-pers) frames of light: scale back to one
    let nrm=1.-p.pers;
    var col=max(v3(0.), xyz_rgb*xyz)*ss*2e-9/256.*nrm;
    let dst=f32(atomicLoad(&atm[(id.x+d.x*id.y)+3u*(d.x*d.y)]))*ss*2e-9/256.*nrm;
    col*=exp(-dst*p.dst*2.2);
    textureStore(out, vec2<i32>(id.xy), v4(col,1.));
}

// clamp half a texel inside: the sampler repeats
fn cuv(uv: v2, d: v2) -> v2 { return clamp(uv, 0.5/d, 1.-0.5/d); }
fn tap0(uv: v2) -> v3 { let d=v2(textureDimensions(t0)); return textureSampleLevel(t0,s0,cuv(uv,d),0.).rgb; }
fn tap1(uv: v2) -> v3 { let d=v2(textureDimensions(t1)); return textureSampleLevel(t1,s1,cuv(uv,d),0.).rgb; }

// P4: TAA
@compute @workgroup_size(16,16,1)
fn taa(@builtin(global_invocation_id) id: u3) {
    let d=vec2<u32>(textureDimensions(out)); if(id.x>=d.x||id.y>=d.y){return;}
    let Rl=v2(d); let uv=(v2(id.xy)+0.5)/Rl;
    let cur=r2y(tap0(uv));
    var mn=cur; var mx=cur;
    for(var y=-1;y<=1;y++){ for(var x=-1;x<=1;x++){
        let n=r2y(tap0(uv+v2(f32(x),f32(y))/Rl)); mn=min(mn,n); mx=max(mx,n);
    }}
    let hr=r2y(tap1(uv));
    let hc=clamp(hr,mn,mx);
    let mo=abs(hr.x-hc.x)/(abs(cur.x)+0.02);
    let w=p.taw*(1.-clamp(mo*3.,0.,0.85));
    let bl=select(w,0.,u_t.frame<4u);
    textureStore(out, vec2<i32>(id.xy), v4(max(v3(0.), y2r(mix(cur,hc,bl))),1.));
}

// bloom mip chain; Karis weights on the first step stop single sparks flickering
fn karis(c: v3) -> f32 { return 1./(1.+dot(c,v3(0.2126,0.7152,0.0722))); }
fn down13(uv: v2, first: bool) -> v3 {
    let e=1./v2(textureDimensions(t0));
    let a=tap0(uv+e*v2(-2.,-2.)); let b=tap0(uv+e*v2(0.,-2.)); let c=tap0(uv+e*v2(2.,-2.));
    let d=tap0(uv+e*v2(-1.,-1.)); let f=tap0(uv+e*v2(1.,-1.));
    let g=tap0(uv+e*v2(-2.,0.)); let h=tap0(uv); let i=tap0(uv+e*v2(2.,0.));
    let j=tap0(uv+e*v2(-1.,1.)); let k=tap0(uv+e*v2(1.,1.));
    let l=tap0(uv+e*v2(-2.,2.)); let m=tap0(uv+e*v2(0.,2.)); let n=tap0(uv+e*v2(2.,2.));
    // five overlapping 2x2 boxes: the centre one counts half
    let q0=(d+f+j+k)*0.25; let q1=(a+b+g+h)*0.25; let q2=(b+c+h+i)*0.25; let q3=(g+h+l+m)*0.25; let q4=(h+i+m+n)*0.25;
    if(first){
        let w0=karis(q0)*0.5; let w1=karis(q1)*0.125; let w2=karis(q2)*0.125; let w3=karis(q3)*0.125; let w4=karis(q4)*0.125;
        return (q0*w0+q1*w1+q2*w2+q3*w3+q4*w4)/(w0+w1+w2+w3+w4);
    }
    return q0*0.5+(q1+q2+q3+q4)*0.125;
}
// only light above white on screen feeds the bloom (soft knee)
fn bright_part(c: v3) -> v3 {
    let t=1./max(p.expo,0.01); let knee=0.5*t;
    let br=max(c.r,max(c.g,c.b));
    var soft=clamp(br-t+knee, 0., 2.*knee); soft=soft*soft/(4.*knee+1e-4);
    return c*(max(soft, br-t)/max(br,1e-4));
}
// tent of the smaller level (t0) + this level (t1)
fn up9(uv: v2) -> v3 {
    let e=1./v2(textureDimensions(t0));
    var c=tap0(uv)*4.;
    c+=(tap0(uv+v2(e.x,0.))+tap0(uv-v2(e.x,0.))+tap0(uv+v2(0.,e.y))+tap0(uv-v2(0.,e.y)))*2.;
    c+=tap0(uv+e)+tap0(uv-e)+tap0(uv+v2(e.x,-e.y))+tap0(uv+v2(-e.x,e.y));
    return c/16.+tap1(uv);
}
fn px_uv(id: u3) -> v2 { return (v2(id.xy)+0.5)/v2(textureDimensions(out)); }
// bloom levels: down to 1/16, then back up
@compute @workgroup_size(16,16,1)
fn bloom_d1(@builtin(global_invocation_id) id: u3) {
    let d=textureDimensions(out); if(id.x>=d.x||id.y>=d.y){return;}
    textureStore(out, vec2<i32>(id.xy), v4(bright_part(down13(px_uv(id), true)),1.));
}
@compute @workgroup_size(16,16,1)
fn bloom_d2(@builtin(global_invocation_id) id: u3) {
    let d=textureDimensions(out); if(id.x>=d.x||id.y>=d.y){return;}
    textureStore(out, vec2<i32>(id.xy), v4(down13(px_uv(id), false),1.));
}
@compute @workgroup_size(16,16,1)
fn bloom_d3(@builtin(global_invocation_id) id: u3) {
    let d=textureDimensions(out); if(id.x>=d.x||id.y>=d.y){return;}
    textureStore(out, vec2<i32>(id.xy), v4(down13(px_uv(id), false),1.));
}
@compute @workgroup_size(16,16,1)
fn bloom_d4(@builtin(global_invocation_id) id: u3) {
    let d=textureDimensions(out); if(id.x>=d.x||id.y>=d.y){return;}
    textureStore(out, vec2<i32>(id.xy), v4(down13(px_uv(id), false),1.));
}
@compute @workgroup_size(16,16,1)
fn bloom_u3(@builtin(global_invocation_id) id: u3) {
    let d=textureDimensions(out); if(id.x>=d.x||id.y>=d.y){return;}
    textureStore(out, vec2<i32>(id.xy), v4(up9(px_uv(id)),1.));
}
@compute @workgroup_size(16,16,1)
fn bloom_u2(@builtin(global_invocation_id) id: u3) {
    let d=textureDimensions(out); if(id.x>=d.x||id.y>=d.y){return;}
    textureStore(out, vec2<i32>(id.xy), v4(up9(px_uv(id)),1.));
}
@compute @workgroup_size(16,16,1)
fn bloom_u1(@builtin(global_invocation_id) id: u3) {
    let d=textureDimensions(out); if(id.x>=d.x||id.y>=d.y){return;}
    textureStore(out, vec2<i32>(id.xy), v4(up9(px_uv(id)),1.));
}

//  post: t0 = TAA result, t1 = bloom
@compute @workgroup_size(16,16,1)
fn main_image(@builtin(global_invocation_id) id: u3) {
    let d=vec2<u32>(textureDimensions(out)); if(id.x>=d.x||id.y>=d.y){return;}
    let Rl=v2(d); let uv=(v2(id.xy)+0.5)/Rl;
    var col=tap0(uv);

    if(p.shp>0.001){ 
        let e=1./Rl;
        let bl=(tap0(uv+v2(e.x,0.))+tap0(uv-v2(e.x,0.))+tap0(uv+v2(0.,e.y))+tap0(uv-v2(0.,e.y)))*0.25;
        col=max(v3(0.), col+(col-bl)*p.shp*2.);
    }

    let l0=dot(col,v3(1.)); var wd=0.;
    for(var k=0;k<8;k++){ let a=f32(k)*0.785398; let dd=v2(cos(a),sin(a));
        wd+=dot(tap0(uv+dd*6./Rl),v3(1.));
        wd+=dot(tap0(uv+dd*14./Rl),v3(1.)); }
    wd*=1./16.;
    let ao=mix(1., clamp(0.15+0.85*l0/(wd+0.003),0.,1.), 0.7);

    // four summed levels, so a quarter is their average
    if(p.blm>0.001){ col+=tap1(uv)*0.25*p.blm; }

    // log density (flame style): compress the dense core, lift thin filaments; level 1 unchanged
    if(p.logd>0.){
        let k=8.; let L=max(max(col.r,col.g),max(col.b,1e-5));
        col*=mix(1., log(1.+k*L)/(k*L)*(k/log(1.+k)), p.logd);
    }

    col*=p.expo*ao;
    let l=dot(col,v3(0.2126,0.7152,0.0722)); col=mix(v3(l),col,p.sat);
    col=aces(col);
    col=pow(max(col,v3(0.)),v3(1./p.gam));
    textureStore(out, vec2<i32>(id.xy), v4(max(col,v3(0.)),1.));
}
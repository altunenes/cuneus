// Enes Altun, 2026; Rorschach Inkblots
// This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
struct TimeUniform { time: f32, delta: f32, frame: u32, _padding: u32 }
@group(0) @binding(0) var<uniform> u_t: TimeUniform;

struct RorschachParams {
    seed: f32,
    zoom: f32,
    threshold: f32,
    distortion: f32,
    particle_speed: f32,
    particle_life: f32,
    trace_steps: f32,
    contrast: f32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
    gamma: f32,
    style: f32, 
    fbm_octaves: f32,
    tint_x: f32,
    tint_y: f32,
    tint_z: f32,
    _pad_final1: f32,
    _pad_final2: f32,
    _pad_final3: f32,
}

@group(1) @binding(0) var out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> p: RorschachParams;

@group(3) @binding(0) var tex0: texture_2d<f32>;
@group(3) @binding(1) var sam0: sampler;
@group(3) @binding(2) var tex1: texture_2d<f32>;
@group(3) @binding(3) var sam1: sampler;

alias v2 = vec2<f32>;
alias v3 = vec3<f32>;
alias v4 = vec4<f32>;
alias iv2 = vec2<i32>;

fn h2(u:v2)->v2{
    var q=fract(v3(u.xyx)*v3(.1031,.1030,.0973));
    q+=dot(q,q.yzx+33.33);
    return fract((q.xx+q.yz)*q.zy);
}

fn nz(u:v2)->f32{
    let i=floor(u);let f=fract(u);let w=f*f*(3.-2.*f);
    let v=mix(mix(dot(h2(i+v2(0.,0.)),f-v2(0.,0.)),dot(h2(i+v2(1.,0.)),f-v2(1.,0.)),w.x),
              mix(dot(h2(i+v2(0.,1.)),f-v2(0.,1.)),dot(h2(i+v2(1.,1.)),f-v2(1.,1.)),w.x),w.y);
    return .5+.5*v;
}

fn fbm(u:v2)->f32{
    var v=0.;var a=.5;var s=u;let oct=i32(p.fbm_octaves);
    for(var i=0;i<oct;i++){v+=a*nz(s);s*=2.;a*=.5;}
    return v;
}

fn wrp(uv:v2,sd:f32)->f32{
    let q=v2(fbm(uv+v2(0.)+sd),fbm(uv+v2(5.2,1.3)+sd));
    let r=v2(fbm(uv+4.*q+v2(1.7,9.2)+sd),fbm(uv+4.*q+v2(8.3,2.8)+sd));
    return fbm(uv+4.*r);
}

// a dynamic color palette that shifts based on ink density and position....
// this mixes a strict base color with a shimmering tint for the highlights.
fn pal(t:f32,uv:v2,off:f32)->v3{
    let base=v3(p.color_r,p.color_g,p.color_b);
    let ph=v3(p.tint_x,p.tint_y,p.tint_z)*6.28;
    let n=fbm(uv*.5+p.seed);
    let hl=.5+.5*cos(ph+t*2.+off*1.5+n);
    return mix(base,mix(hl,base,smoothstep(.2,.8,t)),p.style);
}

// A: Shape Generation
// simple: regular stuff: I calculate the raw heightmap of the inkblot here.
@compute @workgroup_size(16,16,1)
fn buffer_a(@builtin(global_invocation_id) id:vec3<u32>){
    let dim=textureDimensions(out);
    if(id.x>=dim.x||id.y>=dim.y){return;}
    let uv=(v2(id.xy)-.5*v2(f32(dim.x),f32(dim.y)))/f32(dim.y);
    let sym=v2(abs(uv.x),uv.y)*p.zoom;
    let val=wrp(sym,p.seed);
    let shp=smoothstep(p.threshold-.1,p.threshold+.1,val);
    textureStore(out,id.xy,v4(shp,0.,0.,1.));
}

// B: Vector Field
// I calculate the gradient (slope) of the ink shape to determine flow direction.
@compute @workgroup_size(16,16,1)
fn buffer_b(@builtin(global_invocation_id) id:vec3<u32>){
    let dim=textureDimensions(out);
    if(id.x>=dim.x||id.y>=dim.y){return;}
    let c=iv2(id.xy);
    let t=textureLoad(tex0,c+iv2(0,-1),0).x;let b=textureLoad(tex0,c+iv2(0,1),0).x;
    let l=textureLoad(tex0,c+iv2(-1,0),0).x;let r=textureLoad(tex0,c+iv2(1,0),0).x;
    textureStore(out,id.xy,v4((r-l)*.5,(b-t)*.5,0.,1.));
}

// C: Painterly Simulation, important and tricky part...
// I perform Line Integral Convolution (LIC) here. I trace particles through the 
// vector field, accumulating color and density to simulate wet ink flow.
@compute @workgroup_size(16,16,1)
fn buffer_c(@builtin(global_invocation_id) id:vec3<u32>){
    let dim=textureDimensions(out);
    if(id.x>=dim.x||id.y>=dim.y){return;}
    
    // Feedback
    let old=textureLoad(tex0,iv2(id.xy),0);
    
    // Spawn
    let seed=v2(id.xy)+v2(u_t.time*15.,f32(u_t.frame));
    var pos=v2(id.xy)+(h2(seed)-.5)*2.;
    
    // Analysis
    let uv=(pos-.5*v2(f32(dim.x),f32(dim.y)))/f32(dim.y);
    let val=wrp(v2(abs(uv.x),uv.y)*p.zoom,p.seed);
    let msk=smoothstep(p.threshold-.1,p.threshold+.1,val);
    
    var acc=v3(0.);var den=0.;
    let stp=i32(p.trace_steps);
    let spd=p.particle_speed*5.;

    // I trace the streamline, evolving the color at each step but one side only
    for(var i=0;i<stp;i++){
        let ip=iv2(pos);
        let safe=clamp(ip,iv2(0),iv2(dim)-1);
        let grad=textureLoad(tex1,safe,0).xy;
        let gl=length(grad);
        
        let flow=normalize(v2(-grad.y,grad.x)+grad*.2);
        let n=nz(pos*.05+p.seed)-.5;
        let dir=select(v2(cos(n*6.28),sin(n*6.28)),flow,gl>.001);
        pos+=dir*spd;
        
        let prg=f32(i)/f32(stp);
        let uvc=(pos-.5*v2(f32(dim.x),f32(dim.y)))/f32(dim.y);
        let col=pal(val,v2(abs(uvc.x),uvc.y)*2.,prg);
        let w=gl*msk;
        
        acc+=col*w;den+=w;
    }
    
    let nRgb=acc*.15;let nDen=den*.5;
    let fRgb=clamp(old.rgb+nRgb,v3(0.),v3(50.));
    let fDen=clamp(old.a+nDen,0.,50.);
    
    let res=v4(fRgb,fDen)*p.particle_life;
    textureStore(out,id.xy,select(res,v4(0.),u_t.frame==0u));
}

// MAIN: Compositing
// I map the accumulated dynamic range ink onto a paper texture using log tone mapping.
@compute @workgroup_size(16,16,1)
fn main_image(@builtin(global_invocation_id) id:vec3<u32>){
    let dim=textureDimensions(out);
    if(id.x>=dim.x||id.y>=dim.y){return;}
    
    let dat=textureLoad(tex0,iv2(id.xy),0);
    let ink=dat.rgb;let den=dat.a;
    
    // Paper
    let g=nz(v2(id.xy)*.5)*.04;
    let pap=v3(.96,.95,.92)-v3(g);
    
    // Tone Map
    let map=v3(1.)-exp(-ink*(p.contrast*.5));
    let op=smoothstep(0.,1.,den*.5);
    var col=mix(pap,map*pap,op);
    
    // Post
    let uv=v2(id.xy)/v2(dim);
    let v=uv*(1.-uv);
    col*=pow(v.x*v.y*15.,.2);
    textureStore(out,id.xy,v4(pow(col,v3(1./p.gamma)),1.));
}
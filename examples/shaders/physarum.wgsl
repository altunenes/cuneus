// Enes Altun, 2026;
// This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 Unported License.

struct TimeUniform { time: f32, delta: f32, frame: u32, _padding: u32 };
@group(0) @binding(0) var<uniform> u_t: TimeUniform;

struct Params {
    sa: f32, sd: f32, drg: f32, spd: f32, dec: f32, dif: f32, dep: f32, jit: f32,
    rSd: f32, mSc: f32, fSc: f32, sGn: f32, sAt: f32, _s0: f32, str: f32, aSc: f32,
    glw: f32, cSh: f32, spc: f32, gam: f32, cSp: f32, sat: f32, pal: f32, rel: f32,
    tur: f32, rct: f32, stm: f32, vis: f32, org: f32, osz: f32, shn: f32, ohu: f32,
    bgl: f32, fod: f32, ogl: f32, ovr: f32,
    act: f32, bth: f32, spk: f32, eml: f32,
    mit: f32, fcs: f32, apr: f32, fdr: f32,
    sdf: f32, edb: f32, srd: f32, fcx: f32,
    fcy: f32, mtr: f32, vac: f32, pak: f32,
    cal: f32, cyt: f32, tbe: f32, bgf: f32,
    tml: f32, imm: f32, swl: f32, aAB: f32,
    aBC: f32, aCA: f32, stn: f32, dvs: f32,
    prp: f32, _i0: f32, _i1: f32, _i2: f32,
};
@group(1) @binding(0) var out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> p: Params;
@group(2) @binding(0) var<storage, read_write> atm: array<atomic<u32>>;

@group(3) @binding(0) var t0: texture_2d<f32>; @group(3) @binding(1) var s0: sampler;
@group(3) @binding(2) var t1: texture_2d<f32>; @group(3) @binding(3) var s1: sampler;
@group(3) @binding(4) var t2: texture_2d<f32>; @group(3) @binding(5) var s2: sampler;

alias v2 = vec2<f32>; alias v3 = vec3<f32>; alias v4 = vec4<f32>; alias u3 = vec3<u32>; alias i2 = vec2<i32>;
const pi: f32 = 3.14159265; const tau: f32 = 6.28318530;
const asc: f32 = 256.; const aiv: f32 = 1./256.; const msc: f32 = 64.;

fn pcg(s:u32)->u32{var st=s*747796405u+2891336453u;var w=((st>>((st>>28u)+4u))^st)*277803737u;return (w>>22u)^w;}
fn h1(s:u32)->f32{return f32(pcg(s))/4294967295.;}
fn hv(s:u32)->v4{return v4(h1(s),h1(s+1u),h1(s+2u),h1(s+3u))*2.-1.;}
// lengths are in reference pixels at 1080 height
fn rk(h:f32)->f32{return h/1080.;}
fn ld(t:texture_2d<f32>,c:i2,d:i2)->v4{return textureLoad(t,(c%d+d)%d,0);}

// Bivariate Tensor Product Chebyshev behavior map
// per species: 5 terms x [w1, w2, coef, coef, (coef, coef, mc), out]
var<workgroup> rw: array<v4, 90>;

fn rule(k:u32)->v4{
    let sp=k/30u;let r=k%30u;let i=r/6u;
    let b=u32(p.rSd*10000.)+i*24u;let m=u32(p.rSd*10000.)+50000u+sp*7919u+i*12u;let mt=p.mSc;
    switch r%6u {
        case 0u: {return hv(b)+mt*hv(m);}
        case 1u: {return hv(b+4u)+mt*hv(m+4u);}
        case 2u: {return hv(b+8u);}
        case 3u: {return hv(b+12u);}
        case 4u: {return v4(h1(b+16u)*2.-1.,h1(b+17u)*2.-1.,1.+mt*(h1(m+8u)-.5),0.);}
        default: {return hv(b+19u);}
    }
}

fn cheby(ir:v4,sp:u32)->v4{
    var res=v4(0.);let inp=ir*.4;
    for(var i=0u;i<5u;i++){
        let k=sp*30u+i*6u;
        let s=tanh(dot(inp,rw[k]));let t=tanh(dot(inp,rw[k+1u]));
        let a=rw[k+2u];let b=rw[k+3u];let c=rw[k+4u];
        let S2=2.*s*s-1.;let S3=s*(4.*s*s-3.);let T2=2.*t*t-1.;let T3=t*(4.*t*t-3.);
        let v=(a.x*s*t+a.y*s+a.z*t+a.w*S2*.5+b.x*T2*.5+b.y*s*T2*.35+b.z*S2*t*.35+b.w*S2*T2*.15+c.x*S3*t*.08+c.y*s*T3*.08)*c.z;
        res+=rw[k+5u]*v;
    }
    return tanh(res*.15)*.4;
}

// own-species trail
fn fsp(q:v2,cv:v2,sp:u32)->f32{let s=textureSampleLevel(t1,s1,fract(q/cv),0.);return select(select(s.z,s.y,sp==1u),s.x,sp==0u);}

// Sensor Read
fn sns(pt:v2,sp:u32,cv:v2)->v2{
    let s=textureSampleLevel(t1,s1,fract(pt/cv),0.);
    var o=0.;var t=0.;
    switch sp {
        case 0u: {o=s.x;t=s.y+s.z;}
        case 1u: {o=s.y;t=s.x+s.z;}
        default: {o=s.z;t=s.x+s.y;}
    }
    // fresh food attracts
    return v2(o+s.w*p.fod*.5,t*p.sAt);
}

// bilinear deposit, texel centres at i+.5
fn splat(q:v2,a:f32,off:u32,cw:i32,ch:i32){
    let g=q-.5;let fl=floor(g);let f=g-fl;let i=i2(fl);
    let x0=(i.x%cw+cw)%cw;let y0=(i.y%ch+ch)%ch;let x1=(x0+1)%cw;let y1=(y0+1)%ch;
    atomicAdd(&atm[off+u32(y0*cw+x0)],u32(a*(1.-f.x)*(1.-f.y)));
    atomicAdd(&atm[off+u32(y0*cw+x1)],u32(a*f.x*(1.-f.y)));
    atomicAdd(&atm[off+u32(y1*cw+x0)],u32(a*(1.-f.x)*f.y));
    atomicAdd(&atm[off+u32(y1*cw+x1)],u32(a*f.x*f.y));
}

@compute @workgroup_size(16,16,1)
fn agent_update(@builtin(global_invocation_id) id:u3,@builtin(local_invocation_index) li:u32){
    if(li<90u){rw[li]=rule(li);}
    workgroupBarrier();
    // rows [0,h): coarse position + velocity, rows [h,2h): fine position
    let dd=textureDimensions(out);let d=vec2<u32>(dd.x,dd.y/2u);if(id.x>=d.x||id.y>=d.y){return;}
    let cv=v2(textureDimensions(t1));let aid=id.y*d.x+id.x;let lc=vec2<u32>(id.x,id.y+d.y);
    if(aid>=u32(f32(d.x*d.y)*clamp(p.aSc,.05,1.))){return;}

    let sp=aid%3u;let sd=aid*17u+u_t.frame*7919u;
    var pos:v2;var vel:v2;var np:v2;var own=v2(0.);

    if(u_t.frame<2u){
        // Edge Spawn Logic
        let h_1 = h1(sd); let h_2 = h1(sd + 1u);
        let cx = cv * 0.5; let min_dim = min(cv.x, cv.y);
        let dist_from_center = min_dim * 0.38;
        var offset: v2;
        switch sp {
            case 0u: { offset = v2(-dist_from_center, 0.0); }
            case 1u: { offset = v2(dist_from_center, -dist_from_center * 0.6); }
            default: { offset = v2(dist_from_center, dist_from_center * 0.6); }
        }
        let r = sqrt(h_1) * (min_dim * 0.15);
        let theta = h_2 * tau;
        pos = cx + offset + v2(r * cos(theta), r * sin(theta));
        let inward_dir = normalize(-offset);
        let spray_angle = atan2(inward_dir.y, inward_dir.x) + (h_2 - 0.5) * 0.5;
        vel = v2(cos(spray_angle), sin(spray_angle)) * p.spd * rk(cv.y) * 0.5;
        np = pos;
    }else{
        let dt=textureLoad(t0,vec2<i32>(id.xy),0);pos=dt.xy+textureLoad(t0,vec2<i32>(lc),0).xy;vel=dt.zw;
        let hd=select(h1(sd+10u)*tau,atan2(vel.y,vel.x),length(vel)>.01);

        let k=rk(cv.y);let spd=p.spd*k;
        let sa=p.sa;let sdt=p.sd*k;
        let F =sns(pos+v2(cos(hd),sin(hd))*sdt,sp,cv);
        let FL=sns(pos+v2(cos(hd+sa*.5),sin(hd+sa*.5))*sdt,sp,cv);
        let FR=sns(pos+v2(cos(hd-sa*.5),sin(hd-sa*.5))*sdt,sp,cv);
        let L =sns(pos+v2(cos(hd+sa),sin(hd+sa))*sdt,sp,cv);
        let R =sns(pos+v2(cos(hd-sa),sin(hd-sa))*sdt,sp,cv);

        let own_lr = (L.x - R.x) + (FL.x - FR.x) * 0.5;
        let cross_lr = (L.y - R.y) + (FL.y - FR.y) * 0.5;
        let iv = v4(F.x, own_lr, F.y, cross_lr) * p.sGn;

        let bO = cheby(iv, sp);
        let mO = cheby(v4(iv.x, -iv.y, iv.z, -iv.w), sp);

        let fw=v2(cos(hd),sin(hd));let lf=v2(-sin(hd),cos(hd));

        let wF=fw*(bO.x+mO.x)*p.fSc+lf*(bO.y-mO.y)*p.fSc;
        let wS=(fw*(bO.z+mO.z)*p.str+lf*(bO.w-mO.w)*p.str)*k;
        let cross_steer = fw * F.y + lf * cross_lr;
        let jit=(h1(sd+6u)*2.-1.)*p.jit;

        // phase separation: affinities (+ toward, - away) and swirl
        var ps=v2(0.);let e=2.*k;
        if(p.imm>0.||p.swl>0.){
            let cN=textureSampleLevel(t1,s1,fract((pos+v2(0.,e))/cv),0.).xyz;let cS=textureSampleLevel(t1,s1,fract((pos-v2(0.,e))/cv),0.).xyz;
            let cE=textureSampleLevel(t1,s1,fract((pos+v2(e,0.))/cv),0.).xyz;let cW=textureSampleLevel(t1,s1,fract((pos-v2(e,0.))/cv),0.).xyz;
            let om=select(select(v3(1.,1.,0.),v3(1.,0.,1.),sp==1u),v3(0.,1.,1.),sp==0u);
            let af=select(select(v3(p.aCA,p.aBC,0.),v3(p.aAB,0.,p.aBC),sp==1u),v3(0.,p.aAB,p.aCA),sp==0u);
            let aT=dot(cN+cS+cE+cW,om);
            let wd=v2(-(dot(cN,v3(1.))-dot(cS,v3(1.))),dot(cE,v3(1.))-dot(cW,v3(1.)))*p.swl;
            let pb=v2(dot(cE-cW,af),dot(cN-cS,af))*p.imm*6.;
            ps=(wd+pb)*max(smoothstep(.01,1.,aT),h1(aid*113u)*.5)*spd;
        }
        // surface tension (+ round, - fingering)
        if(p.stn!=0.){
            let X=v2(3.*k,0.);let Y=v2(0.,3.*k);let f0=fsp(pos,cv,sp);
            let fx1=fsp(pos+X,cv,sp);let fx0=fsp(pos-X,cv,sp);let fy1=fsp(pos+Y,cv,sp);let fy0=fsp(pos-Y,cv,sp);
            let lxp=fsp(pos+2.*X,cv,sp)+f0+fsp(pos+X+Y,cv,sp)+fsp(pos+X-Y,cv,sp)-4.*fx1;
            let lxm=fsp(pos-2.*X,cv,sp)+f0+fsp(pos-X+Y,cv,sp)+fsp(pos-X-Y,cv,sp)-4.*fx0;
            let lyp=fsp(pos+2.*Y,cv,sp)+f0+fsp(pos+Y+X,cv,sp)+fsp(pos+Y-X,cv,sp)-4.*fy1;
            let lym=fsp(pos-2.*Y,cv,sp)+f0+fsp(pos-Y+X,cv,sp)+fsp(pos-Y-X,cv,sp)-4.*fy0;
            var st=v2(lxp-lxm,lyp-lym)*p.stn*.6*spd;let sl_=length(st);if(sl_>spd*1.5){st*=spd*1.5/sl_;}
            ps+=st;
        }
        // active droplets: size cap, drift to food
        if(p.dvs>0.||p.prp>0.){
            let R=30.*k;let fe=8.*k;
            ps-=v2(fsp(pos+v2(R,0.),cv,sp)-fsp(pos-v2(R,0.),cv,sp),fsp(pos+v2(0.,R),cv,sp)-fsp(pos-v2(0.,R),cv,sp))*p.dvs*.5*spd;
            let fdg=v2(textureSampleLevel(t1,s1,fract((pos+v2(fe,0.))/cv),0.).w-textureSampleLevel(t1,s1,fract((pos-v2(fe,0.))/cv),0.).w,
                       textureSampleLevel(t1,s1,fract((pos+v2(0.,fe))/cv),0.).w-textureSampleLevel(t1,s1,fract((pos-v2(0.,fe))/cv),0.).w);
            ps+=fdg*p.prp*4.*spd;
        }
        let pl_=length(ps);if(pl_>spd*4.){ps*=spd*4./pl_;}
        let acc=(wF*spd+cross_steer*spd*.5+v2(cos(hd+jit),sin(hd+jit))*p.jit*.5*k+ps)*1.25;
        vel=mix(acc,vel,clamp(p.drg,0.,.999));
        let mxS=spd*4.;let cS_=length(vel);
        if(cS_>mxS){vel*=mxS/cS_;}

        // carried by the streaming flow
        let u=textureSampleLevel(t2,s2,fract(pos/cv),0.).xy;
        own=vel+wS;np=pos+own+u;
    }

    let wp=np-cv*floor(np/cv);let wh=floor(wp/8.)*8.;
    textureStore(out,id.xy,v4(wh,vel));textureStore(out,lc,v4(wp-wh,0.,1.));

    // deposit along the path
    let cw=i32(cv.x);let ch=i32(cv.y);let off=sp*u32(cw*ch);
    // same trail density per reference area at any resolution
    let k=rk(cv.y);
    let dp=p.dep*k*k*max(1.5-length(vel)/max(p.spd*4.*k,.01),.5)*asc;
    let sg=np-pos;let n=clamp(u32(length(sg)/(4.*k))+1u,1u,3u);let a=dp/f32(n);
    for(var k=1u;k<=n;k++){splat(pos+sg*(f32(k)/f32(n)),a,off,cw,ch);}

    // momentum into the flow grid
    if(p.stm>0.){
        let hdm=i2(textureDimensions(t2));let hs=u32(hdm.x*hdm.y);let mo=arrayLength(&atm)-3u*hs;
        let c=min(i2(fract(np/cv)*v2(hdm)),hdm-1);let hi=mo+u32(c.y*hdm.x+c.x);
        atomicAdd(&atm[hi],bitcast<u32>(i32(round(own.x*msc))));
        atomicAdd(&atm[hi+hs],bitcast<u32>(i32(round(own.y*msc))));
        atomicAdd(&atm[hi+2u*hs],1u);
    }
}

// half-res flow, velocity in full-res px/frame
@compute @workgroup_size(16,16,1)
fn force(@builtin(global_invocation_id) id:u3){
    let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}
    let px=1./v2(d);let uv=(v2(id.xy)+.5)*px;
    let hs=d.x*d.y;let i=arrayLength(&atm)-3u*hs+id.y*d.x+id.x;
    let m=v2(f32(bitcast<i32>(atomicExchange(&atm[i],0u))),f32(bitcast<i32>(atomicExchange(&atm[i+hs],0u))))/msc;
    let n=f32(atomicExchange(&atm[i+2u*hs],0u));

    // self advection
    let bu=uv-textureSampleLevel(t0,s0,uv,0.).xy*px*.5;
    var u=textureSampleLevel(t0,s0,fract(bu),0.).xy;
    let a=(textureSampleLevel(t0,s0,fract(bu+v2(px.x,0.)),0.).xy+textureSampleLevel(t0,s0,fract(bu-v2(px.x,0.)),0.).xy
          +textureSampleLevel(t0,s0,fract(bu+v2(0.,px.y)),0.).xy+textureSampleLevel(t0,s0,fract(bu-v2(0.,px.y)),0.).xy)*.25;
    u=mix(u,a,clamp(p.vis,0.,.95));

    // agent drag
    u+=(m/max(n,1.)-u)*(n/(n+8.))*p.stm*.1;
    textureStore(out,id.xy,v4(u*.995,0.,1.));
}

@compute @workgroup_size(16,16,1)
fn pressure(@builtin(global_invocation_id) id:u3){
    let d=i2(textureDimensions(out));let c=i2(id.xy);if(c.x>=d.x||c.y>=d.y){return;}
    let X=i2(1,0);let Y=i2(0,1);
    let div=.5*(ld(t0,c+X,d).x-ld(t0,c-X,d).x+ld(t0,c+Y,d).y-ld(t0,c-Y,d).y);
    let pr=(ld(t1,c+X,d).x+ld(t1,c-X,d).x+ld(t1,c+Y,d).x+ld(t1,c-Y,d).x-div)*.25;
    textureStore(out,id.xy,v4(pr*.998,0.,0.,1.));
}

@compute @workgroup_size(16,16,1)
fn flow(@builtin(global_invocation_id) id:u3){
    let d=i2(textureDimensions(out));let c=i2(id.xy);if(c.x>=d.x||c.y>=d.y){return;}
    let X=i2(1,0);let Y=i2(0,1);
    let g=.5*v2(ld(t1,c+X,d).x-ld(t1,c-X,d).x,ld(t1,c+Y,d).x-ld(t1,c-Y,d).x);
    textureStore(out,id.xy,v4(ld(t0,c,d).xy-g,0.,1.));
}

fn cr(t:f32)->v4{let t2=t*t;let t3=t2*t;return v4(-.5*t3+t2-.5*t,1.5*t3-2.5*t2+1.,-1.5*t3+2.*t2+.5*t,.5*t3-.5*t2);}

// cubic advection, clamped to source texels
fn adv(c:i2,u:v2,d:i2)->v4{
    if(dot(u,u)<1e-6){return ld(t0,c,d);}
    let g=v2(c)-u;let fl=floor(g);let f=g-fl;let i=i2(fl);
    let wx=cr(f.x);let wy=cr(f.y);
    var a=v4(0.);var lo=v4(1e9);var hi=v4(-1e9);
    for(var j=0;j<4;j++){for(var k=0;k<4;k++){
        let v=ld(t0,i+i2(k-1,j-1),d);a+=v*wx[k]*wy[j];
        if(j>0&&j<3&&k>0&&k<3){lo=min(lo,v);hi=max(hi,v);}
    }}
    return clamp(a,lo,hi);
}

@compute @workgroup_size(16,16,1)
fn trail_adv(@builtin(global_invocation_id) id:u3){
    let d=i2(textureDimensions(out));let c=i2(id.xy);if(c.x>=d.x||c.y>=d.y){return;}
    let u=textureSampleLevel(t1,s1,(v2(id.xy)+.5)/v2(d),0.).xy;

    let s=adv(c,u,d).xyz;let fd=ld(t0,c,d).w;

    let pi=u32(c.y*d.x+c.x);let st=u32(d.x*d.y);
    let dp=v3(f32(atomicExchange(&atm[pi],0u)),f32(atomicExchange(&atm[pi+st],0u)),f32(atomicExchange(&atm[pi+2u*st],0u)))*aiv*.002;
    // well fed agents deposit more
    let fe=mix(1.,.1+.9*fd,clamp(p.fod,0.,1.));
    textureStore(out,id.xy,v4(s*p.dec+dp*fe*max(v3(0.),v3(1.)-s/3.),fd));
}

@compute @workgroup_size(16,16,1)
fn diffuse_h(@builtin(global_invocation_id) id:u3){
    let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}
    let uv=(v2(id.xy)+.5)/v2(d);let px=rk(f32(d.y))/f32(d.x);

    var s=textureSampleLevel(t0,s0,uv,0.).xyz*.382928;
    s+=textureSampleLevel(t0,s0,fract(uv+v2(px,0.)),0.).xyz*.241732;
    s+=textureSampleLevel(t0,s0,fract(uv-v2(px,0.)),0.).xyz*.241732;
    s+=textureSampleLevel(t0,s0,fract(uv+v2(px*2.,0.)),0.).xyz*.060598;
    s+=textureSampleLevel(t0,s0,fract(uv-v2(px*2.,0.)),0.).xyz*.060598;
    textureStore(out,id.xy,v4(s,1.));
}

@compute @workgroup_size(16,16,1)
fn diffuse_v(@builtin(global_invocation_id) id:u3){
    let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}
    let uv=(v2(id.xy)+.5)/v2(d);let py=rk(f32(d.y))/f32(d.y);

    var s=textureSampleLevel(t0,s0,uv,0.).xyz*.382928;
    s+=textureSampleLevel(t0,s0,fract(uv+v2(0.,py)),0.).xyz*.241732;
    s+=textureSampleLevel(t0,s0,fract(uv-v2(0.,py)),0.).xyz*.241732;
    s+=textureSampleLevel(t0,s0,fract(uv+v2(0.,py*2.)),0.).xyz*.060598;
    s+=textureSampleLevel(t0,s0,fract(uv-v2(0.,py*2.)),0.).xyz*.060598;
    textureStore(out,id.xy,v4(s,1.));
}

@compute @workgroup_size(16,16,1)
fn inhibitor_down(@builtin(global_invocation_id) id:u3){
    let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}
    let uv=(v2(id.xy)+.5)/v2(d);let px=1./v2(d);
    var s=v3(0.);
    for(var y=-1.;y<=1.;y+=1.){for(var x=-1.;x<=1.;x+=1.){s+=textureSampleLevel(t0,s0,fract(uv+v2(x*px.x,y*px.y)),0.).xyz;}}
    textureStore(out,id.xy,v4(s/9.,1.));
}

// trail state: diffusion + turing + cyclic dominance
@compute @workgroup_size(16,16,1)
fn field(@builtin(global_invocation_id) id:u3){
    let d=i2(textureDimensions(out));let c=i2(id.xy);if(c.x>=d.x||c.y>=d.y){return;}
    let a=ld(t0,c,d).xyz;let b=ld(t1,c,d).xyz;
    let inh=textureSampleLevel(t2,s2,(v2(id.xy)+.5)/v2(d),0.).xyz;

    var s=max(v3(0.),mix(a,b,clamp(p.dif,0.,1.))+(b-inh)*p.tur*.1);
    s+=p.rct*.02*s*v3(s.y-s.z,s.z-s.x,s.x-s.y)*max(v3(0.),v3(1.)-s/3.);
    // food: eaten under trails, slowly regrows
    var fd=ld(t0,c,d).w;
    fd=select(clamp(fd+.001*(1.-fd)-.0015*p.fod*fd*(a.x+a.y+a.z),0.,1.),1.,u_t.frame<2u);
    textureStore(out,id.xy,v4(clamp(s,v3(0.),v3(3.)),fd));
}

// organelles
fn ogen(i:u32)->v2{let ph=u_t.time*.04+h1((i>>1u)*7u+1u);return v2(floor(ph),fract(ph));}
// mitosis: pair shares one look, odd one hidden until the split
const MD0:f32=.35;const MD1:f32=.5;
fn odiv(i:u32,gx:f32)->bool{return h1(pcg((i>>1u)*48271u+u32(gx)*7919u+13u))<p.mit;}
fn oaxis(i:u32,gx:f32)->v2{let a=h1(pcg((i>>1u)*16807u+u32(gx)*104729u))*tau;return v2(cos(a),sin(a));}
fn opos(s:v4)->v2{return s.xz+s.yw;}
fn dens(q:v2,cv:v2)->f32{let s=textureSampleLevel(t2,s2,fract(q/cv),0.).xyz;return s.x+s.y+s.z;}

struct Org { pos:v2, dir:v2, r:f32, asp:f32, ty:u32, hs:u32, dv:f32, ext:f32, cc:f32, vc:f32 };

// 0 vesicle, 1 rod, 2 nucleus
// sp: 1 once the pair has split
fn oget(i:u32,pos:v2,cv:v2,sp:f32)->Org{
    let g=ogen(i);let dvp=odiv(i,g.x);
    let hs=pcg(select(i,i&~1u,dvp)*9781u+u32(g.x)*6271u);
    let tv=h1(hs);var ty=0u;if(tv>.55){ty=1u;}if(tv>.8){ty=2u;}if(dvp){ty=2u;}
    let S=(4.+10.*p.osz)*rk(cv.y);
    var r0=S*(.35+.25*h1(hs+1u));if(ty==1u){r0=S*.4;}if(ty==2u){r0=S*(.9+.2*h1(hs+1u));}
    var env=smoothstep(0.,.1,g.y)*(1.-smoothstep(.85,1.,g.y));
    let m=smoothstep(.8,2.2,dens(pos,cv));
    let u=textureSampleLevel(t1,s1,fract(pos/cv),0.).xy;let spd=length(u);
    let a0=h1(hs+2u)*tau;var dir=select(v2(cos(a0),sin(a0)),u/max(spd,1e-4),spd>.05);
    var asp=1.+min(spd*.3,.5);if(ty==1u){asp=2.4;}
    // contractile vacuole
    var vc=0.;
    if(ty==0u&&!dvp&&h1(hs+12u)<p.vac){
        let ph=fract(u_t.time/6.+h1(hs+13u));
        vc=select(mix(1.7,.8,smoothstep(.85,1.,ph)),mix(.8,1.7,smoothstep(0.,.85,ph)),ph<.85);r0*=vc;
    }
    var dv=-1.;var ext=asp;
    if(dvp){
        let s=(g.y-MD0)/(MD1-MD0);
        if(sp<.5){
            if((i&1u)==1u){env=0.;}else if(s>=0.&&s<1.){dv=s;dir=oaxis(i,g.x);asp=1.;ext=2.4;}
        }else{r0*=mix(.8,1.,smoothstep(MD1,MD1+.2,g.y));}
    }
    // dof: depth and blur radius
    let z=clamp((h1(hs+7u)*2.-1.)*.8+.2*sin(u_t.time*.25+h1(hs+8u)*tau),-1.,1.);
    let cc=min(p.apr*max(0.,abs(z-p.fcs)-p.fdr)*14.+rblur(pos/cv,cv),24.)*rk(cv.y);
    return Org(pos,dir,r0*env*m,asp,ty,hs,dv,ext,cc,vc);
}

// height, membrane, inner structure, type + variation
fn oshape(o:Org,q:v2,cv:v2,ex:f32)->v4{
    if(o.r<.3){return v4(0.);}
    var d=q-o.pos;d-=cv*round(d/cv);
    let vb=(floor(h1(o.hs+6u)*7.99)+min(ex,.95))/8.;
    var nd:f32;var ring:f32;var inn=0.;
    if(o.dv>=0.){
        // dividing: two lobes
        let s=smoothstep(.3,1.,o.dv);let r=o.r*mix(1.+.15*smoothstep(0.,.3,o.dv),.8,s);
        let c=o.dir*o.r*1.05*s;
        // field in log space so the distance stays exact far from the cell
        let ea=-3.*dot(d-c,d-c)/(r*r);let eb=-3.*dot(d+c,d+c)/(r*r);let em=max(ea,eb);
        nd=sqrt(max(-(log(mix(.5,1.,s))+em+log(1.+exp(min(ea,eb)-em)))/3.,0.));
        // nuclear envelope
        ring=exp(-pow((nd-.86)/.09,2.))*(1.-smoothstep(.1,.3,o.dv)*(1.-smoothstep(.85,1.,o.dv)));
        // chromosomes
        let xa=dot(d,o.dir);let ya=abs(dot(d,v2(-o.dir.y,o.dir.x)));let xc=o.r*.8*smoothstep(.45,.9,o.dv);
        inn=exp(-pow((abs(xa)-xc)/(.12*o.r),2.))*(1.-smoothstep(.35*o.r,.6*o.r,ya))*smoothstep(.15,.3,o.dv);
    }else{
        let x=dot(d,o.dir);let y=dot(d,v2(-o.dir.y,o.dir.x));
        if(o.ty==1u){nd=length(v2(max(abs(x)-o.r*(o.asp-1.),0.),y))/o.r;}
        else{
            let th=atan2(y,x);let ph=h1(o.hs+3u)*tau;
            nd=length(v2(x/o.asp,y))/(o.r*(1.+.1*sin(3.*th+ph+u_t.time*.6)+.06*sin(5.*th-ph)));
        }
        ring=exp(-pow((nd-.86)/.09,2.));
        if(o.ty==2u){
            let c=v2(.25,.15)*o.r*(v2(h1(o.hs+4u),h1(o.hs+5u))*2.-1.);
            inn=1.-smoothstep(.22,.32,length(v2(x,y)-c)/o.r);
        }else if(o.ty==1u){inn=(.5+.5*sin(x/o.r*7.))*smoothstep(.9,.4,nd)*.6;ring*=.8;}
        else{ring*=select(.6,1.,o.vc>0.);}
    }
    // defocus
    let dp=(nd-1.)*o.r;
    if(dp>=max(o.cc,0.)){return v4(0.);}
    var h=sqrt(max(0.,1.-nd*nd))*select(1.,.55,o.vc>0.);var bb=0.;
    if(o.cc>.05){
        // bokeh disc
        bb=smoothstep(0.,o.r*.6,o.cc);
        let w=max(o.cc*.35,.7);let A=clamp(1.2*o.r/(o.r+o.cc),.25,1.);
        let dk=A*(1.-smoothstep(o.cc-w,o.cc,dp)+.35*exp(-pow((dp-o.cc+w*1.2)/(w*.8),2.)));
        h=mix(h,dk,bb);inn*=1.-bb;
    }
    if(h<=0.){return v4(0.);}
    // y: membrane, or -blur when defocused; w: type + (variation bucket + excitation)/8
    return v4(min(h,1.),select(ring,-bb,bb>.01),inn,select(f32(o.ty),2.,o.dv>=0.)+vb);
}

// signal grid cell, about 8 reference pixels
// radial blur around the focus point
fn rblur(uv:v2,cv:v2)->f32{let r=length((uv-v2(p.fcx,p.fcy))*v2(cv.x/cv.y,1.));return p.edb*smoothstep(p.srd,p.srd+.6,r)*20.;}
fn gcell(h:f32)->i32{return max(4,i32(round(8.*rk(h))));}
// packing grid cell, about 16 reference pixels
fn pcell(h:f32)->i32{return max(8,i32(round(16.*rk(h))));}

@compute @workgroup_size(16,16,1)
fn organ(@builtin(global_invocation_id) id:u3){
    // rows [0,h): position, [h,2h): excitation, refractory, charge, generation, [2h,3h): motor heading, run timer
    let dd=textureDimensions(out);let d=vec2<u32>(dd.x,dd.y/3u);if(id.x>=d.x||id.y>=d.y){return;}
    let i=id.y*d.x+id.x;let cv=v2(textureDimensions(t2));let sc=vec2<u32>(id.x,id.y+d.y);let mc=vec2<u32>(id.x,id.y+2u*d.y);
    let pp0=opos(textureLoad(t0,i2(id.xy),0));var pos=pp0;var ex=textureLoad(t0,i2(sc),0);var mo=textureLoad(t0,i2(mc),0);
    let k=rk(cv.y);let cw=i32(cv.x);let ch=i32(cv.y);let st=u32(cw*ch);
    // packing grid, double buffered: count, x sum, y sum
    let cz=gcell(cv.y);let gw=(cw+cz-1)/cz;let gh=(ch+cz-1)/cz;let gs=u32(gw*gh);
    let pz=pcell(cv.y);let pw=(cw+pz-1)/pz;let ph_=(ch+pz-1)/pz;let pgs=u32(pw*ph_);
    let pb=5u*st+2u*gs;let pwr=pb+(u_t.frame%2u)*3u*pgs;let prd=pb+((u_t.frame+1u)%2u)*3u*pgs;
    let g=ogen(i);let dvp=odiv(i,g.x);
    if(u_t.frame<2u||g.x!=floor(ex.w)){
        // respawn at the densest of 8 candidates
        var bd=-1.;
        for(var k=0u;k<8u;k++){
            let hs=pcg(i*131u+k*7919u+u_t.frame*977u);let c=v2(h1(hs),h1(hs+1u))*cv;let dn=dens(c,cv);
            if(dn>bd){bd=dn;pos=c;}
        }
        ex=v4(0.,0.,0.,g.x);mo=v4(h1(i*977u+u_t.frame)*tau,-h1(i*331u+u_t.frame),0.,0.);
    }else{
        let u=textureSampleLevel(t1,s1,fract(pos/cv),0.).xy;
        let e=3.*k;let gd=v2(dens(pos+v2(e,0.),cv)-dens(pos-v2(e,0.),cv),dens(pos+v2(0.,e),cv)-dens(pos-v2(0.,e),cv));
        let hs=pcg(i*31u+u_t.frame*7919u);let o0=oget(i,pos,cv,fract(ex.w)*2.);
        // brownian ~ 1/sqrt(r)
        let bj=.25*k*clamp(sqrt((4.+10.*p.osz)*k*.6/max(o0.r,.5)),.5,2.);
        pos+=u+clamp(gd*.15,v2(-.4),v2(.4))*k+(v2(h1(hs),h1(hs+1u))-.5)*bj;

        // motor transport
        if(o0.ty==0u&&h1(o0.hs+11u)<p.mtr){
            let dt=u_t.delta;
            if(mo.y>0.){mo.y-=dt;if(mo.y<=0.){mo.y=-(.2+.8*h1(hs+2u));}}
            else{mo.y+=dt;if(mo.y>=0.){mo.y=.5+h1(hs+3u);}}
            if(mo.y>0.){
                let e2=4.*k;let f0=dens(pos,cv);
                let hxx=dens(pos+v2(e2,0.),cv)+dens(pos-v2(e2,0.),cv)-2.*f0;
                let hyy=dens(pos+v2(0.,e2),cv)+dens(pos-v2(0.,e2),cv)-2.*f0;
                let hxy=(dens(pos+v2(e2,e2),cv)-dens(pos+v2(e2,-e2),cv)-dens(pos+v2(-e2,e2),cv)+dens(pos+v2(-e2,-e2),cv))*.25;
                let th=.5*atan2(2.*hxy,hxx-hyy);var ax=v2(cos(th),sin(th));
                if(dot(ax,v2(cos(mo.x),sin(mo.x)))<0.){ax=-ax;}
                // only along a ridge
                let rg=sqrt((hxx-hyy)*(hxx-hyy)+4.*hxy*hxy);
                if(rg>.08){mo.x=atan2(ax.y,ax.x);pos+=ax*2.5*k*smoothstep(0.,.15,mo.y)*smoothstep(.08,.3,rg);}
            }
        }

        // soft packing
        if(p.pak>0.&&o0.r>=.3){
            let pc0=i2(floor(pos/f32(pz)));let ps=i2(floor(pp0/f32(pz)));var pu=v2(0.);
            for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
                let c=((pc0+i2(x,y))%i2(pw,ph_)+i2(pw,ph_))%i2(pw,ph_);let ci=u32(c.y*pw+c.x);
                var n=f32(atomicLoad(&atm[prd+ci]));
                var sm=v2(f32(atomicLoad(&atm[prd+pgs+ci])),f32(atomicLoad(&atm[prd+2u*pgs+ci])))/16.;
                if(all(c==((ps%i2(pw,ph_))+i2(pw,ph_))%i2(pw,ph_))){n-=1.;sm-=pp0-v2(c*pz);}
                if(n>=1.){
                    var dv_=pos-(v2(c*pz)+sm/n);dv_-=cv*round(dv_/cv);let dl=length(dv_);let r0=2.4*o0.r;
                    if(dl<r0&&dl>1e-3){pu+=dv_/dl*(r0-dl)/r0;}
                }
            }}
            let pl=length(pu);if(pl>0.){pos+=pu/pl*min(pl*p.pak*.8,1.5)*k;}
        }
        pos-=cv*floor(pos/cv);
        // split: both daughters jump to the mother's lobe centres
        if(dvp&&fract(ex.w)<.25&&g.y>=MD1){
            let mi=i&~1u;let mp=opos(textureLoad(t0,i2(i32(mi%d.x),i32(mi/d.x)),0));
            pos=mp+oaxis(i,g.x)*oget(mi,mp,cv,0.).r*1.05*select(1.,-1.,(i&1u)==1u);
            pos-=cv*floor(pos/cv);ex.w=g.x+.5;
        }
    }
    let hi=floor(pos/8.)*8.;
    textureStore(out,id.xy,v4(hi.x,pos.x-hi.x,hi.y,pos.y-hi.y));textureStore(out,mc,mo);

    let o=oget(i,pos,cv,fract(ex.w)*2.);let live=f32(i)<f32(d.x*d.y)*p.org&&o.r>=.3;
    if(live&&p.pak>0.){
        let c=min(i2(floor(pos/f32(pz))),i2(pw-1,ph_-1));let ci=u32(c.y*pw+c.x);let rl=(pos-v2(c*pz))*16.;
        atomicAdd(&atm[pwr+ci],1u);atomicAdd(&atm[pwr+pgs+ci],u32(rl.x));atomicAdd(&atm[pwr+2u*pgs+ci],u32(rl.y));
    }

    // action potentials
    let fb=4u*st;
    let wr=fb+(u_t.frame%2u)*gs;let rd=fb+((u_t.frame+1u)%2u)*gs;
    if(live&&p.act>0.){
        let gc=i2(floor(pos/f32(cz)));let ci=u32(gc.y%gh*gw+gc.x%gw);
        ex.z=ex.z*.9+f32(atomicLoad(&atm[rd+ci]))*.35;
        ex.x*=mix(.92,.975,p.cal);ex.y=max(0.,ex.y-1./150.);
        let pk=select(.0004,.004,o.ty==2u)*p.act;
        if(ex.y<=0.&&(ex.z>1.||h1(pcg(i*7717u+u_t.frame*3571u))<pk)){ex=v4(1.,1.,0.,ex.w);}
        if(ex.x>.5&&p.spk<=0.){
            for(var y=-3;y<=3;y++){for(var x=-3;x<=3;x++){
                let c=((gc+i2(x,y))%i2(gw,gh)+i2(gw,gh))%i2(gw,gh);
                atomicAdd(&atm[wr+u32(c.y*gw+c.x)],1u);
            }}
        }
    }else{ex=v4(0.,0.,0.,ex.w);}
    textureStore(out,sc,ex);

    // splat, highest dome wins
    if(!live){return;}
    let R=i32(ceil(o.r*(o.ext*1.4+.2)+o.cc))+1;
    let pc=i2(floor(pos));
    for(var y=-R;y<=R;y++){for(var x=-R;x<=R;x++){
        let pp=pc+i2(x,y);let s=oshape(o,v2(pp)+.5,cv,0.);
        if(s.x>0.){
            let w=i2((pp.x%cw+cw)%cw,(pp.y%ch+ch)%ch);
            atomicMax(&atm[3u*st+u32(w.y*cw+w.x)],((u32(s.x*1048574.)+1u)<<12u)|i);
        }
    }}
}

@compute @workgroup_size(16,16,1)
fn organ_res(@builtin(global_invocation_id) id:u3){
    let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}
    // clear the signal grid read this frame
    let cz=u32(gcell(f32(d.y)));let gw=(d.x+cz-1u)/cz;let gh=(d.y+cz-1u)/cz;
    if(id.x<gw&&id.y<gh){atomicStore(&atm[4u*d.x*d.y+((u_t.frame+1u)%2u)*gw*gh+id.y*gw+id.x],0u);}
    let pz=u32(pcell(f32(d.y)));let pw=(d.x+pz-1u)/pz;let ph_=(d.y+pz-1u)/pz;
    if(id.x<pw&&id.y<ph_){
        let pgs=pw*ph_;let prd=5u*d.x*d.y+2u*gw*gh+((u_t.frame+1u)%2u)*3u*pgs+id.y*pw+id.x;
        atomicStore(&atm[prd],0u);atomicStore(&atm[prd+pgs],0u);atomicStore(&atm[prd+2u*pgs],0u);
    }
    let v=atomicExchange(&atm[3u*d.x*d.y+id.y*d.x+id.x],0u);
    if(v==0u){textureStore(out,id.xy,v4(0.));return;}
    let k=v&4095u;let od=textureDimensions(t0);let cv=v2(d);
    let kc=i2(i32(k%od.x),i32(k/od.x));let ks=textureLoad(t0,kc+i2(0,i32(od.y/3u)),0);
    let o=oget(k,opos(textureLoad(t0,kc,0)),cv,fract(ks.w)*2.);
    textureStore(out,id.xy,oshape(o,v2(id.xy)+.5,cv,ks.x));
}
fn sdir(a:f32)->v2{return v2(cos(a),sin(a));}
fn soff(st:u32,cv:v2)->u32{let cz=gcell(cv.y);let gw=(i32(cv.x)+cz-1)/cz;let gh=(i32(cv.y)+cz-1)/cz;return 4u*st+2u*u32(gw*gh);}

@compute @workgroup_size(16,16,1)
fn spark(@builtin(global_invocation_id) id:u3){
    let dd=textureDimensions(out);let d=vec2<u32>(dd.x,dd.y/2u);if(id.x>=d.x||id.y>=d.y){return;}
    let i=id.y*d.x+id.x;let cv=v2(textureDimensions(t2));let k=rk(cv.y);let sc=vec2<u32>(id.x,id.y+d.y);
    var pos=opos(textureLoad(t0,i2(id.xy),0));var st=textureLoad(t0,i2(sc),0);
    if(u_t.frame<2u){st=v4(0.);}
    let j=i/2u;let od=textureDimensions(t1);let oc=i2(i32(j%od.x),i32(j/od.x));
    let fired=textureLoad(t1,oc+i2(0,i32(od.y/3u)),0).x>.999;
    let hs=pcg(i*7919u+u_t.frame*104729u);
    let cw=i32(cv.x);let ch=i32(cv.y);let stt=u32(cw*ch);

    if(fired&&p.spk>0.&&f32(j)<f32(od.x*od.y/3u)*p.org){
        pos=opos(textureLoad(t1,oc,0));let e=3.*k;
        let g=v2(dens(pos+v2(e,0.),cv)-dens(pos-v2(e,0.),cv),dens(pos+v2(0.,e),cv)-dens(pos-v2(0.,e),cv));
        var a=h1(hs)*tau;if(length(g)>1e-3){a=atan2(g.x,-g.y);}
        st=v4(a+select(0.,pi,i%2u==1u)+(h1(hs+1u)-.5)*.6,1.,0.,0.);
    }else if(st.y>0.){
        var a=st.x;let sd=7.*k;let sa=.45;
        let F=dens(pos+sdir(a)*sd,cv);let L=dens(pos+sdir(a+sa)*sd,cv);let R=dens(pos+sdir(a-sa)*sd,cv);
        if(L>F&&L>R){a+=.25;}else if(R>F&&R>L){a-=.25;}
        a+=(h1(hs)-.5)*.15;
        pos+=sdir(a)*6.*k;pos-=cv*floor(pos/cv);
        st.x=a;st.y-=1./45.;let dc=dens(pos,cv);if(dc<.6){st.y-=.25;}
        // no ridge (sheet): fade fast
        let pr=v2(-sin(a),cos(a))*5.*k;let rg=dc-.5*(dens(pos+pr,cv)+dens(pos-pr,cv));
        st.y-=(1.-smoothstep(.3,1.2,rg))*.08;
        if(st.y>0.){
            let tl=16.*k;let n=i32(tl*2.);let so=soff(stt,cv);
            for(var q=0;q<n;q++){
                let t=f32(q)/f32(n);let pp=pos-sdir(a)*tl*t;
                let g=pp-.5;let fl=floor(g);let f=g-fl;let c=i2(fl);
                let x0=(c.x%cw+cw)%cw;let y0=(c.y%ch+ch)%ch;let x1=(x0+1)%cw;let y1=(y0+1)%ch;
                let w=st.y*(1.-t)*(1.-t)*256.;
                atomicAdd(&atm[so+u32(y0*cw+x0)],u32(w*(1.-f.x)*(1.-f.y)));
                atomicAdd(&atm[so+u32(y0*cw+x1)],u32(w*f.x*(1.-f.y)));
                atomicAdd(&atm[so+u32(y1*cw+x0)],u32(w*(1.-f.x)*f.y));
                atomicAdd(&atm[so+u32(y1*cw+x1)],u32(w*f.x*f.y));
            }
            let cz=gcell(cv.y);let gw=(cw+cz-1)/cz;let gh=(ch+cz-1)/cz;
            let gc=i2(floor(pos/f32(cz)));
            atomicAdd(&atm[4u*stt+(u_t.frame%2u)*u32(gw*gh)+u32(gc.y%gh*gw+gc.x%gw)],3u);
        }
    }
    let hi=floor(pos/8.)*8.;
    textureStore(out,id.xy,v4(hi.x,pos.x-hi.x,hi.y,pos.y-hi.y));
    textureStore(out,sc,st);
}

fn dT(x:f32)->f32{let v=max(0.,x);return v/(v+.6);}
fn hRot(c:v3,a:f32)->v3{let k=v3(.57735);let ca=cos(a);let sa=sin(a);return c*ca+cross(k,c)*sa+k*dot(k,c)*(1.-ca);}
fn aces(x:f32)->f32{return clamp((x*(2.51*x+.03))/(x*(2.43*x+.59)+.14),0.,1.);}
fn lum(c:v3)->f32{return dot(c,v3(.2126,.7152,.0722));}

// palettes: columns edge, core, nucleus; k = palette*3 + species
fn pm(k:u32)->mat3x3<f32>{
    switch k {
        case 3u: {return mat3x3<f32>(.25,.12,.02, 1.,.75,.1, 1.,.95,.6);}
        case 4u: {return mat3x3<f32>(.3,.06,.02, 1.,.45,.05, 1.,.8,.45);}
        case 5u: {return mat3x3<f32>(.15,.14,.03, .8,.85,.2, .95,1.,.7);}
        case 6u: {return mat3x3<f32>(.2,.03,.1, .95,.35,.55, 1.,.75,.85);}
        case 7u: {return mat3x3<f32>(.1,.03,.2, .5,.25,.85, .8,.65,1.);}
        case 8u: {return mat3x3<f32>(.2,.05,.12, .85,.3,.7, 1.,.7,.95);}
        case 9u: {return mat3x3<f32>(0.,.08,.15, .1,.8,1., .7,1.,1.);}
        case 10u: {return mat3x3<f32>(0.,.12,.08, .1,1.,.6, .75,1.,.85);}
        case 11u: {return mat3x3<f32>(.08,.02,.18, .55,.3,1., .9,.8,1.);}
        case 12u: {return mat3x3<f32>(.2,.02,0., 1.,.25,.05, 1.,.85,.5);}
        case 13u: {return mat3x3<f32>(.15,.04,0., .95,.5,.05, 1.,.95,.7);}
        case 14u: {return mat3x3<f32>(.12,0.,.02, .8,.1,.15, 1.,.6,.5);}
        case 15u: {return mat3x3<f32>(.03,.1,.02, .35,.8,.15, .8,1.,.5);}
        case 16u: {return mat3x3<f32>(.1,.08,.02, .7,.6,.15, .95,.9,.55);}
        case 17u: {return mat3x3<f32>(.12,.04,.01, .75,.3,.1, 1.,.7,.4);}
        case 18u: {return mat3x3<f32>(.2,.02,.05, 1.,.2,.3, 1.,.8,.8);}
        case 19u: {return mat3x3<f32>(.02,.15,.05, .2,1.,.4, .8,1.,.85);}
        case 20u: {return mat3x3<f32>(.02,.05,.2, .25,.4,1., .8,.85,1.);}
        default: {
            var e=mat3x3<f32>(.4,.02,.05, .3,.15,.02, .02,.08,.2);var e2=mat3x3<f32>(.02,.2,.3, .2,.02,.3, .3,.2,.02);
            var c=mat3x3<f32>(1.,.4,.1, .8,.9,.1, .1,.99,.4);var c2=mat3x3<f32>(.2,1.,.8, .9,.2,1., 1.,.9,.2);
            var n=mat3x3<f32>(.1,.9,.8, 1.,.1,.7, .8,1.,.1);var n2=mat3x3<f32>(1.,.9,.9, 1.,.4,0., .2,.5,1.);
            let i=k%3u;return mat3x3<f32>(mix(e[i],e2[i],.57),mix(c[i],c2[i],.57),mix(n[i],n2[i],.57));
        }
    }
}
fn bgc(k:u32)->v3{
    switch k {
        case 1u: {return v3(.05,.035,.012);}
        case 2u: {return v3(.05,.025,.045);}
        case 3u: {return v3(0.,.018,.035);}
        case 4u: {return v3(.04,.01,0.);}
        case 5u: {return v3(.02,.03,.012);}
        case 6u: {return v3(.02,.02,.03);}
        default: {return v3(.02,.02,.025);}
    }
}
fn ramp(m:mat3x3<f32>,e:f32)->v3{return mix(mix(m[0],m[1],smoothstep(.1,.6,e)),m[2],smoothstep(.75,.95,e));}

fn hgt(uv:v2)->f32{let s=textureSampleLevel(t0,s0,fract(uv),0.);let x=s.x+s.y+s.z;return x/(x+1.5)+(1.-s.w)*.12;}

// cytoplasm granules
@compute @workgroup_size(16,16,1)
fn cyto(@builtin(global_invocation_id) id:u3){
    let d=i2(textureDimensions(out));let c=i2(id.xy);if(c.x>=d.x||c.y>=d.y){return;}
    let u=textureSampleLevel(t1,s1,(v2(id.xy)+.5)/v2(d),0.).xy;
    let av=adv(c,u,d);var g=av.x;
    let tr=ld(t2,c,d).xyz;let tm=smoothstep(.4,1.5,tr.x+tr.y+tr.z);
    g*=mix(.85,.992,tm);
    // fade in, tissue only
    let kr=rk(f32(d.y));let cs=3.*kr;let q=v2(id.xy)+.5;let cl=floor(q/cs);
    let hs=pcg(u32(cl.x)*73856093u^u32(cl.y)*19349663u^(u_t.frame/8u)*83492791u);
    if(h1(hs)<.012*tm){
        let ce=(cl+.3+.4*v2(h1(hs+1u),h1(hs+2u)))*cs;let r=cs*(.25+.2*h1(hs+3u));
        g+=max(0.,1.-dot(q-ce,q-ce)/(r*r))*.12;
    }
    // slime trace per species
    let mt=max(av.yzw*pow(.5,u_t.delta/max(p.tml,.1)),min(tr,v3(.4)));
    textureStore(out,id.xy,v4(clamp(g,0.,1.),mt));
}

// jump flood: offset from each tissue pixel to the nearest outside pixel (999 = none yet)
@compute @workgroup_size(16,16,1)
fn jinit(@builtin(global_invocation_id) id:u3){
    let d=i2(textureDimensions(out));let c=i2(id.xy);if(c.x>=d.x||c.y>=d.y){return;}
    let f=ld(t0,c,d);
    textureStore(out,id.xy,select(v4(0.,0.,0.,1.),v4(999.,999.,0.,1.),f.x+f.y+f.z>.6));
}
fn jstep(id:u3,k:i32)->v4{
    let d=i2(textureDimensions(out));let c=i2(id.xy);
    var b=ld(t0,c,d).xy;var bl=dot(b,b);
    for(var j=-1;j<=1;j++){for(var i=-1;i<=1;i++){
        let o=i2(i,j)*k;let r=ld(t0,c+o,d).xy;
        if(r.x<900.){let cd=v2(o)+r;let l=dot(cd,cd);if(l<bl){bl=l;b=cd;}}
    }}
    return v4(b,0.,1.);
}
@compute @workgroup_size(16,16,1)
fn j64(@builtin(global_invocation_id) id:u3){let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}textureStore(out,id.xy,jstep(id,64));}
@compute @workgroup_size(16,16,1)
fn j32(@builtin(global_invocation_id) id:u3){let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}textureStore(out,id.xy,jstep(id,32));}
@compute @workgroup_size(16,16,1)
fn j16(@builtin(global_invocation_id) id:u3){let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}textureStore(out,id.xy,jstep(id,16));}
@compute @workgroup_size(16,16,1)
fn j8(@builtin(global_invocation_id) id:u3){let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}textureStore(out,id.xy,jstep(id,8));}
@compute @workgroup_size(16,16,1)
fn j4(@builtin(global_invocation_id) id:u3){let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}textureStore(out,id.xy,jstep(id,4));}
@compute @workgroup_size(16,16,1)
fn j2(@builtin(global_invocation_id) id:u3){let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}textureStore(out,id.xy,jstep(id,2));}
@compute @workgroup_size(16,16,1)
fn j1(@builtin(global_invocation_id) id:u3){let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}textureStore(out,id.xy,jstep(id,1));}

// drawn field: tube shape, cytoplasm, slime trace
@compute @workgroup_size(16,16,1)
fn rfield(@builtin(global_invocation_id) id:u3){
    let d=i2(textureDimensions(out));let c=i2(id.xy);if(c.x>=d.x||c.y>=d.y){return;}
    let f=ld(t0,c,d);let cy=ld(t1,c,d);let g=cy.x;var tr=f.xyz+max(cy.yzw-f.xyz,v3(0.))*min(p.bgf,1.);
    let jr=ld(t2,c,d).xy;
    if(p.tbe>0.&&jr.x<900.&&dot(jr,jr)>.25){
        // local tube radius
        let dd=length(jr);let dr=-jr/dd;let kr=rk(f32(d.y));var R=dd;
        for(var s=1;s<=5;s++){
            let rq=ld(t2,c+i2(round(dr*f32(s)*8.*kr)),d).xy;
            if(rq.x<900.){R=max(R,length(rq));}
        }
        // circle cross-section
        let hr=sqrt(max(0.,2.*R*dd-dd*dd))/max(R,.5);let sz=mix(.75,1.25,smoothstep(2.*kr,16.*kr,R));
        tr*=mix(1.,hr*sz,clamp(p.tbe,0.,1.));
    }
    textureStore(out,id.xy,v4(tr*(1.+p.cyt*g*4.8),f.w));
}

// shared colours
struct Pal { m0:mat3x3<f32>, m1:mat3x3<f32>, m2:mat3x3<f32> };
// species contrast around the mean colour, then hue shift
fn palg()->Pal{
    let pl=u32(clamp(p.pal,0.,6.));let sh=p.cSh*tau;let ro=select(0.,.3*pi,pl==0u);
    var a=pm(pl*3u);var b=pm(pl*3u+1u);var c=pm(pl*3u+2u);
    for(var j=0;j<3;j++){
        let mn=(a[j]+b[j]+c[j])/3.;
        a[j]=hRot(max(mn+(a[j]-mn)*p.cSp,v3(0.)),sh);
        b[j]=hRot(max(mn+(b[j]-mn)*p.cSp,v3(0.)),sh+ro);
        c[j]=hRot(max(mn+(c[j]-mn)*p.cSp,v3(0.)),sh-ro);
    }
    return Pal(a,b,c);
}
fn ohb(w:v3,P:Pal)->v3{
    let cc=P.m0[1]*w.x+P.m1[1]*w.y+P.m2[1]*w.z;let cs=cc-v3(lum(cc));
    return select(v3(1.,.5,.2),v3(.55)+cs/max(length(cs),1e-4)*.55,length(cs)>.02);
}
fn ocol(hb:v3,ty:f32,vr:f32)->v3{let ob=hRot(hb,p.ohu*tau+(select(0.,2.1,ty==1.)-select(0.,2.1,ty==2.)+(vr-.5)*2.)*p.ovr);return ob/max(max(ob.x,max(ob.y,ob.z)),1e-3);}
fn ecol(w:v3,P:Pal)->v3{return mix(P.m0[2]*w.x+P.m1[2]*w.y+P.m2[2]*w.z,v3(.65,.88,1.),.3);}

// emitted light only
fn emitpx(c:i2,d:i2,P:Pal,sk:f32)->v3{
    let gr=ld(t0,c,d);let tr=ld(t1,c,d).xyz;let w=tr/max(tr.x+tr.y+tr.z,1e-4);
    var e=ecol(w,P)*min(sk,3.)*1.8*p.spk;
    if(gr.x>0.&&p.org>0.){
        let ty=floor(gr.w);let fq=fract(gr.w)*8.;let vr=floor(fq)/7.;let ex=min(fract(fq)/.95,1.);
        let ob=ocol(ohb(w,P),ty,vr);let pu=.65+.35*sin(u_t.time*1.3+vr*tau*3.);
        e+=(ob*(select(.3,.15,ty>0.)+gr.z*select(0.,3.,ty==2.))*pu*p.ogl+mix(mix(ob,v3(.75,.92,1.),.55)*ex*ex*4.,v3(.3,1.,.4)*ex*2.2,p.cal))*mix(smoothstep(0.,.3,gr.x),gr.x,max(-gr.y,0.));
    }
    return e;
}

@compute @workgroup_size(16,16,1)
fn emit(@builtin(global_invocation_id) id:u3){
    let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}
    let fd=i2(textureDimensions(t0));let b=i2(id.xy)*2;let P=palg();let so=soff(u32(fd.x*fd.y),v2(fd));
    var e=v3(0.);
    for(var j=0;j<2;j++){for(var i=0;i<2;i++){
        let c=min(b+i2(i,j),fd-1);var sk=0.;
        if(p.spk>0.){sk=f32(atomicLoad(&atm[so+u32(c.y*fd.x+c.x)]))/256.;}
        e+=emitpx(c,fd,P,sk);
    }}
    textureStore(out,id.xy,v4(e*.25,1.));
}
@compute @workgroup_size(16,16,1)
fn eb2(@builtin(global_invocation_id) id:u3){let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}textureStore(out,id.xy,v4(dsmp(id),1.));}
@compute @workgroup_size(16,16,1)
fn eb3(@builtin(global_invocation_id) id:u3){let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}textureStore(out,id.xy,v4(dsmp(id),1.));}
@compute @workgroup_size(16,16,1)
fn eb4(@builtin(global_invocation_id) id:u3){let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}textureStore(out,id.xy,v4(dsmp(id),1.));}
// wider levels weighted up: long light falloff
fn usmpw(id:u3)->v3{let o=textureLoad(t0,i2(id.xy),0).xyz;return o+(usmp(id)-o)*1.6;}
@compute @workgroup_size(16,16,1)
fn eu3(@builtin(global_invocation_id) id:u3){let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}textureStore(out,id.xy,v4(usmpw(id),1.));}
@compute @workgroup_size(16,16,1)
fn eu2(@builtin(global_invocation_id) id:u3){let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}textureStore(out,id.xy,v4(usmpw(id),1.));}
// carries the wide density blur so compose keeps three inputs
@compute @workgroup_size(16,16,1)
fn eu1(@builtin(global_invocation_id) id:u3){
    let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}
    let bl=textureSampleLevel(t2,s2,(v2(id.xy)+.5)/v2(d),0.).xyz;
    textureStore(out,id.xy,v4(usmpw(id),bl.x+bl.y+bl.z));
}

// full linear hdr picture
@compute @workgroup_size(16,16,1)
fn compose(@builtin(global_invocation_id) id:u3){
    let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}
    let uv=(v2(id.xy)+.5)/v2(d);let px=1./v2(d);

    let t4=textureSampleLevel(t0,s0,uv,0.);let tr=t4.xyz;let sl=1.-t4.w;let lt=textureSampleLevel(t1,s1,uv,0.);
    let tt=tr.x+tr.y+tr.z;let bt=lt.w;let w3=tr/max(tt,1e-4);
    let gr=ld(t2,i2(id.xy),i2(d));let sm=smoothstep(.05,.5,tt);

    // relief
    let kr=rk(f32(d.y));var gm=v2(0.);
    for(var k=0;k<8;k++){
        let a=f32(k)*tau/8.;let dr=v2(cos(a),sin(a));
        gm+=dr*(hgt(uv+dr*px*2.5*kr)/10.+hgt(uv+dr*px*5.*kr)/20.);
    }
    let gf=.5*v2(hgt(uv+v2(px.x,0.))-hgt(uv-v2(px.x,0.)),hgt(uv+v2(0.,px.y))-hgt(uv-v2(0.,px.y)));
    // agar out of focus
    var gw=v2(0.);
    for(var k=0;k<8;k++){
        let a=f32(k)*tau/8.+.3927;let dr=v2(cos(a),sin(a));
        gw+=dr*(hgt(uv+dr*px*10.*kr)/40.+hgt(uv+dr*px*16.*kr)/64.);
    }
    // wandering trails
    let wt=smoothstep(.03,.4,tt)*(1.-sm)*p.bgf;
    let fo=max(smoothstep(.05,.6,tt),min(wt,1.));
    let g=mix(gw*.5,gm*.5+gf*.5,fo);
    let nor=normalize(v3(-g*kr*12.*p.rel*(1.+7.*wt),1.));

    // species colour ramps
    let pl=u32(clamp(p.pal,0.,6.));let P=palg();
    let m0=P.m0;let m1=P.m1;let m2=P.m2;
    let c0=ramp(m0,dT(tr.x));let c1=ramp(m1,dT(tr.y));let c2=ramp(m2,dT(tr.z));

    let rt=tr/(tt+3.0001);
    let bs=(c0*rt.x+c1*rt.y+c2*rt.z)*(1.+((rt.x*rt.y+rt.y*rt.z+rt.z*rt.x)*4.*.15));
    var col=bs*pow(smoothstep(0.,1.,tt),1.8);

    let base=col;
    // slime track left on the agar
    let ed=m0[0]+m1[0]+m2[0];
    col+=(bgc(pl)+ed/max(max(ed.x,ed.y),max(ed.z,1e-3))*.8*sl)*p.bgl*(1.-sm);
    // species tint
    let wq=tr/max(tt,1e-4);let et=m0[0]*wq.x+m1[0]*wq.y+m2[0]*wq.z;
    col+=mix(v3(.55),et/max(max(et.x,et.y),max(et.z,1e-3)),.45)*wt*.5;
    let ao=mix(1.,clamp(.1+.9*(tt/(bt+.01)),0.,1.),sm);let lm=max(max(sm,sl*.8),wt*.8);

    // lights
    let l1=normalize(v3(.35,.5,.8));let l2=normalize(v3(-.7,-.4,.8));let l3=normalize(v3(0.,.8,.2));
    let vd=v3(0.,0.,1.);
    let df=(max(0.,dot(nor,l1))+max(0.,dot(nor,l2))*.5+max(0.,dot(nor,l3))*.3)/(l1.z+l2.z*.5+l3.z*.3);
    col*=mix(1.,df*.7+.3*ao,lm);

    // wet specular
    let ex=mix(12.,256.,p.shn)*mix(.15,1.,fo);let nm=(ex+8.)/72.;
    let hv1=normalize(l1+vd);let hv2=normalize(l2+vd);
    let fr=.04+.96*pow(1.-max(dot(nor,vd),0.),5.);
    let sp=(pow(max(dot(nor,hv1),0.),ex)*max(dot(nor,l1),0.)+pow(max(dot(nor,hv2),0.),ex)*max(dot(nor,l2),0.)*.5)*nm;
    col+=v3(1.,.98,.95)*sp*(.5+12.*fr)*p.spc*lm*ao*mix(.35,1.,fo);
    col+=base*fr*3.*ao;

    // sparks
    if(p.spk>0.){
        let so=soff(d.x*d.y,v2(d));
        let sk=f32(atomicExchange(&atm[so+id.y*d.x+id.x],0u))/256.;
        col+=ecol(w3,P)*min(sk,3.)*1.8*p.spk;
    }

    // emitter light
    if(p.eml>0.){
        let ox=v2(px.x*4.*kr,0.);let oy=v2(0.,px.y*4.*kr);let Lc=lt.xyz*.25;
        let gE=v2(lum(textureSampleLevel(t1,s1,fract(uv+ox),0.).xyz)-lum(textureSampleLevel(t1,s1,fract(uv-ox),0.).xyz),
                  lum(textureSampleLevel(t1,s1,fract(uv+oy),0.).xyz)-lum(textureSampleLevel(t1,s1,fract(uv-oy),0.).xyz))*.25;
        let L=normalize(v3(gE/(lum(Lc)+1e-3)*2.,1.));
        let alb=mix(v3(.45),base/max(max(base.x,max(base.y,base.z)),.1)*.6,sm);
        let nl=max(dot(nor,L),0.);let esp=pow(max(dot(nor,normalize(L+vd)),0.),40.);
        col+=(alb*(nl*.8+.2)+v3(esp*1.5))*Lc*mix(1.,.35,smoothstep(1.,2.5,tt))*p.eml*1.5;
    }

    // organelles
    let hb=ohb(w3,P);
    let S=(4.+10.*p.osz)*kr;var ocv=0.;
    if(p.org>0.){
        let ic=i2(id.xy);let di=i2(d);
        // contact shadow, caustic
        let ls=l1.xy/l1.z*S*.5*px;
        let shd=smoothstep(0.,.3,textureSampleLevel(t2,s2,fract(uv+ls),0.).x)*(1.-smoothstep(0.,.3,gr.x));
        col*=1.-shd*.45;
        if(gr.x>0.){
            let ty=floor(gr.w);let fq=fract(gr.w)*8.;let vi=floor(fq);let vr=vi/7.;let ex=min(fract(fq)/.95,1.);
            let ob=ocol(hb,ty,vr);
            let gh=.5*v2(ld(t2,ic+i2(1,0),di).x-ld(t2,ic-i2(1,0),di).x,ld(t2,ic+i2(0,1),di).x-ld(t2,ic-i2(0,1),di).x);
            let on=normalize(v3(-gh*S*.6,1.));
            let od=(max(0.,dot(on,l1))+max(0.,dot(on,l2))*.5+max(0.,dot(on,l3))*.3)/(l1.z+l2.z*.5+l3.z*.3);
            let bm=max(max(max(base.x,base.y),base.z),.08);
            // beer-lambert
            let ab=exp(-(1.-ob)*2.2*gr.x);
            var oc=ab*bm*select(1.4,.9,ty==2.)*(od*.75+.25);
            oc=mix(oc,ob*bm*.08,clamp(gr.y*.85,0.,1.));
            if(ty==1.){oc*=1.-gr.z*.35;}
            let bk=max(-gr.y,0.);let cov=mix(smoothstep(0.,.3,gr.x),gr.x,bk);ocv=cov;
            col=mix(col,oc,cov*select(.65,.92,ty>0.));
            let rim=pow(1.-on.z,2.);
            let oe=mix(100.,600.,p.shn);let gl=pow(max(dot(on,hv1),0.),oe)+pow(max(dot(on,hv2),0.),oe)*.5;
            let cau=pow(textureSampleLevel(t2,s2,fract(uv+l1.xy*S*.3*px),0.).x,6.)*smoothstep(.2,.8,1.-gr.x);
            let pu=.65+.35*sin(u_t.time*1.3+vr*tau*3.);
            col+=(ob*rim*bm*1.2+v3(1.,.98,.95)*gl*3.*p.spc+ab*cau*bm*2.)*cov;
            col+=ob*(select(.3,.15,ty>0.)+gr.z*select(0.,3.,ty==2.))*pu*cov*p.ogl;
            // flash and glitter
            let sh=select(0.,1.,vi>=6.);
            let gc=vec2<u32>(v2(id.xy)/(2.*kr));let gs=pcg(gc.x*1973u^gc.y*9277u);
            let tw=pow(max(0.,sin(u_t.time*(5.+9.*h1(gs))+h1(gs+1u)*tau)),40.)*step(.7-.45*ex,h1(gs+2u))*smoothstep(.15,.5,gr.x)*(1.-bk);
            // calcium look
            col+=(mix(mix(ob,v3(.75,.92,1.),.55)*ex*ex*4.,v3(.3,1.,.4)*ex*2.2,p.cal)+v3(1.,.97,.9)*tw*(sh*1.5+ex*8.*(1.-p.cal*.8))+v3(1.,.98,.95)*gl*sh*4.*p.spc)*cov;
        }
    }
    // scene depth: agar -1, vein tops +1
    let zs=clamp(hgt(uv)/.6*2.-1.,-1.,1.);
    let csn=min(p.sdf*max(0.,abs(zs-p.fcs)-p.fdr)*14.+rblur(uv,v2(d)),30.)*kr*(1.-ocv);
    // alpha: organelle coverage (integer part) + scene blur radius/64
    textureStore(out,id.xy,v4(max(col,v3(0.)),floor(ocv*7.+.5)+min(csn/64.,.99)));
}

// bloom
fn dsmp(id:u3)->v3{
    let uv=(v2(id.xy)+.5)/v2(textureDimensions(out));let sp=1./v2(textureDimensions(t0));
    return (textureSampleLevel(t0,s0,fract(uv+v2(-sp.x,-sp.y)),0.).xyz+textureSampleLevel(t0,s0,fract(uv+v2(sp.x,-sp.y)),0.).xyz
           +textureSampleLevel(t0,s0,fract(uv+v2(-sp.x,sp.y)),0.).xyz+textureSampleLevel(t0,s0,fract(uv+v2(sp.x,sp.y)),0.).xyz)*.25;
}
fn usmp(id:u3)->v3{
    let uv=(v2(id.xy)+.5)/v2(textureDimensions(out));let lp=1./v2(textureDimensions(t1));
    var b=v3(0.);
    for(var j=-1;j<=1;j++){for(var i=-1;i<=1;i++){
        b+=textureSampleLevel(t1,s1,fract(uv+v2(f32(i),f32(j))*lp),0.).xyz*f32((2-abs(i))*(2-abs(j)));
    }}
    return textureLoad(t0,i2(id.xy),0).xyz+b/16.;
}

@compute @workgroup_size(16,16,1)
fn bd1(@builtin(global_invocation_id) id:u3){
    let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}
    let c=dsmp(id);let m=max(c.x,max(c.y,c.z));let k=p.bth*.5;
    let sk=clamp(m-p.bth+k,0.,2.*k);
    textureStore(out,id.xy,v4(c*max(sk*sk/(4.*k+1e-4),m-p.bth)/max(m,1e-4),1.));
}
@compute @workgroup_size(16,16,1)
fn bd2(@builtin(global_invocation_id) id:u3){let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}textureStore(out,id.xy,v4(dsmp(id),1.));}
@compute @workgroup_size(16,16,1)
fn bd3(@builtin(global_invocation_id) id:u3){let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}textureStore(out,id.xy,v4(dsmp(id),1.));}
@compute @workgroup_size(16,16,1)
fn bd4(@builtin(global_invocation_id) id:u3){let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}textureStore(out,id.xy,v4(dsmp(id),1.));}
@compute @workgroup_size(16,16,1)
fn bd5(@builtin(global_invocation_id) id:u3){let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}textureStore(out,id.xy,v4(dsmp(id),1.));}
@compute @workgroup_size(16,16,1)
fn bu4(@builtin(global_invocation_id) id:u3){let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}textureStore(out,id.xy,v4(usmp(id),1.));}
@compute @workgroup_size(16,16,1)
fn bu3(@builtin(global_invocation_id) id:u3){let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}textureStore(out,id.xy,v4(usmp(id),1.));}
@compute @workgroup_size(16,16,1)
fn bu2(@builtin(global_invocation_id) id:u3){let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}textureStore(out,id.xy,v4(usmp(id),1.));}
@compute @workgroup_size(16,16,1)
fn bu1(@builtin(global_invocation_id) id:u3){let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}textureStore(out,id.xy,v4(usmp(id),1.));}

// dof pyramid: 2,4,8,16 px, blur radius in alpha
@compute @workgroup_size(16,16,1)
fn db1(@builtin(global_invocation_id) id:u3){
    let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}
    let uv=(v2(id.xy)+.5)/v2(d);let sp=1./v2(textureDimensions(t0));var c=v3(0.);var r=0.;
    for(var j=0;j<2;j++){for(var i=0;i<2;i++){
        let s=textureSampleLevel(t0,s0,fract(uv+(v2(f32(i),f32(j))-.5)*sp),0.);c+=s.xyz;r+=fract(s.w)*64.;
    }}
    textureStore(out,id.xy,v4(c*.25,r*.25));
}
fn dsa(id:u3)->v4{
    let uv=(v2(id.xy)+.5)/v2(textureDimensions(out));let sp=1./v2(textureDimensions(t0));
    return (textureSampleLevel(t0,s0,fract(uv+v2(-sp.x,-sp.y)),0.)+textureSampleLevel(t0,s0,fract(uv+v2(sp.x,-sp.y)),0.)
           +textureSampleLevel(t0,s0,fract(uv+v2(-sp.x,sp.y)),0.)+textureSampleLevel(t0,s0,fract(uv+v2(sp.x,sp.y)),0.))*.25;
}
@compute @workgroup_size(16,16,1)
fn db2(@builtin(global_invocation_id) id:u3){let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}textureStore(out,id.xy,dsa(id));}
@compute @workgroup_size(16,16,1)
fn db3(@builtin(global_invocation_id) id:u3){let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}textureStore(out,id.xy,dsa(id));}
@compute @workgroup_size(16,16,1)
fn db4(@builtin(global_invocation_id) id:u3){let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}textureStore(out,id.xy,dsa(id));}
// coarse to fine: each level keeps its own blur or takes the coarser one, by the local radius
fn dcomp(id:u3,lv:f32)->v4{
    let o=textureLoad(t0,i2(id.xy),0);let uv=(v2(id.xy)+.5)/v2(textureDimensions(out));let lp=1./v2(textureDimensions(t1));
    var b=v3(0.);
    for(var j=-1;j<=1;j++){for(var i=-1;i<=1;i++){
        b+=textureSampleLevel(t1,s1,fract(uv+v2(f32(i),f32(j))*lp),0.).xyz*f32((2-abs(i))*(2-abs(j)));
    }}
    return v4(mix(o.xyz,b/16.,clamp((o.w-lv)/lv,0.,1.)),o.w);
}
@compute @workgroup_size(16,16,1)
fn du3(@builtin(global_invocation_id) id:u3){let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}textureStore(out,id.xy,dcomp(id,8.));}
@compute @workgroup_size(16,16,1)
fn du2(@builtin(global_invocation_id) id:u3){let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}textureStore(out,id.xy,dcomp(id,4.));}
@compute @workgroup_size(16,16,1)
fn du1(@builtin(global_invocation_id) id:u3){let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}textureStore(out,id.xy,dcomp(id,2.));}

@compute @workgroup_size(16,16,1)
fn main_image(@builtin(global_invocation_id) id:u3){
    let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}
    let uv=(v2(id.xy)+.5)/v2(d);
    let c4=textureLoad(t0,i2(id.xy),0);let ocv=floor(c4.w)/7.;let cn=fract(c4.w)*64.;
    // dof
    let base=mix(c4.xyz,textureSampleLevel(t2,s2,uv,0.).xyz,clamp(cn/2.,0.,1.));
    let E=textureSampleLevel(t1,s1,uv,0.).xyz*.2;
    var col=base+E*p.glw;

    let mx=max(max(col.x,col.y),col.z);col*=aces(mx)/max(mx,1e-5);
    col=mix(v3(lum(col)),col,mix(p.sat,max(p.sat,1.),ocv));
    let vc=(uv-.5)*2.;col*=1.-dot(vc,vc)*.1;

    textureStore(out,id.xy,v4(pow(max(col,v3(0.)),v3(p.gam)),1.));
}

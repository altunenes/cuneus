// Enes Altun, 2026;
// This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 Unported License.

struct TimeUniform { time: f32, delta: f32, frame: u32, _padding: u32 };
@group(0) @binding(0) var<uniform> u_t: TimeUniform;

struct Params {
    sa: f32, sd: f32, drg: f32, spd: f32, dec: f32, dif: f32, dep: f32, jit: f32,
    rSd: f32, mSc: f32, fSc: f32, sGn: f32, sAt: f32, sRp: f32, str: f32, aSc: f32,
    glw: f32, cSh: f32, spc: f32, gam: f32, aCt: f32, aSp: f32, aRd: f32, aSt: f32,
    cSp: f32, sat: f32, pMx: f32, blr: f32,
    c0r: f32, c0g: f32, c0b: f32, c1r: f32, c1g: f32, c1b: f32, c2r: f32, c2g: f32, c2b: f32,
    sub: f32, tSm: f32, wnd: f32, tur: f32, p1: f32, flu: f32, p3: f32,
};
@group(1) @binding(0) var out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> p: Params;
@group(2) @binding(0) var<storage, read_write> atm: array<atomic<u32>>;

@group(3) @binding(0) var t0: texture_2d<f32>; @group(3) @binding(1) var s0: sampler;
@group(3) @binding(2) var t1: texture_2d<f32>; @group(3) @binding(3) var s1: sampler;
@group(3) @binding(4) var t2: texture_2d<f32>; @group(3) @binding(5) var s2: sampler;

alias v2 = vec2<f32>; alias v3 = vec3<f32>; alias v4 = vec4<f32>; alias u3 = vec3<u32>;
const pi: f32 = 3.14159265; const tau: f32 = 6.28318530;
const asc: f32 = 256.; const aiv: f32 = 1./256.;

// hash
fn pcg(s:u32)->u32{var st=s*747796405u+2891336453u;var w=((st>>((st>>28u)+4u))^st)*277803737u;return (w>>22u)^w;}
fn h1(s:u32)->f32{return f32(pcg(s))/4294967295.;}

// chebyshev behavior map (Continuous Non-Linear stuff)
fn cheby(i:v4,bs:u32,ms:u32,ma:f32)->v4{
    var r=v4(0.);let ip=i*.4;
    for(var j=0u;j<5u;j++){
        let b=bs+j*24u;let m=ms+j*12u;
        var w1=v4(h1(b)*2.-1.,h1(b+1u)*2.-1.,h1(b+2u)*2.-1.,h1(b+3u)*2.-1.);
        var w2=v4(h1(b+4u)*2.-1.,h1(b+5u)*2.-1.,h1(b+6u)*2.-1.,h1(b+7u)*2.-1.);
        w1+=ma*v4(h1(m)*2.-1.,h1(m+1u)*2.-1.,h1(m+2u)*2.-1.,h1(m+3u)*2.-1.);
        w2+=ma*v4(h1(m+4u)*2.-1.,h1(m+5u)*2.-1.,h1(m+6u)*2.-1.,h1(m+7u)*2.-1.);
        let s=tanh(dot(ip,w1));let t=tanh(dot(ip,w2));
        let s1=s;let s2=2.*s*s-1.;let s3=s*(4.*s*s-3.);
        let t1=t;let t2=2.*t*t-1.;let t3=t*(4.*t*t-3.);
        let mc=1.+ma*(h1(m+8u)-.5);
        let v=((h1(b+8u)*2.-1.)*s1*t1+(h1(b+9u)*2.-1.)*s1+(h1(b+10u)*2.-1.)*t1+(h1(b+11u)*2.-1.)*s2*.5+(h1(b+12u)*2.-1.)*t2*.5+(h1(b+13u)*2.-1.)*s1*t2*.35+(h1(b+14u)*2.-1.)*s2*t1*.35+(h1(b+15u)*2.-1.)*s2*t2*.15+(h1(b+16u)*2.-1.)*s3*t1*.08+(h1(b+17u)*2.-1.)*s1*t3*.08)*mc;
        r+=v4(h1(b+19u)*2.-1.,h1(b+20u)*2.-1.,h1(b+21u)*2.-1.,h1(b+22u)*2.-1.)*v;
    }
    return tanh(r*.15)*.4;
}

// sensor read
fn sns(pt:v2,sp:u32,tx:texture_2d<f32>,sm:sampler)->v2{
    let cv=v2(textureDimensions(tx));let tp=clamp(pt,v2(0.),cv-v2(1.));
    let s=textureSampleLevel(tx,sm,tp/cv,0.);
    var o=0.;var t=0.;
    switch sp {
        case 0u: {o=s.x;t=s.y+s.z;}
        case 1u: {o=s.y;t=s.x+s.z;}
        default: {o=s.z;t=s.x+s.y;}
    }
    return v2(o,t*p.sAt-t*p.sRp);
}

// attractors
fn att(pos:v2,t:f32,cv:v2)->v2{
    var f=v2(0.);let c=u32(p.aCt);let cx=cv*.5;let m=min(cv.x,cv.y);
    for(var i=0u;i<c;i++){
        let ph=f32(i)*2.399+t*p.aSp;let r=m*(.15+.12*sin(ph*.37+f32(i)*1.7));
        let a=ph+f32(i)*tau/max(f32(c),1.);let df=(cx+v2(r*cos(a),r*sin(a)))-pos;let d=length(df);
        if(d>1.){f+=normalize(df)*p.aSt*(1.-smoothstep(p.aRd*.5,p.aRd,d))/(1.+d*.01);}
    }
    return f;
}

@compute @workgroup_size(16,16,1)
fn agent_update(@builtin(global_invocation_id) id:u3){
    let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}
    let cv=v2(textureDimensions(t1));let aid=id.y*d.x+id.x;
    if(aid>=u32(f32(d.x*d.y)*clamp(p.aSc,.05,1.))){return;}

    let sp=aid%3u;let sd=aid*17u+u_t.frame*7919u;
    var pos:v2;var vel:v2;

    if(u_t.frame<2u){
        // Spawn randomly
        pos=v2(h1(sd)*cv.x,h1(sd+1u)*cv.y);let a=h1(sd+2u)*tau;
        // Random Velocity Direction
        vel=v2(cos(a),sin(a))*p.spd*.5;
    }else{
        let dt=textureLoad(t0,vec2<i32>(id.xy),0);pos=dt.xy*cv;vel=dt.zw*cv;
        var hd=select(h1(sd+10u)*tau,atan2(vel.y,vel.x),length(vel)>.01);

        let sa=p.sa;let sdt=p.sd;
        let F=sns(pos+v2(cos(hd),sin(hd))*sdt,sp,t1,s1);
        let FL=sns(pos+v2(cos(hd+sa*.5),sin(hd+sa*.5))*sdt,sp,t1,s1);
        let FR=sns(pos+v2(cos(hd-sa*.5),sin(hd-sa*.5))*sdt,sp,t1,s1);
        let L=sns(pos+v2(cos(hd+sa),sin(hd+sa))*sdt,sp,t1,s1);
        let R=sns(pos+v2(cos(hd-sa),sin(hd-sa))*sdt,sp,t1,s1);

        // Instantly repels agents from stagnant
        let c=1.5;let pn=3.;
        let Ftx=F.x-max(0.,F.x-c)*pn;let FLtx=FL.x-max(0.,FL.x-c)*pn;let FRtx=FR.x-max(0.,FR.x-c)*pn;
        let Ltx=L.x-max(0.,L.x-c)*pn;let Rtx=R.x-max(0.,R.x-c)*pn;

        // Feed the tricked sensors into the brain
        let iv=v4(Ftx,(Ltx-Rtx)+(FLtx-FRtx)*.5,F.y,(L.y-R.y)+(FL.y-FR.y)*.5)*p.sGn;
        let bs=u32(p.rSd*10000.);let ms=bs+50000u+sp*7919u;
        
        let bO=cheby(iv,bs,ms,p.mSc);let mO=cheby(v4(iv.x,-iv.y,iv.z,-iv.w),bs,ms,p.mSc);
        let fw=v2(cos(hd),sin(hd));let lf=v2(-sin(hd),cos(hd));

        let wF=fw*(bO.x+mO.x)*p.fSc+lf*(bO.y-mO.y)*p.fSc;
        let wS=fw*(bO.z+mO.z)*p.str+lf*(bO.w-mO.w)*p.str;
        let jit=(h1(sd+6u)*2.-1.)*p.jit;
        
        // FLUID FORCES (The Bubble Makers)
        let P=pos/cv;let px=1./cv;
        let cN=textureSampleLevel(t1,s1,fract(P+v2(0.,px.y*2.)),0.).xyz;
        let cS=textureSampleLevel(t1,s1,fract(P-v2(0.,px.y*2.)),0.).xyz;
        let cE=textureSampleLevel(t1,s1,fract(P+v2(px.x*2.,0.)),0.).xyz;
        let cW=textureSampleLevel(t1,s1,fract(P-v2(px.x*2.,0.)),0.).xyz;
        
        // Pseudo-wind pressure based on local gradients
        let wnd=v2(-(dot(cN,v3(1.))-dot(cS,v3(1.))),dot(cE,v3(1.))-dot(cW,v3(1.)))*p.wnd*p.spd;
        var aN=0.;var aS=0.;var aE=0.;var aW=0.;
        if(sp==0u){aN=cN.y+cN.z;aS=cS.y+cS.z;aE=cE.y+cE.z;aW=cW.y+cW.z;}
        else if(sp==1u){aN=cN.x+cN.z;aS=cS.x+cS.z;aE=cE.x+cE.z;aW=cW.x+cW.z;}
        else{aN=cN.x+cN.y;aS=cS.x+cS.y;aE=cE.x+cE.y;aW=cW.x+cW.y;}
        let pb=v2(-(aE-aW),-(aN-aS))*p.sRp*p.spd*6.;

        // BASE MOVEMENT (The Vein Makers)
        vel=vel*p.drg+wF*p.spd+(fw*F.y+lf*iv.w)*p.spd*.5+v2(cos(hd+jit),sin(hd+jit))*p.jit*.5;
        
        // HYBRID COEXISTENCE BLEND
        vel+=(wnd+pb)*max(smoothstep(.01,1.,aN+aS+aE+aW),h1(aid*113u)*.5)*p.flu;

        let mxS=p.spd*4.;let cS_=length(vel);
        if(cS_>mxS){vel*=mxS/cS_;}

        pos=((pos+vel+wS)%cv+cv)%cv;
    }

    textureStore(out,id.xy,v4(pos/cv,vel/cv));

    let dp=p.dep*(1.5-length(vel)/max(p.spd*4.,.01))*asc;
    let fx=fract(pos.x);let fy=fract(pos.y);
    let ix=u32(floor(pos.x));let iy=u32(floor(pos.y));let cw=u32(cv.x);let ch=u32(cv.y);
    
    // Bilinear atomic deposit mapping
    if(ix<cw&&iy<ch){
        let off=sp*cw*ch;let ix1=(ix+1u)%cw;let iy1=(iy+1u)%ch;
        atomicAdd(&atm[iy*cw+ix+off],u32(dp*(1.-fx)*(1.-fy)));
        atomicAdd(&atm[iy*cw+ix1+off],u32(dp*fx*(1.-fy)));
        atomicAdd(&atm[iy1*cw+ix+off],u32(dp*(1.-fx)*fy));
        atomicAdd(&atm[iy1*cw+ix1+off],u32(dp*fx*fy));
    }
}

@compute @workgroup_size(16,16,1)
fn process_trails(@builtin(global_invocation_id) id:u3){
    let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}
    let pi=id.y*d.x+id.x;let st=d.x*d.y;
    let dp=v4(f32(atomicExchange(&atm[pi],0u)),f32(atomicExchange(&atm[pi+st],0u)),f32(atomicExchange(&atm[pi+2u*st],0u)),f32(atomicExchange(&atm[pi+3u*st],0u)))*aiv*.002;

    let uv=(v2(id.xy)+.5)/v2(d);let px=1./v2(d);let o=.5; 
    let c=(textureSampleLevel(t0,s0,fract(uv+v2(-o,-o)*px),0.)+textureSampleLevel(t0,s0,fract(uv+v2(o,-o)*px),0.)+textureSampleLevel(t0,s0,fract(uv+v2(-o,o)*px),0.)+textureSampleLevel(t0,s0,fract(uv+v2(o,o)*px),0.))*.25;

    textureStore(out,id.xy,c*p.dec+dp*max(v4(0.),v4(1.)-c/3.));
}

@compute @workgroup_size(16,16,1)
fn diffuse_h(@builtin(global_invocation_id) id:u3){
    let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}
    let uv=(v2(id.xy)+.5)/v2(d);let px=1./f32(d.x);

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
    let uv=(v2(id.xy)+.5)/v2(d);let py=1./f32(d.y);

    var s=textureSampleLevel(t1,s1,uv,0.).xyz*.382928;
    s+=textureSampleLevel(t1,s1,fract(uv+v2(0.,py)),0.).xyz*.241732;
    s+=textureSampleLevel(t1,s1,fract(uv-v2(0.,py)),0.).xyz*.241732;
    s+=textureSampleLevel(t1,s1,fract(uv+v2(0.,py*2.)),0.).xyz*.060598;
    s+=textureSampleLevel(t1,s1,fract(uv-v2(0.,py*2.)),0.).xyz*.060598;
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

@compute @workgroup_size(16,16,1)
fn turing_resolve(@builtin(global_invocation_id) id:u3){
    let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}
    let uv=(v2(id.xy)+.5)/v2(d);let px=1./v2(d);
    
    // Advection / Chemical Rates
    let N=textureSampleLevel(t1,s1,fract(uv+v2(0.,px.y)),0.).xyz;
    let S=textureSampleLevel(t1,s1,fract(uv-v2(0.,px.y)),0.).xyz;
    let E=textureSampleLevel(t1,s1,fract(uv+v2(px.x,0.)),0.).xyz;
    let W=textureSampleLevel(t1,s1,fract(uv-v2(px.x,0.)),0.).xyz;
    let lDn=(N+S+E+W)*.25;
    
    let gR=v2(E.x-W.x,N.x-S.x);let gG=v2(E.y-W.y,N.y-S.y);let gB=v2(E.z-W.z,N.z-S.z);
    var vel=v2(-(gR.y+gG.y+gB.y),gR.x+gG.x+gB.x)*2.5;
    vel+=(v2(-gR.y,gR.x)*(lDn.y+lDn.z)+v2(-gG.y,gG.x)*(lDn.x+lDn.z)+v2(-gB.y,gB.x)*(lDn.x+lDn.y))*7.5;
    
    let jit=(h1(id.y*d.x+id.x+u_t.frame)-.5)*.5;
    let uva=fract(uv-vel*px+(jit*px));
    
    let rw=textureSampleLevel(t0,s0,uva,0.).xyz;
    let act=textureSampleLevel(t1,s1,uva,0.).xyz;
    let inh=textureSampleLevel(t2,s2,uva,0.).xyz;

    var fS=max(v3(0.),mix(rw,act,v3(p.dif*.7))+(act-inh)*p.tur);
    fS+=v3(1.05*(fS.x*fS.y-fS.z*fS.x),1.05*(fS.y*fS.z-fS.x*fS.y),1.05*(fS.z*fS.x-fS.y*fS.z));
    textureStore(out,id.xy,v4(fS,1.));
}

fn dT(x:f32)->f32{return tanh(log(1.+max(0.,x))*.8);}
fn hRot(c:v3,a:f32)->v3{let k=v3(.57735);let ca=cos(a);let sa=sin(a);return c*ca+cross(k,c)*sa+k*dot(k,c)*(1.-ca);}
fn aces(x:v3)->v3{return clamp((x*(2.51*x+.03))/(x*(2.43*x+.59)+.14),v3(0.),v3(1.));}

@compute @workgroup_size(16,16,1)
fn main_image(@builtin(global_invocation_id) id:u3){
    let d=textureDimensions(out);if(id.x>=d.x||id.y>=d.y){return;}
    let uv=(v2(id.xy)+.5)/v2(d);let px=1./v2(d);

    let tr=textureSampleLevel(t1,s1,uv,0.).xyz;let bl=textureSampleLevel(t2,s2,uv,0.).xyz;
    let tt=tr.x+tr.y+tr.z;let bt=bl.x+bl.y+bl.z;

    // 1. 3D Structure / Normals
    let hr=textureSampleLevel(t1,s1,fract(uv+v2(px.x,0.)),0.);
    let hl=textureSampleLevel(t1,s1,fract(uv-v2(px.x,0.)),0.);
    let hu=textureSampleLevel(t1,s1,fract(uv+v2(0.,px.y)),0.);
    let hd=textureSampleLevel(t1,s1,fract(uv-v2(0.,px.y)),0.);
    
    let dx=(hr.x+hr.y+hr.z)-(hl.x+hl.y+hl.z);
    let dy=(hu.x+hu.y+hu.z)-(hd.x+hd.y+hd.z);
    let nS=smoothstep(0.,.3,sqrt(dx*dx+dy*dy))*8.;
    let nor=normalize(v3(-dx*nS,-dy*nS,1.));

    // 2. TRI-TONE DENSITY COLORS (Edges -> Cores -> Nuclei)
    let t0=dT(tr.x);let t1=dT(tr.y);let t2=dT(tr.z);
    
   // col mats
    let ea=mat3x3<f32>(.4,.02,.05, .3,.15,.02, .02,.08,.2);
    let ca=mat3x3<f32>(1.,.4,.1, .8,.9,.1, .1,.99,.4);
    let na=mat3x3<f32>(.1,.9,.8, 1.,.1,.7, .8,1.,.1);

    let eb=mat3x3<f32>(.02,.2,.3, .2,.02,.3, .3,.2,.02);
    let cb=mat3x3<f32>(.2,1.,.8, .9,.2,1., 1.,.9,.2);
    let nb=mat3x3<f32>(1.,.9,.9, 1.,.4,0., .2,.5,1.);

    var ba=mat3x3<f32>(mix(ea[0],ca[0],smoothstep(.1,.6,t0)),mix(ea[1],ca[1],smoothstep(.1,.6,t1)),mix(ea[2],ca[2],smoothstep(.1,.6,t2)));
    var bb=mat3x3<f32>(mix(eb[0],cb[0],smoothstep(.1,.6,t0)),mix(eb[1],cb[1],smoothstep(.1,.6,t1)),mix(eb[2],cb[2],smoothstep(.1,.6,t2)));

    ba=mat3x3<f32>(mix(ba[0],na[0],smoothstep(.75,.95,t0)),mix(ba[1],na[1],smoothstep(.75,.95,t1)),mix(ba[2],na[2],smoothstep(.75,.95,t2)));
    bb=mat3x3<f32>(mix(bb[0],nb[0],smoothstep(.75,.95,t0)),mix(bb[1],nb[1],smoothstep(.75,.95,t1)),mix(bb[2],nb[2],smoothstep(.75,.95,t2)));

    let sA=p.cSp*tau*.5;let sh=p.cSh*tau;
    let c0=hRot(mix(ba[0],bb[0],p.pMx),sh);
    let c1=hRot(mix(ba[1],bb[1],p.pMx),sh+sA);
    let c2=hRot(mix(ba[2],bb[2],p.pMx),sh-sA);

    let rt=tr/(tt+3.0001);
    var bs=(c0*rt.x+c1*rt.y+c2*rt.z)*(1.+((rt.x*rt.y+rt.y*rt.z+rt.z*rt.x)*4.*.15));

    var col=bs*pow(smoothstep(0.,1.,tt),1.8);
    let ao=clamp(.1+.9*(tt/(bt+.01)),0.,1.);

    let l1=normalize(v3(.6,.5,1.));let l2=normalize(v3(-.7,-.4,.8));let l3=normalize(v3(0.,.8,.2));
    let vd=normalize(v3(.5-uv.x,.5-uv.y,1.));
    
    let df=max(0.,dot(nor,l1)) + max(0.,dot(nor,l2))*.5 + max(0.,dot(nor,l3))*.3;
    let h1_=normalize(l1+vd);let h2_=normalize(l2+vd);
    let tS=(pow(max(dot(nor,h1_),0.),64.) + pow(max(dot(nor,h1_),0.),16.)*.3 + pow(max(dot(nor,h2_),0.),32.)*.4)*p.spc;

    col*=df*.7+.3*ao;
    col+=v3(1.,.98,.95)*tS*3.*ao;
    col+=bs*pow(1.-max(dot(nor,vd),0.),3.)*2.5*ao;

    if(bt>.001){
        let br=bl/(bt+.0001);
        col+=hRot(c0*br.x+c1*br.y+c2*br.z,.15+smoothstep(.1,.9,(br.x*br.y+br.y*br.z+br.z*br.x)*4.)*2.5)*dT(bt)*p.glw*.4;
    }

    col=aces(col);
    col=mix(v3(dot(col,v3(.2126,.7152,.0722))),col,p.sat);
    let vc=(uv-.5)*2.;col*=1.-dot(vc,vc)*.1;
    
    textureStore(out,id.xy,v4(pow(max(col,v3(0.)),v3(1./max(p.gam,.1))),1.));
}
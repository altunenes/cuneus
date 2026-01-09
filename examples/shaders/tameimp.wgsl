// Enes Altun, 2026;
// This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
@group(0) @binding(0) var<uniform> u_t: TimeUniform;

struct Params {
    col_bg: vec4<f32>,
    col_line: vec4<f32>,
    col_core: vec4<f32>,
    col_amber: vec4<f32>,
    ball_offset_x: f32,
    ball_offset_y: f32,
    ball_sink: f32,
    distortion_amt: f32,
    noise_amt: f32,
    stream_width: f32,
    speed: f32,
    scale: f32,
    angle: f32,
    line_freq: f32,
    cam_height: f32,
    cam_distance: f32,
    cam_fov: f32,
    rim_intensity: f32,
    spotlight_intensity: f32,
    spotlight_height: f32,
    gamma: f32,
    saturation: f32,
    contrast: f32,
    orbital_enabled: u32,
    orbital_speed: f32,
    orbital_radius: f32,
    _pad: vec2<f32>,
};

@group(1) @binding(0) var out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> p: Params;

alias v2 = vec2<f32>;
alias v3 = vec3<f32>;
alias v4 = vec4<f32>;
const pi = 3.14159265;

// hash, hash3, cc: dave hoskins, shadertoy
fn h(u:v2)->f32{return fract(sin(dot(u,v2(12.9898,78.233)))*43758.5453);}
fn h3(u:v2)->v3{return v3(h(u),h(u+v2(127.1,311.7)),h(u+v2(269.5,183.3)));}

// noise
fn nz(u:v2)->f32{
    let i=floor(u);let f=fract(u);let w=f*f*(3.-2.*f);
    return mix(mix(h(i+v2(0.,0.)),h(i+v2(1.,0.)),w.x),
               mix(h(i+v2(0.,1.)),h(i+v2(1.,1.)),w.x),w.y);
}

// fbm
fn f(u:v2)->f32{
    var v=0.;var a=.5;var q=u;let m=mat2x2<f32>(.8,.6,-.6,.8);
    for(var i=0;i<4;i++){v+=a*nz(q);q=m*q*2.;a*=.5;}
    return v;
}

// rot=rotate2D, rY=rotateY
fn rot(v:v2,a:f32)->v2{let s=sin(a);let c=cos(a);return v2(v.x*c-v.y*s,v.x*s+v.y*c);}
fn rY(v:v3,a:f32)->v3{let s=sin(a);let c=cos(a);return v3(v.x*c-v.z*s,v.y,v.x*s+v.z*c);}

// pal=iridescent palette
fn pal(t:f32)->v3{
    return v3(.5)+v3(.5)*cos(6.28318*(v3(1.)*t+v3(0.,.33,.67)));
}

fn getRay(uv:v2,ro:v3,ta:v3,fov:f32)->v3{
    let f=normalize(ta-ro);let r=normalize(cross(v3(0.,1.,0.),f));let u=cross(f,r);
    return normalize(f+r*uv.x*tan(fov*.5)+u*uv.y*tan(fov*.5));
}

// iPln=intersectPlane, iSph=intersectSphere
fn iPln(ro:v3,rd:v3,y:f32)->v3{
    if(abs(rd.y)<1e-4){return v3(1e5);}
    let t=(y-ro.y)/rd.y;
    return select(v3(1e5),ro+rd*t,t>=0.);
}
fn iSph(ro:v3,rd:v3,c:v3,r:f32)->f32{
    let oc=ro-c;let b=dot(oc,rd);let h=b*b-(dot(oc,oc)-r*r);
    return select(-b-sqrt(h),-1.,h<0.);
}
fn nSph(p:v3,c:v3)->v3{return normalize(p-c);}

// PBR: D=GGX, G=Smith, F=Fresnel
fn D_ggx(nh:f32,r:f32)->f32{let a=r*r;let a2=a*a;let d=nh*nh*(a2-1.)+1.;return a2/(pi*d*d+1e-4);}
fn G_sch(nv:f32,r:f32)->f32{let k=((r+1.)*(r+1.))/8.;return nv/(nv*(1.-k)+k+1e-4);}
fn G_sm(nv:f32,nl:f32,r:f32)->f32{return G_sch(nv,r)*G_sch(nl,r);}
fn F_sch(ct:f32,f0:v3)->v3{return f0+(1.-f0)*pow(clamp(1.-ct,0.,1.),5.);}

// occ=occlusion, shd=softShadow
fn occ(p:v3,n:v3,c:v3,r:f32)->f32{
    let d=c-p;let l=length(d);if(l<1e-3){return 0.;}
    return clamp(1.-(dot(n,d)*(r*r)/(l*l*l))*1.5,0.,1.);
}
fn shd(p:v3,ld:v3,c:v3,r:f32,k:f32)->f32{
    let oc=p-c;let b=dot(oc,ld);let h=b*b-(dot(oc,oc)-r*r);
    if(h<0.){return 1.;}
    let d=sqrt(max(0.,r*r-h))-r;let t=-b-sqrt(max(h,0.));
    let v=select(.5+.5*clamp(k*d/max(t,1e-3),-1.,1.),1.,b>0.);
    return v*v*(3.-2.*v);
}

// lSpot= spot light
fn lSpot(hp:v3,n:v3,vd:v3,bc:v3,br:f32,ruf:f32,si:f32,sh:f32)->v3{
    let sp=v3(bc.x,sh,bc.z)+v3(0.,0.,-.3);let tl=sp-hp;let ld=normalize(tl);
    let att=1./(1.+.3*length(tl)+.1*dot(tl,tl));
    let sa=dot(-ld,normalize(v3(bc.x,0.,bc.z)-sp));
    let sf=smoothstep(.7,.85,sa);
    let nl=max(dot(n,ld),0.);let hv=normalize(ld+vd);
    let spec=(D_ggx(max(dot(n,hv),0.),ruf)*G_sm(max(dot(n,vd),1e-3),nl,ruf))/(4.*max(dot(n,vd),1e-3)*nl+1e-4);
    return v3(1.,.95,.85)*si*(nl*.3+spec*nl)*att*sf;
}

// sFlu=shade fluid
fn sFlu(col:v3,hp:v3,n:v3,vd:v3,ld:v3,ruf:f32,met:f32,ist:f32,ang:f32,bc:v3,br:f32,si:f32,sh:f32)->v3{
    let f0=mix(v3(.04),col,met);
    let nv=max(dot(n,vd),1e-3);let nl=max(dot(n,ld),0.);
    let hv=normalize(ld+vd);let nh=max(dot(n,hv),0.);let hv_d=max(dot(hv,vd),0.);
    let spec=(D_ggx(nh,ruf)*G_sm(nv,nl,ruf)*F_sch(hv_d,f0))/(4.*nv*nl+1e-4);
    
    let kd=(v3(1.)-F_sch(hv_d,f0))*(1.-met);
    var lo=(kd*col/pi+spec)*v3(1.2,1.14,1.08)*nl; // Main light

    // Side light
    let ls=normalize(rY(v3(.8,.4,-.3),ang));let nls=max(dot(n,ls),0.);
    let hs=normalize(ls+vd);
    let sps=(D_ggx(max(dot(n,hs),0.),ruf+.1)*G_sm(nv,nls,ruf+.1)*F_sch(hv_d,f0))/(4.*nv*nls+1e-4);
    lo+=(kd*col/pi+sps)*v3(.5,.4,.7)*nls*.4;
    
    let spot=lSpot(hp,n,vd,bc,br,ruf,si,sh);
    lo+=spot*(col*.5+.5);

    let sky=mix(v3(.12,.08,.2),v3(.25,.2,.35),reflect(-vd,n).y*.5+.5);
    var fin=lo+sky*F_sch(nv,f0)*.25+col*v3(.06,.05,.1);
    
    return fin+col*ist*.12;
}

// sBal=shadeBall
fn sBal(n:v3,vd:v3,ld:v3,hp:v3,bc:v3,ri:f32,t:f32,ang:f32)->v3{
    let ruf=.12;let met=.98;let col=v3(.96,.96,.98);let f0=mix(v3(.04),col,met);
    let nv=max(dot(n,vd),1e-3);let nl=max(dot(n,ld),1e-3);
    let hv=normalize(ld+vd);let nh=max(dot(n,hv),0.);

    let rf=reflect(-vd,n);let rref=rY(rf,ang);
    let sky=mix(mix(mix(v3(.32,.32,.35),v3(.42,.3,.38),smoothstep(0.,-1.,rf.y)),v3(.58,.35,.32),smoothstep(-.35,-.95,rf.y)),
               mix(v3(.58,.6,.68),v3(.92,.94,1.),smoothstep(.35,.95,rf.y)),smoothstep(0.,1.,rf.y));
    let env=mix(sky,mix(sky,v3(.92,.94,1.),smoothstep(.35,.95,rf.y)),smoothstep(-.1,.1,rf.y))*(.92+.08*sin(atan2(rref.z,rref.x)*2.));
    
    let spec=(D_ggx(nh,ruf)*G_sm(nv,nl,ruf)*F_sch(max(dot(hv,vd),0.),f0))/(4.*nv*nl+1e-4);
    let lo=((v3(1.)-F_sch(max(dot(hv,vd),0.),f0))*(1.-met)*col/pi+spec)*v3(2.8,2.74,2.66)*nl;
    
    // Fill/Side lights
    let lb=normalize(rY(v3(.1,-.8,.2),ang));
    let fill=v3(.58,.35,.32)*D_ggx(max(dot(n,normalize(lb+vd)),0.),ruf+.15)*max(dot(n,lb),0.)*.7;
    let ls=normalize(rY(v3(-.6,.3,.5),ang));
    let side=v3(.4,.45,.55)*D_ggx(max(dot(n,normalize(ls+vd)),0.),ruf+.1)*max(dot(n,ls),0.)*.4;

    var fin=lo+env*F_sch(nv,f0)*.75+fill+side+v3(1.,.99,.97)*pow(nh,600.)*5.;
    let rim=mix(v3(.42,.3,.38),v3(.6,.62,.68),smoothstep(-.3,.5,n.y))*pow(1.-nv,5.)*ri*.12;
    return fin+rim+(h(hp.xz*900.+hp.y*350.+t*.02)-.5)*.01;
}

// op=oscillationPattern
fn op(mn:f32,mx:f32,int:f32,pd:f32,ct:f32)->f32{
    let c=2.*int+pd;let t=ct%c;
    if(t<int){return mix(mx,mn,.5-.5*cos(pi*(t/int)));}
    if(t<int+pd){return mn;}
    return mix(mn,mx,.5-.5*cos(pi*((t-int-pd)/int)));
}

@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
    let dim=textureDimensions(out);
    if(gid.x>=dim.x||gid.y>=dim.y){return;}
    let res=v2(f32(dim.x),f32(dim.y));
    var uv=(v2(f32(gid.x),f32(gid.y))/res-.5)*v2(res.x/res.y,-1.);
    
    // Params extraction
    let T=u_t.time*p.speed;let S=p.scale;let A=p.angle;
    
    // cam
    var ro=v3(0.,p.cam_height,-2.);var ta=v3(.3,0.,.5);
    let bc_orb=v3(.3+p.ball_offset_x*.5,0.,.5+p.ball_offset_y*.5);
    
    if(p.orbital_enabled==1u){
        let oa=u_t.time*p.orbital_speed;
        ro=v3(bc_orb.x+sin(oa)*p.orbital_radius,p.cam_height+sin(u_t.time)*.5,bc_orb.z+cos(oa)*p.orbital_radius);
        ta=bc_orb;
    }
    let rd=getRay(uv,ro,ta,p.cam_fov);let vd=-rd;

    // Ball Logic
    let br=.25*S;
    var bc=v3(.3+p.ball_offset_x*.5,0.,.5+p.ball_offset_y*.5);
    let sink=sin(u_t.time*3.2)*.5+.5;
    bc.y=br*mix(.2,.6,sink)*p.ball_sink+sin(T*.8)*.01;

    // Plane Intersection
    let ph=iPln(ro,rd,0.);
    var col=v3(.02,.01,.05);
    let b2=v2(bc.x,bc.z);let wl_r=sqrt(max(0.,br*br-bc.y*bc.y));
    let ld=normalize(rY(v3(-.4,.8,-.4),A));

    if(ph.x<1e4){
        // Domain warp
        var puv=rot(v2(ph.x,ph.z),A);let bp=rot(b2,A);
        let dst=puv-bp;let r2=dot(dst,dst);let r=sqrt(r2);
        let pr=wl_r;
        let dfo=smoothstep(pr*8.,pr*1.5,r);
        let disp=dst*((pr*pr)/max(r2,1e-3)*dfo);
        var fuv=puv-disp;

        // Turbulence
        let iw=smoothstep(pr*1.2,pr*5.,dst.x)*smoothstep(pr*12.,pr*5.,dst.x);
        let wm=iw*smoothstep(pr*7.,0.,abs(dst.y));
        let ed=smoothstep(0.,1.,1.-abs(fuv.y)*.3);
        let nuv=fuv*4.5-v2(T*4.,0.);
        let n1=f(nuv);let n2=f(nuv+v2(5.2,1.3));
        let ts=p.distortion_amt*.02*wm*ed;
        fuv+=v2((n2-.5)*.5,n1-.5)*ts;

        // Normals & Colors
        let fr=p.line_freq/S;let sw=sin(fuv.y*fr);
        
        let pt=rot(v2((n1-.5)*10.12*wm,cos(fuv.y*fr)*.5+(n2-.5)*10.12*wm),-A);
        let sn=normalize(v3(pt.x,1.,pt.y));
         // sharp line
        let sl=pow(.5+.5*sw,8.);

        let cBg=p.col_bg.rgb;let cLn=p.col_line.rgb;
        let cCo=p.col_core.rgb;let cAm=p.col_amber.rgb;
        
        let oil=smoothstep(.2,.8,abs(n1-.5)*2.*wm);
        var bCol=mix(cBg,mix(cLn,pal(n1+T*.1),oil*.6),sl);

        // Stream
        let swd=p.stream_width;let swW=swd+swd*.4*smoothstep(0.,2.,dst.x);
        let isT=1.-smoothstep(swW*.8,swW,abs(fuv.y-bp.y));
        if(isT>.01){
            let sc=mix(cCo,cAm,smoothstep(0.,3.,dst.x));
            bCol=mix(bCol,mix(sc*.1,sc*1.5,sl),isT);
        }

        col=sFlu(bCol,ph,sn,vd,ld,mix(.4,.25,sl),0.,isT,A,bc,br,p.spotlight_intensity,p.spotlight_height);

        // Meniscus/Shadows
        let ovr=smoothstep(wl_r*.95,wl_r,r)*smoothstep(wl_r*1.15,wl_r*1.02,r);
        col+=v3(.9,.5,.7)*ovr*.5*(.7+.3*sin(atan2(dst.y,dst.x)*2.+1.));
        col*=1.-(smoothstep(wl_r*1.3,wl_r*1.05,r)*smoothstep(wl_r*.8,wl_r,r))*.25; // inner shadow

        let ao=occ(ph,sn,bc,br);let sh=shd(ph,ld,bc,br,10.);
        col*=(op(.4,1.4,5.,1.,u_t.time)+.6*ao)*(.55+.45*sh);
        col=mix(col,v3(.02,.01,.05),smoothstep(3.,p.cam_distance,length(ph-ro)));
    }

    // Sphere 
    let st=iSph(ro,rd,bc,br);
    if(st>0.){
        let hp=ro+rd*st;
        if(hp.y>=0.){col=sBal(nSph(hp,bc),vd,ld,hp,bc,p.rim_intensity,T,A);}
    }

    // Post
    col*=(1.-length(uv)*.35);
    col+=(h(uv+T)-.5)*.012;
    let lum=dot(col,v3(.299,.587,.114));
    col=mix(v3(lum),col,p.saturation);
    col=(col-.5)*p.contrast+.5;
    col=pow(max(col+0.012*(1.-max(col,v3(0.))),v3(0.)),v3(1./p.gamma));

    textureStore(out,gid.xy,v4(col,1.));
}
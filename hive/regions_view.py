"""An interactive browser view of the segmented regions as real 3D surfaces.

:func:`regions_viewer_html` takes the meshes from :func:`hive.regions3d.build_region_meshes` and writes
one self-contained HTML page (WebGL2, no internet, no server). Every region is a lit surface with its own
colour; drag to rotate, scroll to zoom, right-drag (or Shift-drag) to pan.

* **Each region**: its own colour, a show / hide switch and its own transparency (a list in the panel;
  the largest 300 regions get a row, the rest are controlled together). Click a region in the picture to
  find it in the list.
* **Transparency** is order-independent (weighted blended), so see-through regions look right from every
  angle. Browsers without the needed float-texture support fall back to ordinary blending.
* **Light**: ambient, diffuse, specular, shininess, a light that follows the camera or is aimed.
* **Cutting**: a crop box and two free clipping planes. **Scene**: backgrounds, an orientation-axes gizmo,
  a scale bar, the sample's bounding box, auto-rotate, PNG screenshots, a faster render while moving.

For looking only. Measurements come from the analysis, never from this view.
"""

from __future__ import annotations

import base64
import json
import zlib
from typing import Sequence

import numpy as np

from .regions3d import RegionMeshes, boxes_from_volume, build_region_meshes, region_boxes
from .viewer_common import CHROME_HTML, CSS, JS
from .volume_view import auto_depth_exaggeration


def regions_viewer_html(
    meshes: RegionMeshes,
    title: str = "HIVE 3D regions",
    unit: str = "px",
    volume_extent: Sequence[float] | None = None,
    depth_exaggeration: float | None = None,
) -> str:
    """Build the interactive surface page.

    Args:
        meshes: The region meshes (positions in physical units, ordered x, y, z).
        title: Heading inside the page.
        unit: Name of the physical unit, written on the scale bar.
        volume_extent: Physical size ``(x, y, z)`` of the whole sample, drawn as the bounding box and used
            to centre the view. Defaults to the box around the meshes.
        depth_exaggeration: Stretch of the z axis. ``None`` chooses one so a thin sample is visibly deep
            (1 = true to scale); the page has a slider.

    Returns:
        The HTML text.

    Raises:
        ValueError: If there are no surfaces to show.
    """
    if len(meshes.positions) == 0 or len(meshes.faces) == 0:
        raise ValueError("There are no region surfaces to show.")
    pos = meshes.positions.astype(np.float64)
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    size = np.maximum(hi - lo, 1e-9)
    extent = np.maximum(np.asarray(volume_extent, dtype=np.float64) if volume_extent is not None else size, 1e-9)
    stretch = (auto_depth_exaggeration((1, 1, 1), (extent[2], extent[1], extent[0]))
               if depth_exaggeration is None else float(depth_exaggeration))
    n, m = len(pos), len(meshes.faces)
    q = np.rint((pos - lo) / size * 65535.0).astype(np.uint16)
    normals = np.zeros((n, 4), dtype=np.int8)
    normals[:, :3] = np.rint(np.clip(meshes.normals, -1, 1) * 127.0).astype(np.int8)
    blob = b"".join([
        meshes.region.astype("<u4").tobytes(),                   # n * 4 bytes
        meshes.faces.astype("<u4").tobytes(),                    # m * 12 bytes
        q.astype("<u2").tobytes(),                               # n * 6 bytes
        normals.tobytes(),                                       # n * 4 bytes
    ])
    meta = {"title": title, "unit": str(unit), "n": n, "m": m, "stretch": round(stretch, 2),
            "min": lo.tolist(), "size": size.tolist(), "extent": extent.tolist(),
            "regions": [[r["label"], r["volume"]] for r in meshes.table], "grid": meshes.grid_side,
            "skipped": len(meshes.skipped), "omitted": meshes.omitted}
    safe_meta = json.dumps(meta).replace("</", "<\\/")
    payload = base64.b64encode(zlib.compress(blob, 6)).decode("ascii")
    return (_PAGE.replace("__CSS__", CSS + _EXTRA_CSS).replace("__CHROME__", CHROME_HTML).replace("__COMMON__", JS)
            .replace("__META__", safe_meta).replace("__DATA__", payload)
            .replace("__TITLE__", title.replace("<", "&lt;")))


def regions_page_for(labels, measurements, spacing: Sequence[float], unit: str, title: str,
                     progress=None) -> str | None:
    """The surface page for a label volume: boxes, meshes and page in one step.

    Args:
        labels: Label volume ``(Z, Y, X)`` (array or memory map).
        measurements: The 3D measurement table; its ``bbox-0..5`` columns give each region's box, so a
            memory map is read only around the regions. If it has none, the volume is scanned.
        spacing: Physical voxel size ``(z, y, x)``.
        unit: Unit name for the scale bar.
        title: Heading inside the page.
        progress: Optional ``progress(fraction, message)`` callback.

    Returns:
        The HTML text, or ``None`` if there is no region to draw.
    """
    try:
        boxes = region_boxes(measurements)
    except ValueError:
        boxes = boxes_from_volume(labels)
    if not boxes:
        return None
    meshes = build_region_meshes(labels, boxes, tuple(spacing), progress=progress)
    if len(meshes.faces) == 0:
        return None
    z, y, x = labels.shape
    return regions_viewer_html(meshes, title, unit, (x * spacing[2], y * spacing[1], z * spacing[0]))


_EXTRA_CSS = r"""
 #list{max-height:300px;overflow:auto;border:1px solid var(--line);border-radius:5px;margin:4px 0;background:#0e0e15}
 .reg{display:grid;grid-template-columns:18px 26px 1fr 58px;align-items:center;gap:6px;padding:2px 6px;border-bottom:1px solid #1b1b26;cursor:default}
 .reg.sel{background:#2a2550}
 .reg input[type=range]{margin:0;width:58px}.reg input[type=color]{width:24px;height:18px}
 .reg .t{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#cfcfe0}.reg .t i{color:var(--dim);font-style:normal}
 input[type=text]{width:100%;background:#22222e;color:var(--text);border:1px solid #3a3a4a;border-radius:5px;padding:5px 6px;font:inherit}
"""

_PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>__CSS__</style></head><body>__CHROME__
<script>
__COMMON__
const META = __META__;
const B64 = "__DATA__";
const glc = $("gl"), ovc = $("ov");
const gl = glc.getContext("webgl2", {antialias:false, alpha:false, preserveDrawingBuffer:true});
if(!gl){ fail("This view needs <b>WebGL2</b>, which your browser has turned off or does not have.<br>Turn it on (in Firefox: <code>about:config</code> &rarr; <code>webgl.disabled</code> = false), or open this page in another browser.");
} else if(typeof DecompressionStream === "undefined"){ fail("Your browser is too old for this view (no DecompressionStream). Use a current Firefox, Chrome, Edge or Safari.");
} else { main().catch(e => fail("Could not build the 3D view: " + e)); }

async function main(){
  const R = META.regions.length, N = META.n, M = META.m;
  const raw = Uint8Array.from(atob(B64), c => c.charCodeAt(0));
  const buf = await new Response(new Blob([raw]).stream().pipeThrough(new DecompressionStream("deflate"))).arrayBuffer();
  const regArr = new Uint32Array(buf, 0, N), idxArr = new Uint32Array(buf, 4*N, 3*M);
  const posArr = new Uint16Array(buf, 4*N+12*M, 3*N), nrmArr = new Int8Array(buf, 4*N+12*M+6*N, 4*N);

  const hasFloat = !!(gl.getExtension("EXT_color_buffer_float") || gl.getExtension("EXT_color_buffer_half_float"));
  const vs = `#version 300 es
  in vec3 aPos; in vec4 aNrm; in uint aReg;
  uniform vec3 uMin, uSize, uCenter; uniform float uInv, uStretch; uniform mat4 uVP;
  out vec3 vW; out vec3 vN; flat out uint vReg;
  void main(){ vec3 p=aPos*uSize+uMin;
    vec3 w=vec3(p.x-uCenter.x, -(p.y-uCenter.y), (p.z-uCenter.z)*uStretch)*uInv;
    vN=normalize(vec3(aNrm.x,-aNrm.y,aNrm.z/uStretch)); vW=w; vReg=aReg; gl_Position=uVP*vec4(w,1.); }`;
  const fs = `#version 300 es
  precision highp float; precision highp int; precision highp sampler2D;
  in vec3 vW; in vec3 vN; flat in uint vReg; out vec4 o;
  uniform sampler2D uStyle; uniform int uStyleW, uPass, uSel, uOrtho, uSimple;
  uniform float uGlobal, uAmb, uDif, uSpc, uShin; uniform vec3 uEye, uFwd, uLight, uExt, uCropLo, uCropHi;
  uniform vec4 uClipA, uClipB; uniform int uClipOnA, uClipOnB;
  void main(){
    ivec2 tc=ivec2(int(vReg)%uStyleW, int(vReg)/uStyleW);
    vec4 st=texelFetch(uStyle,tc,0); float alpha=st.a*uGlobal;
    if(alpha<0.004) discard;
    vec3 u=vec3(vW.x/(2.*uExt.x)+.5, .5-vW.y/(2.*uExt.y), vW.z/(2.*uExt.z)+.5);
    if(any(lessThan(u,uCropLo))||any(greaterThan(u,uCropHi))) discard;
    if(uClipOnA==1 && dot(uClipA.xyz,vW)+uClipA.w<0.) discard;
    if(uClipOnB==1 && dot(uClipB.xyz,vW)+uClipB.w<0.) discard;
    bool opaque=alpha>=0.995;
    if(uPass==0 && !opaque) discard;
    if((uPass==1||uPass==2) && opaque) discard;
    if(uPass==3){ uint r=vReg+1u; o=vec4(float(r&255u),float((r>>8)&255u),float((r>>16)&255u),255.)/255.; return; }
    vec3 N=normalize(vN); vec3 V= uOrtho==1 ? -uFwd : normalize(uEye-vW); if(dot(N,V)<0.) N=-N;
    vec3 L=normalize(uLight), H=normalize(L+V);
    float df=max(dot(N,L),0.), sp=pow(max(dot(N,H),0.),uShin), rim=pow(1.-max(dot(N,V),0.),3.);
    vec3 base=st.rgb; if(int(vReg)==uSel) base=mix(base,vec3(1.),.5);
    vec3 c=base*(uAmb+(1.-uAmb)*df*uDif)+vec3(sp*uSpc)+base*rim*.12;
    if(uPass==0){ o=vec4(c,1.); }
    else if(uSimple==1){ o=vec4(c,alpha); }
    else if(uPass==1){ float w=clamp(pow(min(1.,alpha*10.)+.01,3.)*1e3*pow(1.-gl_FragCoord.z*.9,3.),1e-2,3e3); o=vec4(c*alpha*w,alpha*w); }
    else { o=vec4(alpha); }
  }`;
  const qvs = `#version 300 es
  in vec2 p; void main(){ gl_Position=vec4(p,0.,1.); }`;
  const qfs = `#version 300 es
  precision highp float; out vec4 o;
  uniform sampler2D tOpaque, tAccum, tReveal; uniform vec2 uRes; uniform vec3 bgTop, bgBot; uniform int uHasT;
  void main(){ vec2 uv=gl_FragCoord.xy/uRes; vec3 bg=mix(bgBot,bgTop,uv.y); vec4 op=texture(tOpaque,uv); vec3 base=mix(bg,op.rgb,op.a);
    if(uHasT==0){ o=vec4(base,1.); return; }
    float rev=texture(tReveal,uv).r; vec4 ac=texture(tAccum,uv); vec3 avg=ac.rgb/max(ac.a,1e-5);
    o=vec4(avg*(1.-rev)+base*rev,1.); }`;
  function sh(type, src){ const s=gl.createShader(type); gl.shaderSource(s,src); gl.compileShader(s);
    if(!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; }
  function program(v,f){ const p=gl.createProgram(); gl.attachShader(p,sh(gl.VERTEX_SHADER,v)); gl.attachShader(p,sh(gl.FRAGMENT_SHADER,f)); gl.linkProgram(p);
    if(!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p)); return p; }
  const pm = program(vs,fs), pq = program(qvs,qfs);
  const UM = n => gl.getUniformLocation(pm, n), UQ = n => gl.getUniformLocation(pq, n);

  // ---- geometry buffers ---------------------------------------------------------------------
  const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
  const buffer = (target, data) => { const b=gl.createBuffer(); gl.bindBuffer(target,b); gl.bufferData(target,data,gl.STATIC_DRAW); return b; };
  buffer(gl.ARRAY_BUFFER, posArr); let l=gl.getAttribLocation(pm,"aPos"); gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l,3,gl.UNSIGNED_SHORT,true,0,0);
  buffer(gl.ARRAY_BUFFER, nrmArr); l=gl.getAttribLocation(pm,"aNrm"); gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l,4,gl.BYTE,true,0,0);
  buffer(gl.ARRAY_BUFFER, regArr); l=gl.getAttribLocation(pm,"aReg"); gl.enableVertexAttribArray(l); gl.vertexAttribIPointer(l,1,gl.UNSIGNED_INT,0,0);
  buffer(gl.ELEMENT_ARRAY_BUFFER, idxArr);
  const qvao = gl.createVertexArray(); gl.bindVertexArray(qvao);
  buffer(gl.ARRAY_BUFFER, new Float32Array([-1,-1,3,-1,-1,3])); l=gl.getAttribLocation(pq,"p"); gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l,2,gl.FLOAT,false,0,0);

  // ---- per-region style: colour, opacity (0 = hidden) -------------------------------------------
  const styleW = 256, styleH = Math.max(1, Math.ceil(R/styleW)), style = new Uint8Array(styleW*styleH*4);
  const reg = META.regions.map(([label,volume],i)=>({label,volume,color:[1,1,1],alpha:1,visible:true,row:null}));
  const hsv = (h,s,v)=>{ const i=Math.floor(h*6), f=h*6-i, p=v*(1-s), q=v*(1-f*s), t=v*(1-(1-f)*s); return [[v,t,p],[q,v,p],[p,v,t],[p,q,v],[t,p,v],[v,p,q]][i%6]; };
  const viridis = u => { const st=[[.267,.005,.329],[.231,.318,.545],[.128,.567,.551],[.369,.789,.383],[.993,.906,.144]], x=Math.min(.9999,Math.max(0,u))*4, i=Math.floor(x), f=x-i; return st[i].map((a,k)=>a+(st[i+1][k]-a)*f); };
  const maxVol = Math.max(...reg.map(r=>r.volume)), minVol = Math.min(...reg.map(r=>r.volume));
  function colourize(scheme){
    reg.forEach((r,i)=>{ r.color = scheme==="one" ? hex2rgb(S.oneCol) : scheme==="vol" ? viridis(Math.log(r.volume/minVol+1)/Math.log(maxVol/minVol+1)) : hsv((i*0.61803398875)%1, 0.62, 0.98); });
  }
  let hasTransparent = false, texture = gl.createTexture();
  function pushStyle(){
    hasTransparent = S.globalOpacity<0.995;
    reg.forEach((r,i)=>{ const o=i*4; style[o]=r.color[0]*255; style[o+1]=r.color[1]*255; style[o+2]=r.color[2]*255; style[o+3]=r.visible ? Math.round(r.alpha*255) : 0;
      if(r.visible && r.alpha<0.995) hasTransparent = true; if(r.row){ r.row.querySelector("input[type=color]").value = "#"+r.color.map(c=>Math.round(c*255).toString(16).padStart(2,"0")).join(""); } });
    gl.bindTexture(gl.TEXTURE_2D, texture); gl.pixelStorei(gl.UNPACK_ALIGNMENT,1);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,styleW,styleH,0,gl.RGBA,gl.UNSIGNED_BYTE,style);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
    poke();
  }

  // ---- framebuffers for the opaque / transparent passes and for picking ---------------------------
  let FB = null;
  function target(W,H){
    if(FB && FB.W===W && FB.H===H) return FB;
    if(FB){ for(const k of ["tOp","tAc","tRe","depth","pickTex","pickDepth"]) { if(FB[k]) (k==="pickDepth"?gl.deleteRenderbuffer:gl.deleteTexture).call(gl, FB[k]); } }
    const tex=(fmt,ifmt,type)=>{ const t=gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D,t); gl.texImage2D(gl.TEXTURE_2D,0,fmt,W,H,0,ifmt,type,null);
      gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE); return t; };
    const f={W,H};
    f.tOp=tex(gl.RGBA8,gl.RGBA,gl.UNSIGNED_BYTE); f.depth=tex(gl.DEPTH_COMPONENT24,gl.DEPTH_COMPONENT,gl.UNSIGNED_INT);
    f.tAc=hasFloat?tex(gl.RGBA16F,gl.RGBA,gl.HALF_FLOAT):null; f.tRe=hasFloat?tex(gl.RGBA8,gl.RGBA,gl.UNSIGNED_BYTE):null;
    const fbo=(color)=>{ const b=gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER,b);
      gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,color,0);
      gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.DEPTH_ATTACHMENT,gl.TEXTURE_2D,f.depth,0); return b; };
    f.fOp=fbo(f.tOp); if(hasFloat){ f.fAc=fbo(f.tAc); f.fRe=fbo(f.tRe); }
    f.pickTex=tex(gl.RGBA8,gl.RGBA,gl.UNSIGNED_BYTE); f.pickDepth=gl.createRenderbuffer(); gl.bindRenderbuffer(gl.RENDERBUFFER,f.pickDepth);
    gl.renderbufferStorage(gl.RENDERBUFFER,gl.DEPTH_COMPONENT24,W,H);
    f.fPick=gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER,f.fPick);
    gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,f.pickTex,0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER,gl.DEPTH_ATTACHMENT,gl.RENDERBUFFER,f.pickDepth);
    gl.bindFramebuffer(gl.FRAMEBUFFER,null); FB=f; return f;
  }

  // ---- scene geometry ---------------------------------------------------------------------------
  const sizes = () => [META.extent[0], META.extent[1], META.extent[2]*S.stretch];
  const maxPhys = () => Math.max(...sizes());
  const halfExt = () => sizes().map(v=>v/maxPhys()/2);
  const scene = { ext: halfExt, physPerWorld: maxPhys, unit: META.unit, fg:[.9,.9,.95] };
  const backgrounds = { dark:["#171722","#06060a"], black:["#000000","#000000"], grey:["#4a4a52","#2c2c32"], white:["#ffffff","#e9e9ee"], sky:["#d7e4f5","#f6f8fb"] };
  const bgColors = () => S.bg==="custom" ? [S.bgTop,S.bgBot] : backgrounds[S.bg];
  const axisN = (axis,az,el,flip)=>{ const n= axis==="x"?[1,0,0]:axis==="y"?[0,1,0]:axis==="z"?[0,0,1]:[Math.cos(el)*Math.sin(az),Math.sin(el),Math.cos(el)*Math.cos(az)]; return flip?n.map(x=>-x):n; };
  function planeFor(p){ const e=halfExt(), n=axisN(S[p+"Axis"],S[p+"Az"]*Math.PI/180,S[p+"El"]*Math.PI/180,S[p+"Flip"]);
    return [n[0],n[1],n[2], -S[p+"Off"]*(Math.abs(n[0])*e[0]+Math.abs(n[1])*e[1]+Math.abs(n[2])*e[2])]; }
  function lightVector(){
    if(S.headlight){ const e=camEye(), f=norm(sub([cam.tx,cam.ty,cam.tz],e)), r=norm(cross(f,[0,1,0])), u=cross(r,f);
      return norm([-f[0]+.45*r[0]+.55*u[0], -f[1]+.45*r[1]+.55*u[1], -f[2]+.45*r[2]+.55*u[2]]); }
    const az=S.lightAz*Math.PI/180, el=S.lightEl*Math.PI/180; return [Math.cos(el)*Math.sin(az),Math.sin(el),Math.cos(el)*Math.cos(az)]; }
  let selected = -1;

  function setMeshUniforms(W,H,pass,vp){
    const e=halfExt(), eye=camEye(), f=norm(sub([cam.tx,cam.ty,cam.tz],eye));
    gl.uniformMatrix4fv(UM("uVP"),false,vp); gl.uniform3fv(UM("uMin"),META.min); gl.uniform3fv(UM("uSize"),META.size);
    gl.uniform3fv(UM("uCenter"),[META.extent[0]/2,META.extent[1]/2,META.extent[2]/2]); gl.uniform1f(UM("uInv"),1/maxPhys()); gl.uniform1f(UM("uStretch"),S.stretch);
    gl.uniform1i(UM("uStyleW"),styleW); gl.uniform1i(UM("uStyle"),0); gl.uniform1i(UM("uPass"),pass); gl.uniform1i(UM("uSel"),selected);
    gl.uniform1i(UM("uOrtho"),cam.ortho?1:0); gl.uniform1i(UM("uSimple"),hasFloat?0:1);
    gl.uniform1f(UM("uGlobal"),S.globalOpacity); gl.uniform1f(UM("uAmb"),S.amb); gl.uniform1f(UM("uDif"),S.dif); gl.uniform1f(UM("uSpc"),S.spc); gl.uniform1f(UM("uShin"),S.shin);
    gl.uniform3fv(UM("uEye"),eye); gl.uniform3fv(UM("uFwd"),f); gl.uniform3fv(UM("uLight"),lightVector()); gl.uniform3fv(UM("uExt"),e);
    gl.uniform3fv(UM("uCropLo"),[S.x0,S.y0,S.z0]); gl.uniform3fv(UM("uCropHi"),[S.x1,S.y1,S.z1]);
    gl.uniform4fv(UM("uClipA"),planeFor("ca")); gl.uniform4fv(UM("uClipB"),planeFor("cb")); gl.uniform1i(UM("uClipOnA"),S.caOn?1:0); gl.uniform1i(UM("uClipOnB"),S.cbOn?1:0);
  }
  const drawMesh = () => { gl.bindVertexArray(vao); gl.drawElements(gl.TRIANGLES, 3*M, gl.UNSIGNED_INT, 0); };

  function draw(fast, scaleOverride){
    const stage=$("stage"), cw=stage.clientWidth, chh=stage.clientHeight, dpr=Math.min(window.devicePixelRatio||1,2);
    const sc = scaleOverride || (fast && S.fastMove ? S.moveScale : S.renderScale);
    const W=Math.max(2,Math.round(cw*dpr*sc)), H=Math.max(2,Math.round(chh*dpr*sc));
    if(glc.width!==W||glc.height!==H){ glc.width=W; glc.height=H; }
    const f=target(W,H), vp=camVP(W/H), [top,bot]=bgColors(), T=hex2rgb(top), B=hex2rgb(bot);
    scene.fg = luma([(T[0]+B[0])/2,(T[1]+B[1])/2,(T[2]+B[2])/2])>.5 ? [.1,.1,.14] : [.9,.9,.95];
    $("hud").style.color = luma(T)>.5 ? "#555" : "#8d8da0";
    gl.useProgram(pm); gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D,texture); gl.viewport(0,0,W,H);
    // 1. opaque regions, with depth
    gl.bindFramebuffer(gl.FRAMEBUFFER,f.fOp); gl.disable(gl.BLEND); gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LESS); gl.depthMask(true); gl.colorMask(true,true,true,true);
    gl.clearColor(0,0,0,0); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT); setMeshUniforms(W,H,0,vp); drawMesh();
    // 2. see-through regions, order independent (or plain blending where float targets are missing)
    if(hasTransparent){
      gl.depthMask(false);
      if(hasFloat){
        gl.bindFramebuffer(gl.FRAMEBUFFER,f.fAc); gl.clearColor(0,0,0,0); gl.clear(gl.COLOR_BUFFER_BIT); gl.enable(gl.BLEND); gl.blendFunc(gl.ONE,gl.ONE);
        setMeshUniforms(W,H,1,vp); drawMesh();
        gl.bindFramebuffer(gl.FRAMEBUFFER,f.fRe); gl.clearColor(1,1,1,1); gl.clear(gl.COLOR_BUFFER_BIT); gl.blendFunc(gl.ZERO,gl.ONE_MINUS_SRC_ALPHA);
        setMeshUniforms(W,H,2,vp); drawMesh();
      } else {
        gl.bindFramebuffer(gl.FRAMEBUFFER,f.fOp); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA); setMeshUniforms(W,H,1,vp); drawMesh();
      }
      gl.disable(gl.BLEND); gl.depthMask(true);
    }
    // 3. together with the background, to the picture
    gl.bindFramebuffer(gl.FRAMEBUFFER,null); gl.viewport(0,0,W,H); gl.disable(gl.DEPTH_TEST); gl.useProgram(pq); gl.bindVertexArray(qvao);
    const tunit=(u,t,name)=>{ gl.activeTexture(gl.TEXTURE0+u); gl.bindTexture(gl.TEXTURE_2D,t); gl.uniform1i(UQ(name),u); };
    tunit(1,f.tOp,"tOpaque"); tunit(2,f.tAc||f.tOp,"tAccum"); tunit(3,f.tRe||f.tOp,"tReveal"); gl.activeTexture(gl.TEXTURE0);
    gl.uniform2f(UQ("uRes"),W,H); gl.uniform3fv(UQ("bgTop"),T); gl.uniform3fv(UQ("bgBot"),B); gl.uniform1i(UQ("uHasT"),hasTransparent&&hasFloat?1:0);
    gl.drawArrays(gl.TRIANGLES,0,3); gl.enable(gl.DEPTH_TEST);
    ovc.width=Math.round(cw*dpr); ovc.height=Math.round(chh*dpr); drawOverlay(ovc.getContext("2d"),ovc.width,ovc.height,scene);
    const shown = reg.filter(r=>r.visible).length;
    $("hud").textContent = shown+" of "+R+" regions shown | "+M.toLocaleString()+" triangles"+(META.skipped?" | "+META.skipped+" too small to draw":"")+(META.omitted?" | "+META.omitted+" smaller regions left out":"")+" | depth x"+S.stretch;
  }
  Q.draw = draw;

  // ---- picking: click a region to find it in the list ---------------------------------------------
  function pick(clientX, clientY){
    const r=glc.getBoundingClientRect(), W=glc.width, H=glc.height; if(!FB) return -1;
    const x=Math.floor((clientX-r.left)/r.width*W), y=Math.floor((1-(clientY-r.top)/r.height)*H);
    gl.useProgram(pm); gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D,texture);
    gl.bindFramebuffer(gl.FRAMEBUFFER,FB.fPick); gl.viewport(0,0,W,H); gl.disable(gl.BLEND); gl.enable(gl.DEPTH_TEST); gl.depthMask(true);
    gl.clearColor(0,0,0,0); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT); setMeshUniforms(W,H,3,camVP(W/H)); drawMesh();
    const px=new Uint8Array(4); gl.readPixels(x,y,1,1,gl.RGBA,gl.UNSIGNED_BYTE,px); gl.bindFramebuffer(gl.FRAMEBUFFER,null);
    return (px[0]|(px[1]<<8)|(px[2]<<16)) - 1;
  }
  function select(i){
    if(selected>=0 && reg[selected].row) reg[selected].row.classList.remove("sel");
    selected=i;
    if(i>=0){ const r=reg[i]; $("hud").title=""; if(r.row){ r.row.classList.add("sel"); r.row.scrollIntoView({block:"nearest"}); }
      $("picked").textContent = "Region #"+r.label+"  |  volume "+(+r.volume.toPrecision(4)).toLocaleString()+" "+META.unit+"³"; }
    else $("picked").textContent = "Click a region in the picture to find it here.";
    poke();
  }
  let down=null;
  glc.addEventListener("pointerdown",e=>{ down={x:e.clientX,y:e.clientY,t:performance.now()}; });
  glc.addEventListener("pointerup",e=>{ if(down && Math.hypot(e.clientX-down.x,e.clientY-down.y)<4 && performance.now()-down.t<500 && !e.shiftKey && e.button===0){ select(pick(e.clientX,e.clientY)); } down=null; });

  // ---- controls -----------------------------------------------------------------------------------
  const ui = makeUI($("panel"));
  const heading=document.createElement("h1"); heading.textContent=META.title;
  const hint=document.createElement("div"); hint.id="hint"; hint.textContent="Drag: rotate · Scroll: zoom · Right-drag or Shift+drag: pan · Click a region: select it · Double-click: reset";
  $("panel").prepend(heading, hint);
  const pct = v=>Math.round(v*100)+"%", x2=v=>v.toFixed(2);

  ui.section("Regions", true);
  const picked=document.createElement("div"); picked.id="picked"; picked.className="note"; picked.textContent="Click a region in the picture to find it here."; ui.box().appendChild(picked);
  ui.range("globalOpacity","Transparency of all regions (opacity)",0.05,1,0.01,1,pct,()=>pushStyle());
  ui.range("topN","Show only the largest",1,R,1,R,v=>v+" of "+R,v=>{ reg.forEach((r,i)=>{ r.visible=i<v; if(r.row) r.row.querySelector("input[type=checkbox]").checked=r.visible; }); pushStyle(); });
  ui.select("scheme","Colours",[["distinct","A different colour for each"],["vol","By volume (viridis)"],["one","One colour"]],"distinct",v=>{ colourize(v); pushStyle(); });
  ui.color("oneCol","  the one colour","#ff4dff",()=>{ if(S.scheme==="one"){ colourize("one"); pushStyle(); } });
  ui.buttons([["Show all",()=>{ reg.forEach(r=>{r.visible=true; if(r.row) r.row.querySelector("input[type=checkbox]").checked=true;}); pushStyle(); }],
              ["Hide all",()=>{ reg.forEach(r=>{r.visible=false; if(r.row) r.row.querySelector("input[type=checkbox]").checked=false;}); pushStyle(); }],
              ["Invert",()=>{ reg.forEach(r=>{r.visible=!r.visible; if(r.row) r.row.querySelector("input[type=checkbox]").checked=r.visible;}); pushStyle(); }]]);
  const search=document.createElement("input"); search.type="text"; search.placeholder="Find a region by its number"; ui.box().appendChild(search);
  const list=document.createElement("div"); list.id="list"; ui.box().appendChild(list);
  const ROWS = Math.min(R, 300);
  for(let i=0;i<ROWS;i++){
    const r=reg[i], row=document.createElement("div"); row.className="reg"; row.dataset.label=r.label;
    row.innerHTML=`<input type="checkbox" checked title="Show or hide this region"><input type="color" title="Colour"><div class="t">#${r.label} <i>${(+r.volume.toPrecision(3)).toLocaleString()}</i></div><input type="range" min="0.05" max="1" step="0.05" value="1" title="Opacity of this region">`;
    row.querySelector("input[type=checkbox]").addEventListener("input",e=>{ r.visible=e.target.checked; pushStyle(); });
    row.querySelector("input[type=color]").addEventListener("input",e=>{ r.color=hex2rgb(e.target.value); pushStyle(); });
    row.querySelector("input[type=range]").addEventListener("input",e=>{ r.alpha=+e.target.value; pushStyle(); });
    row.querySelector(".t").addEventListener("click",()=>select(i)); r.row=row; list.appendChild(row);
  }
  if(R>ROWS){ const note=document.createElement("div"); note.className="note"; note.style.padding="4px 6px";
    note.textContent=(R-ROWS)+" smaller regions are not listed: use the opacity and \"Show only the largest\" controls above, or click one in the picture."; list.appendChild(note); }
  search.addEventListener("input",()=>{ const q=search.value.trim(); reg.forEach(r=>{ if(r.row) r.row.style.display = (!q||String(r.label).includes(q)) ? "" : "none"; }); });
  ui.note("Rows show the region number and its volume ("+META.unit+"³), largest first.");

  ui.section("View", true);
  ui.buttons([["Front",()=>setView(0,0)],["Side",()=>setView(Math.PI/2,0)],["Top",()=>setView(0,Math.PI/2-0.001)],["Back",()=>setView(Math.PI,0)]]);
  ui.select("proj","Projection",[["p","Perspective"],["o","Orthographic (true scale bar)"]],"p",v=>{ cam.ortho=v==="o"; });
  ui.range("stretch","Depth exaggeration",1,20,0.5,META.stretch,v=>"×"+v.toFixed(1));
  ui.range("rotspeed","Rotation speed",0.2,4,0.1,1,x2);
  ui.check("hqRotate","Full quality while rotating",false);

  ui.section("Light", false);
  ui.check("headlight","Light follows the camera",true);
  ui.range("lightAz","Light direction (left-right)",-180,180,1,40,v=>v.toFixed(0)+"°");
  ui.range("lightEl","Light direction (up-down)",-90,90,1,45,v=>v.toFixed(0)+"°");
  ui.range("amb","Ambient light",0,1,0.01,0.3,x2);
  ui.range("dif","Diffuse (shading)",0,1,0.01,0.9,x2);
  ui.range("spc","Specular (shine)",0,1,0.01,0.3,x2);
  ui.range("shin","Shininess",4,128,1,40,v=>v.toFixed(0));

  ui.section("Cut", false);
  ui.note("Crop box (fractions of the sample)");
  [["x","Left - right"],["y","Top - bottom"],["z","Front - back (planes)"]].forEach(([a,lab])=>{ ui.range(a+"0",lab+": from",0,1,0.01,0,pct); ui.range(a+"1",lab+": to",0,1,0.01,1,pct); });
  [["ca","Cutting plane 1"],["cb","Cutting plane 2"]].forEach(([p,lab],idx)=>{
    ui.note(lab); ui.check(p+"On","Use "+lab.toLowerCase(),false);
    ui.select(p+"Axis","  direction",[["x","Across x (left - right)"],["y","Across y (top - bottom)"],["z","Across z (front - back)"],["free","Any direction"]],idx?"y":"x");
    ui.range(p+"Az","  turn (any direction)",-180,180,1,idx?0:35,v=>v.toFixed(0)+"°");
    ui.range(p+"El","  tilt (any direction)",-90,90,1,idx?60:20,v=>v.toFixed(0)+"°");
    ui.range(p+"Off","  position",-1,1,0.01,0,x2); ui.check(p+"Flip","  keep the other side",false); });

  ui.section("Scene", false);
  ui.select("bg","Background",[["dark","Dark"],["black","Black"],["grey","Grey"],["white","White"],["sky","Light"],["custom","Custom colours"]],"dark");
  ui.color("bgTop","  custom: top","#202030"); ui.color("bgBot","  custom: bottom","#05050a");
  ui.check("axes","Orientation axes",true); ui.check("scale","Scale bar",true); ui.check("box","Bounding box of the sample",true);
  ui.note("The scale bar is exact for the centre of the view (and for the whole view in orthographic mode); it measures x and y, not the stretched depth.");

  ui.section("Quality", false);
  ui.range("renderScale","Resolution when still",0.5,2,0.25,1,v=>"×"+v);
  ui.check("fastMove","Faster render while moving",true);
  ui.range("moveScale","Resolution while moving",0.25,1,0.05,0.6,v=>"×"+v.toFixed(2));
  ui.note(hasFloat ? "See-through regions use order-independent transparency." : "This browser lacks float render targets, so see-through regions use plain blending (the order of overlapping regions can show).");

  ui.section("Export", false);
  ui.select("shotScale","Screenshot size",[["1","Window size"],["2","2×"],["4","4× (large)"]],"2");
  ui.buttons([["Save screenshot (PNG)",()=>snapshot()]]);
  function snapshot(){ draw(false, +S.shotScale/Math.min(window.devicePixelRatio||1,2)); savePng(glc,(g,w,h)=>drawOverlay(g,w,h,scene),"HIVE_3D_regions.png"); setTimeout(poke,300); }
  $("btnShot").onclick = snapshot;

  colourize("distinct"); S.autorot=false;
  const layout = () => { $("stage").style.left = $("panel").classList.contains("hidden") ? "0" : "272px"; poke(); };
  wireChrome(layout); attachPointer(glc);
  if(window.innerWidth<720){ $("panel").classList.add("hidden"); $("toggle").textContent="☰ Controls"; }
  layout(); pushStyle();
  window.hiveRegions = { S, cam, reg, render: () => draw(false), pick, select, pushStyle, hasFloat, snapshotCanvas: () => glc.toDataURL("image/png") };
  draw(false);
}
</script></body></html>
"""

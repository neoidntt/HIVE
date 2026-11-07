"""An interactive 3D view of a whole z-stack that runs in the browser.

:func:`volume_viewer_html` turns a volume ``(Z, Y, X)`` (every z-plane) into one self-contained HTML
page with a WebGL2 ray-caster. Drag to rotate, scroll to zoom, right-drag (or Shift-drag) to pan.

What the page offers (the controls are grouped and collapsed so the picture stays uncluttered):

* **Rendering.** Volume rendering with front-to-back alpha compositing, so nearer structures hide
  farther ones; a surface mode that draws the first structure the ray meets; and "brightest voxel",
  the look of a maximum-intensity projection.
* **Light.** Blinn-Phong shading from gradient normals, a light you can aim (or one that follows the
  camera), ambient / diffuse / specular amounts, soft shadows and ambient occlusion.
* **Appearance.** Brightness, black and white points, gamma, opacity, a threshold, colour maps, a colour
  for each channel, and a highlight for segmented objects.
* **Cutting.** A crop box and two free clipping planes (any direction) to look inside the sample.
* **Quality.** Full quality when still, a cheaper render while the view moves.
* **Scene.** Background colours, an orientation-axes gizmo, a scale bar, a bounding box, auto-rotate,
  and PNG screenshots.

Nothing is sent to any server, and the page needs no internet connection. It needs a browser with
WebGL2 turned on (current Firefox, Chrome, Edge and Safari have it); otherwise the page says so.

For looking only. Measurements come from the analysis, never from this view.
"""

from __future__ import annotations

import base64
import json
import math
import zlib
from typing import Sequence

import numpy as np

from .viewer_common import CHROME_HTML, CSS, JS

DEFAULT_MAX_VOXELS = 12_000_000          # voxels sent to the browser (after averaging in y and x)
_SLAB = 8_000_000                        # voxels reduced at a time, to bound temporary memory


def reduce_volume(volume: np.ndarray, max_voxels: int = DEFAULT_MAX_VOXELS, reduce: str = "mean"):
    """Shrink a volume in y and x only, so every z-plane is kept.

    Blocks of ``k x k`` pixels are averaged (``reduce="mean"``) or their maximum is taken
    (``"max"``, for masks). ``k`` is the smallest integer that brings the volume under ``max_voxels``.

    Returns:
        ``(reduced volume, k)``; the volume is returned unchanged (``k = 1``) when it is small enough.
    """
    z, h, w = volume.shape
    k = max(1, math.ceil(math.sqrt(z * h * w / max_voxels)))
    if k == 1:
        return volume, 1
    rows, cols = h // k, w // k
    out = np.empty((z, rows, cols), dtype=np.float32 if reduce == "mean" else volume.dtype)
    step = max(1, _SLAB // max(1, h * w))
    for start in range(0, z, step):
        block = np.asarray(volume[start:start + step, :rows * k, :cols * k])
        block = block.reshape(block.shape[0], rows, k, cols, k)
        out[start:start + step] = block.mean(axis=(2, 4)) if reduce == "mean" else block.max(axis=(2, 4))
    return out, k


def _to_uint8(volume: np.ndarray, low: float = 1.0, high: float = 99.8) -> np.ndarray:
    """Scale to 0-255 between two percentiles, so a few hot voxels do not darken everything."""
    lo, hi = np.percentile(volume, [low, high])
    if hi <= lo:
        hi = lo + 1.0
    return (np.clip((volume - lo) / (hi - lo), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def auto_depth_exaggeration(shape: tuple[int, int, int], scale: tuple[float, float, float],
                            wanted: float = 0.2, cap: float = 12.0) -> float:
    """Stretch that makes a thin sample at least ``wanted`` times as deep as it is wide.

    1.0 (true to scale) when the sample is already deep enough; never more than ``cap``.
    """
    z, h, w = shape
    depth = z * scale[0]
    width = max(h * scale[1], w * scale[2])
    ratio = depth / width if width else 1.0
    return 1.0 if ratio >= wanted else float(min(cap, wanted / max(ratio, 1e-9)))


def volume_viewer_html(
    intensity: np.ndarray,
    mask: np.ndarray | None = None,
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    title: str = "HIVE 3D volume",
    max_voxels: int = DEFAULT_MAX_VOXELS,
    depth_exaggeration: float | None = None,
    unit: str = "px",
    extra_channels: Sequence[np.ndarray] | None = None,
    channel_names: Sequence[str] | None = None,
) -> str:
    """Build the interactive page for a volume.

    Args:
        intensity: ``(Z, Y, X)`` array in any scale (brightness is rescaled for display): channel 1.
        mask: Optional boolean ``(Z, Y, X)`` mask of segmented objects, shown as a highlight that
            can be switched on and off in the page.
        voxel_size: Physical size of a voxel as ``(z, y, x)``, so the shape is true to scale.
        title: Heading inside the page.
        max_voxels: Largest number of voxels sent to the browser; the volume is averaged in y and x
            to fit (all z-planes are always kept).
        depth_exaggeration: Stretch of the z axis. ``None`` chooses one so a thin sample is visibly
            deep (1 = true to scale); the page has a slider to change it.
        unit: Name of the physical unit, written on the scale bar.
        extra_channels: Up to two further ``(Z, Y, X)`` channels, drawn in their own colours.
        channel_names: Names for the channels, in order.

    Returns:
        The HTML text.

    Raises:
        ValueError: If ``intensity`` is not 3D with at least 2 planes, or a shape differs.
    """
    volume = np.asarray(intensity)
    if volume.ndim != 3 or volume.shape[0] < 2:
        raise ValueError(f"A 3D view needs a volume with at least 2 z-planes, got shape {volume.shape}")
    extras = list(extra_channels or [])[:2]
    for other in extras:
        if np.shape(other) != volume.shape:
            raise ValueError(f"channel shape {np.shape(other)} differs from the volume {volume.shape}")
    if mask is not None and np.shape(mask) != volume.shape:
        raise ValueError(f"mask shape {np.shape(mask)} differs from the volume {volume.shape}")

    reduced, k = reduce_volume(volume, max_voxels)
    planes = [_to_uint8(np.asarray(reduced, dtype=np.float32))]
    for other in extras:
        planes.append(_to_uint8(np.asarray(reduce_volume(np.asarray(other), max_voxels)[0], dtype=np.float32)))
    z, h, w = planes[0].shape
    data = np.zeros((z, h, w, 4), dtype=np.uint8)               # R, G, B: channels; A: objects mask
    for index, plane in enumerate(planes):
        data[..., index] = plane
    if mask is not None:
        small, _ = reduce_volume(np.asarray(mask, dtype=np.uint8), max_voxels, reduce="max")
        data[..., 3] = (small[:z, :h, :w] > 0).astype(np.uint8) * 255
    scale = (float(voxel_size[0]), float(voxel_size[1]) * k, float(voxel_size[2]) * k)
    stretch = auto_depth_exaggeration((z, h, w), scale) if depth_exaggeration is None else float(depth_exaggeration)
    names = list(channel_names or [])
    names = [str(names[i]) if i < len(names) else f"Channel {i + 1}" for i in range(len(planes))]
    meta = {"w": w, "h": h, "d": z, "channels": len(planes), "names": names, "hasMask": mask is not None,
            "scale": [scale[2], scale[1], scale[0]], "stretch": round(stretch, 2), "reduced_by": k,
            "source": list(volume.shape), "title": title, "unit": str(unit)}
    payload = base64.b64encode(zlib.compress(np.ascontiguousarray(data).tobytes(), 6)).decode("ascii")
    safe_meta = json.dumps(meta).replace("</", "<\\/")          # a name must not close the <script>
    return (_PAGE.replace("__CSS__", CSS).replace("__CHROME__", CHROME_HTML).replace("__COMMON__", JS)
            .replace("__META__", safe_meta).replace("__DATA__", payload)
            .replace("__TITLE__", title.replace("<", "&lt;")))


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
  const maxTex = gl.getParameter(gl.MAX_3D_TEXTURE_SIZE);
  if(Math.max(META.w, META.h, META.d) > maxTex) throw new Error("volume larger than this graphics card allows (" + maxTex + ")");
  const raw = Uint8Array.from(atob(B64), c => c.charCodeAt(0));
  const vox = new Uint8Array(await new Response(new Blob([raw]).stream().pipeThrough(new DecompressionStream("deflate"))).arrayBuffer());

  const vs = `#version 300 es
  in vec2 p; out vec2 v; void main(){ v=p; gl_Position=vec4(p,0.,1.); }`;
  const fs = `#version 300 es
  precision highp float; precision highp sampler3D;
  in vec2 v; out vec4 o;
  uniform sampler3D tex; uniform mat4 invVP; uniform vec3 ext;
  uniform float steps, eps; uniform int fast, mode, cmap, nch, shade, hl;
  uniform vec3 chColor0, chColor1, chColor2, chOn, chGain, hlColor, lightDir, cropLo, cropHi, bgTop, bgBot;
  uniform float low, high, gamma, thr, density, iso, amb, dif, spc, shin, shadowK, aoK;
  uniform vec4 clipA, clipB; uniform int clipOnA, clipOnB;

  vec3 uvwOf(vec3 q){ return vec3(q.x/(2.*ext.x)+.5, .5-q.y/(2.*ext.y), q.z/(2.*ext.z)+.5); }
  vec4 fetch(vec3 q){ return texture(tex, uvwOf(q)); }
  float win(float x){ float t=clamp((x-low)/max(high-low,1e-4),0.,1.); return pow(t,gamma); }
  vec3 levels(vec3 t){ vec3 s=vec3(win(t.r*chGain.x),win(t.g*chGain.y),win(t.b*chGain.z))*chOn;
                       return clamp((s-thr)/max(1.-thr,1e-3),0.,1.); }
  float drive(vec3 s){ return max(max(s.r,s.g),s.b); }
  float dens(vec3 q){ return drive(levels(fetch(q).rgb)); }
  vec3 pal(float s){
    if(cmap==0) return vec3(s, 0.12*s, s);                                         // magenta
    if(cmap==1) return vec3(min(1.,2.*s), clamp(2.*s-.5,0.,1.)*.85, clamp(3.*s-2.,0.,1.));   // fire
    if(cmap==2) return vec3(s);                                                    // grey
    if(cmap==3) return mix(vec3(.1,.5,1.), vec3(1.,.95,.2), s) * (.25+.75*s);      // cyan-yellow
    if(cmap==4){ float u=s*4.; vec3 a=vec3(.267,.005,.329), b=vec3(.231,.318,.545), c=vec3(.128,.567,.551), d=vec3(.369,.789,.383), e=vec3(.993,.906,.144);
                 return u<1.?mix(a,b,u):u<2.?mix(b,c,u-1.):u<3.?mix(c,d,u-2.):mix(d,e,u-3.); }  // viridis
    if(cmap==5) return vec3(clamp(3.*s-2.,0.,1.), clamp(2.*s,0.,1.)*.95, clamp(1.4*s+.35,0.,1.)*step(.001,s));   // ice
    return chColor0*s;                                                             // one colour
  }
  vec3 colorOf(vec3 s){ if(nch==1) return pal(s.r); return chColor0*s.r + chColor1*s.g + chColor2*s.b; }
  float hash(vec2 q){ return fract(sin(dot(q, vec2(12.9898,78.233))) * 43758.5453); }
  bool clipped(vec3 q){
    vec3 u=uvwOf(q); if(any(lessThan(u,cropLo))||any(greaterThan(u,cropHi))) return true;
    if(clipOnA==1 && dot(clipA.xyz,q)+clipA.w<0.) return true;
    if(clipOnB==1 && dot(clipB.xyz,q)+clipB.w<0.) return true;
    return false; }
  vec3 gradient(vec3 q){ float e=eps;
    return vec3(dens(q+vec3(e,0,0))-dens(q-vec3(e,0,0)), dens(q+vec3(0,e,0))-dens(q-vec3(0,e,0)), dens(q+vec3(0,0,e))-dens(q-vec3(0,0,e))); }
  // Blinn-Phong from the density gradient, with optional soft shadow and ambient occlusion.
  vec3 lighting(vec3 base, vec3 q, vec3 rd){
    if(shade==0) return base;
    vec3 g=gradient(q); float gl2=length(g);
    float w=smoothstep(0.004,0.06,gl2);                              // shade surfaces, not flat interiors
    vec3 n = gl2>1e-5 ? -g/gl2 : -rd; if(dot(n,-rd)<0.) n=-n;
    vec3 L=normalize(lightDir), H=normalize(L-rd);
    float df=max(dot(n,L),0.), sp=pow(max(dot(n,H),0.),shin);
    float lit = mix(1., amb+(1.-amb)*df, w*dif);
    if(fast==0 && shadowK>0.){ float occ=0.; vec3 p=q+n*0.006;
      for(int i=0;i<10;i++){ p+=L*0.045; if(any(greaterThan(abs(p),ext))) break; occ+=dens(p); }
      lit *= mix(1., exp(-occ*shadowK*0.55), 0.9); }
    if(fast==0 && aoK>0.){ float r=0.022, a=dens(q+vec3(r,0,0))+dens(q-vec3(r,0,0))+dens(q+vec3(0,r,0))+dens(q-vec3(0,r,0))+dens(q+vec3(0,0,r))+dens(q-vec3(0,0,r));
      lit *= 1.-aoK*clamp(a/6.,0.,1.)*0.85; }
    return base*lit + vec3(sp*spc*w);
  }
  void main(){
    vec4 n4=invVP*vec4(v,-1.,1.), f4=invVP*vec4(v,1.,1.);
    vec3 a=n4.xyz/n4.w, b=f4.xyz/f4.w, rd=normalize(b-a), ro=a;
    vec3 t1=(-ext-ro)/rd, t2=(ext-ro)/rd, tn=min(t1,t2), tf=max(t1,t2);
    float t0=max(max(tn.x,tn.y),max(tn.z,0.)), te=min(min(tf.x,tf.y),tf.z);
    vec3 bg=mix(bgBot,bgTop,v.y*.5+.5);
    if(te<=t0){ o=vec4(bg,1.); return; }
    float dt=(te-t0)/steps, t=t0+dt*hash(gl_FragCoord.xy);
    vec3 col=vec3(0.); float acc=0.; vec3 mx=vec3(0.); float mxm=0.;
    for(int i=0;i<1024;i++){
      if(float(i)>=steps||t>te) break;
      vec3 q=ro+rd*t; t+=dt;
      if(clipped(q)) continue;
      vec4 tx=fetch(q); vec3 s=levels(tx.rgb); float d=drive(s); bool on=(hl==1 && tx.a>.5);
      if(mode==1){ mx=max(mx,s); if(on) mxm=1.; continue; }
      if(mode==2){                                               // the first structure the ray meets
        if(d>=iso || on){
          vec3 qs=q; float ts=t-2.*dt, tb=t-dt;                  // bisect between the last two samples
          for(int k=0;k<5;k++){ float tm=.5*(ts+tb); vec3 qm=ro+rd*tm; if(dens(qm)>=iso) tb=tm; else ts=tm; }
          qs=ro+rd*tb; vec3 sc=levels(fetch(qs).rgb);
          vec3 c=colorOf(clamp(sc*2.2,0.,1.)); if(on) c=mix(c,hlColor,.65);      // a surface sits at the threshold, so it is brightened
          col=lighting(c,qs,rd); acc=1.; break; }
        continue; }
      if(on) d=max(d,.45);
      if(d<.003) continue;
      vec3 c=colorOf(s); if(on) c=drive(s)<.003 ? hlColor*.85 : mix(c,hlColor,.6);
      float al=1.-exp(-density*pow(d,1.5)*dt*6.);
      c=lighting(c,q,rd);
      col+=(1.-acc)*al*c; acc+=(1.-acc)*al;
      if(acc>.98) break;
    }
    if(mode==1){ vec3 c=colorOf(mx); if(mxm>.5) c=mix(c,hlColor,.6); float al=clamp(max(drive(mx),mxm*.5)*1.5,0.,1.); o=vec4(mix(bg,c,al),1.); return; }
    o=vec4(col+(1.-acc)*bg,1.);
  }`;
  function sh(type, src){ const s=gl.createShader(type); gl.shaderSource(s,src); gl.compileShader(s);
    if(!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; }
  const prog=gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(prog);
  if(!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  gl.useProgram(prog);
  const vb=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, vb);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 3,-1, -1,3]), gl.STATIC_DRAW);
  const loc=gl.getAttribLocation(prog,"p"); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc,2,gl.FLOAT,false,0,0);
  const tex=gl.createTexture(); gl.bindTexture(gl.TEXTURE_3D, tex); gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage3D(gl.TEXTURE_3D,0,gl.RGBA8,META.w,META.h,META.d,0,gl.RGBA,gl.UNSIGNED_BYTE,vox);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  for(const w of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) gl.texParameteri(gl.TEXTURE_3D, w, gl.CLAMP_TO_EDGE);
  const U = n => gl.getUniformLocation(prog, n);

  // ---- geometry helpers -------------------------------------------------------------------
  const sizes = () => { const s=S.stretch; return [META.w*META.scale[0], META.h*META.scale[1], META.d*META.scale[2]*s]; };
  const maxPhys = () => Math.max(...sizes());
  const halfExt = () => sizes().map(v => v/maxPhys()/2);
  const scene = { ext: halfExt, physPerWorld: maxPhys, unit: META.unit, fg: [.9,.9,.95] };
  const axisN = (axis, az, el, flip) => { let n = axis==="x"?[1,0,0]:axis==="y"?[0,1,0]:axis==="z"?[0,0,1]
      : [Math.cos(el)*Math.sin(az), Math.sin(el), Math.cos(el)*Math.cos(az)]; return flip ? n.map(x=>-x) : n; };
  function planeFor(prefix){
    const e=halfExt(), n=axisN(S[prefix+"Axis"], S[prefix+"Az"]*Math.PI/180, S[prefix+"El"]*Math.PI/180, S[prefix+"Flip"]);
    const range=Math.abs(n[0])*e[0]+Math.abs(n[1])*e[1]+Math.abs(n[2])*e[2];
    return [n[0],n[1],n[2], -S[prefix+"Off"]*range];                  // keeps n.q >= offset*range
  }
  const backgrounds = { dark:["#171722","#06060a"], black:["#000000","#000000"], grey:["#4a4a52","#2c2c32"],
                        white:["#ffffff","#e9e9ee"], sky:["#d7e4f5","#f6f8fb"] };
  function bgColors(){ if(S.bg==="custom") return [S.bgTop, S.bgBot]; return backgrounds[S.bg]; }

  // ---- drawing ----------------------------------------------------------------------------
  function lightVector(){
    if(S.headlight){ const e=camEye(), f=norm(sub([cam.tx,cam.ty,cam.tz],e)), r=norm(cross(f,[0,1,0])), u=cross(r,f);
      return norm([-f[0]+.45*r[0]+.55*u[0], -f[1]+.45*r[1]+.55*u[1], -f[2]+.45*r[2]+.55*u[2]]); }
    const az=S.lightAz*Math.PI/180, el=S.lightEl*Math.PI/180; return [Math.cos(el)*Math.sin(az), Math.sin(el), Math.cos(el)*Math.cos(az)];
  }
  function draw(fast, scaleOverride){
    const stage=$("stage"), cw=stage.clientWidth, chh=stage.clientHeight, dpr=Math.min(window.devicePixelRatio||1,2);
    const sc = scaleOverride || (fast && S.fastMove ? S.moveScale : S.renderScale);
    const W=Math.max(2,Math.round(cw*dpr*sc)), H=Math.max(2,Math.round(chh*dpr*sc));
    if(glc.width!==W||glc.height!==H){ glc.width=W; glc.height=H; }
    gl.viewport(0,0,W,H);
    const e=halfExt(), vp=camVP(W/H), [top,bot]=bgColors();
    const T=hex2rgb(top), B=hex2rgb(bot); scene.fg = luma([(T[0]+B[0])/2,(T[1]+B[1])/2,(T[2]+B[2])/2])>.5 ? [.1,.1,.14] : [.9,.9,.95];
    $("hud").style.color = luma(T)>.5 ? "#555" : "#8d8da0";
    const on=[S.ch0On?1:0, S.ch1On?1:0, S.ch2On?1:0].map((v,i)=>i<META.channels?v:0);
    const c0=hex2rgb(S.ch0Col), c1=hex2rgb(S.ch1Col), c2=hex2rgb(S.ch2Col);
    gl.uniformMatrix4fv(U("invVP"),false,inv(vp)); gl.uniform3fv(U("ext"),e);
    gl.uniform1f(U("steps"), fast&&S.fastMove ? Math.max(48,Math.round(S.steps*.4)) : S.steps);
    gl.uniform1f(U("eps"), 1.5/Math.max(META.w,META.h)*(e[0]*2));
    gl.uniform1i(U("fast"), fast&&S.fastMove?1:0); gl.uniform1i(U("mode"), +S.mode); gl.uniform1i(U("cmap"), +S.cmap); gl.uniform1i(U("nch"), META.channels);
    gl.uniform3fv(U("chColor0"),c0); gl.uniform3fv(U("chColor1"),c1); gl.uniform3fv(U("chColor2"),c2);
    gl.uniform3fv(U("chOn"),on); gl.uniform3fv(U("chGain"),[S.gain*S.ch0Gain,S.gain*S.ch1Gain,S.gain*S.ch2Gain]);
    gl.uniform1f(U("low"),Math.min(S.black,S.white-0.02)); gl.uniform1f(U("high"),S.white); gl.uniform1f(U("gamma"),S.gamma);
    gl.uniform1f(U("thr"),S.thr); gl.uniform1f(U("density"),S.density); gl.uniform1f(U("iso"),S.iso);
    gl.uniform1i(U("shade"),S.shade?1:0); gl.uniform3fv(U("lightDir"),lightVector());
    gl.uniform1f(U("amb"),S.amb); gl.uniform1f(U("dif"),S.dif); gl.uniform1f(U("spc"),S.spc); gl.uniform1f(U("shin"),S.shin);
    gl.uniform1f(U("shadowK"),S.shadow); gl.uniform1f(U("aoK"),S.ao);
    gl.uniform1i(U("hl"),(S.hl&&META.hasMask)?1:0); gl.uniform3fv(U("hlColor"),hex2rgb(S.hlCol));
    gl.uniform3fv(U("cropLo"),[S.x0,S.y0,S.z0]); gl.uniform3fv(U("cropHi"),[S.x1,S.y1,S.z1]);
    gl.uniform4fv(U("clipA"),planeFor("ca")); gl.uniform4fv(U("clipB"),planeFor("cb"));
    gl.uniform1i(U("clipOnA"),S.caOn?1:0); gl.uniform1i(U("clipOnB"),S.cbOn?1:0);
    gl.uniform3fv(U("bgTop"),T); gl.uniform3fv(U("bgBot"),B);
    gl.uniform1i(U("tex"),0); gl.drawArrays(gl.TRIANGLES,0,3);
    ovc.width=Math.round(cw*dpr); ovc.height=Math.round(chh*dpr);
    drawOverlay(ovc.getContext("2d"), ovc.width, ovc.height, scene);
    $("hud").textContent = META.source[2]+" x "+META.source[1]+" x "+META.source[0]+" voxels"+(META.reduced_by>1?", shown averaged "+META.reduced_by+"x"+META.reduced_by+" in x,y":"")+
      " | all "+META.d+" planes | depth x"+S.stretch;
  }
  Q.draw = draw;

  // ---- controls ---------------------------------------------------------------------------
  const ui = makeUI($("panel"));
  const heading=document.createElement("h1"); heading.textContent=META.title;
  const hint=document.createElement("div"); hint.id="hint"; hint.textContent="Drag: rotate \u00b7 Scroll: zoom \u00b7 Right-drag or Shift+drag: pan \u00b7 Double-click: reset";
  $("panel").prepend(heading, hint);
  const pct = v => Math.round(v*100)+"%", x2 = v => v.toFixed(2);

  ui.section("View", true);
  ui.buttons([["Front",()=>setView(0,0)],["Side",()=>setView(Math.PI/2,0)],["Top",()=>setView(0,Math.PI/2-0.001)],["Back",()=>setView(Math.PI,0)]]);
  ui.select("proj","Projection",[["p","Perspective"],["o","Orthographic (true scale bar)"]],"p",v=>{ cam.ortho = v==="o"; });
  ui.range("stretch","Depth exaggeration",1,20,0.5,META.stretch,v=>"×"+v.toFixed(1));
  ui.range("rotspeed","Rotation speed",0.2,4,0.1,1,x2);
  ui.check("hqRotate","Full quality while rotating",false);

  ui.section("Appearance", true);
  ui.select("mode","Rendering",[["0","Volume (see-through, depth)"],["2","Surface (first structure hit)"],["1","Brightest voxel (like a projection)"]],"0");
  const cmapBox = ui.select("cmap","Colours",[["0","Magenta"],["1","Fire"],["2","Grey"],["3","Cyan - yellow"],["4","Viridis"],["5","Ice"],["9","One colour (pick below)"]],"0");
  if(META.channels>1){
    cmapBox.disabled = true; ui.note("Each channel uses its own colour, set below.");
    META.names.forEach((nm,i)=>{ ui.check("ch"+i+"On",nm,true); ui.color("ch"+i+"Col","  colour",["#ff4dff","#3cff7a","#4da6ff"][i]); ui.range("ch"+i+"Gain","  brightness",0.2,4,0.05,1,x2); });
  } else { ui.color("ch0Col","Colour (for \"One colour\")","#ff4dff"); S.ch0On=true; S.ch0Gain=1; }
  for(let i=META.channels;i<3;i++){ S["ch"+i+"On"]=false; S["ch"+i+"Col"]="#ffffff"; S["ch"+i+"Gain"]=1; }
  if(META.channels>1){ S.cmap="9"; }
  ui.range("gain","Brightness",0.3,4,0.05,1.6,x2);
  ui.range("black","Black point",0,0.9,0.01,0,x2);
  ui.range("white","White point",0.1,1,0.01,1,x2);
  ui.range("gamma","Gamma (lower = brighter midtones)",0.3,2.5,0.05,1,x2);
  ui.range("density","Opacity",0.5,60,0.5,14,v=>v.toFixed(0));
  ui.range("thr","Hide faint signal",0,0.9,0.01,0.12,x2);
  ui.range("iso","Surface level (Surface mode)",0.05,0.95,0.01,0.3,x2);
  if(META.hasMask){ ui.check("hl","Highlight segmented objects",true); ui.color("hlCol","  highlight colour","#26ff59"); } else { S.hl=false; S.hlCol="#26ff59"; }

  ui.section("Lighting", false);
  ui.check("shade","Shading",true);
  ui.check("headlight","Light follows the camera",true);
  ui.range("lightAz","Light direction (left-right)",-180,180,1,40,v=>v.toFixed(0)+"°");
  ui.range("lightEl","Light direction (up-down)",-90,90,1,45,v=>v.toFixed(0)+"°");
  ui.range("amb","Ambient light",0,1,0.01,0.35,x2);
  ui.range("dif","Diffuse (surface shading)",0,1,0.01,0.85,x2);
  ui.range("spc","Specular (shine)",0,1,0.01,0.25,x2);
  ui.range("shin","Shininess",4,128,1,32,v=>v.toFixed(0));
  ui.range("shadow","Shadows (full quality only)",0,4,0.1,0,x2);
  ui.range("ao","Ambient occlusion (full quality only)",0,1,0.01,0,x2);

  ui.section("Cut", false);
  ui.note("Crop box (fractions of the sample)");
  [["x","Left - right"],["y","Top - bottom"],["z","Front - back (planes)"]].forEach(([a,lab])=>{
    ui.range(a+"0",lab+": from",0,1,0.01,0,pct); ui.range(a+"1",lab+": to",0,1,0.01,1,pct); });
  [["ca","Cutting plane 1"],["cb","Cutting plane 2"]].forEach(([p,lab],idx)=>{
    ui.note(lab);
    ui.check(p+"On","Use "+lab.toLowerCase(),false);
    ui.select(p+"Axis","  direction",[["x","Across x (left - right)"],["y","Across y (top - bottom)"],["z","Across z (front - back)"],["free","Any direction"]],idx?"y":"x");
    ui.range(p+"Az","  turn (any direction)",-180,180,1,idx?0:35,v=>v.toFixed(0)+"°");
    ui.range(p+"El","  tilt (any direction)",-90,90,1,idx?60:20,v=>v.toFixed(0)+"°");
    ui.range(p+"Off","  position",-1,1,0.01,0,x2);
    ui.check(p+"Flip","  keep the other side",false);
  });

  ui.section("Scene", false);
  ui.select("bg","Background",[["dark","Dark"],["black","Black"],["grey","Grey"],["white","White"],["sky","Light"],["custom","Custom colours"]],"dark");
  ui.color("bgTop","  custom: top","#202030"); ui.color("bgBot","  custom: bottom","#05050a");
  ui.check("axes","Orientation axes",true);
  ui.check("scale","Scale bar",true);
  ui.check("box","Bounding box",false);
  ui.note("The scale bar is exact for the centre of the view (and for the whole view in orthographic mode); it measures x and y, not the stretched depth.");

  ui.section("Quality", false);
  ui.range("steps","Quality (samples per ray)",64,768,32,256,v=>v.toFixed(0));
  ui.range("renderScale","Resolution when still",0.5,2,0.25,1,v=>"×"+v);
  ui.check("fastMove","Faster render while moving",true);
  ui.range("moveScale","Resolution while moving",0.25,1,0.05,0.5,v=>"×"+v.toFixed(2));

  ui.section("Export", false);
  ui.select("shotScale","Screenshot size",[["1","Window size"],["2","2×"],["4","4× (large)"]],"2");
  ui.buttons([["Save screenshot (PNG)", () => snapshot()]]);

  function snapshot(){ draw(false, +S.shotScale / Math.min(window.devicePixelRatio||1,2));      // 1x, 2x or 4x the window
    savePng(glc, (g,w,h)=>drawOverlay(g,w,h,scene), "HIVE_3D_view.png"); setTimeout(poke, 300); }
  $("btnShot").onclick = snapshot;

  const layout = () => { $("stage").style.left = $("panel").classList.contains("hidden") ? "0" : "272px"; poke(); };
  wireChrome(layout); attachPointer(glc);
  if(window.innerWidth<720){ $("panel").classList.add("hidden"); $("toggle").textContent="\u2630 Controls"; }   // narrow window: picture first
  layout();
  window.hiveViewer = { S, cam, render: () => draw(false), snapshotCanvas: () => glc.toDataURL("image/png") };
  draw(false);
}
</script></body></html>
"""

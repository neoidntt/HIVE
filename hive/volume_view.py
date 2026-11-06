"""An interactive 3D view of a whole z-stack that runs in the browser.

:func:`volume_viewer_html` turns a volume ``(Z, Y, X)`` (every z-plane) into one self-contained HTML
page with a WebGL2 ray-caster: drag to rotate, scroll to zoom, right-drag (or Shift-drag) to pan. The
brightness along each line of sight is accumulated with occlusion and optional surface shading, so
nearer structures hide farther ones and depth can be seen; the rotating GIF this replaces uses a
maximum-intensity projection, which keeps only the brightest voxel per line of sight and has no depth
cue (a single frame is the same as a flat picture).

Nothing is sent to any server, and the page needs no internet connection. It needs a browser with
WebGL2 turned on (current Firefox, Chrome, Edge and Safari have it); otherwise the page says so.

For looking only. Measurements come from the analysis, never from this view.
"""

from __future__ import annotations

import base64
import json
import math
import zlib

import numpy as np

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
) -> str:
    """Build the interactive page for a volume.

    Args:
        intensity: ``(Z, Y, X)`` array in any scale (brightness is rescaled for display).
        mask: Optional boolean ``(Z, Y, X)`` mask of segmented objects, shown as a highlight that
            can be switched on and off in the page.
        voxel_size: Physical size of a voxel as ``(z, y, x)``, so the shape is true to scale.
        title: Heading inside the page.
        max_voxels: Largest number of voxels sent to the browser; the volume is averaged in y and x
            to fit (all z-planes are always kept).
        depth_exaggeration: Stretch of the z axis. ``None`` chooses one so a thin sample is visibly
            deep (1 = true to scale); the page has a slider to change it.

    Returns:
        The HTML text.

    Raises:
        ValueError: If ``intensity`` is not 3D with at least 2 planes, or the mask's shape differs.
    """
    volume = np.asarray(intensity)
    if volume.ndim != 3 or volume.shape[0] < 2:
        raise ValueError(f"A 3D view needs a volume with at least 2 z-planes, got shape {volume.shape}")
    if mask is not None and np.shape(mask) != volume.shape:
        raise ValueError(f"mask shape {np.shape(mask)} differs from the volume {volume.shape}")
    reduced, k = reduce_volume(volume, max_voxels)
    data = _to_uint8(np.asarray(reduced, dtype=np.float32))
    channels = 1
    if mask is not None:
        small, _ = reduce_volume(np.asarray(mask, dtype=np.uint8), max_voxels, reduce="max")
        data = np.stack([data, (small[:data.shape[0], :data.shape[1], :data.shape[2]] > 0).astype(np.uint8) * 255],
                        axis=-1)
        channels = 2
    z, h, w = data.shape[:3]
    scale = (float(voxel_size[0]), float(voxel_size[1]) * k, float(voxel_size[2]) * k)
    stretch = auto_depth_exaggeration((z, h, w), scale) if depth_exaggeration is None else float(depth_exaggeration)
    meta = {"w": w, "h": h, "d": z, "channels": channels, "scale": [scale[2], scale[1], scale[0]],
            "stretch": round(stretch, 2), "reduced_by": k, "source": list(volume.shape),
            "title": title}
    payload = base64.b64encode(zlib.compress(np.ascontiguousarray(data).tobytes(), 6)).decode("ascii")
    safe_meta = json.dumps(meta).replace("</", "<\\/")          # a name must not close the <script>
    return (_PAGE.replace("__META__", safe_meta).replace("__DATA__", payload)
            .replace("__TITLE__", title.replace("<", "&lt;")))


_PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
 html,body{margin:0;height:100%;background:#0b0b10;color:#e8e8ee;font:13px system-ui,sans-serif;overflow:hidden}
 #wrap{position:relative;width:100%;height:100%}
 canvas{position:absolute;left:230px;right:0;top:0;bottom:0;width:calc(100% - 230px);height:100%;display:block;touch-action:none;cursor:grab}
 #bar{position:absolute;left:0;top:0;bottom:0;width:230px;padding:10px;box-sizing:border-box;overflow:auto;
      background:rgba(16,16,24,.88);border-right:1px solid #2a2a38}
 #bar h1{font-size:14px;margin:0 0 6px}
 label{display:block;margin:7px 0 1px;color:#b9b9c8}
 input[type=range]{width:100%}
 select,button{width:100%;margin:3px 0;background:#23232f;color:#e8e8ee;border:1px solid #3a3a4a;border-radius:4px;padding:4px}
 button:hover{background:#2e2e3d}
 .row{display:flex;gap:4px}.row button{flex:1}
 #msg{position:absolute;inset:0;display:none;align-items:center;justify-content:center;text-align:center;padding:30px;
      background:#0b0b10;font-size:15px}
 #hud{position:absolute;right:10px;bottom:8px;color:#8d8da0;font-size:12px;text-align:right;pointer-events:none}
 small{color:#8d8da0}
</style></head><body><div id="wrap">
<canvas id="c"></canvas>
<div id="bar">
 <h1>__TITLE__</h1>
 <small>Drag: rotate &middot; Scroll: zoom &middot; Right-drag or Shift+drag: pan</small>
 <label>View</label>
 <div class="row"><button id="vf">Front</button><button id="vs">Side</button><button id="vt">Top</button></div>
 <button id="reset">Reset view</button>
 <label>Rendering</label>
 <select id="mode"><option value="0">Volume (depth, occlusion)</option><option value="1">Brightest voxel (like the GIF)</option></select>
 <label>Depth exaggeration &times;<span id="zv"></span> <small>(1 = true to scale)</small></label>
 <input id="stretch" type="range" min="1" max="20" step="0.5">
 <label>Brightness</label><input id="gain" type="range" min="0.3" max="4" step="0.05" value="1.6">
 <label>Hide faint signal</label><input id="thr" type="range" min="0" max="0.9" step="0.01" value="0.12">
 <label>Opacity</label><input id="opac" type="range" min="0.5" max="40" step="0.5" value="14">
 <label>Surface shading <input id="shade" type="checkbox" checked style="width:auto"></label>
 <label id="hlrow">Highlight segmented objects <input id="hl" type="checkbox" checked style="width:auto"></label>
 <label>Colours</label>
 <select id="cmap"><option value="0">Magenta</option><option value="1">Fire</option><option value="2">Grey</option><option value="3">Cyan-yellow</option></select>
 <label>Show planes <span id="zr"></span></label>
 <input id="zlo" type="range" min="0" max="1" step="0.01" value="0">
 <input id="zhi" type="range" min="0" max="1" step="0.01" value="1">
 <label>Quality</label><input id="steps" type="range" min="64" max="512" step="32" value="256">
</div>
<div id="hud"></div>
<div id="msg"></div>
</div>
<script>
const META = __META__;
const B64 = "__DATA__";
const $ = id => document.getElementById(id);
function fail(text){ const m=$("msg"); m.style.display="flex"; m.innerHTML=text; }
const canvas = $("c");
const gl = canvas.getContext("webgl2", {antialias:false, alpha:false, preserveDrawingBuffer:true});
if(!gl){ fail("This view needs <b>WebGL2</b>, which your browser has turned off or does not have.<br>Turn it on (in Firefox: <code>about:config</code> &rarr; <code>webgl.disabled</code> = false), or open this page in another browser.");
} else if(typeof DecompressionStream === "undefined"){ fail("Your browser is too old for this view (no DecompressionStream). Use a current Firefox, Chrome, Edge or Safari.");
} else { main().catch(e => fail("Could not build the 3D view: " + e)); }

async function main(){
  const maxTex = gl.getParameter(gl.MAX_3D_TEXTURE_SIZE);
  if(Math.max(META.w, META.h, META.d) > maxTex) throw new Error("volume larger than this graphics card allows (" + maxTex + ")");
  const raw = Uint8Array.from(atob(B64), c => c.charCodeAt(0));
  const buf = await new Response(new Blob([raw]).stream().pipeThrough(new DecompressionStream("deflate"))).arrayBuffer();
  const vox = new Uint8Array(buf);

  const vs = `#version 300 es
  in vec2 p; out vec2 v; void main(){ v=p; gl_Position=vec4(p,0.,1.); }`;
  const fs = `#version 300 es
  precision highp float; precision highp sampler3D;
  in vec2 v; out vec4 o;
  uniform sampler3D tex; uniform mat4 invVP; uniform vec3 eye, ext;
  uniform float gain, thr, opac, zlo, zhi, steps; uniform int mode, cmap; uniform bool shade, hl;
  vec3 pal(float s){
    if(cmap==0) return vec3(s, 0.12*s, s);
    if(cmap==1) return vec3(min(1.,2.*s), clamp(2.*s-.5,0.,1.)*.85, clamp(3.*s-2.,0.,1.));
    if(cmap==2) return vec3(s);
    return mix(vec3(.1,.5,1.), vec3(1.,.95,.2), s) * (.25+.75*s);
  }
  float hash(vec2 q){ return fract(sin(dot(q, vec2(12.9898,78.233))) * 43758.5453); }
  vec3 uvw(vec3 q){ return vec3(q.x/(2.*ext.x)+.5, .5-q.y/(2.*ext.y), q.z/(2.*ext.z)+.5); }
  vec2 sampleAt(vec3 q){ return texture(tex, uvw(q)).rg; }
  float level(float x){ return clamp((x*gain - thr)/(1.-thr), 0., 1.); }
  void main(){
    vec4 n4 = invVP*vec4(v,-1.,1.), f4 = invVP*vec4(v,1.,1.);
    vec3 a = n4.xyz/n4.w, b = f4.xyz/f4.w, rd = normalize(b-a), ro = a;
    vec3 t1 = (-ext-ro)/rd, t2 = (ext-ro)/rd;
    vec3 tn = min(t1,t2), tf = max(t1,t2);
    float t0 = max(max(tn.x,tn.y),max(tn.z,0.)), te = min(min(tf.x,tf.y),tf.z);
    vec3 bg = vec3(.043,.043,.063);
    if(te<=t0){ o=vec4(bg,1.); return; }
    float dt = (te-t0)/steps;
    float t = t0 + dt*hash(gl_FragCoord.xy);
    vec3 col = vec3(0.); float acc = 0.; float best = 0.; vec3 bestc = vec3(0.);
    float eps = 1.5/float(${META.w});
    for(int i=0;i<1024;i++){
      if(float(i)>=steps || t>te) break;
      vec3 q = ro + rd*t;
      float zz = q.z/(2.*ext.z)+.5;
      if(zz>=zlo && zz<=zhi){
        vec2 s2 = sampleAt(q);
        float s = level(s2.r);
        bool on = hl && s2.g>.5;
        if(mode==1){
          float m = on ? max(s,.55) : s;
          if(m>best){ best=m; bestc = on ? mix(pal(s), vec3(.15,1.,.35), .75) : pal(s); }
        } else if(s>.004 || on){
          float sv = on ? max(s,.45) : s;
          vec3 c = pal(sv);
          if(on) c = mix(c, vec3(.15,1.,.35), .7);
          if(shade){
            vec3 e = vec3(eps)*vec3(1.,1.,ext.z/ext.x*0.0+1.);
            vec3 g = vec3(sampleAt(q+vec3(e.x,0,0)).r - sampleAt(q-vec3(e.x,0,0)).r,
                          -(sampleAt(q+vec3(0,e.x,0)).r - sampleAt(q-vec3(0,e.x,0)).r),
                          sampleAt(q+vec3(0,0,e.x*ext.z/ext.x)).r - sampleAt(q-vec3(0,0,e.x*ext.z/ext.x)).r);
            float gl2 = length(g);
            if(gl2>1e-4){ vec3 nn = -g/gl2; float d = abs(dot(nn,-rd)); c *= mix(.55, 1.15, d); }
          }
          float al = clamp(pow(sv,1.4)*opac*dt*6., 0., 1.);
          col += (1.-acc)*al*c; acc += (1.-acc)*al;
          if(acc>.97) break;
        }
      }
      t += dt;
    }
    if(mode==1){ o = vec4(mix(bg, bestc, clamp(best*1.5,0.,1.)), 1.); return; }
    o = vec4(col + (1.-acc)*bg, 1.);
  }`;
  function sh(type, src){ const s=gl.createShader(type); gl.shaderSource(s,src); gl.compileShader(s);
    if(!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; }
  const prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, vs));
  gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, fs.replace("${META.w}", String(META.w).includes(".")?META.w:META.w+".0")));
  gl.linkProgram(prog);
  if(!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  gl.useProgram(prog);
  const vb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, vb);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 3,-1, -1,3]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, "p"); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc,2,gl.FLOAT,false,0,0);

  const tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_3D, tex);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  const two = META.channels === 2;
  gl.texImage3D(gl.TEXTURE_3D, 0, two ? gl.RG8 : gl.R8, META.w, META.h, META.d, 0, two ? gl.RG : gl.RED, gl.UNSIGNED_BYTE, vox);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  for(const w of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) gl.texParameteri(gl.TEXTURE_3D, w, gl.CLAMP_TO_EDGE);
  const U = n => gl.getUniformLocation(prog, n);
  if(!two) $("hlrow").style.display = "none";

  // ---- camera ------------------------------------------------------------
  const cam = {yaw:0.55, pitch:0.35, dist:1.75, tx:0, ty:0, tz:0};
  const home = JSON.stringify(cam);
  function mul(a,b){ const r=new Float32Array(16); for(let i=0;i<4;i++)for(let j=0;j<4;j++){let s=0;for(let k=0;k<4;k++)s+=a[k*4+j]*b[i*4+k];r[i*4+j]=s;} return r; }
  function persp(fov,asp,n,f){ const t=1/Math.tan(fov/2); return new Float32Array([t/asp,0,0,0, 0,t,0,0, 0,0,(f+n)/(n-f),-1, 0,0,2*f*n/(n-f),0]); }
  function look(e,c,u){ const z=norm(sub(e,c)), x=norm(cross(u,z)), y=cross(z,x);
    return new Float32Array([x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0, -dot(x,e),-dot(y,e),-dot(z,e),1]); }
  const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]], dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
  const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  const norm=a=>{const l=Math.hypot(a[0],a[1],a[2])||1;return [a[0]/l,a[1]/l,a[2]/l];};
  function inv(m){ const i=new Float32Array(16), a=m;
    const b00=a[0]*a[5]-a[1]*a[4],b01=a[0]*a[6]-a[2]*a[4],b02=a[0]*a[7]-a[3]*a[4],b03=a[1]*a[6]-a[2]*a[5],b04=a[1]*a[7]-a[3]*a[5],b05=a[2]*a[7]-a[3]*a[6],
          b06=a[8]*a[13]-a[9]*a[12],b07=a[8]*a[14]-a[10]*a[12],b08=a[8]*a[15]-a[11]*a[12],b09=a[9]*a[14]-a[10]*a[13],b10=a[9]*a[15]-a[11]*a[13],b11=a[10]*a[15]-a[11]*a[14];
    let d=b00*b11-b01*b10+b02*b09+b03*b08-b04*b07+b05*b06; d=1/d;
    i[0]=(a[5]*b11-a[6]*b10+a[7]*b09)*d; i[1]=(a[2]*b10-a[1]*b11-a[3]*b09)*d; i[2]=(a[13]*b05-a[14]*b04+a[15]*b03)*d; i[3]=(a[10]*b04-a[9]*b05-a[11]*b03)*d;
    i[4]=(a[6]*b08-a[4]*b11-a[7]*b07)*d; i[5]=(a[0]*b11-a[2]*b08+a[3]*b07)*d; i[6]=(a[14]*b02-a[12]*b05-a[15]*b01)*d; i[7]=(a[8]*b05-a[10]*b02+a[11]*b01)*d;
    i[8]=(a[4]*b10-a[5]*b08+a[7]*b06)*d; i[9]=(a[1]*b08-a[0]*b10-a[3]*b06)*d; i[10]=(a[12]*b04-a[13]*b02+a[15]*b00)*d; i[11]=(a[9]*b02-a[8]*b04-a[11]*b00)*d;
    i[12]=(a[5]*b07-a[4]*b09-a[6]*b06)*d; i[13]=(a[0]*b09-a[1]*b07+a[2]*b06)*d; i[14]=(a[13]*b01-a[12]*b03-a[14]*b00)*d; i[15]=(a[8]*b03-a[9]*b01+a[10]*b00)*d;
    return i; }

  function extents(){ // half-extents in world units, longest side = 1
    const s = +$("stretch").value;
    const ex = META.w*META.scale[0], ey = META.h*META.scale[1], ez = META.d*META.scale[2]*s, m = Math.max(ex,ey,ez);
    return [ex/m, ey/m, ez/m];
  }
  let dirty = true;
  function draw(){
    dirty = false;
    const r = canvas.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio||1, 2);
    const W = Math.max(2, Math.round((r.width) * dpr)), H = Math.max(2, Math.round(r.height * dpr));
    if(canvas.width!==W || canvas.height!==H){ canvas.width=W; canvas.height=H; }
    gl.viewport(0,0,W,H);
    const ext = extents();
    const cy=Math.cos(cam.pitch), eye=[cam.tx+cam.dist*cy*Math.sin(cam.yaw), cam.ty+cam.dist*Math.sin(cam.pitch), cam.tz+cam.dist*cy*Math.cos(cam.yaw)];
    const view = look(eye,[cam.tx,cam.ty,cam.tz],[0,1,0]);
    const vp = mul(persp(0.8, W/H, 0.05, 20), view);
    gl.uniformMatrix4fv(U("invVP"), false, inv(vp));
    gl.uniform3fv(U("eye"), eye); gl.uniform3fv(U("ext"), ext.map(x=>x/2));
    gl.uniform1f(U("gain"), +$("gain").value); gl.uniform1f(U("thr"), +$("thr").value);
    gl.uniform1f(U("opac"), +$("opac").value); gl.uniform1f(U("steps"), +$("steps").value);
    gl.uniform1f(U("zlo"), Math.min(+$("zlo").value,+$("zhi").value)); gl.uniform1f(U("zhi"), Math.max(+$("zlo").value,+$("zhi").value));
    gl.uniform1i(U("mode"), +$("mode").value); gl.uniform1i(U("cmap"), +$("cmap").value);
    gl.uniform1i(U("shade"), $("shade").checked?1:0); gl.uniform1i(U("hl"), $("hl").checked?1:0);
    gl.uniform1i(U("tex"), 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    $("zv").textContent = $("stretch").value;
    const zl=Math.round(Math.min(+$("zlo").value,+$("zhi").value)*META.d), zh=Math.round(Math.max(+$("zlo").value,+$("zhi").value)*META.d);
    $("zr").textContent = "(" + zl + "-" + zh + " of " + META.d + ")";
    $("hud").textContent = META.source[2]+" x "+META.source[1]+" x "+META.source[0]+" voxels" + (META.reduced_by>1 ? ", shown averaged "+META.reduced_by+"x"+META.reduced_by+" in x,y" : "") +
      " | all "+META.d+" planes | depth x"+$("stretch").value;
  }
  const poke = () => { if(!dirty){ dirty = true; requestAnimationFrame(draw); } };
  $("stretch").value = META.stretch;
  for(const id of ["stretch","gain","thr","opac","steps","zlo","zhi","mode","cmap","shade","hl"]) $(id).addEventListener("input", poke);
  const setView = (yaw,pitch) => { cam.yaw=yaw; cam.pitch=pitch; poke(); };
  $("vf").onclick = () => setView(0,0); $("vs").onclick = () => setView(Math.PI/2,0); $("vt").onclick = () => setView(0,Math.PI/2-0.001);
  $("reset").onclick = () => { Object.assign(cam, JSON.parse(home)); $("stretch").value = META.stretch; poke(); };
  let drag = null;
  canvas.addEventListener("pointerdown", e => { canvas.setPointerCapture(e.pointerId); drag = {x:e.clientX, y:e.clientY, pan: e.shiftKey || e.button===2 || e.button===1}; canvas.style.cursor="grabbing"; });
  canvas.addEventListener("pointerup", () => { drag = null; canvas.style.cursor="grab"; });
  canvas.addEventListener("contextmenu", e => e.preventDefault());
  canvas.addEventListener("pointermove", e => {
    if(!drag) return;
    const dx = e.clientX-drag.x, dy = e.clientY-drag.y; drag.x=e.clientX; drag.y=e.clientY;
    if(drag.pan){
      const k = cam.dist*0.0012, sy=Math.sin(cam.yaw), cyw=Math.cos(cam.yaw);
      cam.tx -= (cyw*dx)*k; cam.tz -= (-sy*dx)*k; cam.ty += dy*k;
    } else { cam.yaw -= dx*0.008; cam.pitch = Math.max(-1.55, Math.min(1.55, cam.pitch + dy*0.008)); }
    poke();
  });
  canvas.addEventListener("wheel", e => { e.preventDefault(); cam.dist = Math.max(0.4, Math.min(8, cam.dist*Math.exp(e.deltaY*0.0012))); poke(); }, {passive:false});
  window.addEventListener("resize", poke);
  draw();
}
</script></body></html>
"""

"""Pieces shared by HIVE's two browser 3D viewers (the volume view and the region-surface view).

Both pages are single self-contained HTML files with a WebGL2 canvas, a collapsible control panel, an
orientation-axes gizmo, a scale bar, auto-rotate and PNG screenshots. This module holds the parts that
are the same: the style sheet and the JavaScript for the camera, the pointer controls, the control
panel builder, the overlay drawing and the screenshot. The pages add their own shaders and controls.

Nothing is loaded from the internet.
"""

from __future__ import annotations

CSS = r"""
 :root{--panel:#12121a;--line:#2a2a38;--text:#e6e6ee;--dim:#8d8da0;--accent:#8b7bff}
 *{box-sizing:border-box}
 html,body{margin:0;height:100%;background:#0b0b10;color:var(--text);font:12.5px/1.35 system-ui,sans-serif;overflow:hidden}
 #wrap{position:relative;width:100%;height:100%}
 #stage{position:absolute;left:0;top:0;right:0;bottom:0}
 canvas{position:absolute;left:0;top:0;width:100%;height:100%;display:block;touch-action:none}
 #gl{cursor:grab}#ov{pointer-events:none}
 #panel{position:absolute;left:0;top:0;bottom:0;width:272px;padding:8px 10px 14px;overflow:auto;z-index:3;
        background:rgba(18,18,26,.94);border-right:1px solid var(--line)}
 #panel.hidden{transform:translateX(-100%)}
 #panel h1{font-size:14px;margin:2px 0 2px;letter-spacing:.2px}
 #hint{color:var(--dim);font-size:11.5px;margin-bottom:6px}
 details{border-top:1px solid var(--line);padding:2px 0}
 summary{cursor:pointer;padding:6px 0;font-weight:600;color:#cfcfe0;list-style:none}
 summary::before{content:"\25B8";display:inline-block;width:14px;color:var(--dim)}
 details[open]>summary::before{content:"\25BE"}
 .sec{padding:0 0 8px}
 label{display:flex;justify-content:space-between;align-items:center;gap:8px;margin:6px 0 1px;color:#b9b9c8}
 label span.v{color:var(--dim);font-variant-numeric:tabular-nums}
 input[type=range]{width:100%;margin:0 0 2px;accent-color:var(--accent)}
 input[type=checkbox]{accent-color:var(--accent)}
 input[type=color]{width:34px;height:22px;padding:0;border:1px solid #3a3a4a;background:none;border-radius:4px}
 select,button{width:100%;margin:3px 0;background:#22222e;color:var(--text);border:1px solid #3a3a4a;border-radius:5px;padding:5px 6px;font:inherit}
 button:hover{background:#2d2d3c}
 .row{display:flex;gap:4px}.row>*{flex:1}
 .chk{display:flex;align-items:center;gap:7px;margin:5px 0;color:#b9b9c8;cursor:pointer}
 .chk input{margin:0}
 .note{color:var(--dim);font-size:11.5px;margin:4px 0}
 #bar{position:absolute;right:10px;top:10px;z-index:4;display:flex;gap:6px}
 #bar button{width:auto;margin:0;padding:6px 10px;background:rgba(18,18,26,.82)}
 #bar button.on{background:#3b3470;border-color:var(--accent)}
 #toggle{position:absolute;left:10px;top:10px;z-index:4;width:auto;margin:0;padding:6px 10px;background:rgba(18,18,26,.82)}
 #panel:not(.hidden)~#toggle{left:282px}
 #hud{position:absolute;right:10px;bottom:8px;color:var(--dim);font-size:11.5px;text-align:right;pointer-events:none;z-index:2}
 #msg{position:absolute;inset:0;display:none;align-items:center;justify-content:center;text-align:center;padding:30px;
      background:#0b0b10;font-size:15px;z-index:9}
 .swatch{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
"""

JS = r"""
const $ = id => document.getElementById(id);
const S = {};                                     // every control's current value
function fail(text){ const m=$("msg"); m.style.display="flex"; m.innerHTML=text; }

// ---- vectors and 4x4 matrices (column-major, as WebGL wants them) ---------------------------
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]], dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const norm=a=>{const l=Math.hypot(a[0],a[1],a[2])||1;return [a[0]/l,a[1]/l,a[2]/l];};
function mul(a,b){ const r=new Float32Array(16); for(let i=0;i<4;i++)for(let j=0;j<4;j++){let s=0;for(let k=0;k<4;k++)s+=a[k*4+j]*b[i*4+k];r[i*4+j]=s;} return r; }
function persp(fov,asp,n,f){ const t=1/Math.tan(fov/2); return new Float32Array([t/asp,0,0,0, 0,t,0,0, 0,0,(f+n)/(n-f),-1, 0,0,2*f*n/(n-f),0]); }
function ortho(l,r,b,t,n,f){ return new Float32Array([2/(r-l),0,0,0, 0,2/(t-b),0,0, 0,0,-2/(f-n),0, -(r+l)/(r-l),-(t+b)/(t-b),-(f+n)/(f-n),1]); }
function look(e,c,u){ const z=norm(sub(e,c)), x=norm(cross(u,z)), y=cross(z,x);
  return new Float32Array([x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0, -dot(x,e),-dot(y,e),-dot(z,e),1]); }
function inv(a){ const i=new Float32Array(16);
  const b00=a[0]*a[5]-a[1]*a[4],b01=a[0]*a[6]-a[2]*a[4],b02=a[0]*a[7]-a[3]*a[4],b03=a[1]*a[6]-a[2]*a[5],b04=a[1]*a[7]-a[3]*a[5],b05=a[2]*a[7]-a[3]*a[6],
        b06=a[8]*a[13]-a[9]*a[12],b07=a[8]*a[14]-a[10]*a[12],b08=a[8]*a[15]-a[11]*a[12],b09=a[9]*a[14]-a[10]*a[13],b10=a[9]*a[15]-a[11]*a[13],b11=a[10]*a[15]-a[11]*a[14];
  let d=b00*b11-b01*b10+b02*b09+b03*b08-b04*b07+b05*b06; d=1/d;
  i[0]=(a[5]*b11-a[6]*b10+a[7]*b09)*d; i[1]=(a[2]*b10-a[1]*b11-a[3]*b09)*d; i[2]=(a[13]*b05-a[14]*b04+a[15]*b03)*d; i[3]=(a[10]*b04-a[9]*b05-a[11]*b03)*d;
  i[4]=(a[6]*b08-a[4]*b11-a[7]*b07)*d; i[5]=(a[0]*b11-a[2]*b08+a[3]*b07)*d; i[6]=(a[14]*b02-a[12]*b05-a[15]*b01)*d; i[7]=(a[8]*b05-a[10]*b02+a[11]*b01)*d;
  i[8]=(a[4]*b10-a[5]*b08+a[7]*b06)*d; i[9]=(a[1]*b08-a[0]*b10-a[3]*b06)*d; i[10]=(a[12]*b04-a[13]*b02+a[15]*b00)*d; i[11]=(a[9]*b02-a[8]*b04-a[11]*b00)*d;
  i[12]=(a[5]*b07-a[4]*b09-a[6]*b06)*d; i[13]=(a[0]*b09-a[1]*b07+a[2]*b06)*d; i[14]=(a[13]*b01-a[12]*b03-a[14]*b00)*d; i[15]=(a[8]*b03-a[9]*b01+a[10]*b00)*d;
  return i; }
function hex2rgb(h){ const n=parseInt(h.slice(1),16); return [(n>>16&255)/255,(n>>8&255)/255,(n&255)/255]; }
function luma(c){ return 0.2126*c[0]+0.7152*c[1]+0.0722*c[2]; }

// ---- orbit camera: left-drag rotate, wheel zoom, right-drag / Shift-drag pan, pinch zoom -----
const cam = { yaw:0.55, pitch:0.35, dist:1.75, tx:0, ty:0, tz:0, ortho:false, fov:0.8 };
const camHome = Object.assign({}, cam);
function camEye(){ const cp=Math.cos(cam.pitch); return [cam.tx+cam.dist*cp*Math.sin(cam.yaw), cam.ty+cam.dist*Math.sin(cam.pitch), cam.tz+cam.dist*cp*Math.cos(cam.yaw)]; }
function camView(){ return look(camEye(),[cam.tx,cam.ty,cam.tz],[0,1,0]); }
function camProj(aspect){ const n=0.05,f=20;
  if(cam.ortho){ const h=cam.dist*Math.tan(cam.fov/2); return ortho(-h*aspect,h*aspect,-h,h,n,f); }
  return persp(cam.fov,aspect,n,f); }
function camVP(aspect){ return mul(camProj(aspect), camView()); }
function pxPerUnit(heightPx){ return heightPx/(2*cam.dist*Math.tan(cam.fov/2)); }   // at the centre plane
function setView(yaw,pitch){ cam.yaw=yaw; cam.pitch=pitch; poke(); }
function resetCamera(){ Object.assign(cam, camHome); poke(); }

// ---- quality scheduling: cheap while moving, full quality when still ------------------------
const Q = { interacting:false, rotating:false, timer:null, dirty:false, draw:null };
function poke(){ if(!Q.dirty && Q.draw){ Q.dirty=true; requestAnimationFrame(()=>{ Q.dirty=false; Q.draw(Q.interacting||Q.rotating); }); } }
function interact(){ Q.interacting=true; clearTimeout(Q.timer); Q.timer=setTimeout(()=>{ Q.interacting=false; poke(); }, 260); poke(); }
function attachPointer(canvas){
  const pts = new Map(); let last = 0;
  canvas.addEventListener("contextmenu", e => e.preventDefault());
  canvas.addEventListener("pointerdown", e => { canvas.setPointerCapture(e.pointerId);
    pts.set(e.pointerId,{x:e.clientX,y:e.clientY,pan:e.shiftKey||e.button===2||e.button===1}); canvas.style.cursor="grabbing"; });
  const up = e => { pts.delete(e.pointerId); if(!pts.size) canvas.style.cursor="grab"; };
  canvas.addEventListener("pointerup", up); canvas.addEventListener("pointercancel", up);
  canvas.addEventListener("pointermove", e => {
    const p = pts.get(e.pointerId); if(!p) return;
    const dx=e.clientX-p.x, dy=e.clientY-p.y; p.x=e.clientX; p.y=e.clientY;
    if(pts.size===2){                                   // two fingers: pinch to zoom
      const [a,b]=[...pts.values()]; const d=Math.hypot(a.x-b.x,a.y-b.y);
      if(last) cam.dist=Math.max(0.3,Math.min(10,cam.dist*last/d)); last=d; interact(); return; }
    last=0;
    if(p.pan){
      const r=canvas.getBoundingClientRect(), k=2*cam.dist*Math.tan(cam.fov/2)/r.height;
      const e0=camEye(), f=norm(sub([cam.tx,cam.ty,cam.tz],e0)), right=norm(cross(f,[0,1,0])), up=cross(right,f);
      cam.tx+=(-dx*right[0]+dy*up[0])*k; cam.ty+=(-dx*right[1]+dy*up[1])*k; cam.tz+=(-dx*right[2]+dy*up[2])*k;
    } else { cam.yaw-=dx*0.008; cam.pitch=Math.max(-1.55,Math.min(1.55,cam.pitch+dy*0.008)); }
    interact();
  });
  canvas.addEventListener("wheel", e => { e.preventDefault(); cam.dist=Math.max(0.3,Math.min(10,cam.dist*Math.exp(e.deltaY*0.0012))); interact(); }, {passive:false});
  canvas.addEventListener("dblclick", () => resetCamera());
}

// ---- auto-rotate ----------------------------------------------------------------------------
let rotLast = 0;
function rotateTick(t){
  if(!S.autorot){ Q.rotating=false; poke(); return; }
  Q.rotating = !S.hqRotate;
  const dt = rotLast ? Math.min(0.1,(t-rotLast)/1000) : 0; rotLast = t;
  cam.yaw += S.rotspeed*dt*0.6; poke(); requestAnimationFrame(rotateTick);
}
function setAutoRotate(on){ S.autorot=on; $("btnRot").classList.toggle("on",on); rotLast=0; if(on) requestAnimationFrame(rotateTick); }

// ---- control panel builder ------------------------------------------------------------------
function makeUI(root){
  let box = root;
  const el=(tag,cls,html)=>{ const e=document.createElement(tag); if(cls) e.className=cls; if(html!=null) e.innerHTML=html; return e; };
  const ui = {
    section(title, open){ const d=el("details"); d.open=!!open; d.appendChild(el("summary",null,title)); box=el("div","sec"); d.appendChild(box); root.appendChild(d); },
    range(key,label,min,max,step,val,fmt,onchange){
      const l=el("label",null,`<span>${label}</span><span class="v"></span>`), v=l.querySelector(".v");
      const r=el("input"); r.type="range"; r.min=min; r.max=max; r.step=step; r.value=val; S[key]=+val;
      const show=()=>{ v.textContent = fmt ? fmt(+r.value) : (+r.value).toFixed(step<1?2:0); };
      r.addEventListener("input",()=>{ S[key]=+r.value; show(); if(onchange) onchange(S[key]); poke(); });
      show(); box.appendChild(l); box.appendChild(r); return r; },
    check(key,label,val,onchange){
      const l=el("label","chk"); const c=el("input"); c.type="checkbox"; c.checked=!!val; S[key]=!!val;
      c.addEventListener("input",()=>{ S[key]=c.checked; if(onchange) onchange(c.checked); poke(); });
      l.appendChild(c); l.appendChild(document.createTextNode(label)); box.appendChild(l); return c; },
    select(key,label,options,val,onchange){
      box.appendChild(el("label",null,`<span>${label}</span>`)); const s=el("select");
      options.forEach(([v,t])=>{ const o=el("option",null,t); o.value=v; s.appendChild(o); }); s.value=val; S[key]=val;
      s.addEventListener("input",()=>{ S[key]=s.value; if(onchange) onchange(s.value); poke(); }); box.appendChild(s); return s; },
    color(key,label,val,onchange){
      const l=el("label",null,`<span>${label}</span>`); const c=el("input"); c.type="color"; c.value=val; S[key]=val;
      c.addEventListener("input",()=>{ S[key]=c.value; if(onchange) onchange(c.value); poke(); }); l.appendChild(c); box.appendChild(l); return c; },
    buttons(list){ const r=el("div","row"); list.forEach(([t,f])=>{ const b=el("button",null,t); b.addEventListener("click",f); r.appendChild(b); }); box.appendChild(r); },
    note(text){ box.appendChild(el("div","note",text)); },
    box(){ return box; }
  };
  return ui;
}

// ---- overlay: bounding box, orientation axes, scale bar --------------------------------------
function niceLength(x){ const p=Math.pow(10,Math.floor(Math.log10(x))); for(const m of [1,2,5,10]) if(m*p>=x*0.999) return m*p; return 10*p; }
function drawOverlay(ctx, W, H, scene){
  ctx.clearRect(0,0,W,H);
  const dpr = H/($("stage").clientHeight||H), fg = scene.fg, vp = camVP(W/H);
  ctx.lineWidth = Math.max(1,dpr); ctx.font = `${Math.round(12*dpr)}px system-ui,sans-serif`;
  const toScreen = p => { const x=vp[0]*p[0]+vp[4]*p[1]+vp[8]*p[2]+vp[12], y=vp[1]*p[0]+vp[5]*p[1]+vp[9]*p[2]+vp[13], w=vp[3]*p[0]+vp[7]*p[1]+vp[11]*p[2]+vp[15];
    return [(x/w*0.5+0.5)*W, (1-(y/w*0.5+0.5))*H]; };
  if(S.box){
    const e=scene.ext(); const c=[]; for(const x of[-1,1])for(const y of[-1,1])for(const z of[-1,1]) c.push(toScreen([x*e[0],y*e[1],z*e[2]]));
    ctx.strokeStyle = `rgba(${fg.map(v=>Math.round(v*255)).join(",")},.30)`; ctx.beginPath();
    for(let i=0;i<8;i++)for(const b of[1,2,4]) if(!(i&b)){ const j=i|b; ctx.moveTo(c[i][0],c[i][1]); ctx.lineTo(c[j][0],c[j][1]); }
    ctx.stroke(); }
  if(S.axes){
    const view=camView(), cx=46*dpr+8*dpr, cy=H-46*dpr-8*dpr, L=34*dpr;
    const axes=[["X",[1,0,0],"#ff5d5d"],["Y",[0,1,0],"#5ddc6f"],["Z",[0,0,1],"#5da8ff"]].map(([n,a,col])=>{
      const x=view[0]*a[0]+view[4]*a[1]+view[8]*a[2], y=view[1]*a[0]+view[5]*a[1]+view[9]*a[2], z=view[2]*a[0]+view[6]*a[1]+view[10]*a[2];
      return {n,col,x,y,z}; }).sort((p,q)=>p.z-q.z);
    ctx.fillStyle=`rgba(${fg.map(v=>Math.round(v*255)).join(",")},.5)`; ctx.beginPath(); ctx.arc(cx,cy,3*dpr,0,7); ctx.fill();
    axes.forEach(a=>{ const ex=cx+a.x*L, ey=cy-a.y*L; ctx.strokeStyle=a.col; ctx.lineWidth=2*dpr; ctx.globalAlpha=0.55+0.45*Math.max(0,a.z);
      ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(ex,ey); ctx.stroke(); ctx.fillStyle=a.col;
      ctx.beginPath(); ctx.arc(ex,ey,3.2*dpr,0,7); ctx.fill(); ctx.fillText(a.n, ex+(a.x>=0?5:-13)*dpr, ey+4*dpr); ctx.globalAlpha=1; });
  }
  if(S.scale){
    const phys = scene.physPerWorld(), pxu = pxPerUnit(H), physPerPx = phys/pxu;
    const len = niceLength(W*0.16*physPerPx), px = len/physPerPx, x1=W-24*dpr, x0=x1-px, y=H-34*dpr;
    const label = (len>=1000 && scene.unit==="um") ? `${len/1000} mm` : `${len} ${scene.unit}`;
    ctx.strokeStyle=`rgb(${fg.map(v=>Math.round(v*255)).join(",")})`; ctx.fillStyle=ctx.strokeStyle; ctx.lineWidth=2.4*dpr;
    ctx.beginPath(); ctx.moveTo(x0,y); ctx.lineTo(x1,y); ctx.moveTo(x0,y-5*dpr); ctx.lineTo(x0,y+5*dpr); ctx.moveTo(x1,y-5*dpr); ctx.lineTo(x1,y+5*dpr); ctx.stroke();
    ctx.textAlign="center"; ctx.fillText(label,(x0+x1)/2,y-9*dpr); ctx.textAlign="start";
  }
}

// ---- screenshot: the GL canvas plus the overlay, as a PNG download ---------------------------
function savePng(glCanvas, drawOv, name){
  const c=document.createElement("canvas"); c.width=glCanvas.width; c.height=glCanvas.height;
  const g=c.getContext("2d"); g.drawImage(glCanvas,0,0); drawOv(g,c.width,c.height);
  c.toBlob(b=>{ const a=document.createElement("a"); a.href=URL.createObjectURL(b); a.download=name; document.body.appendChild(a); a.click();
    setTimeout(()=>{ URL.revokeObjectURL(a.href); a.remove(); }, 2000); }, "image/png");
}

// ---- panel show / hide and the top-right toolbar --------------------------------------------
function wireChrome(onResize){
  $("toggle").onclick = () => { const p=$("panel"); p.classList.toggle("hidden"); $("toggle").textContent = p.classList.contains("hidden") ? "☰ Controls" : "✕"; onResize(); };
  $("btnReset").onclick = () => resetCamera();
  $("btnRot").onclick = () => setAutoRotate(!S.autorot);
  $("btnFull").onclick = () => { if(document.fullscreenElement) document.exitFullscreen(); else $("wrap").requestFullscreen && $("wrap").requestFullscreen(); };
  window.addEventListener("resize", onResize);
}
"""

CHROME_HTML = r"""
<div id="wrap">
 <div id="stage"><canvas id="gl"></canvas><canvas id="ov"></canvas></div>
 <div id="panel"></div>
 <button id="toggle" title="Show or hide the controls">&#10005;</button>
 <div id="bar">
  <button id="btnRot" title="Spin the view around the vertical axis">&#9654; Rotate</button>
  <button id="btnShot" title="Save a PNG of this view (with the scale bar and axes)">&#128247; Screenshot</button>
  <button id="btnReset" title="Back to the starting view (or double-click the picture)">&#10226; Reset</button>
  <button id="btnFull" title="Full screen">&#9974;</button>
 </div>
 <div id="hud"></div>
 <div id="msg"></div>
</div>
"""

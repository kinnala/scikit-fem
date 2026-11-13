/*
 * scikit-fem web viewer.
 *
 * A dependency-free viewer for the payload produced by
 * skfem.visuals.data.VisData.to_dict().  Renders with WebGL2 and falls back
 * to a software rasterizer on a 2D canvas.  Usage:
 *
 *   const handle = SkfemViewer.mount(element, data, options);
 *   handle.update(newData, newOptions);
 *   handle.png();      // data URL of the current view, with colorbar
 *   handle.png({colorbar: false});
 *   handle.destroy();
 *
 * Part of scikit-fem, BSD-3-Clause license.
 */
(function (root) {
  'use strict';
  if (root.SkfemViewer) return;

  /* ------------------------------------------------------------------ */
  /* Styles (scoped under .skv, injected once per document)              */
  /* ------------------------------------------------------------------ */
  const CSS = `
.skv{--skv-bg:#ffffff;--skv-canvas:#f6f7f9;--skv-ink:#1d2330;--skv-muted:#667085;--skv-line:#e3e6eb;
--skv-accent:#2f6fde;--skv-face:#c9d3df;--skv-panel:rgba(255,255,255,.86);--skv-ctl:#ffffff;--skv-ctl-line:#cfd5de;
position:relative;display:flex;flex-direction:column;width:100%;height:100%;min-height:240px;box-sizing:border-box;
font:12px/1.35 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:var(--skv-ink);background:var(--skv-bg);
border:1px solid var(--skv-line);border-radius:8px;overflow:hidden}
@media (prefers-color-scheme:dark){.skv:not([data-theme="light"]){--skv-bg:#14171c;--skv-canvas:#1b1f26;--skv-ink:#e6e9ee;
--skv-muted:#98a2b3;--skv-line:#2c323c;--skv-accent:#6ea2ff;--skv-face:#5b6675;--skv-panel:rgba(20,23,28,.82);--skv-ctl:#20252d;--skv-ctl-line:#3a414c}}
.skv[data-theme="dark"]{--skv-bg:#14171c;--skv-canvas:#1b1f26;--skv-ink:#e6e9ee;--skv-muted:#98a2b3;--skv-line:#2c323c;
--skv-accent:#6ea2ff;--skv-face:#5b6675;--skv-panel:rgba(20,23,28,.82);--skv-ctl:#20252d;--skv-ctl-line:#3a414c}
.skv *{box-sizing:border-box}
.skv-bar{display:flex;flex-wrap:wrap;gap:4px 10px;align-items:center;padding:6px 8px;border-bottom:1px solid var(--skv-line)}
.skv-bar[hidden]{display:none}
.skv-grp{display:inline-flex;align-items:center;gap:4px;white-space:nowrap}
.skv-grp[hidden]{display:none}
.skv-grp>span{color:var(--skv-muted)}
.skv select,.skv button{font:inherit;color:var(--skv-ink);background:var(--skv-ctl);border:1px solid var(--skv-ctl-line);
border-radius:5px;padding:2px 6px;height:24px;cursor:pointer;max-width:12em}
.skv button:hover,.skv select:hover{border-color:var(--skv-accent)}
.skv input[type=range]{width:72px;accent-color:var(--skv-accent)}
.skv input[type=checkbox]{accent-color:var(--skv-accent);margin:0}
.skv label{display:inline-flex;align-items:center;gap:4px;cursor:pointer}
.skv-stage{position:relative;flex:1 1 auto;min-height:0;background:var(--skv-canvas)}
.skv-stage canvas.skv-main{position:absolute;inset:0;width:100%;height:100%;touch-action:none;cursor:grab;display:block}
.skv-stage canvas.skv-main.skv-drag{cursor:grabbing}
.skv-stage canvas[hidden]{display:none}
.skv-info,.skv-readout,.skv-hint{position:absolute;padding:4px 7px;border-radius:5px;background:var(--skv-panel);
pointer-events:none;font-size:11px}
.skv-info{left:8px;top:8px;max-width:calc(100% - 120px)}
.skv-info b{font-weight:600}
.skv-readout{left:8px;bottom:8px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.skv-hint{right:8px;bottom:8px;color:var(--skv-muted)}
.skv-info[hidden],.skv-readout[hidden],.skv-hint[hidden],.skv-cbar[hidden]{display:none}
.skv-cbar{position:absolute;right:12px;top:50%;transform:translateY(-50%);height:min(60%,320px);display:flex;gap:6px;pointer-events:none}
.skv-cbar canvas{width:14px;height:100%;border-radius:3px;border:1px solid var(--skv-line)}
.skv-ticks{position:relative;width:52px;height:100%;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:10.5px}
.skv-ticks span{position:absolute;left:0;transform:translateY(-50%);white-space:nowrap;background:var(--skv-panel);padding:0 3px;border-radius:3px}
.skv-msg{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:var(--skv-muted)}
`;
  function injectStyle(doc) {
    if (doc.getElementById('skv-style')) return;
    const s = doc.createElement('style');
    s.id = 'skv-style';
    s.textContent = CSS;
    doc.head.appendChild(s);
  }

  /* ------------------------------------------------------------------ */
  /* Colormaps                                                           */
  /* ------------------------------------------------------------------ */
  const CMAPS = {
    viridis: ['#440154', '#482878', '#3e4989', '#31688e', '#26828e', '#1f9e89', '#35b779', '#6ece58', '#b5de2b', '#fde725'],
    turbo: ['#30123b', '#4145ab', '#4675ed', '#39a2fc', '#1bcfd4', '#24eca6', '#61fc6c', '#a4fc3b', '#d1e834', '#f3c63a', '#fe9b2d', '#f36315', '#d93806', '#b11901', '#7a0402'],
    jet: ['#00007f', '#0000ff', '#007fff', '#00ffff', '#7fff7f', '#ffff00', '#ff7f00', '#ff0000', '#7f0000'],
    coolwarm: ['#3b4cc0', '#6282ea', '#8db0fe', '#b8d0f9', '#dddcdc', '#f5c4ac', '#f39b7a', '#de604d', '#b40426'],
    magma: ['#000004', '#1c1044', '#4f127b', '#812581', '#b5367a', '#e55964', '#fb8761', '#fec287', '#fcfdbf'],
    plasma: ['#0d0887', '#47039f', '#7301a8', '#9c179e', '#bd3786', '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921'],
    grey: ['#1d1d1f', '#f2f2f2'],
  };
  function hex2rgb(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
  function buildLut(name) {
    const stops = (CMAPS[name] || CMAPS.viridis).map(hex2rgb), k = stops.length - 1;
    const lut = new Uint8Array(256 * 4);
    for (let i = 0; i < 256; i++) {
      const t = (i / 255) * k, j = Math.min(k - 1, Math.floor(t)), f = t - j;
      for (let c = 0; c < 3; c++) lut[4 * i + c] = Math.round(stops[j][c] + (stops[j + 1][c] - stops[j][c]) * f);
      lut[4 * i + 3] = 255;
    }
    return lut;
  }

  /* ------------------------------------------------------------------ */
  /* Helpers                                                             */
  /* ------------------------------------------------------------------ */
  function decode(x, Type) {
    if (x === undefined || x === null) return null;
    if (x instanceof Type) return x;
    if (ArrayBuffer.isView(x)) return new Type(x.buffer.slice(x.byteOffset, x.byteOffset + x.byteLength));
    if (x instanceof ArrayBuffer) return new Type(x);
    if (Array.isArray(x)) return Type.from(x);
    const bin = atob(x), n = bin.length, u8 = new Uint8Array(n);
    for (let i = 0; i < n; i++) u8[i] = bin.charCodeAt(i);
    return new Type(u8.buffer);
  }
  const rotX = (a) => { const c = Math.cos(a), s = Math.sin(a); return [1, 0, 0, 0, c, -s, 0, s, c]; };
  const rotY = (a) => { const c = Math.cos(a), s = Math.sin(a); return [c, 0, s, 0, 1, 0, -s, 0, c]; };
  const rotZ = (a) => { const c = Math.cos(a), s = Math.sin(a); return [c, -s, 0, s, c, 0, 0, 0, 1]; };
  function mul(A, B) {
    const C = new Array(9);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[3 * i + j] = A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j];
    return C;
  }
  const VIEWS = {
    xy: () => [1, 0, 0, 0, 1, 0, 0, 0, 1],
    xz: () => rotX(-Math.PI / 2),
    yz: () => mul(rotX(-Math.PI / 2), rotZ(-Math.PI / 2)),
    iso: () => mul(rotX(-1.0), rotZ(-0.65)),
  };
  function fmt(v) {
    if (!isFinite(v)) return String(v);
    const a = Math.abs(v);
    if (a === 0) return '0';
    if (a >= 1e5 || a < 1e-3) return v.toExponential(2).replace('e+', 'e');
    return String(+v.toPrecision(4));
  }
  function esc(s) { return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]); }
  function niceTicks(a, b, n) {
    const span = b - a; if (!(span > 0)) return [a];
    const step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map((x) => x * mag).find((x) => span / x <= n) || 10 * mag;
    const out = []; for (let v = Math.ceil(a / step) * step; v <= b + 1e-9 * span; v += step) out.push(Math.abs(v) < 1e-12 * step ? 0 : v);
    return out;
  }
  function el(doc, tag, cls, html) { const e = doc.createElement(tag); if (cls) e.className = cls; if (html !== undefined) e.innerHTML = html; return e; }
  function cssRgb(node, name) {
    const c = getComputedStyle(node).getPropertyValue(name).trim();
    if (c.startsWith('#')) return hex2rgb(c.length === 4 ? '#' + [...c.slice(1)].map((x) => x + x).join('') : c);
    const m = c.match(/\d+(\.\d+)?/g);
    return m ? m.slice(0, 3).map(Number) : [128, 128, 128];
  }

  /* ------------------------------------------------------------------ */
  /* WebGL2 renderer                                                     */
  /* ------------------------------------------------------------------ */
  const VS = `#version 300 es
in vec3 aPos; in float aVal; in vec3 aDisp;
uniform mat3 uR; uniform vec3 uCenter; uniform float uScale; uniform vec2 uPan; uniform vec2 uSize; uniform float uDepth;
uniform float uDef; uniform float uWarp; uniform float uVmin; uniform float uVmax;
out float vVal; out vec3 vView;
void main() {
  vec3 p = aPos + uDef * aDisp;
  p.z += uWarp * (aVal - uVmin) / (uVmax - uVmin);
  vec3 q = uR * (p - uCenter);
  vView = q; vVal = aVal;
  vec2 s = q.xy * uScale + uPan;
  gl_Position = vec4(2.0 * s.x / uSize.x, 2.0 * s.y / uSize.y, -q.z / uDepth, 1.0);
}`;
  const FS_FACE = `#version 300 es
precision highp float;
in float vVal; in vec3 vView;
uniform sampler2D uLut; uniform int uHasField; uniform float uBands; uniform vec3 uFlat; uniform float uVmin; uniform float uVmax;
out vec4 o;
void main() {
  vec3 n = normalize(cross(dFdx(vView), dFdy(vView)));
  float light = 0.38 + 0.62 * abs(n.z);
  vec3 c = uFlat;
  if (uHasField == 1) {
    float q = clamp((vVal - uVmin) / (uVmax - uVmin), 0.0, 1.0);
    if (uBands > 1.0) q = min(uBands - 1.0, floor(q * uBands)) / (uBands - 1.0);
    c = texture(uLut, vec2((q * 255.0 + 0.5) / 256.0, 0.5)).rgb;
  }
  o = vec4(c * light, 1.0);
}`;
  const FS_LINE = `#version 300 es
precision highp float;
in float vVal; in vec3 vView;
uniform sampler2D uLut; uniform int uUseLut; uniform vec4 uColor; uniform float uBands; uniform float uVmin; uniform float uVmax;
out vec4 o;
void main() {
  if (uUseLut == 1) {
    float q = clamp((vVal - uVmin) / (uVmax - uVmin), 0.0, 1.0);
    if (uBands > 1.0) q = min(uBands - 1.0, floor(q * uBands)) / (uBands - 1.0);
    o = vec4(texture(uLut, vec2((q * 255.0 + 0.5) / 256.0, 0.5)).rgb, 1.0);
  } else o = uColor;
}`;

  function compile(gl, vs, fs) {
    const mk = (type, src) => {
      const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, mk(gl.VERTEX_SHADER, vs)); gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs));
    gl.bindAttribLocation(p, 0, 'aPos'); gl.bindAttribLocation(p, 1, 'aVal'); gl.bindAttribLocation(p, 2, 'aDisp');
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const u = {}, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) { const name = gl.getActiveUniform(p, i).name; u[name] = gl.getUniformLocation(p, name); }
    return { p, u };
  }

  class GLRenderer {
    constructor(canvas) {
      const gl = canvas.getContext('webgl2', { antialias: true, preserveDrawingBuffer: false, premultipliedAlpha: false });
      if (!gl) throw new Error('WebGL2 not available');
      this.gl = gl;
      this.face = compile(gl, VS, FS_FACE);
      this.line = compile(gl, VS, FS_LINE);
      this.vao = gl.createVertexArray();
      this.bPos = gl.createBuffer(); this.bVal = gl.createBuffer(); this.bDisp = gl.createBuffer();
      this.bTri = gl.createBuffer(); this.bEdge = gl.createBuffer();
      this.lut = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, this.lut);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      this.ntri = 0; this.nedge = 0;
    }
    setPositions(P3) { const gl = this.gl; gl.bindVertexArray(this.vao); gl.bindBuffer(gl.ARRAY_BUFFER, this.bPos); gl.bufferData(gl.ARRAY_BUFFER, P3, gl.STATIC_DRAW); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0); }
    setValues(V) { const gl = this.gl; gl.bindVertexArray(this.vao); gl.bindBuffer(gl.ARRAY_BUFFER, this.bVal); gl.bufferData(gl.ARRAY_BUFFER, V, gl.DYNAMIC_DRAW); gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 0, 0); }
    setDisp(D) { const gl = this.gl; gl.bindVertexArray(this.vao); gl.bindBuffer(gl.ARRAY_BUFFER, this.bDisp); gl.bufferData(gl.ARRAY_BUFFER, D, gl.DYNAMIC_DRAW); gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 0, 0); }
    setTris(T) { const gl = this.gl; gl.bindVertexArray(this.vao); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.bTri); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, T, gl.DYNAMIC_DRAW); this.ntri = T.length; }
    setEdges(E) { this.edgeData = E; this.nedge = E.length; const gl = this.gl; gl.bindVertexArray(this.vao); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.bEdge); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, E, gl.DYNAMIC_DRAW); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.bTri); }
    setLut(lut) { const gl = this.gl; gl.bindTexture(gl.TEXTURE_2D, this.lut); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, lut); }
    draw(st) {
      const gl = this.gl;
      gl.viewport(0, 0, st.W, st.H);
      gl.clearColor(st.bg[0] / 255, st.bg[1] / 255, st.bg[2] / 255, 1);
      gl.clearDepth(1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      gl.bindVertexArray(this.vao);
      const R = st.R, Rt = [R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]];
      const common = (prog) => {
        const u = prog.u;
        gl.useProgram(prog.p);
        gl.uniformMatrix3fv(u.uR, false, Rt);
        gl.uniform3f(u.uCenter, st.center[0], st.center[1], st.center[2]);
        gl.uniform1f(u.uScale, st.scale);
        gl.uniform2f(u.uPan, st.panX, -st.panY);
        gl.uniform2f(u.uSize, st.W, st.H);
        gl.uniform1f(u.uDepth, st.depth);
        gl.uniform1f(u.uDef, st.def);
        gl.uniform1f(u.uWarp, st.warp);
        gl.uniform1f(u.uVmin, st.vmin);
        gl.uniform1f(u.uVmax, st.vmax);
        if (u.uBands) gl.uniform1f(u.uBands, st.bands);
        if (u.uLut) { gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.lut); gl.uniform1i(u.uLut, 0); }
      };
      if (this.ntri) {
        common(this.face);
        gl.uniform1i(this.face.u.uHasField, st.hasField ? 1 : 0);
        gl.uniform3f(this.face.u.uFlat, st.face[0] / 255, st.face[1] / 255, st.face[2] / 255);
        gl.enable(gl.POLYGON_OFFSET_FILL); gl.polygonOffset(1, 1);
        gl.depthMask(true); gl.disable(gl.BLEND); gl.depthFunc(gl.LESS);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.bTri);
        gl.drawElements(gl.TRIANGLES, this.ntri, gl.UNSIGNED_INT, 0);
        gl.disable(gl.POLYGON_OFFSET_FILL);
      }
      if (this.nedge && (st.edges || st.lineField)) {
        common(this.line);
        gl.uniform1i(this.line.u.uUseLut, st.lineField ? 1 : 0);
        gl.uniform4f(this.line.u.uColor, st.edgeColor[0] / 255, st.edgeColor[1] / 255, st.edgeColor[2] / 255, st.edgeAlpha);
        gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.depthMask(false); gl.depthFunc(gl.LEQUAL);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.bEdge);
        gl.drawElements(gl.LINES, this.nedge, gl.UNSIGNED_INT, 0);
        gl.depthMask(true); gl.disable(gl.BLEND);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.bTri);
      }
    }
    destroy() { const ext = this.gl.getExtension('WEBGL_lose_context'); if (ext) ext.loseContext(); }
  }

  /* ------------------------------------------------------------------ */
  /* Software (Canvas 2D) renderer                                       */
  /* ------------------------------------------------------------------ */
  class CanvasRenderer {
    constructor(canvas) { this.canvas = canvas; this.ctx = canvas.getContext('2d'); this.img = null; }
    setLut(lut) { this.lut = lut; }
    draw(st, scene) {
      const W = st.W, H = st.H, ctx = this.ctx;
      if (!this.img || this.img.width !== W || this.img.height !== H) { this.img = ctx.createImageData(W, H); this.zbuf = new Float32Array(W * H); }
      const pix = new Uint32Array(this.img.data.buffer), zbuf = this.zbuf, bg = st.bg, mf = st.face, lut = this.lut;
      pix.fill((255 << 24) | (bg[2] << 16) | (bg[1] << 8) | bg[0]);
      zbuf.fill(-Infinity);
      const sx = scene.sx, sy = scene.sy, sz = scene.sz, V = scene.V, hasF = st.hasField;
      const vmin = st.vmin, ir = 1 / (st.vmax - st.vmin), nb = st.bands;
      const T = scene.tris;
      const col = (v, light) => {
        let q = (v - vmin) * ir; q = q < 0 ? 0 : q > 1 ? 1 : q;
        if (nb > 1) q = Math.min(nb - 1, Math.floor(q * nb)) / (nb - 1);
        const k = 4 * ((q * 255 + 0.5) | 0);
        return (255 << 24) | (((lut[k + 2] * light) | 0) << 16) | (((lut[k + 1] * light) | 0) << 8) | ((lut[k] * light) | 0);
      };
      for (let t = 0; t < T.length; t += 3) {
        const a = T[t], b = T[t + 1], d = T[t + 2];
        const x0 = sx[a], y0 = sy[a], z0 = sz[a], x1 = sx[b], y1 = sy[b], z1 = sz[b], x2 = sx[d], y2 = sy[d], z2 = sz[d];
        const area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        if (Math.abs(area) < 1e-9) continue;
        const ux = x1 - x0, uy = y1 - y0, uz = z1 - z0, vx = x2 - x0, vy = y2 - y0, vz = z2 - z0;
        const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
        const light = 0.38 + 0.62 * Math.abs(nz) / (Math.hypot(nx, ny, nz) || 1);
        const minX = Math.max(0, Math.floor(Math.min(x0, x1, x2))), maxX = Math.min(W - 1, Math.ceil(Math.max(x0, x1, x2)));
        const minY = Math.max(0, Math.floor(Math.min(y0, y1, y2))), maxY = Math.min(H - 1, Math.ceil(Math.max(y0, y1, y2)));
        if (minX > maxX || minY > maxY) continue;
        const ia = 1 / area;
        const A0 = (y1 - y2) * ia, B0 = (x2 - x1) * ia, C0 = (x1 * y2 - x2 * y1) * ia;
        const A1 = (y2 - y0) * ia, B1 = (x0 - x2) * ia, C1 = (x2 * y0 - x0 * y2) * ia;
        const va = hasF ? V[a] : 0, vb = hasF ? V[b] : 0, vd = hasF ? V[d] : 0;
        const flat = (255 << 24) | (((mf[2] * light) | 0) << 16) | (((mf[1] * light) | 0) << 8) | ((mf[0] * light) | 0);
        for (let y = minY; y <= maxY; y++) {
          const py = y + 0.5;
          let w0 = A0 * (minX + 0.5) + B0 * py + C0, w1 = A1 * (minX + 0.5) + B1 * py + C1;
          let row = y * W + minX;
          for (let x = minX; x <= maxX; x++, w0 += A0, w1 += A1, row++) {
            const w2 = 1 - w0 - w1;
            if (w0 < -1e-5 || w1 < -1e-5 || w2 < -1e-5) continue;
            const z = w0 * z0 + w1 * z1 + w2 * z2;
            if (z <= zbuf[row]) continue;
            zbuf[row] = z;
            pix[row] = hasF ? col(w0 * va + w1 * vb + w2 * vd, light) : flat;
          }
        }
      }
      const Ed = scene.edgesIdx;
      if (Ed && (st.edges || st.lineField)) {
        const ec = st.edgeColor, alpha = st.lineField ? 1 : st.edgeAlpha, zeps = 2.0 * st.dpr + 0.004 * st.scale * scene.radius;
        for (let k = 0; k < Ed.length; k += 2) {
          const a = Ed[k], b = Ed[k + 1];
          const x0 = sx[a], y0 = sy[a], z0 = sz[a], dx = sx[b] - x0, dy = sy[b] - y0, dz = sz[b] - z0;
          const steps = Math.ceil(Math.max(Math.abs(dx), Math.abs(dy)));
          if (steps > 20000) continue;
          for (let i = 0; i <= steps; i++) {
            const f = steps ? i / steps : 0;
            const px = (x0 + dx * f) | 0, py = (y0 + dy * f) | 0;
            if (px < 0 || py < 0 || px >= W || py >= H) continue;
            const idx = py * W + px, z = z0 + dz * f;
            if (z < zbuf[idx] - zeps) continue;
            if (st.lineField) { pix[idx] = col(V[a] + (V[b] - V[a]) * f, 1); continue; }
            const v = pix[idx], r = v & 255, g = (v >> 8) & 255, bb = (v >> 16) & 255;
            pix[idx] = (255 << 24) | (((bb + (ec[2] - bb) * alpha) | 0) << 16) | (((g + (ec[1] - g) * alpha) | 0) << 8) | ((r + (ec[0] - r) * alpha) | 0);
          }
        }
      }
      ctx.putImageData(this.img, 0, 0);
    }
    destroy() {}
  }

  /* ------------------------------------------------------------------ */
  /* Viewer                                                              */
  /* ------------------------------------------------------------------ */
  const TOOLBAR = `
<span class="skv-grp" data-g="field"><span>field</span><select data-c="field"></select></span>
<span class="skv-grp" data-g="comp"><span>comp</span><select data-c="comp"></select></span>
<span class="skv-grp" data-g="cmap"><select data-c="cmap" title="colormap"></select>
<select data-c="bands" title="color bands"><option value="0">smooth</option><option>5</option><option>8</option><option>12</option><option>20</option></select></span>
<span class="skv-grp" data-g="edges"><label><input type="checkbox" data-c="edges">edges</label></span>
<span class="skv-grp" data-g="warp"><label><input type="checkbox" data-c="warp">warp</label><input type="range" data-c="warpAmt" min="2" max="150" value="35"></span>
<span class="skv-grp" data-g="deform"><span>deform</span><select data-c="deform"></select><input type="range" data-c="defAmt" min="0" max="100" value="50"></span>
<span class="skv-grp" data-g="clip"><span>clip</span><select data-c="clip"><option value="-1">off</option><option value="0">x</option><option value="1">y</option><option value="2">z</option></select>
<input type="range" data-c="clipPos" min="0" max="1000" value="500"><label title="flip"><input type="checkbox" data-c="clipFlip">flip</label></span>
<span class="skv-grp" data-g="view"><button data-view="xy">xy</button><button data-view="xz">xz</button><button data-view="yz">yz</button><button data-view="iso">iso</button></span>
<span class="skv-grp" data-g="png"><button data-c="png" title="save PNG">png</button></span>`;

  class Viewer {
    constructor(host, data, options) {
      const doc = host.ownerDocument || document;
      this.doc = doc;
      injectStyle(doc);
      this.host = host;
      this.root = el(doc, 'div', 'skv');
      this.bar = el(doc, 'div', 'skv-bar', TOOLBAR);
      this.stage = el(doc, 'div', 'skv-stage');
      this.glCanvas = el(doc, 'canvas', 'skv-main');
      this.cv = el(doc, 'canvas', 'skv-main');
      this.info = el(doc, 'div', 'skv-info');
      this.readout = el(doc, 'div', 'skv-readout'); this.readout.hidden = true;
      this.hint = el(doc, 'div', 'skv-hint');
      this.cbar = el(doc, 'div', 'skv-cbar', '<canvas width="14" height="256"></canvas><div class="skv-ticks"></div>');
      this.stage.append(this.glCanvas, this.cv, this.info, this.cbar, this.readout, this.hint);
      this.root.append(this.bar, this.stage);
      host.appendChild(this.root);
      this.q = (sel) => this.root.querySelector(sel);
      this.c = (name) => this.root.querySelector('[data-c="' + name + '"]');
      const cm = this.c('cmap');
      for (const k of Object.keys(CMAPS)) cm.add(new Option(k, k));
      this.view = { R: VIEWS.xy(), zoom: 1, panX: 0, panY: 0, base: 1, x0: 0, x1: 1 };
      this.W = 1; this.H = 1; this.dpr = 1;
      this.renderQueued = false;
      this.gl = null; this.sw = null;
      this.wire();
      this.ro = new ResizeObserver(() => this.resize());
      this.ro.observe(this.stage);
      this.mql = root.matchMedia ? root.matchMedia('(prefers-color-scheme: dark)') : null;
      this.onTheme = () => this.requestRender();
      if (this.mql && this.mql.addEventListener) this.mql.addEventListener('change', this.onTheme);
      this.update(data, options);
    }

    /* ----- data ----- */
    update(data, options) {
      if (options) this.options = Object.assign({}, this.options || {}, options);
      const o = this.options || {};
      if (data) this.load(data);
      this.applyOptions(o, !!data || !!options);
    }

    load(d) {
      if (typeof d === 'string') d = JSON.parse(d);
      if (d.data && d.points === undefined) { if (d.options) this.options = Object.assign({}, d.options, this.options); d = d.data; }
      const g = d.gdim, P = decode(d.points, Float32Array), n = P.length / g;
      const s = {
        d, meta: d.meta || {}, dim: d.dim, gdim: g, n, nloc: d.nloc, nsub: d.nsub, nel: d.nel, nvc: d.nvc,
        P, C: decode(d.cells, Int32Array), E: decode(d.edges, Int32Array), Wd: decode(d.weld, Int32Array),
        fields: (d.fields || []).map((f) => Object.assign({}, f, { data: decode(f.data, Float32Array) })),
      };
      s.mode = s.dim === 1 ? (g === 1 ? '1d' : 'curve') : s.dim === 3 ? '3d' : g === 3 ? 'surf' : '2d';
      const P3 = new Float32Array(n * 3);
      const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      for (let i = 0; i < n; i++) for (let k = 0; k < g; k++) {
        const x = P[i * g + k]; P3[i * 3 + k] = x;
        if (x < lo[k]) lo[k] = x; if (x > hi[k]) hi[k] = x;
      }
      for (let k = g; k < 3; k++) { lo[k] = 0; hi[k] = 0; }
      if (!n) { lo.fill(0); hi.fill(1); }
      s.P3 = P3; s.lo = lo; s.hi = hi;
      s.radius = Math.max(1e-12, 0.5 * Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]));
      s.cent = new Float32Array(s.nel * 3);
      for (let e = 0; e < s.nel; e++) for (let k = 0; k < 3; k++) {
        let acc = 0; for (let j = 0; j < s.nloc; j++) acc += P3[(e * s.nloc + j) * 3 + k];
        s.cent[e * 3 + k] = acc / s.nloc;
      }
      s.sx = new Float32Array(n); s.sy = new Float32Array(n); s.sz = new Float32Array(n);
      s.V = new Float32Array(n); s.D = new Float32Array(n * 3);
      const prevKind = this.scene ? this.scene.mode + '|' + this.scene.meta.mesh : null;
      this.scene = s;
      this.sameKind = prevKind === s.mode + '|' + s.meta.mesh;
      // field / deformation menus
      const fs = this.c('field'); fs.innerHTML = '';
      s.fields.forEach((f, i) => fs.add(new Option(f.name, String(i))));
      if (!s.fields.length) fs.add(new Option('(none)', '-1'));
      const ds = this.c('deform'); ds.innerHTML = ''; ds.add(new Option('off', '-1'));
      s.fields.forEach((f, i) => { if (f.ncomp === g && g > 1) ds.add(new Option(f.name, String(i))); });
      this.setupRenderer();
      if (this.gl) this.gl.setPositions(P3);
    }

    setupRenderer() {
      const s = this.scene;
      const want = (this.options && this.options.renderer) || 'auto';
      const use1d = s.mode === '1d';
      if (!use1d && want !== 'canvas' && !this.gl && !this.glFailed) {
        try { this.gl = new GLRenderer(this.glCanvas); }
        catch (e) { this.glFailed = true; this.gl = null; }
      }
      if (want === 'canvas' && this.gl) { this.gl.destroy(); this.gl = null; }
      this.useGL = !use1d && !!this.gl;
      if (!this.useGL && !use1d && !this.sw) this.sw = new CanvasRenderer(this.cv);
      this.glCanvas.hidden = !this.useGL;
      this.cv.hidden = this.useGL;
      this.mainCanvas = this.useGL ? this.glCanvas : this.cv;
      if (this.lut) { if (this.gl) this.gl.setLut(this.lut); if (this.sw) this.sw.setLut(this.lut); }
    }

    get rendererName() { return this.scene.mode === '1d' ? 'canvas2d' : this.useGL ? 'webgl2' : 'canvas'; }

    applyOptions(o, fresh) {
      const s = this.scene, g = s.gdim;
      const st = this.st = this.st || {};
      if (fresh || st.cmap === undefined) {
        st.cmap = o.cmap in CMAPS ? o.cmap : (st.cmap || 'viridis');
        st.bands = o.bands | 0;
        st.edges = o.edges !== undefined ? !!o.edges : (st.edges !== undefined ? st.edges : true);
        st.warp = !!o.warp; st.warpAmt = typeof o.warp === 'number' ? o.warp : (o.warp_scale || 0.35);
        st.clipAxis = o.clip === undefined || o.clip === null || o.clip === false ? -1 : ({ x: 0, y: 1, z: 2 }[o.clip] !== undefined ? { x: 0, y: 1, z: 2 }[o.clip] : +o.clip);
        st.clipPos = o.clip_position !== undefined ? +o.clip_position : 0.5;
        st.clipFlip = !!o.clip_flip;
        st.defAmt = 0.5; st.defScaleUser = o.deform_scale;
        let fi = 0;
        if (o.field !== undefined && o.field !== null) {
          const k = s.fields.findIndex((f) => f.name === o.field);
          if (k >= 0) fi = k;
        }
        st.field = s.fields.length ? fi : -1;
        st.comp = o.component !== undefined && o.component !== null ? String(o.component) : 'mag';
        st.deform = -1;
        if (o.deform) {
          const name = o.deform === true ? null : o.deform;
          const k = s.fields.findIndex((f) => f.ncomp === g && g > 1 && (name === null || f.name === name));
          st.deform = k;
        }
        st.vminUser = o.vmin; st.vmaxUser = o.vmax;
      }
      // controls
      this.c('cmap').value = st.cmap;
      this.c('bands').value = String(st.bands);
      if (this.c('bands').value !== String(st.bands)) { this.c('bands').add(new Option(String(st.bands), String(st.bands))); this.c('bands').value = String(st.bands); }
      this.c('edges').checked = st.edges;
      this.c('warp').checked = st.warp;
      this.c('warpAmt').value = String(Math.round(st.warpAmt * 100));
      this.c('clip').value = String(st.clipAxis);
      this.c('clipPos').value = String(Math.round(st.clipPos * 1000));
      this.c('clipFlip').checked = st.clipFlip;
      this.c('field').value = String(st.field);
      this.c('deform').value = String(st.deform);
      const show = (g2, on) => { this.q('[data-g="' + g2 + '"]').hidden = !on; };
      const hasF = st.field >= 0;
      show('field', s.fields.length > 1);
      show('cmap', hasF && s.mode !== '1d');
      show('edges', s.mode !== '1d' && s.mode !== 'curve');
      show('warp', s.mode === '2d' && hasF);
      show('deform', this.c('deform').options.length > 1 && s.mode !== '1d');
      show('clip', s.mode === '3d' && !s.meta.no_clip);
      show('view', s.mode !== '1d');
      this.bar.hidden = o.toolbar === false;
      if (o.theme === 'light' || o.theme === 'dark') this.root.dataset.theme = o.theme; else delete this.root.dataset.theme;
      this.fillComponents();
      this.lut = buildLut(st.cmap);
      if (this.gl) this.gl.setLut(this.lut);
      if (this.sw) this.sw.setLut(this.lut);
      this.computeValues();
      this.computeDisp();
      if (!this.sameKind || fresh && o.view) {
        const v = this.view;
        v.R = o.view && VIEWS[o.view] ? VIEWS[o.view]() : (s.mode === '3d' || s.mode === 'surf') ? VIEWS.iso() : (st.warp ? mul(rotX(-0.95), rotZ(-0.5)) : VIEWS.xy());
        v.zoom = 1; v.panX = 0; v.panY = 0; v.x0 = s.lo[0]; v.x1 = s.hi[0];
        if (!(v.x1 > v.x0)) { v.x0 -= 0.5; v.x1 += 0.5; }
      }
      this.sameKind = true;
      this.prepareGeometry();
      this.fitView(false);
      this.updateInfo();
      this.drawColorbar();
      this.resize(true);
      this.requestRender();
    }

    fillComponents() {
      const s = this.scene, st = this.st, cs = this.c('comp');
      cs.innerHTML = '';
      const f = st.field >= 0 ? s.fields[st.field] : null;
      if (!f || f.ncomp === 1) { this.q('[data-g="comp"]').hidden = true; st.compIdx = 0; return; }
      this.q('[data-g="comp"]').hidden = false;
      cs.add(new Option('magnitude', 'mag'));
      const sh = f.shape && f.shape.length ? f.shape : [f.ncomp];
      const lbl = 'xyz';
      for (let k = 0; k < f.ncomp; k++) {
        let name;
        if (sh.length === 1) name = sh[0] <= 3 ? lbl[k] : String(k);
        else { const j = k % sh[1], i = (k - j) / sh[1]; name = (sh[0] <= 3 && sh[1] <= 3) ? lbl[i] + lbl[j] : i + ',' + j; }
        cs.add(new Option(name, String(k)));
      }
      if (![...cs.options].some((op) => op.value === st.comp)) {
        const named = [...cs.options].find((op) => op.text === st.comp);
        st.comp = named ? named.value : 'mag';
      }
      cs.value = st.comp;
    }

    computeValues() {
      const s = this.scene, st = this.st, V = s.V, n = s.n;
      const f = st.field >= 0 ? s.fields[st.field] : null;
      s.hasField = !!f;
      if (!f) { V.fill(0); st.vmin = 0; st.vmax = 1; }
      else {
        const nc = f.ncomp, D = f.data;
        let r;
        if (nc === 1) { V.set(D.subarray(0, n)); r = f.range[0]; }
        else if (st.comp === 'mag') {
          for (let i = 0; i < n; i++) { let a = 0; for (let k = 0; k < nc; k++) a += D[i * nc + k] * D[i * nc + k]; V[i] = Math.sqrt(a); }
          r = f.magrange;
        } else {
          const k = +st.comp; for (let i = 0; i < n; i++) V[i] = D[i * nc + k];
          r = f.range[k];
        }
        let vmin = st.vminUser !== undefined && st.vminUser !== null ? +st.vminUser : r[0];
        let vmax = st.vmaxUser !== undefined && st.vmaxUser !== null ? +st.vmaxUser : r[1];
        st.dmin = r[0]; st.dmax = r[1];
        if (!(vmax > vmin)) { const c = vmin, h = 0.5 * (Math.abs(c) || 1); vmin = c - h; vmax = c + h; }
        st.vmin = vmin; st.vmax = vmax;
      }
      if (this.gl) this.gl.setValues(V);
      this.projDirty = true;
    }

    computeDisp() {
      const s = this.scene, st = this.st, Dd = s.D, g = s.gdim, n = s.n;
      Dd.fill(0); st.dispMax = 0;
      if (st.deform >= 0) {
        const f = s.fields[st.deform].data;
        let m = 0;
        for (let i = 0; i < n; i++) {
          let a = 0;
          for (let k = 0; k < g; k++) { const v = f[i * g + k]; Dd[i * 3 + k] = v; a += v * v; }
          if (a > m) m = a;
        }
        st.dispMax = Math.sqrt(m);
      }
      if (this.gl) this.gl.setDisp(Dd);
      this.projDirty = true;
    }

    defScale() {
      const st = this.st, s = this.scene;
      if (st.deform < 0 || !st.dispMax) return 0;
      if (st.defScaleUser !== undefined && st.defScaleUser !== null && st.defAmt === 0.5) return +st.defScaleUser;
      const base = st.defScaleUser !== undefined && st.defScaleUser !== null ? +st.defScaleUser : 0.12 * s.radius / st.dispMax;
      return base * Math.pow(10, (st.defAmt - 0.5) * 3);
    }
    warpHeight() { const s = this.scene; return s.mode === '2d' && this.st.warp && s.hasField ? this.st.warpAmt * 2 * s.radius : 0; }

    /* ----- geometry: visible triangles and wireframe ----- */
    prepareGeometry() {
      const s = this.scene, st = this.st;
      if (s.mode === '1d') return;
      const nvc = s.nvc, nsub = s.nsub, C = s.C, ncells = C.length / nvc;
      const keep = new Uint8Array(s.nel).fill(1);
      if (s.mode === '3d' && st.clipAxis >= 0) {
        const a = st.clipAxis, lo = s.lo[a], span = (s.hi[a] - lo) || 1;
        for (let e = 0; e < s.nel; e++) {
          const t = (s.cent[e * 3 + a] - lo) / span;
          keep[e] = st.clipFlip ? (t >= st.clipPos ? 1 : 0) : (t <= st.clipPos ? 1 : 0);
        }
      }
      const tris = [];
      if (s.mode === '3d') {
        const Wd = s.Wd, faces = s.meta.faces;
        let maxW = 0; for (let i = 0; i < Wd.length; i++) if (Wd[i] > maxW) maxW = Wd[i];
        const N = maxW + 1, big = N > 200000;
        const cnt = new Map(), ids = [0, 0, 0, 0];
        for (let c = 0; c < ncells; c++) {
          if (!keep[(c / nsub) | 0]) continue;
          for (let f = 0; f < faces.length; f++) {
            const fl = faces[f];
            for (let j = 0; j < fl.length; j++) ids[j] = Wd[C[c * nvc + fl[j]]];
            const so = ids.slice(0, fl.length).sort((x, y) => x - y);
            const key = big ? so[0] + ',' + so[1] + ',' + so[2] : so[0] + N * (so[1] + N * so[2]);
            const prev = cnt.get(key);
            if (prev === undefined) cnt.set(key, c * 8 + f); else cnt.set(key, -1);
          }
        }
        for (const v of cnt.values()) {
          if (v < 0) continue;
          const c = Math.floor(v / 8), fl = faces[v % 8], p0 = C[c * nvc + fl[0]];
          for (let j = 1; j + 1 < fl.length; j++) tris.push(p0, C[c * nvc + fl[j]], C[c * nvc + fl[j + 1]]);
        }
      } else if (s.dim === 2) {
        const poly = s.meta.poly;
        for (let c = 0; c < ncells; c++) {
          const p0 = C[c * nvc + poly[0]];
          for (let j = 1; j + 1 < poly.length; j++) tris.push(p0, C[c * nvc + poly[j]], C[c * nvc + poly[j + 1]]);
        }
      }
      s.tris = new Uint32Array(tris);
      let edges = [];
      if (s.mode === 'curve') edges = Array.from(C);
      else {
        const E = s.E, epe = (E.length / 2) / s.nel, Wd = s.Wd, seen = new Set();
        let N = 0; for (let i = 0; i < Wd.length; i++) if (Wd[i] >= N) N = Wd[i] + 1;
        // in 3D only draw edges of visible faces: endpoints on the visible surface
        let onSurf = null;
        if (s.mode === '3d') { onSurf = new Uint8Array(N); for (let i = 0; i < tris.length; i++) onSurf[Wd[tris[i]]] = 1; }
        for (let k = 0; k < E.length / 2; k++) {
          if (!keep[(k / epe) | 0]) continue;
          const a = E[2 * k], b = E[2 * k + 1], wa = Wd[a], wb = Wd[b];
          if (onSurf && !(onSurf[wa] && onSurf[wb])) continue;
          const key = wa < wb ? wa * N + wb : wb * N + wa;
          if (seen.has(key)) continue;
          seen.add(key);
          edges.push(a, b);
        }
      }
      s.edgesIdx = new Uint32Array(edges);
      if (this.useGL) { this.gl.setTris(s.tris); this.gl.setEdges(s.edgesIdx); }
    }

    /* ----- view ----- */
    center() {
      const s = this.scene, wh = this.warpHeight();
      return [(s.lo[0] + s.hi[0]) / 2, (s.lo[1] + s.hi[1]) / 2, (s.lo[2] + s.hi[2]) / 2 + wh / 2];
    }
    fitView(reset) {
      const s = this.scene, v = this.view;
      if (s.mode === '1d') return;
      if (reset) { v.zoom = 1; v.panX = 0; v.panY = 0; }
      const wh = this.warpHeight(), R = v.R, c = this.center();
      let mx = 1e-12, my = 1e-12;
      for (let i = 0; i < 8; i++) {
        const x = (i & 1 ? s.hi[0] : s.lo[0]) - c[0], y = (i & 2 ? s.hi[1] : s.lo[1]) - c[1], z = (i & 4 ? s.hi[2] + wh : s.lo[2]) - c[2];
        mx = Math.max(mx, Math.abs(R[0] * x + R[1] * y + R[2] * z));
        my = Math.max(my, Math.abs(R[3] * x + R[4] * y + R[5] * z));
      }
      const d = this.dpr, padTop = 40 * d, padSide = (s.hasField ? 90 : 24) * d;
      v.base = Math.min((this.W - 2 * padSide) / (2 * mx), (this.H - 2 * padTop) / (2 * my));
      if (!(v.base > 0)) v.base = Math.min(this.W, this.H) / (2 * s.radius);
    }
    resize(force) {
      const d = Math.min(2, root.devicePixelRatio || 1);
      const w = Math.max(1, Math.round(this.stage.clientWidth * d)), h = Math.max(1, Math.round(this.stage.clientHeight * d));
      if (force || w !== this.W || h !== this.H || d !== this.dpr) {
        this.dpr = d; this.W = w; this.H = h;
        for (const c of [this.glCanvas, this.cv]) { if (c.width !== w) c.width = w; if (c.height !== h) c.height = h; }
        if (this.scene) this.fitView(false);
        this.projDirty = true;
        this.requestRender();
      }
    }
    requestRender() {
      if (this.renderQueued || this.destroyed) return;
      this.renderQueued = true;
      (root.requestAnimationFrame || setTimeout)(() => this.render());
    }
    state() {
      const s = this.scene, st = this.st, v = this.view;
      const dark = cssRgb(this.root, '--skv-canvas').reduce((a, b) => a + b, 0) < 300;
      return {
        W: this.W, H: this.H, dpr: this.dpr, R: v.R, center: this.center(), scale: v.base * v.zoom, panX: v.panX, panY: v.panY,
        depth: 20 * s.radius + 4 * this.warpHeight(), def: this.defScale(), warp: this.warpHeight(),
        vmin: st.vmin, vmax: st.vmax, bands: st.bands, hasField: s.hasField, edges: st.edges, lineField: s.mode === 'curve' && s.hasField,
        bg: cssRgb(this.root, '--skv-canvas'), face: cssRgb(this.root, '--skv-face'),
        edgeColor: s.hasField ? [0, 0, 0] : (dark ? [235, 242, 248] : [10, 16, 24]),
        edgeAlpha: s.hasField ? (dark ? 0.4 : 0.3) : 0.55,
      };
    }
    project(st) {
      // CPU projection to device pixels (software renderer and picking)
      const s = this.scene, n = s.n, P = s.P3, D = s.D, V = s.V, R = st.R, sc = st.scale;
      const cx = st.W / 2 + st.panX, cy = st.H / 2 + st.panY, c = st.center, ds = st.def, wh = st.warp;
      const ir = 1 / (st.vmax - st.vmin);
      for (let i = 0; i < n; i++) {
        let x = P[3 * i] + ds * D[3 * i] - c[0], y = P[3 * i + 1] + ds * D[3 * i + 1] - c[1], z = P[3 * i + 2] + ds * D[3 * i + 2] - c[2];
        if (wh) z += (V[i] - st.vmin) * ir * wh;
        s.sx[i] = cx + sc * (R[0] * x + R[1] * y + R[2] * z);
        s.sy[i] = cy - sc * (R[3] * x + R[4] * y + R[5] * z);
        s.sz[i] = sc * (R[6] * x + R[7] * y + R[8] * z);
      }
      this.projDirty = false;
      this.projKey = this.viewKey(st);
    }
    viewKey(st) { return [st.R.join(','), st.scale, st.panX, st.panY, st.W, st.H, st.def, st.warp].join('|'); }
    render() {
      this.renderQueued = false;
      if (!this.scene || this.destroyed) return;
      if (this.scene.mode === '1d') return this.render1d();
      const st = this.state();
      if (this.useGL) this.gl.draw(st);
      else { this.project(st); this.sw.draw(st, this.scene); }
    }

    /* ----- 1D plots ----- */
    render1d() {
      const s = this.scene, st = this.st, ctx = this.cv.getContext('2d'), d = this.dpr;
      const W = this.W, H = this.H, bg = cssRgb(this.root, '--skv-canvas');
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = `rgb(${bg})`; ctx.fillRect(0, 0, W, H);
      ctx.setTransform(d, 0, 0, d, 0, 0);
      const w = W / d, h = H / d, L = 64, Rm = 24, T = 44, B = 72;
      let ylo = -1, yhi = 1;
      if (s.hasField) { const pad = 0.08 * (st.vmax - st.vmin); ylo = st.vmin - pad; yhi = st.vmax + pad; }
      const x0 = this.view.x0, x1 = this.view.x1;
      const X = (x) => L + (x - x0) / (x1 - x0) * (w - L - Rm);
      const Y = (y) => h - B - (y - ylo) / (yhi - ylo) * (h - T - B);
      this.plot1d = { L, Rm, w, x0, x1 };
      const css = (n) => getComputedStyle(this.root).getPropertyValue(n).trim();
      const ink = css('--skv-ink'), muted = css('--skv-muted'), line = css('--skv-line'), accent = css('--skv-accent');
      ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace'; ctx.lineWidth = 1;
      ctx.strokeStyle = line; ctx.fillStyle = muted; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (const t of niceTicks(x0, x1, Math.max(3, Math.floor((w - L - Rm) / 90)))) {
        const px = Math.round(X(t)) + 0.5;
        ctx.beginPath(); ctx.moveTo(px, T); ctx.lineTo(px, h - B); ctx.stroke(); ctx.fillText(fmt(t), px, h - B + 10);
      }
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      if (s.hasField) for (const t of niceTicks(ylo, yhi, Math.max(3, Math.floor((h - T - B) / 50)))) {
        const py = Math.round(Y(t)) + 0.5;
        ctx.beginPath(); ctx.moveTo(L, py); ctx.lineTo(w - Rm, py); ctx.stroke(); ctx.fillText(fmt(t), L - 8, py);
      }
      ctx.save(); ctx.beginPath(); ctx.rect(L, T - 8, w - L - Rm, h - T - B + 16); ctx.clip();
      const P = s.P3, C = s.C, nvc = s.nvc, V = s.V, nl = s.nloc, ya = h - B;
      ctx.strokeStyle = ink; ctx.globalAlpha = 0.6; ctx.beginPath();
      for (let e = 0; e < s.nel; e++) for (const j of [0, 1]) { const px = Math.round(X(P[3 * (e * nl + j)])) + 0.5; ctx.moveTo(px, ya - 4); ctx.lineTo(px, ya + 4); }
      ctx.stroke(); ctx.globalAlpha = 1;
      ctx.strokeStyle = accent; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.beginPath();
      for (let c = 0; c < C.length; c += nvc) { const a = C[c], b = C[c + 1]; ctx.moveTo(X(P[3 * a]), Y(s.hasField ? V[a] : 0)); ctx.lineTo(X(P[3 * b]), Y(s.hasField ? V[b] : 0)); }
      ctx.stroke();
      if (st.edges) {
        ctx.fillStyle = accent;
        for (let e = 0; e < s.nel; e++) for (const j of [0, 1]) { const i = e * nl + j; ctx.beginPath(); ctx.arc(X(P[3 * i]), Y(s.hasField ? V[i] : 0), 2.4, 0, 2 * Math.PI); ctx.fill(); }
      }
      ctx.restore();
      if (this.hover1d && s.hasField) {
        const px = X(this.hover1d.x);
        ctx.strokeStyle = muted; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(px, T); ctx.lineTo(px, h - B); ctx.stroke(); ctx.setLineDash([]);
        if (this.hover1d.v !== null) { ctx.fillStyle = ink; ctx.beginPath(); ctx.arc(px, Y(this.hover1d.v), 3.5, 0, 2 * Math.PI); ctx.fill(); }
      }
      ctx.fillStyle = muted; ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText('x', (L + w - Rm) / 2, h - B + 26);
      ctx.setTransform(1, 0, 0, 1, 0, 0);
    }
    sample1d(x) {
      const s = this.scene, C = s.C, P = s.P3, nvc = s.nvc;
      for (let c = 0; c < C.length; c += nvc) {
        const a = C[c], b = C[c + 1], xa = P[3 * a], xb = P[3 * b];
        if ((x - xa) * (x - xb) <= 0 && xa !== xb) return s.V[a] + (x - xa) / (xb - xa) * (s.V[b] - s.V[a]);
      }
      return null;
    }

    /* ----- picking ----- */
    pick(px, py) {
      const s = this.scene, T = s.tris;
      if (!T || !T.length || T.length > 6e6) return null;
      const st = this.state();
      if (this.projDirty || this.projKey !== this.viewKey(st)) this.project(st);
      const sx = s.sx, sy = s.sy, sz = s.sz;
      let best = null, bz = -Infinity;
      for (let t = 0; t < T.length; t += 3) {
        const a = T[t], b = T[t + 1], d = T[t + 2];
        const x0 = sx[a], y0 = sy[a], x1 = sx[b], y1 = sy[b], x2 = sx[d], y2 = sy[d];
        if (px < Math.min(x0, x1, x2) || px > Math.max(x0, x1, x2) || py < Math.min(y0, y1, y2) || py > Math.max(y0, y1, y2)) continue;
        const area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        if (Math.abs(area) < 1e-12) continue;
        const w1 = ((px - x0) * (y2 - y0) - (x2 - x0) * (py - y0)) / area, w2 = ((x1 - x0) * (py - y0) - (px - x0) * (y1 - y0)) / area, w0 = 1 - w1 - w2;
        if (w0 < -1e-6 || w1 < -1e-6 || w2 < -1e-6) continue;
        const z = w0 * sz[a] + w1 * sz[b] + w2 * sz[d];
        if (z > bz) {
          bz = z;
          const P = s.P3, V = s.V;
          best = { v: w0 * V[a] + w1 * V[b] + w2 * V[d], x: [0, 1, 2].map((k) => w0 * P[3 * a + k] + w1 * P[3 * b + k] + w2 * P[3 * d + k]) };
        }
      }
      return best;
    }

    /* ----- overlays ----- */
    updateInfo() {
      const s = this.scene, st = this.st, m = s.meta, o = this.options || {};
      const lines = [];
      if (o.title) lines.push('<b>' + esc(o.title) + '</b>');
      if (m.mesh) lines.push((o.title ? '' : '<b>' + esc(m.mesh) + '</b> · ') + (m.nelements || s.nel).toLocaleString() + ' elements' + (m.nrefs ? ' · nrefs ' + m.nrefs : ''));
      if (s.hasField) {
        const f = s.fields[st.field], en = m.elements && m.elements[f.name];
        lines.push('<code>' + esc(f.name) + '</code>' + (en ? ' · ' + esc(en) : '') + ' · min ' + fmt(st.dmin) + ' · max ' + fmt(st.dmax));
      }
      if (st.deform >= 0) lines.push('deformation ×' + fmt(this.defScale()));
      this.info.innerHTML = lines.join('<br>');
      this.info.hidden = o.info === false || !lines.length;
      this.hint.hidden = o.hint === false;
      this.hint.textContent = s.mode === '1d' ? 'wheel zoom · drag pan · double-click reset'
        : this.isFlat() ? 'drag pan · wheel zoom · alt-drag rotate · double-click reset'
        : 'drag rotate · shift-drag pan · wheel zoom · double-click reset';
    }
    drawColorbar() {
      const s = this.scene, st = this.st, o = this.options || {};
      const on = s.hasField && s.mode !== '1d' && o.colorbar !== false;
      this.cbar.hidden = !on;
      if (!on) return;
      const c = this.cbar.querySelector('canvas'), g = c.getContext('2d'), h = c.height, nb = st.bands, lut = this.lut;
      for (let y = 0; y < h; y++) {
        let q = 1 - y / (h - 1);
        if (nb > 1) q = Math.min(nb - 1, Math.floor(q * nb)) / (nb - 1);
        const k = 4 * Math.round(q * 255);
        g.fillStyle = `rgb(${lut[k]},${lut[k + 1]},${lut[k + 2]})`; g.fillRect(0, y, c.width, 1);
      }
      const ticks = this.cbar.querySelector('.skv-ticks'); ticks.innerHTML = '';
      for (let i = 0; i <= 4; i++) {
        let v = st.vmax - (st.vmax - st.vmin) * i / 4;
        if (Math.abs(v) < 1e-9 * (st.vmax - st.vmin)) v = 0;
        const sp = this.doc.createElement('span'); sp.style.top = (i * 25) + '%'; sp.textContent = fmt(v); ticks.appendChild(sp);
      }
    }
    // a planar mesh viewed from above: drag pans.  Deformation keeps the mesh in the plane
    // because the displacement has as many components as the mesh has dimensions.
    isFlat() { const s = this.scene; return s.mode === '2d' && !this.st.warp && Math.abs(this.view.R[8] - 1) < 1e-9; }

    /* ----- export ----- */
    png(opts) {
      opts = opts || {};
      if (this.scene.mode === '1d') { this.render1d(); return this.cv.toDataURL('image/png'); }
      const st = this.state();
      let src;
      if (this.useGL) { this.gl.draw(st); src = this.glCanvas; }
      else { this.project(st); this.sw.draw(st, this.scene); src = this.cv; }
      const withBar = opts.colorbar !== false && !this.cbar.hidden;
      if (!withBar) return src.toDataURL('image/png');
      // composite the colorbar onto a copy of the rendered image
      const out = this.doc.createElement('canvas');
      out.width = this.W; out.height = this.H;
      const g = out.getContext('2d');
      g.drawImage(src, 0, 0);
      this.drawColorbarOn(g);
      return out.toDataURL('image/png');
    }

    drawColorbarOn(g) {
      // same layout as the on-screen colorbar (right 12px, bar 14px, gap 6px, ticks 52px)
      const s = this.scene, st = this.st, d = this.dpr, W = this.W, H = this.H, lut = this.lut, nb = st.bands;
      const css = (n) => getComputedStyle(this.root).getPropertyValue(n).trim();
      const ink = css('--skv-ink'), line = css('--skv-line'), panel = css('--skv-panel');
      const bh = Math.round(Math.min(0.6 * H, 320 * d)), bw = Math.round(14 * d);
      const bx = Math.round(W - (12 + 52 + 6 + 14) * d), by = Math.round((H - bh) / 2);
      for (let y = 0; y < bh; y++) {
        let q = 1 - y / (bh - 1);
        if (nb > 1) q = Math.min(nb - 1, Math.floor(q * nb)) / (nb - 1);
        const k = 4 * Math.round(q * 255);
        g.fillStyle = `rgb(${lut[k]},${lut[k + 1]},${lut[k + 2]})`;
        g.fillRect(bx, by + y, bw, 1);
      }
      g.strokeStyle = line; g.lineWidth = d; g.strokeRect(bx + 0.5 * d, by + 0.5 * d, bw - d, bh - d);
      g.font = (10.5 * d) + 'px ui-monospace, SFMono-Regular, Menlo, monospace';
      g.textBaseline = 'middle'; g.textAlign = 'left';
      const tx = bx + bw + 6 * d;
      for (let i = 0; i <= 4; i++) {
        let v = st.vmax - (st.vmax - st.vmin) * i / 4;
        if (Math.abs(v) < 1e-9 * (st.vmax - st.vmin)) v = 0;
        const txt = fmt(v), ty = by + bh * i / 4, tw = g.measureText(txt).width;
        g.fillStyle = panel; g.fillRect(tx - 3 * d, ty - 7 * d, tw + 6 * d, 14 * d);
        g.fillStyle = ink; g.fillText(txt, tx, ty);
      }
      // field name (and component) above the bar
      const f = s.fields[st.field];
      let label = f.name;
      if (f.ncomp > 1) { const cs = this.c('comp'); label += ' (' + (cs.selectedIndex >= 0 ? cs.options[cs.selectedIndex].text : st.comp) + ')'; }
      g.font = (11 * d) + 'px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
      g.textAlign = 'right'; g.textBaseline = 'bottom';
      const lx = Math.min(W - 6 * d, bx + bw + 52 * d), lw = g.measureText(label).width;
      g.fillStyle = panel; g.fillRect(lx - lw - 4 * d, by - 22 * d, lw + 8 * d, 16 * d);
      g.fillStyle = ink; g.fillText(label, lx, by - 8 * d);
    }

    /* ----- events ----- */
    wire() {
      const on = (name, ev, fn) => this.c(name).addEventListener(ev, fn);
      const regeo = () => { this.prepareGeometry(); this.requestRender(); };
      on('field', 'change', (e) => { this.st.field = +e.target.value; this.st.vminUser = this.st.vmaxUser = undefined; this.fillComponents(); this.computeValues(); this.updateInfo(); this.drawColorbar(); this.fitView(false); this.requestRender(); });
      on('comp', 'change', (e) => { this.st.comp = e.target.value; this.st.vminUser = this.st.vmaxUser = undefined; this.computeValues(); this.updateInfo(); this.drawColorbar(); this.requestRender(); });
      on('cmap', 'change', (e) => { this.st.cmap = e.target.value; this.lut = buildLut(this.st.cmap); if (this.gl) this.gl.setLut(this.lut); if (this.sw) this.sw.setLut(this.lut); this.drawColorbar(); this.requestRender(); });
      on('bands', 'change', (e) => { this.st.bands = +e.target.value; this.drawColorbar(); this.requestRender(); });
      on('edges', 'change', (e) => { this.st.edges = e.target.checked; this.requestRender(); });
      on('warp', 'change', (e) => {
        this.st.warp = e.target.checked;
        this.view.R = this.st.warp ? mul(rotX(-0.95), rotZ(-0.5)) : VIEWS.xy();
        this.fitView(true); this.updateInfo(); this.requestRender();
      });
      on('warpAmt', 'input', (e) => { this.st.warpAmt = e.target.value / 100; this.requestRender(); });
      on('deform', 'change', (e) => { this.st.deform = +e.target.value; this.computeDisp(); this.updateInfo(); this.requestRender(); });
      on('defAmt', 'input', (e) => { this.st.defAmt = e.target.value / 100; this.updateInfo(); this.requestRender(); });
      on('clip', 'change', (e) => {
        this.st.clipAxis = +e.target.value;
        if (this.st.clipAxis >= 0) { this.st.clipFlip = this.view.R[6 + this.st.clipAxis] < 0; this.c('clipFlip').checked = this.st.clipFlip; }
        regeo();
      });
      on('clipFlip', 'change', (e) => { this.st.clipFlip = e.target.checked; regeo(); });
      let clipQueued = false;
      on('clipPos', 'input', (e) => {
        this.st.clipPos = e.target.value / 1000;
        if (this.st.clipAxis < 0 || clipQueued) return;
        clipQueued = true;
        requestAnimationFrame(() => { clipQueued = false; regeo(); });
      });
      on('png', 'click', () => {
        const a = this.doc.createElement('a');
        a.href = this.png(); a.download = ((this.options && this.options.title) || 'skfem') + '.png';
        this.doc.body.appendChild(a); a.click(); a.remove();
      });
      this.root.querySelectorAll('[data-view]').forEach((b) => b.addEventListener('click', () => {
        this.view.R = VIEWS[b.dataset.view](); this.fitView(true); this.updateInfo(); this.requestRender();
      }));

      const pointers = new Map();
      let drag = null, pinch = null;
      for (const cv of [this.glCanvas, this.cv]) {
        cv.addEventListener('pointerdown', (e) => {
          if (!this.scene) return;
          cv.setPointerCapture(e.pointerId);
          pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
          if (pointers.size === 2) {
            const [a, b] = [...pointers.values()];
            pinch = { d: Math.hypot(a.x - b.x, a.y - b.y), zoom: this.view.zoom, x0: this.view.x0, x1: this.view.x1 };
            drag = null; return;
          }
          const pan = e.shiftKey || e.button === 1 || e.button === 2 || e.ctrlKey || (this.isFlat() && !e.altKey) || this.scene.mode === '1d';
          drag = { x: e.clientX, y: e.clientY, pan };
          cv.classList.add('skv-drag');
        });
        cv.addEventListener('pointermove', (e) => {
          if (!this.scene) return;
          const v = this.view;
          if (pointers.has(e.pointerId)) pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
          if (pinch && pointers.size === 2) {
            const [a, b] = [...pointers.values()], f = Math.hypot(a.x - b.x, a.y - b.y) / (pinch.d || 1);
            if (this.scene.mode === '1d') { const mid = (pinch.x0 + pinch.x1) / 2, hw = (pinch.x1 - pinch.x0) / 2 / f; v.x0 = mid - hw; v.x1 = mid + hw; }
            else v.zoom = pinch.zoom * f;
            this.requestRender(); return;
          }
          if (drag) {
            const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
            drag.x = e.clientX; drag.y = e.clientY;
            if (this.scene.mode === '1d') {
              const p = this.plot1d, span = v.x1 - v.x0, wpx = p ? p.w - p.L - p.Rm : 400;
              v.x0 -= dx / wpx * span; v.x1 -= dx / wpx * span;
            } else if (drag.pan) { v.panX += dx * this.dpr; v.panY += dy * this.dpr; }
            else { v.R = mul(rotX(dy * 0.008), mul(rotY(dx * 0.008), v.R)); }
            this.requestRender();
            return;
          }
          this.hover(e, cv);
        });
        const end = (e) => {
          pointers.delete(e.pointerId);
          if (pointers.size < 2) pinch = null;
          if (!pointers.size) { drag = null; cv.classList.remove('skv-drag'); this.updateInfo(); }
        };
        cv.addEventListener('pointerup', end);
        cv.addEventListener('pointercancel', end);
        cv.addEventListener('pointerleave', () => { this.readout.hidden = true; if (this.hover1d) { this.hover1d = null; this.requestRender(); } });
        cv.addEventListener('contextmenu', (e) => e.preventDefault());
        cv.addEventListener('wheel', (e) => {
          if (!this.scene) return;
          e.preventDefault();
          const v = this.view, f = Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015)), r = cv.getBoundingClientRect();
          if (this.scene.mode === '1d') {
            const p = this.plot1d; if (!p) return;
            const xm = p.x0 + (e.clientX - r.left - p.L) / (p.w - p.L - p.Rm) * (p.x1 - p.x0);
            v.x0 = xm - (xm - v.x0) / f; v.x1 = xm + (v.x1 - xm) / f;
          } else {
            const mx = (e.clientX - r.left) * this.dpr - this.W / 2, my = (e.clientY - r.top) * this.dpr - this.H / 2;
            v.panX = mx - (mx - v.panX) * f; v.panY = my - (my - v.panY) * f; v.zoom *= f;
          }
          this.requestRender();
        }, { passive: false });
        cv.addEventListener('dblclick', () => {
          const s = this.scene, v = this.view;
          if (s.mode === '1d') { v.x0 = s.lo[0]; v.x1 = s.hi[0]; }
          else { v.R = (s.mode === '3d' || s.mode === 'surf') ? VIEWS.iso() : this.st.warp ? mul(rotX(-0.95), rotZ(-0.5)) : VIEWS.xy(); this.fitView(true); }
          this.updateInfo(); this.requestRender();
        });
      }
    }
    hover(e, cv) {
      const s = this.scene, r = cv.getBoundingClientRect(), ro = this.readout;
      if (s.mode === '1d') {
        const p = this.plot1d; if (!p || !s.hasField) return;
        const x = p.x0 + (e.clientX - r.left - p.L) / (p.w - p.L - p.Rm) * (p.x1 - p.x0), v = this.sample1d(x);
        this.hover1d = { x, v };
        ro.hidden = false; ro.textContent = 'x = ' + fmt(x) + (v !== null ? '   value = ' + fmt(v) : '');
        this.requestRender(); return;
      }
      if (this.hoverQueued) return;
      this.hoverQueued = true;
      const px = (e.clientX - r.left) * this.dpr, py = (e.clientY - r.top) * this.dpr;
      requestAnimationFrame(() => {
        this.hoverQueued = false;
        const hit = this.pick(px, py);
        if (!hit) { ro.hidden = true; return; }
        const coords = 'xyz'.slice(0, s.gdim).split('').map((c, k) => c + ' = ' + fmt(hit.x[k])).join('   ');
        ro.textContent = coords + (s.hasField ? '   value = ' + fmt(hit.v) : '');
        ro.hidden = false;
      });
    }

    destroy() {
      this.destroyed = true;
      this.ro.disconnect();
      if (this.mql && this.mql.removeEventListener) this.mql.removeEventListener('change', this.onTheme);
      if (this.gl) this.gl.destroy();
      this.root.remove();
    }
  }

  function mount(host, data, options) {
    if (typeof host === 'string') host = document.querySelector(host);
    const v = new Viewer(host, data, options || {});
    return {
      viewer: v,
      update: (d, o) => v.update(d, o),
      render: () => v.render(),
      png: (o) => v.png(o),
      get renderer() { return v.rendererName; },
      destroy: () => v.destroy(),
    };
  }

  root.SkfemViewer = { mount, version: '1', colormaps: Object.keys(CMAPS) };
})(typeof window !== 'undefined' ? window : globalThis);

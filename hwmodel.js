/* 隨手掃描 — 本機 AI 去手寫（小型 U-Net，onnxruntime-web） */
(function (root) {
'use strict';
const MODEL_W = 1000;   // the model was trained on pages ~1000 px wide
let session = null, loading = null;

function loadScript(src){ return new Promise((res, rej) => { const s = document.createElement('script'); s.src = src; s.onload = res; s.onerror = () => rej(new Error('load ' + src)); document.head.appendChild(s); }); }
async function load(onStatus){
  if (session) return session;
  if (loading) return loading;
  loading = (async () => {
    onStatus && onStatus('載入 AI 模型…');
    if (!root.ort) await loadScript('ort.wasm.min.js');
    root.ort.env.wasm.wasmPaths = './';
    root.ort.env.wasm.numThreads = 1;
    try { root.ort.env.wasm.simd = true; } catch (e) {}
    const resp = await fetch('hw.model.json'); if (!resp.ok) throw new Error('model fetch ' + resp.status);
    const b64 = (await resp.json()).b64, bin = atob(b64), buf = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
    session = await root.ort.InferenceSession.create(buf, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    return session;
  })();
  try { return await loading; } catch (e){ loading = null; throw e; }
}

/* whole-stroke decisions: a pen stroke is one connected blob of ink. If the model marks most of a blob,
 * take the whole blob (no half-erased letters left behind); if it marks only a sliver of a blob, leave the
 * blob alone (printed characters the model brushed against stay intact). */
const PRINT_SURE = -2.5; // logit below this: the model is confident the ink is print (p < 0.08)
function completeStrokes(mask, rgba, w, h, logits){
  const n = w * h, ink = new Uint8Array(n);
  for (let i = 0; i < n; i++){
    const r = rgba[i * 4], g = rgba[i * 4 + 1], b = rgba[i * 4 + 2];
    const mn = Math.min(r, g, b), mx = Math.max(r, g, b);
    if (mn < 150 || mx - mn > 60) ink[i] = 1;
  }
  // connected components (4-neighbour, two-pass union-find on a label array)
  const lab = new Int32Array(n), parent = [0];
  const find = a => { while (parent[a] !== a){ parent[a] = parent[parent[a]]; a = parent[a]; } return a; };
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++){
    const p = y * w + x; if (!ink[p]) continue;
    const up = y > 0 && ink[p - w] ? lab[p - w] : 0, left = x > 0 && ink[p - 1] ? lab[p - 1] : 0;
    if (!up && !left){ lab[p] = parent.length; parent.push(parent.length); }
    else if (up && left){ const a = find(up), b = find(left); lab[p] = a; if (a !== b) parent[b] = a; }
    else lab[p] = up || left;
  }
  const size = new Int32Array(parent.length), hit = new Int32Array(parent.length);
  const bx0 = new Int32Array(parent.length).fill(w), bx1 = new Int32Array(parent.length).fill(-1), by0 = new Int32Array(parent.length).fill(h), by1 = new Int32Array(parent.length).fill(-1);
  for (let i = 0; i < n; i++){
    if (!ink[i]) continue; const r = find(lab[i]); lab[i] = r; size[r]++; if (mask[i]) hit[r]++;
    const x = i % w, y = (i / w) | 0; if (x < bx0[r]) bx0[r] = x; if (x > bx1[r]) bx1[r] = x; if (y < by0[r]) by0[r] = y; if (y > by1[r]) by1[r] = y;
  }
  // longest straight horizontal / vertical run of ink in each blob: axes, table rules and frames have them, pen strokes do not
  const runMax = new Int32Array(parent.length);
  for (let y = 0; y < h; y++){ let len = 0; for (let x = 0; x <= w; x++){ const p = y * w + x; if (x < w && ink[p]) len++; else { if (len > 0){ const r = lab[p - 1]; if (len > runMax[r]) runMax[r] = len; } len = 0; } } }
  for (let x = 0; x < w; x++){ let len = 0; for (let y = 0; y <= h; y++){ const p = y * w + x; if (y < h && ink[p]) len++; else { if (len > 0){ const r = lab[p - w]; if (len > runMax[r]) runMax[r] = len; } len = 0; } } }
  const ruled = Math.round(w * 0.04);
  // a filled upright rectangle (section bar, table rule, black box) is print, whatever the model thinks
  const solid = r => { const bw = bx1[r] - bx0[r] + 1, bh = by1[r] - by0[r] + 1; return Math.min(bw, bh) >= 2 && Math.max(bw, bh) >= 8 && size[r] >= 0.8 * bw * bh; };
  const big = n * 0.02; // a blob this large is a frame or table, never a single stroke: leave the model's pixels as they are
  let added = 0, cleared = 0, before = 0;
  for (let i = 0; i < n; i++) before += mask[i];
  for (let i = 0; i < n; i++){
    if (!ink[i]) continue;
    const r = lab[i], f = hit[r] / size[r];
    if (size[r] > big) continue;
    if (solid(r) || (runMax[r] >= ruled && f < 0.5)){ mask[i] = 0; continue; } // a drawing with ruled lines (graph, table) that is not mostly pen: print
    if (f >= 0.75){ if (!mask[i] && logits[i] > PRINT_SURE){ added++; mask[i] = 1; } }  // take the rest of the stroke, but not a printed character it touches
    else if (f < 0.12 && size[r] > 12){ if (mask[i]) cleared++; mask[i] = 0; } // a brush against print: leave it
    else if (f < 0.3 && mask[i] && logits[i] < 1.0){ cleared++; mask[i] = 0; }  // mostly print: only what the model is really sure about goes
  }
  // Print comes first. Where a pen stroke runs across a printed character, erasing the stroke cuts the character.
  // Take the ink the model is sure is print, close the small gaps in it (the width of a pen stroke), and keep whatever
  // marked ink lies in those gaps: the character stays whole and at worst a speck of pen stays inside it.
  const P = new Uint8Array(n), tmp = new Uint8Array(n), D = new Uint8Array(n), C = new Uint8Array(n);
  for (let i = 0; i < n; i++) P[i] = ink[i] && !mask[i] && logits[i] < PRINT_SURE ? 1 : 0;
  const R = 2; let thick = 0;
  const pass = (src, dst, horizontal, wantAll) => { // separable min/max filter of radius R
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++){
      let v = wantAll ? 1 : 0;
      for (let k = -R; k <= R; k++){
        const X = horizontal ? x + k : x, Y = horizontal ? y : y + k;
        const s = X < 0 || Y < 0 || X >= w || Y >= h ? (wantAll ? 1 : 0) : src[Y * w + X];
        if (wantAll){ if (!s){ v = 0; break; } } else if (s){ v = 1; break; }
      }
      dst[y * w + x] = v;
    }
  };
  // Thick ink is print: a pen line is a few pixels wide here, so anything that survives a 5×5 erosion (bold type,
  // brackets, section bars, black boxes) is not a pen stroke, however the model scored it.
  {
    const T = new Uint8Array(n), O = new Uint8Array(n);
    pass(ink, tmp, true, true); pass(tmp, T, false, true);     // erode
    pass(T, tmp, true, false); pass(tmp, O, false, false);     // dilate → opening
    const seen = new Uint8Array(n), stack = [];
    for (let s0 = 0; s0 < n; s0++){
      if (!O[s0] || seen[s0]) continue;
      const comp = []; let x0 = w, x1 = -1, y0 = h, y1 = -1; stack.push(s0); seen[s0] = 1;
      while (stack.length){
        const p = stack.pop(); comp.push(p); const x = p % w, y = (p / w) | 0;
        if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
        if (x > 0 && O[p - 1] && !seen[p - 1]){ seen[p - 1] = 1; stack.push(p - 1); }
        if (x < w - 1 && O[p + 1] && !seen[p + 1]){ seen[p + 1] = 1; stack.push(p + 1); }
        if (y > 0 && O[p - w] && !seen[p - w]){ seen[p - w] = 1; stack.push(p - w); }
        if (y < h - 1 && O[p + w] && !seen[p + w]){ seen[p + w] = 1; stack.push(p + w); }
      }
      if (comp.length >= 25 && comp.length >= 0.45 * (x1 - x0 + 1) * (y1 - y0 + 1)) for (const p of comp) if (mask[p]){ mask[p] = 0; thick++; }
    }
  }
  for (let i = 0; i < n; i++) P[i] = ink[i] && !mask[i] && logits[i] < PRINT_SURE ? 1 : 0;
  pass(P, tmp, true, false); pass(tmp, D, false, false);   // dilate
  pass(D, tmp, true, true); pass(tmp, C, false, true);     // erode → closing
  let kept = 0;
  const protect = new Uint8Array(n);
  for (let i = 0; i < n; i++){
    if (mask[i] && ink[i] && C[i]){ mask[i] = 0; kept++; }
    if (ink[i] && !mask[i]) protect[i] = 1;                // growing the mask must not bite into ink that stays
  }
  root.HWModel && (root.HWModel.lastStats = { w, h, before, added, cleared, kept, thick, comps: parent.length - 1 });
  return protect;
}

function downscale(img, w, h){
  const src = document.createElement('canvas'); src.width = img.width; src.height = img.height;
  src.getContext('2d').putImageData(img instanceof ImageData ? img : new ImageData(img.data, img.width, img.height), 0, 0);
  const sm = document.createElement('canvas'); sm.width = w; sm.height = h;
  const sx = sm.getContext('2d', { willReadFrequently: true }); sx.imageSmoothingQuality = 'high'; sx.drawImage(src, 0, 0, w, h);
  const d = sx.getImageData(0, 0, w, h).data; src.width = 0; sm.width = 0; return d;
}
/**
 * img: {data,width,height} RGBA — the straightened photo BEFORE cleaning (the model was trained on photos, not on
 *      cleaned scans; on a pure-white cleaned page it finds almost nothing)
 * cleaned: optional {data,width,height} of the cleaned page, same size — used to decide where the ink is
 * returns a canvas of the same size: white where handwriting, transparent elsewhere
 */
async function handwritingMask(img, onStatus, cleaned){
  const sess = await load(onStatus);
  onStatus && onStatus('AI 正在找手寫…');
  const W = img.width, H = img.height;
  const O_ = root.__hwOpt || {};
  const scale = Math.min(O_.up || 1.6, MODEL_W / Math.min(W, H)); // short side (page width) ≈ 1000, as in training; small photos are enlarged
  const w = Math.max(8, Math.floor(W * scale / 8) * 8), h = Math.max(8, Math.floor(H * scale / 8) * 8);
  const d = downscale(img, w, h);
  const inkSrc = cleaned && cleaned.width === W && cleaned.height === H ? downscale(cleaned, w, h) : d;
  const n = w * h, x = new Float32Array(3 * n);
  for (let i = 0; i < n; i++){ x[i] = d[i * 4] / 255; x[n + i] = d[i * 4 + 1] / 255; x[2 * n + i] = d[i * 4 + 2] / 255; }
  const input = new root.ort.Tensor('float32', x, [1, 3, h, w]);
  const out = await sess.run({ input });
  const logits = out[Object.keys(out)[0]].data;
  // mask at model resolution → RGBA canvas (white where handwriting), slightly grown
  const mask = new Uint8Array(n);
  for (let i = 0; i < n; i++) mask[i] = logits[i] > (O_.th === undefined ? -0.85 : O_.th) ? 1 : 0;   // p ≈ 0.3
  const raw = root.__hwDebug ? mask.slice() : null;
  const protect = completeStrokes(mask, inkSrc, w, h, logits);
  const grown = new Uint8Array(n);
  for (let yy = 0; yy < h; yy++) for (let xx = 0; xx < w; xx++){
    const p = yy * w + xx; if (!mask[p]) continue;
    for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++){ if (Math.abs(dx) + Math.abs(dy) > 3) continue; const X = xx + dx, Y = yy + dy; if (X >= 0 && Y >= 0 && X < w && Y < h && !protect[Y * w + X]) grown[Y * w + X] = 1; }
  }
  if (root.__hwDebug) root.__hwDebug({ w, h, d, raw, mask, grown, logits });
  const mc = document.createElement('canvas'); mc.width = w; mc.height = h;
  const mid = mc.getContext('2d').createImageData(w, h), md = mid.data;
  let count = 0;
  for (let i = 0; i < n; i++){ if (grown[i]){ md[i * 4] = 255; md[i * 4 + 1] = 255; md[i * 4 + 2] = 255; md[i * 4 + 3] = 255; count++; } }
  mc.getContext('2d').putImageData(mid, 0, 0);
  const full = document.createElement('canvas'); full.width = W; full.height = H;
  const fx = full.getContext('2d'); fx.imageSmoothingEnabled = true; fx.drawImage(mc, 0, 0, W, H);
  mc.width = 0;
  return { canvas: full, fraction: count / n };
}
root.HWModel = { load, handwritingMask, get ready(){ return !!session; } };
})(window);

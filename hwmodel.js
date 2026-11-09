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
function completeStrokes(mask, rgba, w, h){
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
  for (let i = 0; i < n; i++){ if (!ink[i]) continue; const r = find(lab[i]); lab[i] = r; size[r]++; if (mask[i]) hit[r]++; }
  const big = n * 0.02; // a blob this large is a frame or table, never a single stroke: leave the model's pixels as they are
  let added = 0, cleared = 0, before = 0;
  for (let i = 0; i < n; i++) before += mask[i];
  for (let i = 0; i < n; i++){
    if (!ink[i]) continue;
    const r = lab[i], f = hit[r] / size[r];
    if (size[r] > big) continue;
    if (f >= 0.75){ if (!mask[i]) added++; mask[i] = 1; }            // take the rest of the stroke
    else if (f < 0.12 && size[r] > 12){ if (mask[i]) cleared++; mask[i] = 0; } // a brush against print: leave it
  }
  root.HWModel && (root.HWModel.lastStats = { w, h, before, added, cleared, comps: parent.length - 1 });
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
  const scale = Math.min(1, MODEL_W / Math.min(W, H)); // scale so the short side (page width) ≈ 1000
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
  for (let i = 0; i < n; i++) mask[i] = logits[i] > -0.85 ? 1 : 0;   // p ≈ 0.3
  completeStrokes(mask, inkSrc, w, h);
  const grown = new Uint8Array(n);
  for (let yy = 0; yy < h; yy++) for (let xx = 0; xx < w; xx++){
    const p = yy * w + xx; if (!mask[p]) continue;
    for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++){ if (Math.abs(dx) + Math.abs(dy) > 3) continue; const X = xx + dx, Y = yy + dy; if (X >= 0 && Y >= 0 && X < w && Y < h) grown[Y * w + X] = 1; }
  }
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

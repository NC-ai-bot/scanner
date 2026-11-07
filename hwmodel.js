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

/**
 * img: {data,width,height} RGBA (the processed, upright scan)
 * returns a canvas of the same size: white where handwriting, transparent elsewhere
 */
async function handwritingMask(img, onStatus){
  const sess = await load(onStatus);
  onStatus && onStatus('AI 正在找手寫…');
  const W = img.width, H = img.height;
  const k = Math.min(1, MODEL_W / Math.min(W, H)) * (W <= H ? 1 : 1); // scale so the short side (page width) ≈ 1000
  const scale = Math.min(1, MODEL_W / Math.min(W, H));
  const w = Math.max(8, Math.floor(W * scale / 8) * 8), h = Math.max(8, Math.floor(H * scale / 8) * 8);
  // downscale with canvas
  const src = document.createElement('canvas'); src.width = W; src.height = H;
  src.getContext('2d').putImageData(img instanceof ImageData ? img : new ImageData(img.data, W, H), 0, 0);
  const sm = document.createElement('canvas'); sm.width = w; sm.height = h;
  const sx = sm.getContext('2d', { willReadFrequently: true }); sx.imageSmoothingQuality = 'high'; sx.drawImage(src, 0, 0, w, h);
  const d = sx.getImageData(0, 0, w, h).data;
  const n = w * h, x = new Float32Array(3 * n);
  for (let i = 0; i < n; i++){ x[i] = d[i * 4] / 255; x[n + i] = d[i * 4 + 1] / 255; x[2 * n + i] = d[i * 4 + 2] / 255; }
  const input = new root.ort.Tensor('float32', x, [1, 3, h, w]);
  const out = await sess.run({ input });
  const logits = out[Object.keys(out)[0]].data;
  // mask at model resolution → RGBA canvas (white where handwriting), slightly grown
  const mask = new Uint8Array(n);
  for (let i = 0; i < n; i++) mask[i] = logits[i] > -0.85 ? 1 : 0;   // p ≈ 0.3
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
  src.width = 0; sm.width = 0; mc.width = 0;
  return { canvas: full, fraction: count / n };
}
root.HWModel = { load, handwritingMask, get ready(){ return !!session; } };
})(window);

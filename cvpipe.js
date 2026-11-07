/* 隨手掃描 — OpenCV 影像管線（在 Web Worker 或主執行緒都可用）
 * 需要全域 `cv`（opencv.js）已就緒。
 * API：CVPipe.detect(imageData) / CVPipe.warp(imageData, quad, maxSide) / CVPipe.process(imageData, opts)
 */
(function (root) {
'use strict';
let cv = null;
function use(cvModule){ cv = cvModule; }

const DET_SIZE = 640;

/* ---------- helpers ---------- */
function toMat(img){ return cv.matFromImageData(img); }
function matToImageData(m){ // m: RGBA 8U → plain {data,width,height} (works in workers without ImageData)
  return { data: new Uint8ClampedArray(m.data), width: m.cols, height: m.rows };
}
function del(...ms){ for (const m of ms) if (m && !m.isDeleted || (m && m.delete)) { try { m.delete(); } catch (e) {} } }
function dist(a, b){ return Math.hypot(a.x - b.x, a.y - b.y); }
function order(pts){ // tl, tr, br, bl
  const s = pts.map(p => p.x + p.y), d = pts.map(p => p.x - p.y);
  const tl = pts[s.indexOf(Math.min(...s))], br = pts[s.indexOf(Math.max(...s))];
  const tr = pts[d.indexOf(Math.max(...d))], bl = pts[d.indexOf(Math.min(...d))];
  return [tl, tr, br, bl];
}
function polyArea(q){ let s = 0; for (let i = 0; i < q.length; i++){ const a = q[i], b = q[(i+1)%q.length]; s += a.x*b.y - b.x*a.y; } return Math.abs(s)/2; }
function angles(q){
  const out = [];
  for (let i = 0; i < 4; i++){
    const p = q[(i+3)%4], c = q[i], n = q[(i+1)%4];
    const v1 = {x: p.x-c.x, y: p.y-c.y}, v2 = {x: n.x-c.x, y: n.y-c.y};
    const cos = (v1.x*v2.x + v1.y*v2.y) / (Math.hypot(v1.x,v1.y)*Math.hypot(v2.x,v2.y) + 1e-9);
    out.push(Math.acos(Math.max(-1, Math.min(1, cos))) * 180 / Math.PI);
  }
  return out;
}

/* ---------- detection ---------- */
function quadsFromMask(mask, imgArea, w, h, found){
  const contours = new cv.MatVector(), hier = new cv.Mat();
  cv.findContours(mask, contours, hier, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
  const list = [];
  for (let i = 0; i < contours.size(); i++){
    const c = contours.get(i), a = Math.abs(cv.contourArea(c));
    if (a > imgArea * 0.08) list.push({ c, a }); else c.delete();
  }
  list.sort((p, q) => q.a - p.a);
  for (const { c } of list.slice(0, 12)){
    const peri = cv.arcLength(c, true);
    let got = false;
    for (const eps of [0.015, 0.025, 0.04, 0.06, 0.09]){
      const ap = new cv.Mat(); cv.approxPolyDP(c, ap, eps * peri, true);
      if (ap.rows === 4 && cv.isContourConvex(ap)){
        const pts = []; for (let k = 0; k < 4; k++) pts.push({ x: ap.data32S[k*2], y: ap.data32S[k*2+1] });
        found.push(pts); got = true;
      }
      ap.delete(); if (got) break;
    }
    if (!got){ // convex hull → approx
      const hull = new cv.Mat(); cv.convexHull(c, hull, false, true);
      const hp = cv.arcLength(hull, true);
      for (const eps of [0.03, 0.06, 0.1]){
        const ap = new cv.Mat(); cv.approxPolyDP(hull, ap, eps * hp, true);
        if (ap.rows === 4){ const pts = []; for (let k = 0; k < 4; k++) pts.push({ x: ap.data32S[k*2], y: ap.data32S[k*2+1] }); found.push(pts); ap.delete(); break; }
        ap.delete();
      }
      hull.delete();
    }
    c.delete();
  }
  contours.delete(); hier.delete();
}
function edgeSupport(q, edge, w, h){ // fraction of the quad's outline that lies on an image edge (per side, worst side counts most)
  const o = order(q), sides = [];
  for (let i = 0; i < 4; i++){
    const a = o[i], b = o[(i+1)%4], len = dist(a, b), n = Math.max(8, Math.ceil(len / 2));
    let hit = 0;
    for (let k = 0; k <= n; k++){
      const t = k / n, x = Math.round(a.x + (b.x - a.x) * t), y = Math.round(a.y + (b.y - a.y) * t);
      if (x < 0 || y < 0 || x >= w || y >= h) continue;
      if (edge[y * w + x]) hit++;
    }
    sides.push(hit / (n + 1));
  }
  const mean = sides.reduce((p, c) => p + c, 0) / 4, min = Math.min(...sides);
  return 0.7 * mean + 0.3 * min;
}
function pointInQuad(q, x, y){ // q ordered
  let sign = 0;
  for (let i = 0; i < 4; i++){ const a = q[i], b = q[(i+1)%4], cr = (b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x); const s2 = cr > 0 ? 1 : cr < 0 ? -1 : 0; if (!s2) continue; if (!sign) sign = s2; else if (s2 !== sign) return false; }
  return true;
}
function textCoverage(q, pts){ // share of text pixels that fall inside the quad (the paper holds the text)
  if (!pts || pts.length < 40) return 1;
  const o = order(q); let inside = 0; const n = pts.length / 2;
  for (let i = 0; i < pts.length; i += 2) if (pointInQuad(o, pts[i], pts[i+1])) inside++;
  return inside / n;
}
// is the strip just inside the outline as bright as the middle? (paper is uniform; a quad that includes desk is not)
function bandSimilarity(q, gray, w, h){
  const o = order(q), cx = o.reduce((p, c) => p + c.x, 0) / 4, cy = o.reduce((p, c) => p + c.y, 0) / 4;
  const sample = (k) => { // k: 1 = on the outline, 0 = centre
    let sum = 0, n = 0;
    for (let i = 0; i < 4; i++){
      const a = o[i], b = o[(i+1)%4];
      for (let t = 0.1; t <= 0.9; t += 0.1){
        const px = a.x + (b.x - a.x) * t, py = a.y + (b.y - a.y) * t;
        const x = Math.round(cx + (px - cx) * k), y = Math.round(cy + (py - cy) * k);
        if (x < 0 || y < 0 || x >= w || y >= h) continue;
        vals.push(gray[y * w + x]);
      }
    }
  };
  let vals = []; sample(0.6); if (!vals.length) return 0.5;
  const innerVals = vals.slice().sort((p, c) => p - c), ref = innerVals[Math.floor(innerVals.length * 0.7)]; // bright-ish paper level, ignoring ink
  vals = []; sample(0.96); if (!vals.length) return 0.5;
  let ok = 0; for (const v of vals) if (Math.abs(v - ref) < 45) ok++;
  return ok / vals.length;
}
function scoreQuad(q, w, h, edge, textPts, gray){
  const a = polyArea(q), imgA = w * h;
  if (a < imgA * 0.1 || a > imgA * 0.985) return -1;
  const ang = angles(q); if (ang.some(x => x < 50 || x > 130)) return -1;
  // sides ratio sanity (paper is not a sliver)
  const o = order(q), wTop = dist(o[0], o[1]), wBot = dist(o[3], o[2]), hL = dist(o[0], o[3]), hR = dist(o[1], o[2]);
  const ar = Math.max(wTop, wBot) / Math.max(1, Math.max(hL, hR));
  if (ar < 0.2 || ar > 5) return -1;
  // penalise quads glued to the image frame on all sides (that is just the frame)
  const m = 3; let touch = 0;
  if (o.some(p => p.x <= m)) touch++; if (o.some(p => p.x >= w-1-m)) touch++; if (o.some(p => p.y <= m)) touch++; if (o.some(p => p.y >= h-1-m)) touch++;
  let s = Math.sqrt(Math.min(1, (a / imgA) / 0.25));
  if (touch === 4 && a > imgA * 0.93) s *= 0.5;
  // prefer right-ish angles
  const angPen = ang.reduce((acc, x) => acc + Math.abs(x - 90), 0) / 4; s *= 1 - Math.min(0.5, angPen / 90);
  // the outline must actually sit on edges in the image (a shadow or the frame alone does not make a paper)
  if (edge){ const sup = edgeSupport(q, edge, w, h); s *= 0.25 + 0.75 * sup; }
  if (textPts){ const cov = textCoverage(q, textPts); s *= 0.1 + 0.9 * Math.pow(cov, 6); }
  if (gray){ const sim = bandSimilarity(q, gray, w, h); s *= 0.2 + 0.8 * sim; }
  return s;
}
/* straight-line candidates: pairs of near-parallel lines in two orientations → quads (white paper on a white desk) */
function lineAngle(l){ let a = Math.atan2(l.y2 - l.y1, l.x2 - l.x1) * 180 / Math.PI; a = ((a % 180) + 180) % 180; return a; }
function angDiff(a, b){ let d = Math.abs(a - b) % 180; return Math.min(d, 180 - d); }
function intersect(l, m){
  const x1 = l.x1, y1 = l.y1, x2 = l.x2, y2 = l.y2, x3 = m.x1, y3 = m.y1, x4 = m.x2, y4 = m.y2;
  const den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4); if (Math.abs(den) < 1e-6) return null;
  const t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den;
  return { x: x1 + t * (x2 - x1), y: y1 + t * (y2 - y1) };
}
function ptLineDist(p, l){ const dx = l.x2 - l.x1, dy = l.y2 - l.y1, L = Math.hypot(dx, dy) || 1; return Math.abs((p.x - l.x1) * dy - (p.y - l.y1) * dx) / L; }
function extend(m){ // longest extent of all member segments along the line direction
  const a = m.a * Math.PI / 180, dx = Math.cos(a), dy = Math.sin(a), ref = m.segs[0];
  let tMin = Infinity, tMax = -Infinity;
  for (const sg of m.segs) for (const p of [[sg.x1, sg.y1], [sg.x2, sg.y2]]){ const t = (p[0] - ref.x1) * dx + (p[1] - ref.y1) * dy; if (t < tMin) tMin = t; if (t > tMax) tMax = t; }
  return { x1: ref.x1 + dx * tMin, y1: ref.y1 + dy * tMin, x2: ref.x1 + dx * tMax, y2: ref.y1 + dy * tMax, len: m.len, a: m.a, segs: m.segs };
}
function mergeLines(lines){
  // 1) merge near-collinear segments (same angle, same offset)
  const groups = [];
  for (const l of lines){
    const a = lineAngle(l), nx = -Math.sin(a * Math.PI / 180), ny = Math.cos(a * Math.PI / 180), off = nx * l.x1 + ny * l.y1;
    let m = groups.find(o => angDiff(o.a, a) < 3 && Math.abs(o.off - off) < 6);
    if (!m){ groups.push({ a, off, segs: [l], len: l.len }); continue; }
    m.segs.push(l); m.len += l.len;
  }
  let out = groups.map(extend).sort((p, q) => q.len - p.len);
  // 2) chain: a slightly curved paper edge comes out as several short lines at slightly different angles — join them
  let changed = true;
  while (changed){
    changed = false;
    for (let i = 0; i < out.length && !changed; i++) for (let j = i + 1; j < out.length; j++){
      const A = out[i], B = out[j];
      if (angDiff(A.a, B.a) > 6) continue;
      if (ptLineDist({ x: B.x1, y: B.y1 }, A) > 9 || ptLineDist({ x: B.x2, y: B.y2 }, A) > 9) continue;
      // refit angle from the two far ends
      const segs = A.segs.concat(B.segs), len = A.len + B.len;
      const pts = []; for (const sg of segs){ pts.push({ x: sg.x1, y: sg.y1 }, { x: sg.x2, y: sg.y2 }); }
      let far = [pts[0], pts[1], 0]; for (let p = 0; p < pts.length; p++) for (let q = p + 1; q < pts.length; q++){ const d = dist(pts[p], pts[q]); if (d > far[2]) far = [pts[p], pts[q], d]; }
      const a = lineAngle({ x1: far[0].x, y1: far[0].y, x2: far[1].x, y2: far[1].y });
      out.splice(j, 1); out[i] = extend({ a, segs, len }); out.sort((p, q) => q.len - p.len); changed = true; break;
    }
  }
  return out;
}
function quadsFromLines(edges, w, h, found){
  const lines = new cv.Mat();
  cv.HoughLinesP(edges, lines, 1, Math.PI / 180, 25, Math.max(w, h) * 0.08, 20);
  const segs = [], d = lines.data32S, nL = d.length / 4; // opencv.js returns a 1×N matrix
  for (let i = 0; i < nL; i++){ const x1 = d[i*4], y1 = d[i*4+1], x2 = d[i*4+2], y2 = d[i*4+3]; segs.push({ x1, y1, x2, y2, len: Math.hypot(x2 - x1, y2 - y1) }); }
  lines.delete();
  if (segs.length < 4) return;
  // dominant orientation (length-weighted histogram, 5° bins)
  const hist = new Float32Array(36); for (const sg of segs) hist[Math.floor(lineAngle(sg) / 5) % 36] += sg.len;
  let bestBin = 0; for (let i = 0; i < 36; i++){ const v = hist[i] + hist[(i + 18) % 36]; if (v > hist[bestBin] + hist[(bestBin + 18) % 36]) bestBin = i; }
  const A = bestBin * 5 + 2.5;
  const g1 = mergeLines(segs.filter(sg => angDiff(lineAngle(sg), A) < 22)).sort((p, q) => q.len - p.len).slice(0, 10);
  const g2 = mergeLines(segs.filter(sg => angDiff(lineAngle(sg), A + 90) < 22)).sort((p, q) => q.len - p.len).slice(0, 10);
  if (root.__cvDebugLines) root.__cvDebugLines({ A, g1, g2, n: segs.length });
  if (g1.length < 2 || g2.length < 2) return;
  for (let i = 0; i < g1.length; i++) for (let j = i + 1; j < g1.length; j++)
    for (let k = 0; k < g2.length; k++) for (let l = k + 1; l < g2.length; l++){
      const p = [intersect(g1[i], g2[k]), intersect(g1[i], g2[l]), intersect(g1[j], g2[l]), intersect(g1[j], g2[k])];
      if (p.some(x => !x || x.x < -w * 0.05 || x.y < -h * 0.05 || x.x > w * 1.05 || x.y > h * 1.05)) continue;
      const q = p.map(x => ({ x: Math.min(w - 1, Math.max(0, x.x)), y: Math.min(h - 1, Math.max(0, x.y)) }));
      if (isConvexQ(q)) found.push(q);
    }
}
function isConvexQ(q){
  let sign = 0;
  for (let i = 0; i < 4; i++){ const a = q[i], b = q[(i+1)%4], c = q[(i+2)%4], cr = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x); if (Math.abs(cr) < 1e-6) return false; const s2 = Math.sign(cr); if (!sign) sign = s2; else if (s2 !== sign) return false; }
  return true;
}
function detectMat(src, W, H){
  const s = Math.min(1, DET_SIZE / Math.max(W, H));
  const w = Math.max(32, Math.round(W * s)), h = Math.max(32, Math.round(H * s));
  const small = new cv.Mat(); cv.resize(src, small, new cv.Size(w, h), 0, 0, cv.INTER_AREA);
  const rgb = new cv.Mat(); cv.cvtColor(small, rgb, cv.COLOR_RGBA2RGB);
  const gray = new cv.Mat(); cv.cvtColor(rgb, gray, cv.COLOR_RGB2GRAY);
  const blur = new cv.Mat(); cv.GaussianBlur(gray, blur, new cv.Size(5, 5), 0);
  const hsv = new cv.Mat(); cv.cvtColor(rgb, hsv, cv.COLOR_RGB2HSV);
  const ch = new cv.MatVector(); cv.split(hsv, ch); const sat = ch.get(1), val = ch.get(2);
  const found = [], imgArea = w * h;
  const k3 = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
  const k5 = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(5, 5));
  // sensitive edge map used both for line candidates and for scoring (does the outline sit on real edges?)
  // high-pass + gain makes a white paper edge on a white desk visible
  const edgeFine = new cv.Mat(), textPts = [];
  {
    const bl = new cv.Mat(); cv.GaussianBlur(gray, bl, new cv.Size(0, 0), 12);
    const g16 = new cv.Mat(), b16 = new cv.Mat(), d16 = new cv.Mat(); gray.convertTo(g16, cv.CV_16S); bl.convertTo(b16, cv.CV_16S); cv.subtract(g16, b16, d16);
    const boosted = new cv.Mat(); d16.convertTo(boosted, cv.CV_8U, 4, 128); cv.GaussianBlur(boosted, boosted, new cv.Size(3, 3), 0);
    cv.Canny(boosted, edgeFine, 30, 90);
    const es = new cv.Mat(); cv.Canny(sat, es, 25, 80); cv.bitwise_or(edgeFine, es, edgeFine);
    del(bl, g16, b16, d16, boosted, es);
    // the image border itself is never a paper edge: blank a thin frame so border artefacts cannot vote
    { const fb = 4, ed = edgeFine.data; for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (x < fb || y < fb || x >= w - fb || y >= h - fb) ed[y * w + x] = 0; }
    // drop small edge blobs (text, noise): paper edges, table rules and shadows are long structures
    const labels = new cv.Mat(), stats = new cv.Mat(), cent = new cv.Mat();
    const n = cv.connectedComponentsWithStats(edgeFine, labels, stats, cent, 8, cv.CV_32S);
    const minLen = Math.max(w, h) * 0.06, keep = new Uint8Array(n);
    for (let i = 1; i < n; i++){ const bw = stats.data32S[i * 5 + 2], bh = stats.data32S[i * 5 + 3]; if (Math.max(bw, bh) >= minLen) keep[i] = 1; }
    const ld = labels.data32S, ed = edgeFine.data;
    for (let i = 0; i < w * h; i++) if (ed[i] && !keep[ld[i]]){ ed[i] = 0; if ((i & 3) === 0) textPts.push(i % w, (i / w) | 0); }
    del(labels, stats, cent);
  }
  const edgeWide = new cv.Mat(); cv.dilate(edgeFine, edgeWide, k5); const edgeArr = new Uint8Array(edgeWide.data);
  { const e = new cv.Mat(); cv.dilate(edgeFine, e, k3); quadsFromLines(e, w, h, found); quadsFromMask(e, imgArea, w, h, found); e.delete(); }
  for (const [lo, hi] of [[40, 120], [20, 70], [80, 200]]){
    const e = new cv.Mat(); cv.Canny(blur, e, lo, hi); cv.dilate(e, e, k3);
    quadsFromMask(e, imgArea, w, h, found); e.delete();
  }
  {
    const t = new cv.Mat(); cv.threshold(blur, t, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
    cv.morphologyEx(t, t, cv.MORPH_CLOSE, k5); cv.morphologyEx(t, t, cv.MORPH_OPEN, k5);
    quadsFromMask(t, imgArea, w, h, found); t.delete();
  }
  {
    const sb = new cv.Mat(); cv.GaussianBlur(sat, sb, new cv.Size(7, 7), 0);
    const t = new cv.Mat(); cv.threshold(sb, t, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
    const vb = new cv.Mat(); cv.threshold(val, vb, 90, 255, cv.THRESH_BINARY); cv.bitwise_and(t, vb, t);
    cv.morphologyEx(t, t, cv.MORPH_CLOSE, k5); cv.morphologyEx(t, t, cv.MORPH_OPEN, k5);
    quadsFromMask(t, imgArea, w, h, found); sb.delete(); t.delete(); vb.delete();
  }
  {
    const e = new cv.Mat(); cv.Canny(sat, e, 30, 90); cv.dilate(e, e, k3); quadsFromMask(e, imgArea, w, h, found); e.delete();
  }
  let best = null, bestS = 0;
  const grayArr = new Uint8Array(blur.data);
  for (const q of found){ const sc = scoreQuad(q, w, h, edgeArr, textPts, grayArr); if (sc > bestS){ bestS = sc; best = q; } }
  if (root.__cvDebug) root.__cvDebug(found.map(q => ({ q: order(q).map(p => [Math.round(p.x / s), Math.round(p.y / s)]), s: scoreQuad(q, w, h, edgeArr, textPts, grayArr) })).filter(x => x.s > 0).sort((a, b) => b.s - a.s).slice(0, 8));
  del(small, rgb, gray, blur, hsv, sat, val, k3, k5, edgeFine, edgeWide); ch.delete();
  if (!best) return null;
  return { quad: order(best).map(p => ({ x: (p.x + 0.5) / s, y: (p.y + 0.5) / s })), score: bestS };
}
// an upright rectangle with square corners and sides parallel to the image: a screenshot frame or a photo in a photo
function isAxisRect(q, W, H){
  const o = order(q), tol = Math.max(W, H) * 0.015;
  const horiz = Math.abs(o[0].y - o[1].y) < tol && Math.abs(o[3].y - o[2].y) < tol;
  const vert = Math.abs(o[0].x - o[3].x) < tol && Math.abs(o[1].x - o[2].x) < tol;
  return horiz && vert;
}
// is the area just outside the rectangle dark (a phone UI, a black border)?
function darkFrame(src, q, W, H){
  const o = order(q), m = Math.round(Math.max(W, H) * 0.03);
  const x0 = Math.round(Math.min(o[0].x, o[3].x)), x1 = Math.round(Math.max(o[1].x, o[2].x)), y0 = Math.round(Math.min(o[0].y, o[1].y)), y1 = Math.round(Math.max(o[2].y, o[3].y));
  const strips = [];
  if (y0 > m) strips.push(new cv.Rect(x0, y0 - m, x1 - x0, m));
  if (H - y1 > m) strips.push(new cv.Rect(x0, y1, x1 - x0, m));
  if (x0 > m) strips.push(new cv.Rect(x0 - m, y0, m, y1 - y0));
  if (W - x1 > m) strips.push(new cv.Rect(x1, y0, m, y1 - y0));
  if (!strips.length) return false;
  let sum = 0, n = 0;
  for (const r of strips){ const roi = src.roi(r); const mean = cv.mean(roi); roi.delete(); sum += (mean[0] + mean[1] + mean[2]) / 3; n++; }
  return sum / n < 80;
}
function detect(img){
  const W = img.width, H = img.height, src = toMat(img);
  let res = detectMat(src, W, H);
  // a perfectly upright rectangle is usually not the paper but a frame around the photo (chat screenshot, photo of a screen):
  // look inside it for the real paper
  if (res && isAxisRect(res.quad, W, H) && darkFrame(src, res.quad, W, H)){
    const o = order(res.quad);
    const x0 = Math.max(0, Math.round(Math.min(o[0].x, o[3].x)) + 2), y0 = Math.max(0, Math.round(Math.min(o[0].y, o[1].y)) + 2);
    const x1 = Math.min(W, Math.round(Math.max(o[1].x, o[2].x)) - 2), y1 = Math.min(H, Math.round(Math.max(o[2].y, o[3].y)) - 2);
    if (x1 - x0 > 64 && y1 - y0 > 64){
      const roi = src.roi(new cv.Rect(x0, y0, x1 - x0, y1 - y0)), inner = roi.clone(); roi.delete();
      const r2 = detectMat(inner, x1 - x0, y1 - y0); inner.delete();
      if (r2 && r2.score > 0.12){
        const q2 = r2.quad.map(p => ({ x: p.x + x0, y: p.y + y0 }));
        res = { quad: q2, score: r2.score, frame: res.quad };
      }
    }
  }
  src.delete();
  if (!res) return null;
  let o = res.quad.map(p => ({ x: Math.min(W, Math.max(0, p.x)), y: Math.min(H, Math.max(0, p.y)) }));
  const cx = o.reduce((a, p) => a + p.x, 0) / 4, cy = o.reduce((a, p) => a + p.y, 0) / 4, inset = 0.004;
  o = o.map(p => ({ x: Math.min(W, Math.max(0, p.x + (cx - p.x) * inset)), y: Math.min(H, Math.max(0, p.y + (cy - p.y) * inset)) }));
  return { quad: o, score: res.score };
}

/* ---------- warp ---------- */
function warp(img, quad, maxSide){
  const q = order(quad);
  let W = Math.max(dist(q[0], q[1]), dist(q[3], q[2])), H = Math.max(dist(q[0], q[3]), dist(q[1], q[2]));
  const k = Math.min(1, maxSide / Math.max(W, H));
  W = Math.max(16, Math.round(W * k)); H = Math.max(16, Math.round(H * k));
  const src = toMat(img), dst = new cv.Mat();
  const from = cv.matFromArray(4, 1, cv.CV_32FC2, [q[0].x, q[0].y, q[1].x, q[1].y, q[2].x, q[2].y, q[3].x, q[3].y]);
  const to = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, W, 0, W, H, 0, H]);
  const M = cv.getPerspectiveTransform(from, to);
  cv.warpPerspective(src, dst, M, new cv.Size(W, H), cv.INTER_LINEAR, cv.BORDER_REPLICATE);
  const out = matToImageData(dst); del(src, dst, from, to, M); return out;
}

/* ---------- processing ---------- */
function percentile(gray, p){ // gray: 8U1
  const hist = new cv.Mat(), mv = new cv.MatVector(); mv.push_back(gray);
  cv.calcHist(mv, [0], new cv.Mat(), hist, [256], [0, 256]);
  const total = gray.rows * gray.cols; let acc = 0, v = 0;
  for (v = 0; v < 256; v++){ acc += hist.data32F[v]; if (acc >= total * p) break; }
  hist.delete(); mv.delete(); return v;
}
function estimateBackground(rgb){
  // paper colour / illumination map: close out the dark ink at low res, blur, upsample
  const f = Math.max(1, Math.round(Math.max(rgb.cols, rgb.rows) / 400));
  const sw = Math.max(8, Math.round(rgb.cols / f)), sh = Math.max(8, Math.round(rgb.rows / f));
  const small = new cv.Mat(); cv.resize(rgb, small, new cv.Size(sw, sh), 0, 0, cv.INTER_AREA);
  const ks = Math.max(5, (Math.round(Math.max(sw, sh) / 22) | 1));
  const k = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(ks, ks));
  cv.morphologyEx(small, small, cv.MORPH_CLOSE, k);
  cv.medianBlur(small, small, 5);
  cv.GaussianBlur(small, small, new cv.Size(ks | 1, ks | 1), 0);
  const bg = new cv.Mat(); cv.resize(small, bg, new cv.Size(rgb.cols, rgb.rows), 0, 0, cv.INTER_LINEAR);
  del(small, k); return bg;
}
function normalize(rgb, bg){
  const a = new cv.Mat(), b = new cv.Mat(), n = new cv.Mat();
  rgb.convertTo(a, cv.CV_32FC3); bg.convertTo(b, cv.CV_32FC3);
  cv.max(b, new cv.Mat(b.rows, b.cols, cv.CV_32FC3, new cv.Scalar(24, 24, 24)), b);
  cv.divide(a, b, n, 255);
  const out = new cv.Mat(); n.convertTo(out, cv.CV_8UC3); // saturates at 255
  del(a, b, n); return out;
}
function levels(m, lowP, highP, gamma){ // in-place contrast stretch on 8U (1 or 3 ch) using luminance percentiles
  let g;
  if (m.channels() === 3){ g = new cv.Mat(); cv.cvtColor(m, g, cv.COLOR_RGB2GRAY); } else g = m;
  // white point: anything lighter than ~82% after normalisation is paper (kills bleed-through and faint shadows)
  const lo = Math.min(90, percentile(g, lowP)), hi = Math.max(lo + 40, Math.min(210, percentile(g, highP)));
  if (g !== m) g.delete();
  const lut = new cv.Mat(1, 256, cv.CV_8U);
  for (let i = 0; i < 256; i++){ let t = (i - lo) / (hi - lo); t = Math.min(1, Math.max(0, t)); lut.data[i] = Math.round(Math.pow(t, gamma) * 255); }
  cv.LUT(m, lut, m); lut.delete();
}
function unsharp(m, amount, sigma){
  const b = new cv.Mat(); cv.GaussianBlur(m, b, new cv.Size(0, 0), sigma);
  cv.addWeighted(m, 1 + amount, b, -amount, 0, m); b.delete();
}
function fillMask(m, mask, value){ m.setTo(new cv.Scalar(value, value, value, 255), mask); }

function handwritingMasks(norm, hw){
  const W = norm.cols, H = norm.rows, out = { color: null, pencil: null };
  if (hw && hw.color){
    // chroma = max(R,G,B) - min(R,G,B): absolute, so dark printed pixels with a little colour noise stay neutral
    const ch = new cv.MatVector(); cv.split(norm, ch); const R = ch.get(0), G = ch.get(1), B = ch.get(2);
    const mx = new cv.Mat(), mn = new cv.Mat(), chroma = new cv.Mat();
    cv.max(R, G, mx); cv.max(mx, B, mx); cv.min(R, G, mn); cv.min(mn, B, mn); cv.subtract(mx, mn, chroma);
    const m = new cv.Mat(); cv.threshold(chroma, m, 42, 255, cv.THRESH_BINARY);          // coloured ink
    const vm = new cv.Mat(); cv.threshold(mx, vm, 250, 255, cv.THRESH_BINARY_INV); cv.bitwise_and(m, vm, m); // and not paper
    const k = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(5, 5));
    const grown = new cv.Mat(); cv.dilate(m, grown, k, new cv.Point(-1, -1), 2);
    // protect dark neutral (printed) pixels from the grow step
    const dark = new cv.Mat(); cv.threshold(mx, dark, 130, 255, cv.THRESH_BINARY_INV);
    const lowC = new cv.Mat(); cv.threshold(chroma, lowC, 30, 255, cv.THRESH_BINARY_INV); cv.bitwise_and(dark, lowC, dark);
    cv.bitwise_not(dark, dark); cv.bitwise_and(grown, dark, grown);
    out.color = grown; del(R, G, B, mx, mn, chroma, m, vm, k, dark, lowC); ch.delete();
  }
  if (hw && hw.pencil){
    const T = Math.round((hw.pencilT || 0.4) * 255);
    const g = new cv.Mat(); cv.cvtColor(norm, g, cv.COLOR_RGB2GRAY);
    const ink = new cv.Mat(); cv.threshold(g, ink, 232, 255, cv.THRESH_BINARY_INV);   // anything that is not paper
    if (out.color){ const nc = new cv.Mat(); cv.bitwise_not(out.color, nc); cv.bitwise_and(ink, nc, ink); nc.delete(); }
    const k = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3)); cv.morphologyEx(ink, ink, cv.MORPH_CLOSE, k);
    const labels = new cv.Mat(), stats = new cv.Mat(), cent = new cv.Mat();
    const n = cv.connectedComponentsWithStats(ink, labels, stats, cent, 8, cv.CV_32S);
    // a component that has a really dark core is print (or a pen); the rest is pencil
    const hasDark = new Uint8Array(n), gd = g.data, ld = labels.data32S, N = W * H;
    for (let i = 0; i < N; i++){ const l = ld[i]; if (l && gd[i] < T) hasDark[l] = 1; }
    const pm = new cv.Mat.zeros(H, W, cv.CV_8U), pd = pm.data, inkd = ink.data;
    for (let i = 0; i < N; i++){ const l = ld[i]; if (l && inkd[i] && !hasDark[l]) pd[i] = 255; }
    const k2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(5, 5)); cv.dilate(pm, pm, k2);
    out.pencil = pm; del(g, ink, k, labels, stats, cent, k2);
  }
  return out;
}

/**
 * opts: { filter: 'original'|'enhance'|'gray'|'bw', hw: {color, pencil, pencilT} }
 * returns RGBA ImageData of the same size
 */
function process(img, opts){
  const filter = opts.filter || 'enhance', hw = opts.hw || {};
  const src = toMat(img), rgb = new cv.Mat(); cv.cvtColor(src, rgb, cv.COLOR_RGBA2RGB); src.delete();
  const W = rgb.cols, H = rgb.rows;
  const needHw = !!(hw.color || hw.pencil);
  if (filter === 'original' && !needHw){ const o = new cv.Mat(); cv.cvtColor(rgb, o, cv.COLOR_RGB2RGBA); const r = matToImageData(o); del(rgb, o); return r; }
  const bg = estimateBackground(rgb);
  const norm = normalize(rgb, bg);
  const masks = needHw ? handwritingMasks(norm, hw) : {};
  let out;
  if (filter === 'original'){
    out = rgb.clone();
    if (masks.color) bg.copyTo(out, masks.color);
    if (masks.pencil) bg.copyTo(out, masks.pencil);
  } else {
    if (masks.color) fillMask(norm, masks.color, 255);
    if (masks.pencil) fillMask(norm, masks.pencil, 255);
    if (filter === 'enhance'){
      out = norm.clone();
      levels(out, 0.004, 0.80, 1.15);
      // boost ink colours a little
      const hsv = new cv.Mat(); cv.cvtColor(out, hsv, cv.COLOR_RGB2HSV);
      const ch = new cv.MatVector(); cv.split(hsv, ch); const S = ch.get(1);
      S.convertTo(S, -1, 1.3, 0); ch.set(1, S); cv.merge(ch, hsv); cv.cvtColor(hsv, out, cv.COLOR_HSV2RGB);
      del(hsv, S); ch.delete();
      unsharp(out, 0.6, 1.6);
    } else {
      const g = new cv.Mat(); cv.cvtColor(norm, g, cv.COLOR_RGB2GRAY);
      levels(g, 0.004, 0.80, 1.1);
      if (filter === 'gray'){
        unsharp(g, 0.6, 1.6);
      } else { // bw
        const soft = new cv.Mat(); cv.GaussianBlur(g, soft, new cv.Size(3, 3), 0);
        const block = Math.max(15, (Math.round(Math.max(W, H) / 45) | 1));
        const ad = new cv.Mat(); cv.adaptiveThreshold(soft, ad, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, block, 11);
        const glob = new cv.Mat(); cv.threshold(soft, glob, 185, 255, cv.THRESH_BINARY);
        cv.bitwise_or(ad, glob, ad); // black only where both say black
        // despeckle: drop tiny black blobs
        const inv = new cv.Mat(); cv.bitwise_not(ad, inv);
        const labels = new cv.Mat(), stats = new cv.Mat(), cent = new cv.Mat();
        const n = cv.connectedComponentsWithStats(inv, labels, stats, cent, 8, cv.CV_32S);
        const minA = Math.max(4, Math.round(Math.pow(Math.max(W, H) / 700, 2)));
        const kill = new Uint8Array(n); for (let i = 1; i < n; i++) if (stats.data32S[i * 5 + 4] < minA) kill[i] = 1;
        const ld = labels.data32S, od = ad.data; for (let i = 0; i < W * H; i++) if (kill[ld[i]]) od[i] = 255;
        ad.copyTo(g); del(soft, ad, glob, inv, labels, stats, cent);
      }
      out = new cv.Mat(); cv.cvtColor(g, out, cv.COLOR_GRAY2RGB); g.delete();
    }
  }
  const rgba = new cv.Mat(); cv.cvtColor(out, rgba, cv.COLOR_RGB2RGBA);
  const r = matToImageData(rgba);
  del(rgb, bg, norm, out, rgba, masks.color, masks.pencil);
  return r;
}

root.CVPipe = { use, detect, warp, process, order };
})(typeof self !== 'undefined' ? self : this);

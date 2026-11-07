/* 隨手掃描 — 影像處理 Worker */
importScripts('opencv.js', 'cvpipe.js');
const ready = (async () => {
  let m = self.cv;
  if (m && typeof m.then === 'function') m = await m;
  else if (m && m.ready && typeof m.ready.then === 'function') m = await m.ready;
  if (!m || !m.Mat) throw new Error('opencv failed to load');
  CVPipe.use(m);
})();
self.onmessage = async e => {
  const { id, op, img, quad, maxSide, opts } = e.data;
  try {
    await ready;
    let res, transfer = [];
    if (op === 'ping') res = true;
    else if (op === 'detect') res = CVPipe.detect(img);
    else if (op === 'warp'){ res = CVPipe.warp(img, quad, maxSide); transfer = [res.data.buffer]; }
    else if (op === 'process'){ res = CVPipe.process(img, opts); transfer = [res.data.buffer]; }
    else if (op === 'scan'){ // warp + process in one go, returns both
      const w = CVPipe.warp(img, quad, maxSide); const p = CVPipe.process(w, opts);
      res = { warped: w, out: p }; transfer = [w.data.buffer, p.data.buffer];
    }
    else throw new Error('unknown op ' + op);
    self.postMessage({ id, ok: true, res }, transfer);
  } catch (err){
    let msg = String(err && err.message || err);
    if (typeof err === 'number' && self.cv && self.cv.exceptionFromPtr){ try { msg = self.cv.exceptionFromPtr(err).msg; } catch (e) {} }
    self.postMessage({ id, ok: false, err: msg });
  }
};

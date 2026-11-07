/* 離線快取：整個 app 只有幾個檔案，全部 cache-first；改版時把 VER 加一 */
const VER = 'v3';
const SHELL = ['./', 'index.html', 'jspdf.umd.min.js', 'opencv.js', 'cvpipe.js', 'cvworker.js', 'manifest.webmanifest', 'icons/icon-192.png', 'icons/icon-512.png', 'icons/apple-touch-icon.png'];
self.addEventListener('install', e => {
  e.waitUntil(caches.open('scan-' + VER).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k !== 'scan-' + VER).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  if (url.origin !== location.origin || e.request.method !== 'GET') return;
  // the page itself: network-first so updates show up; offline falls back to cache
  if (e.request.mode === 'navigate') {
    e.respondWith(fetch(e.request).then(r => { if (r.ok) caches.open('scan-' + VER).then(c => c.put('index.html', r.clone())); return r; })
      .catch(() => caches.match('index.html')));
    return;
  }
  e.respondWith(caches.match(e.request).then(r => r || fetch(e.request)));
});

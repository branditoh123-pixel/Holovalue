// HoloValue offline cache: app page and prices update from the network first; libraries and images are cached.
const CACHE = 'holovalue-v2';
const SHELL = ['./', 'index.html', 'manifest.webmanifest', 'icon-192.png', 'icon-512.png'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});

const networkFirst = async req => {
  const c = await caches.open(CACHE);
  try { const r = await fetch(req, { cache: 'no-cache' }); if (r.ok) c.put(req, r.clone()); return r; }
  catch (e) { const hit = await c.match(req, { ignoreSearch: true }); if (hit) return hit; throw e; }
};
const cacheFirst = async req => {
  const c = await caches.open(CACHE);
  const hit = await c.match(req); if (hit) return hit;
  const r = await fetch(req); if (r.ok || r.type === 'opaque') c.put(req, r.clone()); return r;
};

self.addEventListener('fetch', e => {
  const req = e.request; if (req.method !== 'GET') return;
  const u = new URL(req.url);
  if (u.origin === location.origin) { e.respondWith(networkFirst(req)); return; }
  if (/cdn\.jsdelivr\.net|tessdata|fonts\.(googleapis|gstatic)\.com|tcgplayer-cdn\.tcgplayer\.com|images\.pokemontcg\.io|assets\.tcgdex\.net/.test(u.host + u.pathname)) {
    e.respondWith(cacheFirst(req));
  }
  // price APIs: straight to the network
});

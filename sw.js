// HoloValue offline cache: app page and prices update from the network first; libraries and images are cached.
const CACHE = 'holovalue-v3';
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

// ---------- price alerts while the app is closed (Periodic Background Sync, installed app on Android Chrome) ----------
function idbGet(k) {
  return new Promise((res) => {
    const r = indexedDB.open('holovalue', 1);
    r.onupgradeneeded = () => r.result.createObjectStore('kv');
    r.onsuccess = () => { try { const g = r.result.transaction('kv').objectStore('kv').get(k); g.onsuccess = () => res(g.result); g.onerror = () => res(null); } catch (e) { res(null); } };
    r.onerror = () => res(null);
  });
}
function idbPut(k, v) {
  return new Promise((res) => {
    const r = indexedDB.open('holovalue', 1);
    r.onupgradeneeded = () => r.result.createObjectStore('kv');
    r.onsuccess = () => { const t = r.result.transaction('kv', 'readwrite'); t.objectStore('kv').put(v, k); t.oncomplete = () => res(); t.onerror = () => res(); };
    r.onerror = () => res();
  });
}
const usd = (n) => '$' + Number(n).toFixed(2);
function marketOf(row, sub) {
  const p = row && row[3] || {};
  if (sub && p[sub] && p[sub][1] != null) return p[sub][1];
  const v = Object.values(p).map((a) => a && a[1]).filter((x) => x != null);
  return v.length ? Math.max(...v) : null;
}
async function priceCheck() {
  const w = await idbGet('watch'); if (!w) return;
  const r = await fetch(w.url, { cache: 'no-cache' }); if (!r.ok) return;
  const data = await r.json(); const stamp = data.source || data.updated;
  const seen = (await idbGet('alerted')) || {};
  if (seen._stamp === stamp) return; // already checked this price update
  const find = (it) => (data.cards[it.key] || []).find((row) => +row[2] === +it.pid);
  const msgs = [];
  for (const it of w.wish || []) { const m = marketOf(find(it), it.sub); if (m != null && m <= it.target) msgs.push(`🎯 ${it.name} is ${usd(m)} (your target ${usd(it.target)})`); }
  for (const it of w.own || []) { const m = marketOf(find(it), it.sub); if (m != null && it.price) { const d = m - it.price; if (Math.abs(d) >= 2 && Math.abs(d) / it.price >= 0.15) msgs.push(`${d > 0 ? '📈' : '📉'} ${it.name} ${d > 0 ? '+' : '−'}${usd(Math.abs(d))}`); } }
  seen._stamp = stamp; await idbPut('alerted', seen);
  if (!msgs.length) return;
  await self.registration.showNotification('HoloValue price alert', { body: msgs.slice(0, 4).join('\n') + (msgs.length > 4 ? `\n+${msgs.length - 4} more` : ''), icon: 'icon-192.png', badge: 'icon-192.png', tag: 'price-alert', data: { url: (w.app || './') + '#wish' } });
}
self.addEventListener('periodicsync', (e) => { if (e.tag === 'price-check') e.waitUntil(priceCheck().catch(() => {})); });
self.addEventListener('notificationclick', (e) => {
  e.notification.close();
  const url = (e.notification.data && e.notification.data.url) || './';
  e.waitUntil(self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then((cs) => { for (const c of cs) { if ('focus' in c) { c.navigate(url).catch(() => {}); return c.focus(); } } return self.clients.openWindow(url); }));
});

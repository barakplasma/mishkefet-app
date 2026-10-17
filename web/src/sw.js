/* Mishkefet offline shell. Build __BUILD_ID__.
 * - navigations: network first, cached copy when offline
 * - hashed assets and ONNX Runtime files: cache first (immutable)
 * Cross-origin requests (model download, analytics) are not touched; the model
 * has its own Cache managed by the engine worker. */
const SHELL = 'mk-shell-__BUILD_ID__'
const BASE = '__BASE__'
const PRECACHE = __PRECACHE__
// Hosts add Vary (Origin, Accept-Encoding); cached entries are URL-keyed and immutable.
const MATCH = { ignoreVary: true }

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(SHELL)
      .then((c) => c.addAll(PRECACHE))
      .catch(() => undefined)
      .then(() => self.skipWaiting()),
  )
})

// The page sends the ONNX Runtime files it actually loaded (WebGPU or WASM
// build) once the model is ready, so they are available offline too.
self.addEventListener('message', (event) => {
  const data = event.data
  if (!data || data.type !== 'cache' || !Array.isArray(data.urls)) return
  const urls = data.urls.filter((u) => typeof u === 'string' && new URL(u, self.location.href).origin === self.location.origin)
  event.waitUntil(
    caches.open(SHELL).then((c) =>
      Promise.all(urls.map((u) => c.match(u, MATCH).then((hit) => hit || c.add(u).catch(() => undefined)))),
    ),
  )
})

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith('mk-shell-') && k !== SHELL).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  )
})

self.addEventListener('fetch', (event) => {
  const req = event.request
  if (req.method !== 'GET') return
  const url = new URL(req.url)
  if (url.origin !== self.location.origin || !url.pathname.startsWith(BASE)) return

  if (req.mode === 'navigate') {
    event.respondWith(
      fetch(req)
        .then((res) => {
          if (res.ok) {
            const copy = res.clone()
            caches.open(SHELL).then((c) => c.put(req, copy))
          }
          return res
        })
        .catch(() => caches.match(req, MATCH).then((r) => r || caches.match(BASE, MATCH))),
    )
    return
  }

  const immutable = url.pathname.startsWith(`${BASE}assets/`) || url.pathname.startsWith(`${BASE}ort/`)
  if (immutable) {
    event.respondWith(
      caches.match(req, MATCH).then(
        (hit) =>
          hit ||
          fetch(req).then((res) => {
            if (res.ok) {
              const copy = res.clone()
              caches.open(SHELL).then((c) => c.put(req, copy))
            }
            return res
          }),
      ),
    )
    return
  }

  // Everything else under the app (icons, manifest, filters): stale-while-revalidate.
  event.respondWith(
    caches.match(req, MATCH).then((hit) => {
      const net = fetch(req)
        .then((res) => {
          if (res.ok) {
            const copy = res.clone()
            caches.open(SHELL).then((c) => c.put(req, copy))
          }
          return res
        })
        .catch(() => hit)
      return hit || net
    }),
  )
})

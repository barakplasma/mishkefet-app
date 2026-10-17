/**
 * Model download + Cache API storage, used from the worker. DOM-free (fetch,
 * caches and crypto.subtle exist in workers).
 *
 * Every file is fetched once from the pinned Hugging Face revision with a
 * streaming reader (for byte progress, resuming with a Range request if the
 * connection drops), checked against its pinned SHA-256, and only then put in
 * a Cache keyed by revision. Later visits read straight from the cache and
 * work offline.
 */

import { MODEL_FILES, MODEL_REVISION, MODEL_TOTAL_BYTES, modelUrl, type ModelFile } from './constants.ts'

export const CACHE_PREFIX = 'mishkefet-model-'
export const CACHE_NAME = `${CACHE_PREFIX}${MODEL_REVISION.slice(0, 8)}`

export type ModelBytes = Record<ModelFile['key'], Uint8Array>

export interface LoadProgress {
  loaded: number
  total: number
  /** True when nothing had to be downloaded. */
  fromCache: boolean
  stage: 'checking' | 'downloading' | 'verifying' | 'done'
}

export class ModelDownloadError extends Error {
  readonly offline: boolean
  constructor(message: string, offline: boolean, options?: { cause?: unknown }) {
    super(message, options)
    this.name = 'ModelDownloadError'
    this.offline = offline
  }
}

const TOTAL_BYTES = MODEL_TOTAL_BYTES

async function openCache(): Promise<Cache | null> {
  try {
    if (typeof caches === 'undefined') return null
    // Drop copies of older revisions.
    for (const name of await caches.keys()) {
      if (name.startsWith(CACHE_PREFIX) && name !== CACHE_NAME) await caches.delete(name)
    }
    return await caches.open(CACHE_NAME)
  } catch {
    return null
  }
}

/** True when every file is already cached (no network needed). */
export async function isModelCached(): Promise<boolean> {
  const cache = await openCache()
  if (!cache) return false
  for (const f of MODEL_FILES) if (!(await cache.match(modelUrl(f.path)))) return false
  return true
}

export async function clearModelCache(): Promise<void> {
  try {
    if (typeof caches !== 'undefined') await caches.delete(CACHE_NAME)
  } catch {
    /* ignore */
  }
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', bytes as Uint8Array<ArrayBuffer>)
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('')
}

async function download(f: ModelFile, onBytes: (n: number) => void, signal?: AbortSignal, baseUrl?: string): Promise<Uint8Array> {
  const url = baseUrl ? `${baseUrl.replace(/\/$/, '')}/${f.path}` : modelUrl(f.path)
  let buf = new Uint8Array(f.bytes)
  let received = 0
  let attempts = 0
  for (;;) {
    try {
      const headers: Record<string, string> = {}
      if (received > 0) headers.Range = `bytes=${received}-`
      const res = await fetch(url, { headers, signal, cache: 'no-store' })
      if (received > 0 && res.status !== 206) {
        // Server ignored the range: start over.
        onBytes(-received)
        received = 0
      }
      if (!res.ok) throw new Error(`HTTP ${res.status} for ${f.path}`)
      const len = Number(res.headers.get('content-length') ?? 0)
      if (received === 0 && len && len !== buf.length) buf = new Uint8Array(len)
      if (!res.body) {
        const all = new Uint8Array(await res.arrayBuffer())
        if (received + all.length > buf.length) {
          const grown = new Uint8Array(received + all.length)
          grown.set(buf.subarray(0, received))
          buf = grown
        }
        buf.set(all, received)
        received += all.length
        onBytes(all.length)
      } else {
        const reader = res.body.getReader()
        for (;;) {
          const { done, value } = await reader.read()
          if (done) break
          if (received + value.length > buf.length) {
            const grown = new Uint8Array(Math.max(buf.length * 2, received + value.length))
            grown.set(buf.subarray(0, received))
            buf = grown
          }
          buf.set(value, received)
          received += value.length
          onBytes(value.length)
        }
      }
      return received === buf.length ? buf : buf.slice(0, received)
    } catch (e) {
      if (signal?.aborted) throw e
      attempts++
      const offline = typeof navigator !== 'undefined' && navigator.onLine === false
      if (attempts >= 4 || offline) {
        throw new ModelDownloadError(`download failed: ${f.path}: ${(e as Error).message}`, offline || received === 0, { cause: e })
      }
      await new Promise((r) => setTimeout(r, 800 * attempts))
    }
  }
}

/**
 * Returns the model files, from the cache when possible, downloading the rest.
 * `baseUrl` swaps the download origin (tests, mirrors); bytes are still checked
 * against the pinned SHA-256 and cached under the canonical Hugging Face URL.
 */
export async function loadModelFiles(onProgress: (p: LoadProgress) => void, opts: { signal?: AbortSignal; baseUrl?: string } = {}): Promise<ModelBytes> {
  const { signal, baseUrl } = opts
  const cache = await openCache()
  const out: Partial<ModelBytes> = {}
  let loaded = 0
  let downloaded = false
  const report = (stage: LoadProgress['stage']) => onProgress({ loaded, total: TOTAL_BYTES, fromCache: !downloaded, stage })
  report('checking')

  for (const f of MODEL_FILES) {
    const url = modelUrl(f.path)
    const hit = cache ? await cache.match(url).catch(() => undefined) : undefined
    if (hit) {
      const bytes = new Uint8Array(await hit.arrayBuffer())
      if (bytes.length === f.bytes) {
        out[f.key] = bytes
        loaded += f.bytes
        report('checking')
        continue
      }
      await cache?.delete(url)
    }
    downloaded = true
    const bytes = await download(
      f,
      (n) => {
        loaded += n
        report('downloading')
      },
      signal,
      baseUrl,
    )
    report('verifying')
    const sha = await sha256Hex(bytes)
    if (sha !== f.sha256) throw new ModelDownloadError(`integrity check failed for ${f.path}`, false)
    if (cache) {
      try {
        await cache.put(
          url,
          new Response(bytes as Uint8Array<ArrayBuffer>, {
            headers: { 'content-type': 'application/octet-stream', 'content-length': String(bytes.length) },
          }),
        )
      } catch {
        // Quota or private mode: still usable for this visit.
      }
    }
    out[f.key] = bytes
  }
  report('done')
  return out as ModelBytes
}

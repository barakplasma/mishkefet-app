import react from '@vitejs/plugin-react'
import { createHash } from 'node:crypto'
import { readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { dirname, join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig, type Plugin } from 'vite'

const here = dirname(fileURLToPath(import.meta.url))
const BASE = '/mishkefet-app/'
const ortDist = resolve(here, 'node_modules/onnxruntime-web/dist')
const ORT_VERSION: string = JSON.parse(readFileSync(resolve(here, 'node_modules/onnxruntime-web/package.json'), 'utf8')).version

/**
 * Files served next to the app that are not part of the module graph:
 * - charset.json straight from ../spec (single source of truth, shared with iOS)
 * - ONNX Runtime's wasm + glue for the two builds we use (plain WASM and WebGPU),
 *   self-hosted under ort/<version>/ instead of a CDN.
 */
const STATIC: Record<string, string> = {
  'charset.json': resolve(here, '../spec/charset.json'),
}
for (const f of [
  'ort-wasm-simd-threaded.mjs',
  'ort-wasm-simd-threaded.wasm',
  'ort-wasm-simd-threaded.asyncify.mjs',
  'ort-wasm-simd-threaded.asyncify.wasm',
]) {
  STATIC[`ort/${ORT_VERSION}/${f}`] = resolve(ortDist, f)
}

const TYPES: Record<string, string> = {
  '.mjs': 'text/javascript',
  '.js': 'text/javascript',
  '.wasm': 'application/wasm',
  '.bin': 'application/octet-stream',
  '.json': 'application/json',
}

function listFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name)
    return statSync(p).isDirectory() ? listFiles(p) : [p]
  })
}

function mkStatic(): Plugin {
  let outDir = 'dist'
  return {
    name: 'mk-static',
    configResolved(config) {
      outDir = resolve(config.root, config.build.outDir)
    },
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = (req.url ?? '').split('?')[0]
        if (!url.startsWith(BASE)) return next()
        const file = STATIC[url.slice(BASE.length)]
        if (!file) return next()
        const ext = url.slice(url.lastIndexOf('.'))
        res.setHeader('Content-Type', TYPES[ext] ?? 'application/octet-stream')
        res.end(readFileSync(file))
      })
    },
    generateBundle() {
      for (const [fileName, file] of Object.entries(STATIC)) {
        this.emitFile({ type: 'asset', fileName, source: readFileSync(file) })
      }
    },
    closeBundle() {
      // The service worker precaches the whole app shell except the two large
      // ONNX Runtime binaries; the page asks it to cache the one it uses.
      let files: string[]
      try {
        files = listFiles(outDir).map((f) => relative(outDir, f).split('\\').join('/'))
      } catch {
        return
      }
      const shell = files.filter((f) => !f.endsWith('.wasm') && !f.endsWith('.map') && f !== 'sw.js').sort()
      const h = createHash('sha256')
      for (const f of shell) h.update(f).update(readFileSync(join(outDir, f)))
      const buildId = h.digest('hex').slice(0, 12)
      const precache = ['', ...shell.filter((f) => f !== 'index.html')].map((f) => BASE + f)
      const sw = readFileSync(resolve(here, 'src/sw.js'), 'utf8')
        .replaceAll('__BUILD_ID__', buildId)
        .replaceAll('__BASE__', BASE)
        .replace('__PRECACHE__', JSON.stringify(precache))
      writeFileSync(join(outDir, 'sw.js'), sw)
    },
  }
}

export default defineConfig({
  base: BASE,
  plugins: [react(), mkStatic()],
  define: {
    __ORT_VERSION__: JSON.stringify(ORT_VERSION),
  },
  resolve: {
    alias: [
      // The "extern wasm" entries load ort-wasm-*.mjs/.wasm from env.wasm.wasmPaths
      // (our self-hosted copies) instead of inlining them into the bundle.
      { find: /^onnxruntime-web\/wasm$/, replacement: resolve(ortDist, 'ort.wasm.min.mjs') },
      { find: /^onnxruntime-web\/webgpu$/, replacement: resolve(ortDist, 'ort.webgpu.min.mjs') },
    ],
  },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  worker: { format: 'es' },
  build: {
    target: 'es2022',
    chunkSizeWarningLimit: 1200,
    rollupOptions: {
      input: {
        main: resolve(here, 'index.html'),
        privacy: resolve(here, 'privacy.html'),
      },
    },
  },
  preview: { port: 4173, strictPort: true },
  // spec/ (adapt.json, train-words.json) sits next to web/ and is imported directly
  server: { port: 5173, fs: { allow: ['..'] } },
})

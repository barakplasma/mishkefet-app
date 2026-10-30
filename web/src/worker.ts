/**
 * Engine worker. Owns ONNX Runtime, the backbone session, the language models
 * and the trainer; the page sends crops, pages, strokes and training views and
 * receives text, features and heads (src/engine/api.ts is the contract).
 *
 * Backend: WebGPU when the browser has an adapter with shader-f16 (the
 * backbone is fp16), otherwise single-threaded WASM: the app is served
 * without COOP/COEP, so there is no SharedArrayBuffer for threads.
 *
 * Requests run one at a time (a session cannot run twice at once); 'cancel'
 * is handled as soon as it arrives and is honoured between lines and steps.
 */

/// <reference lib="webworker" />

import type * as Ort from 'onnxruntime-web'
import type { Backend, HeadData, Reply, Request, TrainView } from './engine/api.ts'
import { parseCharset, textToLabels } from './engine/charset.ts'
import { EN_INPUT, FEATURE_DIM, HE_INPUT } from './engine/constants.ts'
import { applyHead, parseHead } from './engine/head.ts'
import { rgbaToGray, rgbToRgba } from './engine/image.ts'
import { parseCharLM, parseWordLM } from './engine/lm.ts'
import { createOrtMatMul, matmulModel, type MatMul } from './engine/matmul.ts'
import { characterAccuracy } from './engine/metrics.ts'
import { createBackbone } from './engine/model.ts'
import { CancelledError, decode, grayFeatures, readBoxes, readLine, readPage, readStrokes, renderWord, type Engine } from './engine/reader.ts'
import { loadModelFiles } from './engine/store.ts'
import { createTrainer } from './engine/trainer.ts'

declare const self: DedicatedWorkerGlobalScope

const BASE = import.meta.env.BASE_URL
const ORT_VERSION = __ORT_VERSION__

/** Requests without their id (the reply helpers add it back). */
type Body<T extends Request['type']> = Extract<Request, { type: T }>

interface Loaded {
  engine: Engine
  backend: Backend
  matmul?: MatMul
  ortFiles: string[]
}

let loaded: Loaded | null = null
let loading: Promise<Loaded> | null = null
const cancelled = new Set<number>()
let queue: Promise<void> = Promise.resolve()

function post(m: Reply, transfer: Transferable[] = []): void {
  self.postMessage(m, transfer)
}

function checkCancel(id: number): void {
  if (cancelled.has(id)) throw new CancelledError()
}

/** Let queued messages (cancel) through during long loops. */
const yieldToEvents = () => new Promise<void>((r) => setTimeout(r, 0))

// ---------------------------------------------------------------- loading

async function webgpuUsable(): Promise<boolean> {
  try {
    const gpu = (navigator as Navigator & { gpu?: GPU }).gpu
    if (!gpu) return false
    const adapter = await gpu.requestAdapter()
    return !!adapter && adapter.features.has('shader-f16')
  } catch {
    return false
  }
}

async function importOrt(backend: Backend): Promise<typeof Ort> {
  const ort = backend === 'webgpu' ? await import('onnxruntime-web/webgpu') : await import('onnxruntime-web/wasm')
  ort.env.wasm.wasmPaths = `${BASE}ort/${ORT_VERSION}/`
  // No COOP/COEP on the host, so no SharedArrayBuffer: say so instead of probing.
  ort.env.wasm.numThreads = 1
  ort.env.logLevel = 'error'
  return ort
}

/** The ONNX Runtime files this backend loads, for the service worker's offline cache. */
function ortFilesFor(backend: Backend): string[] {
  const stem = backend === 'webgpu' ? 'ort-wasm-simd-threaded.asyncify' : 'ort-wasm-simd-threaded'
  return [`${BASE}ort/${ORT_VERSION}/${stem}.mjs`, `${BASE}ort/${ORT_VERSION}/${stem}.wasm`]
}

/** One small line through the backbone (fixed-width models: their width); throws when the output is missing or not finite. */
async function selfTest(ort: typeof Ort, session: Ort.InferenceSession, width: number): Promise<void> {
  const x = new Float32Array(64 * width)
  for (let i = 0; i < x.length; i++) x[i] = (i % 7) / 14
  const features = await createBackbone(ort, session).features(x, width)
  if (features.frames !== width / 4) throw new Error(`self-test: ${features.frames} frames for width ${width}`)
  for (let i = 0; i < features.values.length; i += 31) {
    if (!Number.isFinite(features.values[i])) throw new Error('self-test: backbone produced NaN')
  }
}

async function createSession(ort: typeof Ort, backend: Backend, bytes: Uint8Array, width = 64): Promise<Ort.InferenceSession> {
  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: [backend],
    graphOptimizationLevel: 'all',
    logSeverityLevel: 3,
  })
  try {
    await selfTest(ort, session, width)
  } catch (e) {
    await session.release().catch(() => undefined)
    throw e
  }
  return session
}

/** The trainer's matmul on the same runtime, checked on a tiny product; undefined means JS loops. */
async function createMatMul(ort: typeof Ort, backend: Backend): Promise<MatMul | undefined> {
  try {
    const s = await ort.InferenceSession.create(matmulModel(), { executionProviders: [backend], logSeverityLevel: 3 })
    const mm = createOrtMatMul(ort, s)
    const c = await mm(Float32Array.of(1, 2, 3, 4, 5, 6), Float32Array.of(1, 0, 0, 1, 1, 1), 2, 3, 2)
    const want = [4, 5, 10, 11]
    if (want.some((v, i) => Math.abs(c[i] - v) > 1e-4)) throw new Error(`matmul self-test gave ${[...c]}`)
    return mm
  } catch (e) {
    console.warn('[mishkefet] runtime matmul unavailable, training on the CPU', e)
    return undefined
  }
}

async function load(id: number, req: Body<'load'>): Promise<Loaded> {
  const t0 = performance.now()
  const lang = req.lang ?? 'he'
  const files = await loadModelFiles((p) => post({ id, type: 'progress', loaded: p.loaded, total: p.total }), { baseUrl: req.baseUrl, lang })
  const charset = parseCharset(new TextDecoder().decode(files.charset))
  const baseHead = parseHead(files.head)
  const lm = files.charlm && parseCharLM(files.charlm)
  const words = files.words && parseWordLM(files.words, charset)
  const input = lang === 'en' ? EN_INPUT : HE_INPUT

  let backend: Backend = !req.forceWasm && (await webgpuUsable()) ? 'webgpu' : 'wasm'
  let ort: typeof Ort
  let session: Ort.InferenceSession
  try {
    ort = await importOrt(backend)
    session = await createSession(ort, backend, files.backbone, input.width)
  } catch (e) {
    if (backend === 'wasm') throw e
    console.warn('[mishkefet] WebGPU failed, falling back to WASM', e)
    backend = 'wasm'
    ort = await importOrt('wasm')
    session = await createSession(ort, 'wasm', files.backbone, input.width)
  }
  const matmul = await createMatMul(ort, backend)
  const engine: Engine = { charset, lm, words, backbone: createBackbone(ort, session), baseHead, input }
  console.info(`[mishkefet] model ready on ${backend} in ${Math.round(performance.now() - t0)} ms`)
  return { engine, backend, matmul, ortFiles: ortFilesFor(backend) }
}

function ensureLoaded(id: number, req: Body<'load'> = { id, type: 'load' }): Promise<Loaded> {
  if (loaded) return Promise.resolve(loaded)
  loading ??= load(id, req).then(
    (l) => (loaded = l),
    (e) => {
      loading = null
      throw e
    },
  )
  return loading
}

// ---------------------------------------------------------------- helpers

function checkHead(h: HeadData | undefined, engine: Engine): HeadData | undefined {
  if (!h) return undefined
  const b = engine.baseHead
  if (h.classes !== b.classes || h.dim !== b.dim || h.weight.length !== b.classes * b.dim || h.bias.length !== b.classes) {
    throw new Error(`head is ${h.classes} x ${h.dim}, the model needs ${b.classes} x ${b.dim}`)
  }
  return h
}

type ImageRequest = { rgba: Uint8ClampedArray; width: number; height: number }

function checkImage(req: ImageRequest): void {
  if (!(req.width >= 1 && req.height >= 1) || req.rgba.length < req.width * req.height * 4) throw new Error('bad image')
}

function grayOf(req: ImageRequest) {
  checkImage(req)
  return rgbaToGray(req.rgba, req.width, req.height)
}

// ---------------------------------------------------------------- requests

async function handle(req: Request): Promise<void> {
  const id = req.id
  if (req.type === 'load') {
    const l = await ensureLoaded(id, req)
    post({ id, type: 'ready', backend: l.backend, baseHead: l.engine.baseHead, ortFiles: l.ortFiles })
    return
  }
  const { engine, matmul } = await ensureLoaded(id)
  checkCancel(id)
  switch (req.type) {
    case 'readLine': {
      const result = await readLine(engine, grayOf(req), { tta: req.tta, head: checkHead(req.head, engine) })
      post({ id, type: 'done', lines: [result] })
      return
    }
    case 'readPage': {
      checkImage(req)
      const { seg, lines } = await readPage(
        engine,
        req,
        { tta: req.tta, head: checkHead(req.head, engine) },
        {
          onSegmented: ({ paper, turns, seg: s }) => {
            const image = { rgba: rgbToRgba(s.deskewed), width: s.deskewed.width, height: s.deskewed.height }
            post(
              { id, type: 'segmented', angle: s.angle, turns, paper: paper ?? undefined, boxes: s.boxes, width: image.width, height: image.height, image },
              [image.rgba.buffer as ArrayBuffer],
            )
          },
          onLine: (index, total, result) => post({ id, type: 'line', index, total, result }),
          shouldStop: () => cancelled.has(id),
        },
      )
      post({ id, type: 'done', lines, angle: seg.angle })
      return
    }
    case 'readBoxes': {
      checkImage(req)
      const lines = await readBoxes(
        engine,
        req,
        req.boxes,
        { tta: req.tta, head: checkHead(req.head, engine) },
        {
          onLine: (index, total, result) => post({ id, type: 'line', index, total, result }),
          shouldStop: () => cancelled.has(id),
        },
      )
      post({ id, type: 'done', lines })
      return
    }
    case 'readStrokes': {
      const text = await readStrokes(engine, req.strokes, req.word, checkHead(req.head, engine))
      post({ id, type: 'text', text })
      return
    }
    case 'sampleFeatures': {
      const label = textToLabels(engine.charset, req.word)
      const seeds = [0, ...req.seeds.filter((s) => s !== 0)]
      const views: TrainView[] = []
      let preview: { gray: Uint8Array; width: number; height: number } | undefined
      for (const seed of seeds) {
        checkCancel(id)
        const g = renderWord(req.strokes, req.word, seed)
        if (!g) throw new Error('the sample has no ink')
        if (seed === 0) preview = { gray: g.data, width: g.width, height: g.height }
        const f = await grayFeatures(engine, g)
        views.push({ features: f.values, frames: f.frames, label })
      }
      const transfer: Transferable[] = views.map((v) => v.features.buffer as ArrayBuffer)
      if (preview) transfer.push(preview.gray.buffer as ArrayBuffer)
      post({ id, type: 'features', views, preview }, transfer)
      return
    }
    case 'lineFeatures': {
      const f = await grayFeatures(engine, grayOf(req))
      const views = [{ features: f.values, frames: f.frames, label: textToLabels(engine.charset, req.text) }]
      post({ id, type: 'features', views }, [f.values.buffer as ArrayBuffer])
      return
    }
    case 'train': {
      const views = req.views.filter((v) => v.frames > 0 && v.label.length && v.features.length >= v.frames * FEATURE_DIM)
      if (!views.length) throw new Error('no usable training views')
      const trainer = createTrainer(engine.baseHead, views, { lr: req.lr, l2: req.l2, matmul })
      const every = Math.max(1, Math.round(req.steps / 50))
      const t0 = performance.now()
      for (let step = 1; step <= req.steps; step++) {
        checkCancel(id)
        const loss = await trainer.step()
        if (step % every === 0 || step === req.steps) post({ id, type: 'trainStep', step, steps: req.steps, loss })
        await yieldToEvents()
      }
      console.info(`[mishkefet] trained ${req.steps} steps on ${views.length} views in ${((performance.now() - t0) / 1000).toFixed(1)} s`)
      const head = trainer.head()
      post({ id, type: 'trained', head }, [head.weight.buffer as ArrayBuffer, head.bias.buffer as ArrayBuffer])
      return
    }
    case 'evaluate': {
      const before = checkHead(req.before, engine) ?? engine.baseHead
      const after = checkHead(req.after, engine)!
      const reads: Array<{ before: string; after: string }> = []
      const pairsBefore: Array<{ target: string; read: string }> = []
      const pairsAfter: Array<{ target: string; read: string }> = []
      for (const s of req.samples) {
        checkCancel(id)
        const g = renderWord(s.strokes, s.word)
        let rb = ''
        let ra = ''
        if (g) {
          // one backbone pass, two heads
          const f = await grayFeatures(engine, g)
          rb = decode(engine, applyHead(f.values, f.frames, before))
          ra = decode(engine, applyHead(f.values, f.frames, after))
        }
        reads.push({ before: rb, after: ra })
        pairsBefore.push({ target: s.word, read: rb })
        pairsAfter.push({ target: s.word, read: ra })
      }
      post({ id, type: 'evaluated', before: characterAccuracy(pairsBefore), after: characterAccuracy(pairsAfter), reads })
      return
    }
  }
}

self.addEventListener('message', (event: MessageEvent<Request>) => {
  const req = event.data
  if (req.type === 'cancel') {
    cancelled.add(req.target)
    return
  }
  queue = queue.then(async () => {
    try {
      await handle(req)
    } catch (e) {
      const message = e instanceof CancelledError || cancelled.has(req.id) ? 'cancelled' : ((e as Error)?.message ?? String(e))
      if (message !== 'cancelled') console.error(`[mishkefet] ${req.type} failed`, e)
      post({ id: req.id, type: 'error', message })
    } finally {
      cancelled.delete(req.id)
    }
  })
})


/**
 * Main-thread handle on the engine worker: model loading state plus
 * promise-based calls for every request in engine/api.ts.
 */

import type { Backend, HeadData, LineBox, LineResult, Reply, Request, Stroke, TrainView } from '../engine/api.ts'

export interface PreparedPage {
  boxes: LineBox[]
  angle: number
  turns: number
  image: { rgba: Uint8ClampedArray; width: number; height: number }
}

export interface ModelState {
  phase: 'idle' | 'loading' | 'ready' | 'error'
  loaded: number
  total: number
  backend?: Backend
  error?: string
}

export class CancelledError extends Error {
  constructor() {
    super('cancelled')
    this.name = 'CancelledError'
  }
}

type Pending = { onReply: (r: Reply) => void; resolve: (r: Reply) => void; reject: (e: Error) => void; final: Reply['type'][] }

/** Same crash-loop guard as Ozen: a GPU fault during session creation falls back to WASM for a day. */
const gpuGuard = {
  PENDING: 'mk-gpu-pending',
  SKIP: 'mk-gpu-skip-until',
  shouldSkip(): boolean {
    try {
      const ls = window.localStorage
      if (ls.getItem(this.PENDING)) {
        ls.removeItem(this.PENDING)
        ls.setItem(this.SKIP, String(Date.now() + 24 * 3600 * 1000))
        return true
      }
      return Number(ls.getItem(this.SKIP) ?? 0) > Date.now()
    } catch {
      return false
    }
  },
  begin() {
    try {
      window.localStorage.setItem(this.PENDING, String(Date.now()))
    } catch {
      /* storage unavailable */
    }
  },
  end() {
    try {
      window.localStorage.removeItem(this.PENDING)
    } catch {
      /* storage unavailable */
    }
  },
}

type Distribute<T> = T extends unknown ? Omit<T, 'id'> : never
type Body = Distribute<Request>

export class EngineClient {
  private worker: Worker
  private listeners = new Set<(s: ModelState) => void>()
  private pending = new Map<number, Pending>()
  private nextId = 1
  private loadPromise: Promise<void> | null = null
  baseHead: HeadData | null = null
  state: ModelState = { phase: 'idle', loaded: 0, total: 0 }

  constructor() {
    this.worker = new Worker(new URL('../worker.ts', import.meta.url), { type: 'module', name: 'mishkefet-engine' })
    this.worker.addEventListener('message', (e: MessageEvent<Reply>) => this.onMessage(e.data))
    this.worker.addEventListener('error', (e) => {
      this.set({ phase: 'error', error: e.message || 'worker failed' })
      for (const p of this.pending.values()) p.reject(new Error(e.message || 'worker failed'))
      this.pending.clear()
    })
  }

  dispose(): void {
    this.worker.terminate()
  }

  subscribe(fn: (s: ModelState) => void): () => void {
    this.listeners.add(fn)
    fn(this.state)
    return () => this.listeners.delete(fn)
  }

  private set(patch: Partial<ModelState>) {
    this.state = { ...this.state, ...patch }
    for (const fn of this.listeners) fn(this.state)
  }

  /** `?backend=wasm` forces the CPU path, `?modelBase=<url>` downloads from a mirror (still SHA-256 pinned). */
  load(): Promise<void> {
    if (this.loadPromise && this.state.phase !== 'error') return this.loadPromise
    const q = new URLSearchParams(location.search)
    const forceWasm = q.get('backend') === 'wasm' || gpuGuard.shouldSkip()
    if (!forceWasm) gpuGuard.begin()
    this.set({ phase: 'loading', error: undefined })
    this.loadPromise = this.call(
      { type: 'load', baseUrl: q.get('modelBase') ?? undefined, forceWasm },
      ['ready'],
      (r) => {
        if (r.type === 'progress') this.set({ loaded: r.loaded, total: r.total })
      },
    ).then(
      (r) => {
        gpuGuard.end()
        if (r.type === 'ready') {
          this.baseHead = r.baseHead
          this.set({ phase: 'ready', backend: r.backend })
          cacheRuntimeForOffline(r.ortFiles)
        }
      },
      (e: Error) => {
        gpuGuard.end()
        this.set({ phase: 'error', error: e.message })
        throw e
      },
    )
    return this.loadPromise
  }

  /** TTA 3 where it is affordable; one view on the CPU path keeps a page under a few minutes. */
  get tta(): number {
    return this.state.backend === 'webgpu' ? 3 : 1
  }

  async readLine(rgba: ImageData, head?: HeadData): Promise<LineResult> {
    await this.load()
    const r = await this.call({ type: 'readLine', rgba: rgba.data, width: rgba.width, height: rgba.height, head, tta: this.tta }, ['done'])
    return r.type === 'done' ? r.lines[0] : { text: '', confidence: 0 }
  }

  /**
   * Streams the prepared page (paper crop, upright, deskewed: the image the boxes
   * refer to and the one to store) and each line; resolves with every line.
   */
  readPage(
    rgba: ImageData,
    head: HeadData | undefined,
    on: { segmented?: (page: PreparedPage) => void; line?: (index: number, total: number, r: LineResult) => void },
  ): { done: Promise<LineResult[]>; cancel: () => void } {
    const id = this.nextId
    const done = this.load().then(() =>
      this.call({ type: 'readPage', rgba: rgba.data, width: rgba.width, height: rgba.height, head, tta: this.tta }, ['done'], (r) => {
        if (r.type === 'segmented') on.segmented?.({ boxes: r.boxes, angle: r.angle, turns: r.turns, image: r.image })
        if (r.type === 'line') on.line?.(r.index, r.total, r.result)
      }),
    ).then((r) => (r.type === 'done' ? r.lines : []))
    return { done, cancel: () => this.post({ id: this.nextId++, type: 'cancel', target: id }) }
  }

  /** Reads given boxes of an already prepared page, one result per box in order (re-reads). */
  async readBoxes(rgba: ImageData, boxes: LineBox[], head?: HeadData): Promise<LineResult[]> {
    await this.load()
    const r = await this.call({ type: 'readBoxes', rgba: rgba.data, width: rgba.width, height: rgba.height, boxes, head, tta: this.tta }, ['done'])
    return r.type === 'done' ? r.lines : []
  }

  async readStrokes(strokes: Stroke[], word: string, head?: HeadData): Promise<string> {
    await this.load()
    const r = await this.call({ type: 'readStrokes', strokes, word, head }, ['text'])
    return r.type === 'text' ? r.text : ''
  }

  async sampleFeatures(strokes: Stroke[], word: string, seeds: number[]) {
    await this.load()
    const r = await this.call({ type: 'sampleFeatures', strokes, word, seeds }, ['features'])
    if (r.type !== 'features') throw new Error('unexpected reply')
    return r
  }

  async lineFeatures(rgba: ImageData, text: string): Promise<TrainView[]> {
    await this.load()
    const r = await this.call({ type: 'lineFeatures', rgba: rgba.data, width: rgba.width, height: rgba.height, text }, ['features'])
    return r.type === 'features' ? r.views : []
  }

  async train(views: TrainView[], cfg: { steps: number; lr: number; l2: number }, onStep?: (step: number, steps: number) => void): Promise<HeadData> {
    await this.load()
    const r = await this.call({ type: 'train', views, ...cfg }, ['trained'], (m) => {
      if (m.type === 'trainStep') onStep?.(m.step, m.steps)
    })
    if (r.type !== 'trained') throw new Error('unexpected reply')
    return r.head
  }

  async evaluate(samples: Array<{ strokes: Stroke[]; word: string }>, before: HeadData | undefined, after: HeadData) {
    await this.load()
    const r = await this.call({ type: 'evaluate', samples, before, after }, ['evaluated'])
    if (r.type !== 'evaluated') throw new Error('unexpected reply')
    return r
  }

  private call(body: Body, final: Reply['type'][], onReply: (r: Reply) => void = () => {}): Promise<Reply> {
    const id = this.nextId++
    return new Promise<Reply>((resolve, reject) => {
      this.pending.set(id, { onReply, resolve, reject, final })
      this.post({ ...body, id } as Request)
    })
  }

  private post(m: Request) {
    this.worker.postMessage(m)
  }

  private onMessage(r: Reply) {
    const p = this.pending.get(r.id)
    if (!p) return
    if (r.type === 'error') {
      this.pending.delete(r.id)
      p.reject(r.message === 'cancelled' ? new CancelledError() : new Error(r.message))
      return
    }
    if (p.final.includes(r.type)) {
      this.pending.delete(r.id)
      p.resolve(r)
      return
    }
    p.onReply(r)
  }
}

/** Asks the service worker to keep the ONNX Runtime build in use for offline visits. */
function cacheRuntimeForOffline(urls: string[]) {
  if (!import.meta.env.PROD || !('serviceWorker' in navigator) || !urls.length) return
  navigator.serviceWorker.ready.then((reg) => reg.active?.postMessage({ type: 'cache', urls })).catch(() => undefined)
}

let singleton: EngineClient | null = null

export function getEngine(): EngineClient {
  if (!singleton) {
    singleton = new EngineClient()
    window.addEventListener('pagehide', (e) => {
      if (!e.persisted) singleton?.dispose()
    })
  }
  return singleton
}

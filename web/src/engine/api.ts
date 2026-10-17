/**
 * The contract between the UI (main thread, src/lib/engine.ts) and the engine
 * worker (src/worker.ts). Everything the worker does is in spec/*.md; this
 * file only fixes the shapes that cross the postMessage boundary.
 */

export type LineBox = { x: number; y: number; w: number; h: number }

export type LineResult = { text: string; confidence: number; box?: LineBox }

/** Strokes from the training canvas, in CSS pixels, y down. */
export type Stroke = Array<[number, number]>

/** A head file's contents (head.bin format, spec/model.md). */
export type HeadData = { classes: number; dim: number; weight: Float32Array; bias: Float32Array }

/** One cached training view: backbone features for a rendered sample. */
export type TrainView = { features: Float32Array; frames: number; label: number[] }

export type Backend = 'webgpu' | 'wasm'

/** Main thread to worker. Every request carries an id echoed by its reply. */
export type Request =
  | { id: number; type: 'load'; baseUrl?: string; forceWasm?: boolean }
  /** Read a single line crop (RGBA from a canvas). */
  | { id: number; type: 'readLine'; rgba: Uint8ClampedArray; width: number; height: number; head?: HeadData; tta: number }
  /** Segment and read a page. Emits 'segmented' then one 'line' per line, then 'done'. */
  | { id: number; type: 'readPage'; rgba: Uint8ClampedArray; width: number; height: number; head?: HeadData; tta: number }
  /** Training check: render strokes, read with beam (tta 1). */
  | { id: number; type: 'readStrokes'; strokes: Stroke[]; word: string; head?: HeadData }
  /** Render the sample's views and return backbone features for each. */
  | { id: number; type: 'sampleFeatures'; strokes: Stroke[]; word: string; seeds: number[] }
  /** Features for a line crop (edited scan lines become training samples). */
  | { id: number; type: 'lineFeatures'; rgba: Uint8ClampedArray; width: number; height: number; text: string }
  /** Fine-tune the head. Emits 'trainStep' progress, then 'trained'. */
  | { id: number; type: 'train'; views: TrainView[]; steps: number; lr: number; l2: number }
  /** Read held-out samples with two heads, for the before/after numbers. */
  | { id: number; type: 'evaluate'; samples: Array<{ strokes: Stroke[]; word: string }>; before?: HeadData; after: HeadData }
  | { id: number; type: 'cancel'; target: number }

/** Worker to main thread. */
export type Reply =
  | { id: number; type: 'progress'; loaded: number; total: number }
  | { id: number; type: 'ready'; backend: Backend; baseHead: HeadData; ortFiles: string[] }
  | { id: number; type: 'line'; index: number; total: number; result: LineResult }
  | { id: number; type: 'segmented'; angle: number; boxes: LineBox[]; width: number; height: number }
  | { id: number; type: 'done'; lines: LineResult[]; angle?: number }
  | { id: number; type: 'text'; text: string }
  | { id: number; type: 'features'; views: TrainView[]; preview?: { gray: Uint8Array; width: number; height: number } }
  | { id: number; type: 'trainStep'; step: number; steps: number; loss: number }
  | { id: number; type: 'trained'; head: HeadData }
  | { id: number; type: 'evaluated'; before: number; after: number; reads: Array<{ before: string; after: string }> }
  | { id: number; type: 'error'; message: string }

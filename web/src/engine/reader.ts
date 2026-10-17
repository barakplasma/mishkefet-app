/**
 * Reading (spec/decoder.md "Reading a line", "Page"): crop to features to
 * log-probabilities to text, with TTA selection, plus page and stroke reads.
 * Shared by the worker and the Node golden script.
 */

import type { HeadData, LineResult, Stroke } from './api.ts'
import { labelsToText, textToLabels, type Charset } from './charset.ts'
import { TTA_SCALES } from './constants.ts'
import { beamSearch, confidence, greedy, type LogProbs } from './decoder.ts'
import { applyHead } from './head.ts'
import { rgbaToRgb, rgbToGray, type Gray } from './image.ts'
import type { CharLM, WordLM } from './lm.ts'
import type { Backbone, Features } from './model.ts'
import { preprocess, ttaView } from './preprocess.ts'
import { lineCrop, segmentPage, toOriginal, type Segmentation } from './segmenter.ts'
import { augmentParams, renderStrokes } from './strokes.ts'

export interface Engine {
  charset: Charset
  lm: CharLM
  words: WordLM
  backbone: Backbone
  baseHead: HeadData
}

export class CancelledError extends Error {
  constructor() {
    super('cancelled')
    this.name = 'CancelledError'
  }
}

/** Gray crop (any size) to backbone features at the model's input width. */
export async function grayFeatures(engine: Engine, gray: Gray): Promise<Features> {
  const x = preprocess(gray)
  return engine.backbone.features(x.data, x.width)
}

export function decode(engine: Engine, lp: LogProbs, mode: 'beam' | 'greedy' = 'beam'): string {
  const labels = mode === 'beam' ? beamSearch(lp, engine.lm, engine.words) : greedy(lp)
  return labelsToText(engine.charset, labels)
}

export interface ReadOptions {
  tta?: number
  head?: HeadData
  mode?: 'beam' | 'greedy'
}

/**
 * One line crop. With TTA > 1, keeps the first view with the strictly highest
 * emitting-frame confidence (blank reads score -inf); the stored confidence is
 * the all-frame confidence of the scale-1.0 view.
 */
export async function readLine(engine: Engine, gray: Gray, opts: ReadOptions = {}): Promise<LineResult> {
  const tta = Math.max(1, Math.min(opts.tta ?? 3, TTA_SCALES.length))
  const head = opts.head ?? engine.baseHead
  let bestText = ''
  let bestScore = -Infinity
  let baseConf = 0
  for (const scale of TTA_SCALES.slice(0, tta)) {
    const f = await grayFeatures(engine, ttaView(gray, scale))
    const lp = applyHead(f.values, f.frames, head)
    const text = decode(engine, lp, opts.mode)
    if (scale === 1) baseConf = confidence(lp, false)
    if (tta <= 1) return { text, confidence: baseConf }
    const score = text.trim() ? confidence(lp, true) : -Infinity
    if (score > bestScore) {
      bestScore = score
      bestText = text
    }
  }
  return { text: bestText, confidence: baseConf }
}

export interface PageHooks {
  onSegmented?: (seg: Segmentation, boxes: Array<{ x: number; y: number; w: number; h: number }>) => void
  onLine?: (index: number, total: number, result: LineResult) => void
  /** Checked between lines; return true to stop with a CancelledError. */
  shouldStop?: () => boolean
}

/**
 * Segment and read a page. Boxes in the results are in the original image's
 * pixel grid. Every line is returned, in reading order and one per box; the
 * transcript is the lines whose text is not blank (`transcript`).
 */
export async function readPage(
  engine: Engine,
  page: { rgba: ArrayLike<number>; width: number; height: number },
  opts: ReadOptions = {},
  hooks: PageHooks = {},
): Promise<{ seg: Segmentation; lines: LineResult[] }> {
  const rgb = rgbaToRgb(page.rgba, page.width, page.height)
  const gray = rgbToGray(rgb)
  const stop = () => {
    if (hooks.shouldStop?.()) throw new CancelledError()
  }
  const seg = segmentPage(gray, { checkpoint: stop })
  const boxes = seg.boxes.map((b) => toOriginal(b, seg.angle, gray.width, gray.height))
  hooks.onSegmented?.(seg, boxes)
  const lines: LineResult[] = []
  for (let i = 0; i < seg.boxes.length; i++) {
    stop()
    const r = await readLine(engine, lineCrop(gray, seg.angle, seg.boxes[i], rgb), opts)
    const result = { ...r, box: boxes[i] }
    hooks.onLine?.(i, seg.boxes.length, result)
    lines.push(result)
  }
  return { seg, lines }
}

/** The lines that make it into a page's text: blank reads dropped, nothing else. */
export function transcript(lines: readonly LineResult[]): LineResult[] {
  return lines.filter((l) => l.text.trim())
}

/** Plain render of canvas strokes, or null when there is no ink. */
export function renderWord(strokes: readonly Stroke[], word: string, seed = 0): Gray | null {
  return renderStrokes(strokes, Array.from(word).length, augmentParams(seed))
}

/** The training check: plain render, beam search, TTA 1. */
export async function readStrokes(engine: Engine, strokes: readonly Stroke[], word: string, head?: HeadData): Promise<string> {
  const g = renderWord(strokes, word)
  if (!g) return ''
  return (await readLine(engine, g, { tta: 1, head })).text
}

export function labelsFor(engine: Engine, text: string): number[] {
  return textToLabels(engine.charset, text)
}

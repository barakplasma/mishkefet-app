/**
 * Reading (spec/decoder.md "Reading a line", "Page"): crop to features to
 * log-probabilities to text, with straightening and TTA selection, plus
 * page (paper crop, quarter turns, deskew, segmentation), box and stroke
 * reads. Shared by the worker and the Node golden script.
 */

import type { HeadData, LineResult, Stroke } from './api.ts'
import { labelsToText, textToLabels, type Charset } from './charset.ts'
import { TTA_SCALES, type InputSpec } from './constants.ts'
import { beamSearch, confidence, greedy, type LogProbs } from './decoder.ts'
import { applyHead } from './head.ts'
import { rgbaToRgb, type Gray, type Rect, type Rgb } from './image.ts'
import type { CharLM, WordLM } from './lm.ts'
import type { Backbone, Features } from './model.ts'
import { cropToPaper, findPaper, orientPage } from './orientation.ts'
import { preprocess, ttaView } from './preprocess.ts'
import { lineCrop, type Box, type Segmentation } from './segmenter.ts'
import { straightenLine } from './straighten.ts'
import { augmentParams, renderStrokes } from './strokes.ts'

export interface Engine {
  charset: Charset
  /** Absent for the English model: lines are decoded greedily. */
  lm?: CharLM
  words?: WordLM
  backbone: Backbone
  baseHead: HeadData
  /** Line image format the backbone expects (default: Hebrew). */
  input?: InputSpec
}

export class CancelledError extends Error {
  constructor() {
    super('cancelled')
    this.name = 'CancelledError'
  }
}

/** Gray crop (any size) to backbone features at the model's input width. */
export async function grayFeatures(engine: Engine, gray: Gray): Promise<Features> {
  const x = preprocess(gray, engine.input)
  return engine.backbone.features(x.data, x.width)
}

export function decode(engine: Engine, lp: LogProbs, mode: 'beam' | 'greedy' = 'beam'): string {
  const labels = mode === 'beam' && engine.lm && engine.words ? beamSearch(lp, engine.lm, engine.words) : greedy(lp)
  return labelsToText(engine.charset, labels)
}

export interface ReadOptions {
  tta?: number
  head?: HeadData
  mode?: 'beam' | 'greedy'
  /** Straighten the crop first (default). Canvas renders skip it: they are straight already. */
  straighten?: boolean
}

/**
 * One line crop: straightened (usually a no-op), then with TTA > 1 the first
 * view with the strictly highest emitting-frame confidence wins (blank reads
 * score -inf); the stored confidence is the all-frame confidence of the
 * scale-1.0 view.
 */
export async function readLine(engine: Engine, crop: Gray, opts: ReadOptions = {}): Promise<LineResult> {
  const tta = Math.max(1, Math.min(opts.tta ?? 3, TTA_SCALES.length))
  const head = opts.head ?? engine.baseHead
  const gray = opts.straighten === false ? crop : straightenLine(crop)
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

/** A page image as it crosses the worker boundary (canvas RGBA). */
export interface RgbaImage {
  rgba: ArrayLike<number>
  width: number
  height: number
}

export interface PageResult {
  /** The paper crop in the input image, when it was a screenshot-like frame. */
  paper: Rect | null
  /** Quarter turns counter-clockwise that made the page upright. */
  turns: number
  /** Segmentation of the upright page; boxes refer to `seg.deskewed`. */
  seg: Segmentation
  /** One result per box, in reading order, blank texts included. */
  lines: LineResult[]
}

export interface PageHooks {
  onSegmented?: (page: Omit<PageResult, 'lines'>) => void
  onLine?: (index: number, total: number, result: LineResult) => void
  /** Checked between stages and lines; return true to stop with a CancelledError. */
  shouldStop?: () => boolean
}

/** The orientation vote's reader: greedy single view with the general head, emitting confidence. */
function voteScore(engine: Engine) {
  return async (gray: Gray) => {
    const f = await grayFeatures(engine, gray)
    return confidence(applyHead(f.values, f.frames, engine.baseHead), true)
  }
}

/** Read each box of a page (padded crop, straightened, TTA), in order. */
async function readCrops(engine: Engine, page: Rgb, boxes: readonly Box[], opts: ReadOptions, hooks: PageHooks): Promise<LineResult[]> {
  const lines: LineResult[] = []
  for (let i = 0; i < boxes.length; i++) {
    if (hooks.shouldStop?.()) throw new CancelledError()
    const r = await readLine(engine, lineCrop(page, boxes[i]), opts)
    const result = { ...r, box: { ...boxes[i] } }
    hooks.onLine?.(i, boxes.length, result)
    lines.push(result)
  }
  return lines
}

/**
 * Read a page: crop to the paper (screenshots), turn it upright, deskew and
 * segment, then read every line. Boxes are in the upright deskewed page
 * (`seg.deskewed`), which is what the UI stores and shows.
 */
export async function readPage(engine: Engine, image: RgbaImage, opts: ReadOptions = {}, hooks: PageHooks = {}): Promise<PageResult> {
  const checkpoint = () => {
    if (hooks.shouldStop?.()) throw new CancelledError()
  }
  const rgb = rgbaToRgb(image.rgba, image.width, image.height)
  const paper = findPaper(rgb)
  checkpoint()
  const { turns, seg } = await orientPage(cropToPaper(rgb, paper), voteScore(engine), { checkpoint })
  hooks.onSegmented?.({ paper, turns, seg })
  const lines = await readCrops(engine, seg.deskewed, seg.boxes, opts, hooks)
  return { paper, turns, seg, lines }
}

/** Re-read given boxes of an (already deskewed) page: one result per box, in order. */
export async function readBoxes(engine: Engine, image: RgbaImage, boxes: readonly Box[], opts: ReadOptions = {}, hooks: PageHooks = {}): Promise<LineResult[]> {
  const rgb = rgbaToRgb(image.rgba, image.width, image.height)
  return readCrops(engine, rgb, boxes, opts, hooks)
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
  return (await readLine(engine, g, { tta: 1, head, straighten: false })).text
}

export function labelsFor(engine: Engine, text: string): number[] {
  return textToLabels(engine.charset, text)
}

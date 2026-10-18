/**
 * Getting a page upright (spec/orientation.md, reference find_paper and
 * orient_page): crop screenshots to the paper, then find the quarter turns
 * that make the text upright. Sideways pages are told by the stripe score of
 * the row profile; upside-down ones by asking the model which way the middles
 * of its widest lines read more confidently.
 */

import { crop, cropRgb, otsuThreshold, resize, rgbToGray, rot90, roundHalfEven, type Gray, type Rect, type Rgb } from './image.ts'
import { morph, rectKernel, type Kernel, type MorphOp } from './morph.ts'
import { binarize, components, estimateSkewWide, lineCrop, segmentPage, type Segmentation } from './segmenter.ts'

/** Erode or dilate where pixels outside the image count as dark (scipy's border_value=0). */
function morphDark(a: Uint8Array, w: number, h: number, k: Kernel, op: MorphOp): Uint8Array {
  if (op === 'dilate') return morph(a, w, h, k, op) // outside never helps a dilation anyway
  const r = Math.max(k.rows, k.cols)
  const W = w + 2 * r
  const H = h + 2 * r
  const p = new Uint8Array(W * H)
  for (let y = 0; y < h; y++) p.set(a.subarray(y * w, y * w + w), (y + r) * W + r)
  const e = morph(p, W, H, k, 'erode')
  const out = new Uint8Array(w * h)
  for (let y = 0; y < h; y++) out.set(e.subarray((y + r) * W + r, (y + r) * W + r + w), y * w)
  return out
}

/**
 * The paper's rectangle when it sits inside near-black surroundings (a
 * screenshot's app bars, a photo of a screen), else null. Worked out at 256 px.
 */
export function findPaper(rgb: Rgb, work = 256): Rect | null {
  const g = rgbToGray(rgb)
  const { width: W, height: H } = g
  const s = work / Math.max(H, W)
  const w = Math.max(1, roundHalfEven(W * s))
  const h = Math.max(1, roundHalfEven(H * s))
  const small = resize(g, w, h, 'area', W / w, H / h)
  const t = otsuThreshold(small.data)
  let bright: Uint8Array = new Uint8Array(w * h)
  for (let i = 0; i < bright.length; i++) bright[i] = small.data[i] > t ? 255 : 0
  const k = Math.max(3, roundHalfEven(Math.min(w, h) * 0.03)) | 1
  const sq = rectKernel(k, k)
  bright = morphDark(morphDark(bright, w, h, sq, 'dilate'), w, h, sq, 'erode')
  const k3 = rectKernel(3, 3)
  bright = morphDark(morphDark(bright, w, h, k3, 'erode'), w, h, k3, 'dilate')
  const comps = components(bright, w, h, 4)
  if (!comps.length) return null
  let big = comps[0]
  for (const c of comps) if (c.area > big.area) big = c
  if (big.area < 0.2 * w * h) return null
  const { x: x0, y: y0 } = big.box
  const x1 = x0 + big.box.w
  const y1 = y0 + big.box.h
  if ((y1 - y0) * (x1 - x0) > 0.92 * w * h) return null
  let sum = 0
  let n = 0
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x >= x0 && x < x1 && y >= y0 && y < y1) continue
      sum += small.data[y * w + x]
      n++
    }
  }
  if (!n || sum / n > 25 || n / (w * h) < 0.25) return null
  const m = 0.015
  const X0 = Math.max(0, Math.trunc((x0 / w + m) * W))
  const Y0 = Math.max(0, Math.trunc((y0 / h + m) * H))
  const X1 = Math.min(W, Math.trunc((x1 / w - m) * W))
  const Y1 = Math.min(H, Math.trunc((y1 / h - m) * H))
  if (X1 - X0 < 16 || Y1 - Y0 < 16) return null
  return { x: X0, y: Y0, w: X1 - X0, h: Y1 - Y0 }
}

/** Emitting confidence of a greedy single-view read of a gray crop with the general head. */
export type CropScore = (gray: Gray) => Promise<number>

/** The vote reads the 6 widest lines, each cut to its middle 10 crop heights (letters, not whole lines). */
const VOTE_LINES = 6
const VOTE_WINDOW = 10

/** Mean score of the widest crops' middle windows (stable by full width, widest first), as is or turned 180 degrees. */
async function votes(seg: Segmentation, flip: boolean, score: CropScore, checkpoint: () => void): Promise<number> {
  const crops = seg.boxes.map((b) => lineCrop(seg.deskewed, b))
  const widest = crops
    .map((c, i) => ({ c, i }))
    .sort((a, b) => b.c.width - a.c.width || a.i - b.i)
    .slice(0, VOTE_LINES)
  if (!widest.length) return 0
  let sum = 0
  for (const { c } of widest) {
    checkpoint()
    const win = Math.trunc(VOTE_WINDOW * c.height)
    const mid = c.width > win ? crop(c, { x: Math.floor((c.width - win) / 2), y: 0, w: win, h: c.height }) : c
    sum += await score(flip ? rot90(mid, 2) : mid)
  }
  return sum / widest.length
}

export interface Oriented {
  /** Quarter turns counter-clockwise (as displayed) that make the page upright. */
  turns: number
  /** Segmentation of the upright page. */
  seg: Segmentation
}

/**
 * Quarter turns and the upright page's segmentation. Upright vs sideways by
 * stripe score when one is at least 3x the other, else both are tried; each
 * candidate is then read both ways up and the first strictly most confident
 * (base, flip) wins.
 */
export async function orientPage(rgb: Rgb, score: CropScore, opts: { checkpoint?: () => void; maxSide?: number } = {}): Promise<Oriented> {
  const cp = opts.checkpoint ?? (() => {})
  const maxSide = opts.maxSide ?? 2000
  const binarized = binarize(rgbToGray(rgb), maxSide)
  const s0 = estimateSkewWide(binarized.binary).score
  const s1 = estimateSkewWide(rot90(binarized.binary, 1)).score
  cp()
  const bases = Math.max(s0, s1) >= 3 * Math.min(s0, s1) ? [s0 >= s1 ? 0 : 1] : [0, 1]
  let best: { c: number; base: number; flip: boolean; seg: Segmentation } | null = null
  for (const base of bases) {
    // the upright page's mask is the one already computed; a turned page is binarized afresh
    const seg = segmentPage(rot90(rgb, base, 3), { checkpoint: cp, maxSide, binarized: base === 0 ? binarized : undefined })
    for (const flip of [false, true]) {
      const c = await votes(seg, flip, score, cp)
      if (!best || c > best.c) best = { c, base, flip, seg }
    }
  }
  const { base, flip } = best!
  const turns = (base + (flip ? 2 : 0)) % 4
  const seg = flip ? segmentPage(rot90(rgb, turns, 3), { checkpoint: cp, maxSide }) : best!.seg
  return { turns, seg }
}

/** The paper crop of a page, or the page itself. */
export function cropToPaper(rgb: Rgb, paper: Rect | null): Rgb {
  return paper ? cropRgb(rgb, paper) : rgb
}

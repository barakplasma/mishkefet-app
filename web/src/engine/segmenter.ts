/**
 * Page to line boxes (spec/segmenter.md, reference.segment_page): binarize,
 * deskew, measure the line pitch, remove ruled lines, find text lines as
 * connected components of a smeared mask, split tall boxes, merge, order.
 * Every threshold scales with the measured pitch.
 *
 * Binary masks are uint8 with ink = 255. Boxes come out in the deskewed
 * page's pixel grid at full resolution, and the deskewed page (on an
 * expanded canvas beyond 6 degrees) is returned with them: it is what gets
 * stored and shown, and what line crops are cut from.
 */

import {
  blurLarge,
  convolveSep,
  cropRgb,
  expandedSize,
  gaussianKernel,
  resize,
  rgbToGray,
  rotateRgb,
  roundHalfEven,
  roundU8,
  type Gray,
  type Rect,
  type Rgb,
} from './image.ts'
import { lineKernel, maxInto, morph, opening, rectKernel, subSat, transpose } from './morph.ts'

export type Box = Rect

export interface Mask {
  data: Uint8Array
  width: number
  height: number
}

export interface Segmentation {
  /** Skew in degrees, counter-clockwise on screen. */
  angle: number
  pitch: number
  /** Working resolution / full resolution. */
  scale: number
  /** Line boxes in reading order, in `deskewed` coordinates. */
  boxes: Box[]
  /** The deskewed page (the input itself when no rotation was needed). */
  deskewed: Rgb
}

// ---------------------------------------------------------------- binarize

/** Background-flattened, adaptively thresholded, lightly opened ink mask. */
export function binarize(gray: Gray, maxSide = 2000): { binary: Mask; scale: number } {
  const scale = Math.min(1, maxSide / Math.max(gray.height, gray.width))
  let g = gray
  if (scale < 1) {
    g = resize(gray, roundHalfEven(gray.width * scale), roundHalfEven(gray.height * scale), 'area', 1 / scale, 1 / scale)
  }
  const { width: w, height: h } = g
  // flatten: divide by an exact Gaussian background
  const bg = blurLarge(g, Math.max(h, w) / 30)
  const flat = new Uint8Array(w * h)
  for (let i = 0; i < flat.length; i++) {
    const v = Math.fround(Math.fround(g.data[i] / Math.max(bg[i], 1)) * 128)
    flat[i] = v >= 255 ? 255 : Math.trunc(v)
  }
  // adaptive threshold against a Gaussian local mean (replicate border)
  const block = Math.max(15, Math.floor(Math.min(h, w) / 40) | 1)
  const k = gaussianKernel(block, 0)
  const mean = convolveSep(flat, w, h, k, k, 'replicate')
  let binary: Uint8Array = new Uint8Array(w * h)
  for (let i = 0; i < binary.length; i++) binary[i] = flat[i] - roundU8(mean[i]) <= -12 ? 255 : 0
  binary = opening(binary, w, h, rectKernel(2, 2))
  return { binary: { data: binary, width: w, height: h }, scale }
}

// ---------------------------------------------------------------- rotation

/** M^-1 of OpenCV's getRotationMatrix2D about (cx, cy): dst (x, y) samples src M^-1 (x, y). */
function inverseRotation(cx: number, cy: number, angle: number): number[] {
  const a = angle * (Math.PI / 180)
  const c = Math.cos(a)
  const s = Math.sin(a)
  const tx = (1 - c) * cx - s * cy
  const ty = s * cx + (1 - c) * cy
  const det = c * c + s * s
  const i00 = c / det
  const i01 = -s / det
  const i10 = s / det
  const i11 = c / det
  return [i00, i01, -(i00 * tx + i01 * ty), i10, i11, -(i10 * tx + i11 * ty)]
}

/** warpAffine(INTER_NEAREST) about the centre, 0 outside. */
export function rotateNearest(m: Mask, angle: number): Mask {
  const { width: w, height: h, data } = m
  const inv = inverseRotation(w / 2, h / 2, angle)
  const out = new Uint8Array(w * h)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const sx = Math.floor(inv[0] * x + inv[1] * y + inv[2] + 0.5)
      const sy = Math.floor(inv[3] * x + inv[4] * y + inv[5] + 0.5)
      if (sx >= 0 && sx < w && sy >= 0 && sy < h) out[y * w + x] = data[sy * w + sx]
    }
  }
  return { data: out, width: w, height: h }
}

/** Row sums of the nearest-rotated mask, without materializing it. */
function rotatedRowSums(m: Mask, angle: number): Float64Array {
  const { width: w, height: h, data } = m
  const inv = inverseRotation(w / 2, h / 2, angle)
  const prof = new Float64Array(h)
  for (let y = 0; y < h; y++) {
    let s = 0
    for (let x = 0; x < w; x++) {
      const sx = Math.floor(inv[0] * x + inv[1] * y + inv[2] + 0.5)
      const sy = Math.floor(inv[3] * x + inv[4] * y + inv[5] + 0.5)
      if (sx >= 0 && sx < w && sy >= 0 && sy < h) s += data[sy * w + sx]
    }
    prof[y] = s
  }
  return prof
}

/**
 * Nearest-neighbour rotation onto the expanded canvas, centre to centre:
 * destination (x, y) samples (c X - s Y + w/2, s X + c Y + h/2) rounded,
 * with X = x - W'/2, Y = y - H'/2; 0 outside.
 */
export function rotateNearestExpand(m: Mask, angle: number): Mask {
  const { width: w, height: h, data } = m
  const { width: W, height: H } = expandedSize(w, h, angle)
  const a = angle * (Math.PI / 180)
  const c = Math.cos(a)
  const s = Math.sin(a)
  const out = new Uint8Array(W * H)
  for (let y = 0; y < H; y++) {
    const dy = y - H / 2
    for (let x = 0; x < W; x++) {
      const dx = x - W / 2
      const sx = Math.floor(c * dx - s * dy + w / 2 + 0.5)
      const sy = Math.floor(s * dx + c * dy + h / 2 + 0.5)
      if (sx >= 0 && sx < w && sy >= 0 && sy < h) out[y * W + x] = data[sy * w + sx]
    }
  }
  return { data: out, width: W, height: H }
}

/** Population variance of the first differences of `prof`. */
function diffVariance(prof: Float64Array): number {
  const m = prof.length - 1
  if (m <= 0) return NaN
  let mean = 0
  for (let j = 0; j < m; j++) mean += prof[j + 1] - prof[j]
  mean /= m
  let v = 0
  for (let j = 0; j < m; j++) {
    const d = prof[j + 1] - prof[j] - mean
    v += d * d
  }
  return v / m
}

/** Row-profile sharpness at `angle`, per unit width (comparable across orientations). */
export function stripeScore(small: Mask, angle: number): number {
  const prof = rotatedRowSums(small, angle)
  const norm = 255 * small.width
  for (let i = 0; i < prof.length; i++) prof[i] /= norm
  return diffVariance(prof)
}

const WIDE_LIMIT = 44

/**
 * (angle, stripe score): a coarse search over +-44 degrees (step 2) at
 * quarter resolution; within +-6 the original fine search decides (ordinary
 * pages deskew exactly as before), beyond it a 0.5-degree search around the
 * coarse answer at half resolution. A page where every angle scores the same
 * (no ink) stays level. The score is used to tell sideways pages.
 */
export function estimateSkewWide(binary: Mask): { angle: number; score: number } {
  const { width: w, height: h } = binary
  const q = resize(binary, Math.max(1, roundHalfEven(w / 4)), Math.max(1, roundHalfEven(h / 4)), 'area', 4, 4)
  let coarse = 0
  let best = -1
  let worst = Infinity
  for (let i = 0; i <= WIDE_LIMIT; i++) {
    const a = -WIDE_LIMIT + 2 * i
    const sc = stripeScore(q, a)
    if (sc < worst) worst = sc // Python's min(worst, sc)
    if (sc > best) {
      coarse = a
      best = sc
    }
  }
  const half = resize(binary, roundHalfEven(w * 0.5), roundHalfEven(h * 0.5), 'area', 2, 2)
  // every angle scores the same (a page without ink): leave it level
  if (best <= worst) return { angle: 0, score: stripeScore(half, 0) }
  let angle: number
  if (Math.abs(coarse) <= 6) {
    angle = estimateSkew(binary)
  } else {
    angle = coarse - 2
    best = -1
    for (let i = 0; i < 9; i++) {
      const a = coarse - 2 + 0.5 * i
      const sc = stripeScore(half, a)
      if (sc > best) {
        angle = a
        best = sc
      }
    }
  }
  return { angle, score: stripeScore(half, angle) }
}

/** The angle in -6..6 (step 0.5) whose row profile has the sharpest transitions. */
export function estimateSkew(binary: Mask, limit = 6, step = 0.5): number {
  const small = resize(binary, roundHalfEven(binary.width * 0.5), roundHalfEven(binary.height * 0.5), 'area', 2, 2)
  let bestAngle = 0
  let best = -1
  const n = Math.round((2 * limit) / step)
  for (let i = 0; i <= n; i++) {
    const angle = -limit + i * step
    const score = diffVariance(rotatedRowSums(small, angle))
    if (score > best) {
      bestAngle = angle
      best = score
    }
  }
  return bestAngle
}

// ---------------------------------------------------------------- pitch

/** Line pitch from the first peak of the row profile's autocorrelation. */
export function estimateLinePitch(binary: Mask): number {
  const { width: w, height: n, data } = binary
  const prof = new Float64Array(n)
  let mean = 0
  for (let y = 0; y < n; y++) {
    let s = 0
    for (let x = 0; x < w; x++) s += data[y * w + x]
    prof[y] = s
    mean += s
  }
  mean /= n
  let any = false
  for (let y = 0; y < n; y++) {
    prof[y] -= mean
    if (prof[y] !== 0) any = true
  }
  if (!any) return 30
  const lo = Math.max(8, Math.trunc(n * 0.004))
  // up to half the height, not hebocr's 0.15: a few big lines would otherwise shatter (spec/segmenter.md)
  const hi = Math.min(n - 1, Math.trunc(n / 2))
  if (hi <= lo + 2) return 30
  const win = new Float64Array(hi - lo)
  for (let lag = lo; lag < hi; lag++) {
    let s = 0
    for (let t = 0; t + lag < n; t++) s += prof[t] * prof[t + lag]
    win[lag - lo] = s
  }
  for (let i = 1; i < win.length - 1; i++) {
    if (win[i] > win[i - 1] && win[i] >= win[i + 1] && win[i] > 0) return lo + i
  }
  let arg = 0
  for (let i = 1; i < win.length; i++) if (win[i] > win[arg]) arg = i
  return lo + arg
}

// ---------------------------------------------------------------- rules

const RULE_ANGLES = [-2, -1, -0.4, 0, 0.4, 1, 2]

/** Remove long thin horizontal and vertical strokes (ruled paper, margins, frames). */
export function removeRules(binary: Mask, pitch: number): Mask {
  const { width: w, height: h, data } = binary
  const length = Math.max(40, Math.trunc(pitch * 2))
  const thickness = Math.max(3, Math.trunc(pitch * 0.28))
  const horiz = new Uint8Array(w * h)
  const vertT = new Uint8Array(w * h)
  const dataT = transpose(data, w, h)
  // Small angles often round to the same kernel; open each distinct one once.
  const seen = new Set<string>()
  for (const angle of RULE_ANGLES) {
    const k = lineKernel(length, angle)
    const key = JSON.stringify(k.runs)
    if (seen.has(key)) continue
    seen.add(key)
    maxInto(horiz, opening(data, w, h, k))
    // the transposed kernel on the image == the kernel on the transposed image
    maxInto(vertT, opening(dataT, h, w, k))
  }
  const vert = transpose(vertT, h, w)
  const thin = rectKernel(thickness, 1)
  // horizontal runs: keep what survives a 1 x t (vertical) opening
  const horizThick = transpose(opening(transpose(horiz, w, h), h, w, thin), h, w)
  const vertThick = opening(vert, w, h, thin)
  const box3 = rectKernel(3, 3)
  let out = data
  for (const [runs, thick] of [
    [horiz, horizThick],
    [vert, vertThick],
  ]) {
    const rules = subSat(runs, thick)
    out = subSat(out, morph(rules, w, h, box3, 'dilate'))
  }
  return { data: out, width: w, height: h }
}

// ---------------------------------------------------------------- lines

interface Component {
  box: Box
  area: number
}

/** Connected components (8- or 4-connected) of non-zero pixels, in raster order of first pixel. */
export function components(a: Uint8Array, w: number, h: number, connectivity: 4 | 8 = 8): Component[] {
  const seen = new Uint8Array(w * h)
  const stack = new Int32Array(w * h)
  const out: Component[] = []
  for (let start = 0; start < a.length; start++) {
    if (!a[start] || seen[start]) continue
    let sp = 0
    stack[sp++] = start
    seen[start] = 1
    let x0 = w
    let x1 = -1
    let y0 = h
    let y1 = -1
    let area = 0
    while (sp) {
      const p = stack[--sp]
      const y = (p / w) | 0
      const x = p - y * w
      area++
      if (x < x0) x0 = x
      if (x > x1) x1 = x
      if (y < y0) y0 = y
      if (y > y1) y1 = y
      for (let dy = -1; dy <= 1; dy++) {
        const ny = y + dy
        if (ny < 0 || ny >= h) continue
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx
          if (nx < 0 || nx >= w || (connectivity === 4 && dx && dy)) continue
          const q = ny * w + nx
          if (a[q] && !seen[q]) {
            seen[q] = 1
            stack[sp++] = q
          }
        }
      }
    }
    out.push({ box: { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1 }, area })
  }
  return out
}

/** Smear words into lines, then keep the connected components that look like lines. */
export function findLines(binary: Mask, pitch: number): Box[] {
  const { width: w, height: h } = binary
  const sx = Math.max(8, Math.trunc(pitch * 0.55))
  const sy = Math.max(1, Math.trunc(pitch * 0.05))
  const wide = rectKernel(sx, 1)
  let sm = morph(binary.data, w, h, wide, 'dilate')
  if (sy > 1) sm = transpose(morph(transpose(sm, w, h), h, w, rectKernel(sy, 1), 'dilate'), h, w)
  sm = morph(morph(sm, w, h, wide, 'dilate'), w, h, wide, 'erode')
  const boxes: Box[] = []
  for (const c of components(sm, w, h)) {
    const b = c.box
    if (b.w < pitch * 0.8 || b.h < pitch * 0.25 || c.area < pitch * pitch * 0.12) continue
    // tall and mostly empty is a box outline; tall and dense is a paragraph that smeared together (spec/segmenter.md)
    if (b.h > pitch * 12 && c.area / (b.w * b.h) < 0.35) continue
    boxes.push(b)
  }
  return boxes
}

/** Cut boxes that span several lines at the profile minima near each expected boundary. */
export function splitTall(binary: Mask, boxes: Box[], pitch: number): Box[] {
  const { width: W, data } = binary
  const k5 = gaussianKernel(5, 0)
  const one = Float64Array.of(1)
  const out: Box[] = []
  for (const b of boxes) {
    const n = roundHalfEven(b.h / pitch)
    if (n <= 1) {
      out.push(b)
      continue
    }
    const rowSum = (y: number) => {
      let s = 0
      for (let x = 0; x < b.w; x++) s += data[(b.y + y) * W + b.x + x]
      return s
    }
    const prof = new Float64Array(b.h)
    for (let y = 0; y < b.h; y++) prof[y] = rowSum(y)
    const smooth = convolveSep(prof, 1, b.h, one, k5)
    const cuts = [0]
    const reach = Math.trunc(pitch * 0.35)
    for (let k = 1; k < n; k++) {
      const centre = Math.trunc((k * b.h) / n)
      const lo = Math.max(cuts[cuts.length - 1] + 2, centre - reach)
      const hi = Math.min(b.h - 2, centre + reach)
      if (hi > lo) {
        let arg = lo
        for (let i = lo + 1; i < hi; i++) if (smooth[i] < smooth[arg]) arg = i
        cuts.push(arg)
      }
    }
    cuts.push(b.h)
    for (let i = 0; i + 1 < cuts.length; i++) {
      const top = cuts[i]
      const bottom = cuts[i + 1]
      let r0 = -1
      let r1 = -1
      let c0 = b.w
      let c1 = -1
      for (let y = top; y < bottom; y++) {
        const row = (b.y + y) * W + b.x
        let any = false
        for (let x = 0; x < b.w; x++) {
          if (data[row + x]) {
            any = true
            if (x < c0) c0 = x
            if (x > c1) c1 = x
          }
        }
        if (any) {
          if (r0 < 0) r0 = y
          r1 = y
        }
      }
      if (r0 < 0 || c1 < 0) continue
      out.push({ x: b.x + c0, y: b.y + r0, w: c1 - c0 + 1, h: r1 - r0 + 1 })
    }
  }
  return out
}

/** Merge boxes that overlap vertically by more than half and sit close horizontally. */
export function mergeOverlapping(boxes: Box[], pitch: number): Box[] {
  const sorted = [...boxes].sort((a, b) => a.y - b.y || a.x - b.x)
  const merged: Box[] = []
  for (const b of sorted) {
    let done = false
    for (let i = 0; i < merged.length; i++) {
      const o = merged[i]
      const v = Math.min(b.y + b.h, o.y + o.h) - Math.max(b.y, o.y)
      const gap = Math.max(b.x, o.x) - Math.min(b.x + b.w, o.x + o.w)
      if (v > Math.min(b.h, o.h) * 0.5 && gap < pitch * 1.2) {
        const x0 = Math.min(b.x, o.x)
        const y0 = Math.min(b.y, o.y)
        const x1 = Math.max(b.x + b.w, o.x + o.w)
        const y1 = Math.max(b.y + b.h, o.y + o.h)
        merged[i] = { x: x0, y: y0, w: x1 - x0, h: y1 - y0 }
        done = true
        break
      }
    }
    if (!done) merged.push(b)
  }
  return merged
}

/** Rows top to bottom; inside a row, right to left (Hebrew). */
export function readingOrder(boxes: Box[], pitch: number): Box[] {
  if (!boxes.length) return []
  const cy = (b: Box) => b.y + b.h / 2
  const ordered = [...boxes].sort((a, b) => cy(a) - cy(b))
  const rows: Box[][] = [[ordered[0]]]
  for (const b of ordered.slice(1)) {
    const row = rows[rows.length - 1]
    const rc = row.reduce((s, o) => s + cy(o), 0) / row.length
    if (Math.abs(cy(b) - rc) < pitch * 0.45) row.push(b)
    else rows.push([b])
  }
  return rows.flatMap((row) => [...row].sort((a, b) => b.x + b.w - (a.x + a.w)))
}

// ---------------------------------------------------------------- page

export interface SegmentOptions {
  /** Checked between stages; throw from it to abort. */
  checkpoint?: () => void
  maxSide?: number
  /** binarize(gray of this page), when the caller already has it. */
  binarized?: { binary: Mask; scale: number }
}

/** Page (RGB) to line boxes, plus the deskewed page they refer to. */
export function segmentPage(rgb: Rgb, opts: SegmentOptions = {}): Segmentation {
  const cp = opts.checkpoint ?? (() => {})
  const { binary: b0, scale } = opts.binarized ?? binarize(rgbToGray(rgb), opts.maxSide ?? 2000)
  cp()
  let binary = b0
  const { angle } = estimateSkewWide(binary)
  let deskewed = rgb
  if (Math.abs(angle) > 6) {
    // a strongly tilted photo: a canvas big enough to keep the corners
    binary = rotateNearestExpand(binary, angle)
    deskewed = rotateRgb(rgb, angle, true)
  } else if (Math.abs(angle) > 1e-3) {
    binary = rotateNearest(binary, angle)
    deskewed = rotateRgb(rgb, angle)
  }
  cp()
  let pitch = estimateLinePitch(binary)
  binary = removeRules(binary, pitch)
  pitch = estimateLinePitch(binary)
  cp()
  let boxes = findLines(binary, pitch)
  boxes = splitTall(binary, boxes, pitch)
  boxes = mergeOverlapping(boxes, pitch)
  boxes = readingOrder(boxes, pitch)
  const inv = 1 / scale
  const full = boxes.map((b) => ({
    x: Math.trunc(b.x * inv),
    y: Math.trunc(b.y * inv),
    w: Math.trunc(b.w * inv),
    h: Math.trunc(b.h * inv),
  }))
  return { angle, pitch, scale, boxes: full, deskewed }
}

/** The padded crop rectangle of a line box, clamped to the page. */
export function cropRect(b: Box, width: number, height: number): Rect {
  const py = Math.trunc(b.h * 0.12)
  const px = Math.trunc(b.h * 0.05)
  const x0 = Math.min(Math.max(b.x - px, 0), width - 1)
  const y0 = Math.min(Math.max(b.y - py, 0), height - 1)
  const x1 = Math.min(b.x + b.w + px, width)
  const y1 = Math.min(b.y + b.h + py, height)
  return { x: x0, y: y0, w: Math.max(1, x1 - x0), h: Math.max(1, y1 - y0) }
}

/** The padded RGB crop of line box `b` from the (deskewed) page. */
export function lineCropRgb(page: Rgb, b: Box): Rgb {
  return cropRgb(page, cropRect(b, page.width, page.height))
}

/** The padded gray crop of line box `b`: crop the RGB, then convert, as the reference does. */
export function lineCrop(page: Rgb, b: Box): Gray {
  return rgbToGray(lineCropRgb(page, b))
}

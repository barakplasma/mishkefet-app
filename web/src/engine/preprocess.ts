/**
 * Line crop to model input (spec/preprocess.md): contrast normalization at the
 * crop's own resolution, resize to height 64, invert. Plus the TTA views.
 */

import { LINE_HEIGHT, MAX_WIDTH, MIN_WIDTH } from './constants.ts'
import { bicubicResizeX, blurLarge, percentileSorted, resize, roundHalfEven, type Gray } from './image.ts'

const EPS32 = Math.fround(1e-3)

/** Divide by a heavily blurred copy, then stretch the 2nd..98th percentile to 0..255. */
export function contrastNormalize(g: Gray): Gray {
  const blur = blurLarge(g, Math.max(g.height / 8, 3))
  const n = g.data.length
  const flat = new Float64Array(n)
  for (let i = 0; i < n; i++) flat[i] = Math.fround(g.data[i] / Math.max(blur[i], EPS32))
  const sorted = Float64Array.from(flat).sort()
  const lo = percentileSorted(sorted, 2)
  const hi = percentileSorted(sorted, 98)
  const out = new Uint8Array(n)
  if (hi - lo < 1e-6) {
    out.fill(255)
  } else {
    for (let i = 0; i < n; i++) {
      const v = ((flat[i] - lo) / (hi - lo)) * 255
      out[i] = v <= 0 ? 0 : v >= 255 ? 255 : Math.trunc(v)
    }
  }
  return { data: out, width: g.width, height: g.height }
}

/** Width the model input will have for a crop of this size. */
export function modelWidth(width: number, height: number): number {
  const scale = LINE_HEIGHT / Math.max(height, 1)
  return Math.max(MIN_WIDTH, Math.min(roundHalfEven(width * scale), MAX_WIDTH))
}

/** uint8 line crop to float32 (64, W), ink near 1, paper near 0. */
export function preprocess(gray: Gray): { data: Float32Array; width: number } {
  const g = contrastNormalize(gray)
  const scale = LINE_HEIGHT / Math.max(g.height, 1)
  const newW = modelWidth(g.width, g.height)
  const r = resize(g, newW, LINE_HEIGHT, scale < 1 ? 'area' : 'linear')
  const out = new Float32Array(r.data.length)
  for (let i = 0; i < out.length; i++) out[i] = 1 - Math.fround(r.data[i] / 255)
  return { data: out, width: newW }
}

/** A TTA view: horizontal-only bicubic resize to max(8, trunc(w * s)) columns. */
export function ttaView(gray: Gray, scale: number): Gray {
  if (scale === 1) return gray
  return bicubicResizeX(gray, Math.max(8, Math.trunc(gray.width * scale)))
}

/**
 * Binary morphology with OpenCV's conventions (spec/segmenter.md): kernel
 * anchor (cols // 2, rows // 2), no kernel reflection for dilate, and pixels
 * outside the image never count (erode sees 255 there, dilate 0).
 *
 * Every kernel used by the segmenter has a single run of ones per row, so an
 * erode/dilate is a min/max over shifted 1-D running windows; the windows are
 * computed with van Herk / Gil-Werman in O(1) per pixel whatever their length.
 * Kernels whose columns are runs are applied to the transposed image.
 */

import { roundHalfEven } from './image.ts'

/** A 0/1 kernel stored as one [first, last] column run per row (null: empty row). */
export interface Kernel {
  rows: number
  cols: number
  runs: Array<[number, number] | null>
}

export type MorphOp = 'erode' | 'dilate'

export function rectKernel(w: number, h: number): Kernel {
  return { rows: h, cols: w, runs: Array.from({ length: h }, () => [0, w - 1] as [number, number]) }
}

/** OpenCV LineIterator (connectivity 8) pixel sequence from (x0, y0) to (x1, y1). */
export function cvLine(x0: number, y0: number, x1: number, y1: number): Array<[number, number]> {
  let dx = x1 - x0
  let dy = y1 - y0
  const s = dx < 0 ? -1 : 0
  dx = (dx ^ s) - s
  const sy = dy < 0 ? -1 : 0
  dy = (dy ^ sy) - sy
  const stepx = s ? -1 : 1
  const stepy = sy ? -1 : 1
  const swap = dy > dx
  if (swap) [dx, dy] = [dy, dx]
  let err = dx - (dy + dy)
  const plusDelta = dx + dx
  const minusDelta = -(dy + dy)
  const count = dx + 1
  let x = x0
  let y = y0
  const pts: Array<[number, number]> = []
  for (let i = 0; i < count; i++) {
    pts.push([x, y])
    const mask = err < 0 ? -1 : 0
    err += minusDelta + (plusDelta & mask)
    if (swap) {
      y += stepy
      if (mask) x += stepx
    } else {
      x += stepx
      if (mask) y += stepy
    }
  }
  return pts
}

/** A thin line `length` px long at `angle` degrees, as reference.oriented_line_kernel. */
export function lineKernel(length: number, angle: number): Kernel {
  const drop = roundHalfEven(Math.tan(angle * (Math.PI / 180)) * (length - 1))
  const rows = Math.abs(drop) + 1
  const y0 = drop >= 0 ? 0 : rows - 1
  const runs: Array<[number, number] | null> = Array.from({ length: rows }, () => null)
  for (const [x, y] of cvLine(0, y0, length - 1, y0 + drop)) {
    const r = runs[y]
    if (!r) runs[y] = [x, x]
    else {
      if (x < r[0] - 1 || x > r[1] + 1) throw new Error('line kernel row is not a single run')
      r[0] = Math.min(r[0], x)
      r[1] = Math.max(r[1], x)
    }
  }
  return { rows, cols: length, runs }
}

export function transposeKernel(k: Kernel): Kernel {
  // Only used on kernels whose columns are single runs.
  const runs: Array<[number, number] | null> = Array.from({ length: k.cols }, () => null)
  k.runs.forEach((run, r) => {
    if (!run) return
    for (let c = run[0]; c <= run[1]; c++) {
      const t = runs[c]
      if (!t) runs[c] = [r, r]
      else {
        t[0] = Math.min(t[0], r)
        t[1] = Math.max(t[1], r)
      }
    }
  })
  return { rows: k.cols, cols: k.rows, runs }
}

/** Transpose in 64 x 64 tiles (cache-friendly on page-sized masks). */
export function transpose(a: Uint8Array, w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h)
  const B = 64
  for (let y0 = 0; y0 < h; y0 += B) {
    const y1 = Math.min(y0 + B, h)
    for (let x0 = 0; x0 < w; x0 += B) {
      const x1 = Math.min(x0 + B, w)
      for (let y = y0; y < y1; y++) {
        const r = y * w
        for (let x = x0; x < x1; x++) out[x * h + y] = a[r + x]
      }
    }
  }
  return out
}

/**
 * R[y][x' + L - 1] = op(a[y][x' .. x' + L - 1]) over its in-bounds part, for
 * x' in [-(L - 1), w - 1]. Row stride w + L - 1. Van Herk / Gil-Werman: a
 * forward and a backward running extreme inside blocks of L, combined.
 */
function runExtreme(a: Uint8Array, w: number, h: number, L: number, isMin: boolean): Uint8Array {
  const neutral = isMin ? 255 : 0
  const stride = w + L - 1
  const out = new Uint8Array(h * stride)
  const n = w + 2 * (L - 1)
  const p = new Uint8Array(n)
  const g = new Uint8Array(n)
  const hh = new Uint8Array(n)
  for (let y = 0; y < h; y++) {
    p.fill(neutral)
    p.set(a.subarray(y * w, y * w + w), L - 1)
    for (let b0 = 0; b0 < n; b0 += L) {
      const b1 = Math.min(b0 + L, n)
      let acc = p[b0]
      g[b0] = acc
      if (isMin) {
        for (let i = b0 + 1; i < b1; i++) {
          const v = p[i]
          if (v < acc) acc = v
          g[i] = acc
        }
        acc = p[b1 - 1]
        hh[b1 - 1] = acc
        for (let i = b1 - 2; i >= b0; i--) {
          const v = p[i]
          if (v < acc) acc = v
          hh[i] = acc
        }
      } else {
        for (let i = b0 + 1; i < b1; i++) {
          const v = p[i]
          if (v > acc) acc = v
          g[i] = acc
        }
        acc = p[b1 - 1]
        hh[b1 - 1] = acc
        for (let i = b1 - 2; i >= b0; i--) {
          const v = p[i]
          if (v > acc) acc = v
          hh[i] = acc
        }
      }
    }
    const o = y * stride
    const off = L - 1
    if (isMin) {
      for (let i = 0; i < stride; i++) {
        const u = hh[i]
        const v = g[i + off]
        out[o + i] = u < v ? u : v
      }
    } else {
      for (let i = 0; i < stride; i++) {
        const u = hh[i]
        const v = g[i + off]
        out[o + i] = u > v ? u : v
      }
    }
  }
  return out
}

/** Erode or dilate a w x h uint8 image with a kernel whose rows are single runs. */
export function morph(a: Uint8Array, w: number, h: number, k: Kernel, op: MorphOp): Uint8Array {
  const isMin = op === 'erode'
  const neutral = isMin ? 255 : 0
  const ax = k.cols >> 1
  const ay = k.rows >> 1
  const out = new Uint8Array(w * h).fill(neutral)
  const cache = new Map<number, Uint8Array>()
  for (let r = 0; r < k.rows; r++) {
    const run = k.runs[r]
    if (!run) continue
    const L = run[1] - run[0] + 1
    let R = cache.get(L)
    if (!R) {
      R = runExtreme(a, w, h, L, isMin)
      cache.set(L, R)
    }
    const stride = w + L - 1
    const dx = run[0] - ax
    const dy = r - ay
    // window [x + dx, x + dx + L - 1] must touch [0, w)
    const xLo = Math.max(0, -(L - 1) - dx)
    const xHi = Math.min(w - 1, w - 1 - dx)
    for (let y = 0; y < h; y++) {
      const sy = y + dy
      if (sy < 0 || sy >= h) continue
      const src = sy * stride + dx + L - 1
      const o = y * w
      if (isMin) {
        for (let x = xLo; x <= xHi; x++) {
          const v = R[src + x]
          if (v < out[o + x]) out[o + x] = v
        }
      } else {
        for (let x = xLo; x <= xHi; x++) {
          const v = R[src + x]
          if (v > out[o + x]) out[o + x] = v
        }
      }
    }
  }
  return out
}

export function opening(a: Uint8Array, w: number, h: number, k: Kernel): Uint8Array {
  return morph(morph(a, w, h, k, 'erode'), w, h, k, 'dilate')
}

/** Element-wise max into `acc`. */
export function maxInto(acc: Uint8Array, b: Uint8Array): void {
  for (let i = 0; i < acc.length; i++) if (b[i] > acc[i]) acc[i] = b[i]
}

/** Saturating a - b. */
export function subSat(a: Uint8Array, b: Uint8Array): Uint8Array {
  const out = new Uint8Array(a.length)
  for (let i = 0; i < a.length; i++) out[i] = a[i] > b[i] ? a[i] - b[i] : 0
  return out
}

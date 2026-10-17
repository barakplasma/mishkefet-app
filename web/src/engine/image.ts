/**
 * Image primitives shared by line preprocessing and page segmentation.
 * Grayscale images are row-major Uint8Array (one byte per pixel).
 *
 * Every function mirrors the numpy reference (tools/model/reference.py): the
 * arithmetic is in double unless the reference works in float32, and the
 * rounding rules are the reference's, because a level or two of difference in
 * a blurred background moves binarization thresholds and so line boxes.
 */

import { createFFT } from './fft.ts'

export interface Gray {
  data: Uint8Array
  width: number
  height: number
}

/**
 * RGBA (canvas ImageData order) to gray: alpha composited onto white, then
 * PIL's "L" conversion `(R*19595 + G*38470 + B*7471 + 0x8000) >> 16` (BT.601).
 */
export function rgbaToGray(rgba: ArrayLike<number>, width: number, height: number): Gray {
  const n = width * height
  const out = new Uint8Array(n)
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    let r = rgba[p]
    let g = rgba[p + 1]
    let b = rgba[p + 2]
    const a = rgba[p + 3]
    if (a !== 255) {
      // Over white: c * a + 255 * (1 - a), rounded like PIL's alpha_composite.
      const ia = 255 - a
      r = Math.floor((r * a + 255 * ia) / 255 + 0.5)
      g = Math.floor((g * a + 255 * ia) / 255 + 0.5)
      b = Math.floor((b * a + 255 * ia) / 255 + 0.5)
    }
    out[i] = (r * 19595 + g * 38470 + b * 7471 + 0x8000) >>> 16
  }
  return { data: out, width, height }
}

/** OpenCV saturate_cast<uchar>: round half away from zero (for x >= 0), clamp. */
export function roundU8(x: number): number {
  const v = Math.floor(x + 0.5)
  return v < 0 ? 0 : v > 255 ? 255 : v
}

/** Python's round(): half to even. */
export function roundHalfEven(x: number): number {
  const f = Math.floor(x)
  const d = x - f
  if (d > 0.5) return f + 1
  if (d < 0.5) return f
  return f % 2 === 0 ? f : f + 1
}

/** OpenCV getGaussianKernel, including its fixed small-kernel tables. */
export function gaussianKernel(ksize: number, sigma: number): Float64Array {
  if (sigma <= 0) {
    const fixed: Record<number, number[]> = {
      1: [1],
      3: [0.25, 0.5, 0.25],
      5: [0.0625, 0.25, 0.375, 0.25, 0.0625],
      7: [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125],
    }
    if (fixed[ksize]) return Float64Array.from(fixed[ksize])
    sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
  }
  const k = new Float64Array(ksize)
  let sum = 0
  for (let i = 0; i < ksize; i++) {
    const x = i - (ksize - 1) / 2
    k[i] = Math.exp(-(x * x) / (2 * sigma * sigma))
    sum += k[i]
  }
  for (let i = 0; i < ksize; i++) k[i] /= sum
  return k
}

/** OpenCV's kernel size for GaussianBlur(ksize=(0,0)) on 8-bit input. */
export function gaussianKsizeU8(sigma: number): number {
  return Math.floor(sigma * 6 + 1 + 0.5) | 1
}

export type Border = 'reflect101' | 'replicate'

/** Source indices for a signal of length n padded by r on each side. */
export function padIndex(n: number, r: number, border: Border): Int32Array {
  const out = new Int32Array(n + 2 * r)
  for (let k = 0; k < out.length; k++) {
    const i = k - r
    if (n === 1) out[k] = 0
    else if (border === 'replicate') out[k] = i < 0 ? 0 : i >= n ? n - 1 : i
    else {
      const period = 2 * (n - 1)
      const m = ((i % period) + period) % period
      out[k] = m >= n ? period - m : m
    }
  }
  return out
}

/** Kernels longer than this go through the FFT (a page background blur has 300+ taps). */
const FFT_TAPS = 64

/** Correlate every row of a (width x height) with a symmetric kernel, directly. */
function correlateRowsDirect(a: Float64Array, width: number, height: number, k: Float64Array, border: Border): Float64Array {
  const r = k.length >> 1
  const idx = padIndex(width, r, border)
  const row = new Float64Array(width + 2 * r)
  const out = new Float64Array(width * height)
  const kc = k[r]
  for (let y = 0; y < height; y++) {
    const base = y * width
    for (let i = 0; i < row.length; i++) row[i] = a[base + idx[i]]
    for (let x = 0; x < width; x++) {
      let s = kc * row[x + r]
      for (let j = 0; j < r; j++) s += k[j] * (row[x + j] + row[x + 2 * r - j])
      out[base + x] = s
    }
  }
  return out
}

/**
 * The same through the FFT, two rows per complex transform. With the padded
 * row p (length width + 2r) and a symmetric kernel, the correlation at x is
 * the circular convolution at x + 2r, which never wraps when n >= width + 2r.
 */
function correlateRowsFFT(a: Float64Array, width: number, height: number, k: Float64Array, border: Border): Float64Array {
  const r = k.length >> 1
  const idx = padIndex(width, r, border)
  let n = 1
  while (n < width + 2 * r) n <<= 1
  const fft = createFFT(n)
  const kr = new Float64Array(n)
  const ki = new Float64Array(n)
  kr.set(k)
  fft.transform(kr, ki, false)
  const re = new Float64Array(n)
  const im = new Float64Array(n)
  const out = new Float64Array(width * height)
  for (let y = 0; y < height; y += 2) {
    const b1 = y * width
    const two = y + 1 < height
    const b2 = b1 + width
    re.fill(0)
    im.fill(0)
    for (let i = 0; i < idx.length; i++) {
      re[i] = a[b1 + idx[i]]
      if (two) im[i] = a[b2 + idx[i]]
    }
    fft.transform(re, im, false)
    for (let i = 0; i < n; i++) {
      const xr = re[i]
      const xi = im[i]
      re[i] = xr * kr[i] - xi * ki[i]
      im[i] = xr * ki[i] + xi * kr[i]
    }
    fft.transform(re, im, true)
    for (let x = 0; x < width; x++) {
      out[b1 + x] = re[x + 2 * r]
      if (two) out[b2 + x] = im[x + 2 * r]
    }
  }
  return out
}

function transpose64(a: Float64Array, w: number, h: number): Float64Array {
  const out = new Float64Array(w * h)
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[x * h + y] = a[y * w + x]
  return out
}

/**
 * Separable correlation in double, rows then columns. Kernels must be
 * symmetric (all of ours are), which halves the multiplies.
 */
export function convolveSep(
  src: ArrayLike<number>,
  width: number,
  height: number,
  kx: Float64Array,
  ky: Float64Array,
  border: Border = 'reflect101',
): Float64Array {
  const rx = kx.length >> 1
  const ry = ky.length >> 1
  let a: Float64Array = Float64Array.from(src)
  if (rx) a = (kx.length > FFT_TAPS ? correlateRowsFFT : correlateRowsDirect)(a, width, height, kx, border)
  if (ry && ky.length > FFT_TAPS) {
    a = transpose64(correlateRowsFFT(transpose64(a, width, height), height, width, ky, border), height, width)
  } else if (ry) {
    const idx = padIndex(height, ry, border)
    const out = new Float64Array(width * height)
    const kc = ky[ry]
    for (let y = 0; y < height; y++) {
      const o = y * width
      const c = idx[y + ry] * width
      for (let x = 0; x < width; x++) out[o + x] = kc * a[c + x]
      for (let j = 0; j < ry; j++) {
        const p = idx[y + j] * width
        const q = idx[y + 2 * ry - j] * width
        const w = ky[j]
        for (let x = 0; x < width; x++) out[o + x] += w * (a[p + x] + a[q + x])
      }
    }
    a = out
  }
  return a
}

/** Gaussian blur of an 8-bit image with OpenCV's kernel size, exact, as uint8 (reflect101). */
export function blurLarge(g: Gray, sigma: number): Uint8Array {
  const k = gaussianKernel(gaussianKsizeU8(sigma), sigma)
  const f = convolveSep(g.data, g.width, g.height, k, k)
  const out = new Uint8Array(f.length)
  for (let i = 0; i < f.length; i++) out[i] = roundU8(f[i])
  return out
}

/** Sparse resampling weights: output i reads src[start[i] + j] * w[off[i] + j] for j < len[i]. */
interface Weights {
  start: Int32Array
  len: Int32Array
  off: Int32Array
  w: Float64Array
}

function packWeights(rows: Array<{ start: number; w: number[] }>): Weights {
  const n = rows.length
  const start = new Int32Array(n)
  const len = new Int32Array(n)
  const off = new Int32Array(n)
  let total = 0
  rows.forEach((r, i) => {
    start[i] = r.start
    len[i] = r.w.length
    off[i] = total
    total += r.w.length
  })
  const w = new Float64Array(total)
  rows.forEach((r, i) => w.set(r.w, off[i]))
  return { start, len, off, w }
}

/** Box-filter weights: dst i covers src [i*scale, (i+1)*scale), normalized. */
function areaWeights(nSrc: number, nDst: number, scale: number): Weights {
  const rows: Array<{ start: number; w: number[] }> = []
  for (let i = 0; i < nDst; i++) {
    const a = i * scale
    const b = Math.min((i + 1) * scale, nSrc)
    const j0 = Math.floor(a)
    const j1 = Math.min(Math.ceil(b), nSrc)
    const w: number[] = []
    let s = 0
    for (let j = j0; j < j1; j++) {
      const v = Math.min(b, j + 1) - Math.max(a, j)
      w.push(v)
      s += v
    }
    if (s > 0) for (let k = 0; k < w.length; k++) w[k] /= s
    rows.push({ start: j0, w })
  }
  return packWeights(rows)
}

/** Bilinear weights with half-pixel centres, edges clamped. */
function linearWeights(nSrc: number, nDst: number, scale: number): Weights {
  const rows: Array<{ start: number; w: number[] }> = []
  for (let i = 0; i < nDst; i++) {
    let f = (i + 0.5) * scale - 0.5
    let j = Math.floor(f)
    f -= j
    if (j < 0) {
      j = 0
      f = 0
    }
    if (j >= nSrc - 1) {
      j = nSrc - 1
      f = 0
    }
    rows.push({ start: j, w: f ? [1 - f, f] : [1] })
  }
  return packWeights(rows)
}

/** PIL BICUBIC (a = -0.5) weights, support widened when shrinking. */
function bicubicWeights(nSrc: number, nDst: number): Weights {
  const scale = nSrc / nDst
  const fscale = Math.max(scale, 1)
  const support = 2 * fscale
  const cubic = (x: number) => {
    x = Math.abs(x)
    if (x < 1) return (1.5 * x - 2.5) * x * x + 1
    if (x < 2) return ((-0.5 * x + 2.5) * x - 4) * x + 2
    return 0
  }
  const rows: Array<{ start: number; w: number[] }> = []
  for (let i = 0; i < nDst; i++) {
    const centre = (i + 0.5) * scale
    const lo = Math.max(Math.trunc(centre - support + 0.5), 0)
    const hi = Math.min(Math.trunc(centre + support + 0.5), nSrc)
    const w: number[] = []
    let s = 0
    for (let j = lo; j < hi; j++) {
      const v = cubic((j - centre + 0.5) / fscale)
      w.push(v)
      s += v
    }
    if (s) for (let k = 0; k < w.length; k++) w[k] /= s
    rows.push({ start: lo, w })
  }
  return packWeights(rows)
}

function applyRows(src: ArrayLike<number>, width: number, wy: Weights, outH: number): Float64Array {
  const out = new Float64Array(outH * width)
  for (let i = 0; i < outH; i++) {
    const o = i * width
    for (let k = 0; k < wy.len[i]; k++) {
      const w = wy.w[wy.off[i] + k]
      if (!w) continue
      const p = (wy.start[i] + k) * width
      for (let x = 0; x < width; x++) out[o + x] += w * src[p + x]
    }
  }
  return out
}

function applyCols(src: ArrayLike<number>, width: number, height: number, wx: Weights, outW: number): Uint8Array {
  const out = new Uint8Array(height * outW)
  for (let y = 0; y < height; y++) {
    const p = y * width
    const o = y * outW
    for (let i = 0; i < outW; i++) {
      let s = 0
      const st = p + wx.start[i]
      const off = wx.off[i]
      for (let k = 0; k < wx.len[i]; k++) s += wx.w[off + k] * src[st + k]
      out[o + i] = roundU8(s)
    }
  }
  return out
}

/**
 * Area (shrinking) or bilinear (enlarging) resize to uint8, separable, in
 * double. `sx`, `sy` override source-per-destination, the way OpenCV uses
 * 1/fx rather than w/new_w when a resize is given as a factor.
 */
export function resize(g: Gray, newW: number, newH: number, mode: 'area' | 'linear', sx?: number, sy?: number): Gray {
  const make = mode === 'area' ? areaWeights : linearWeights
  const wx = make(g.width, newW, sx ?? g.width / newW)
  const wy = make(g.height, newH, sy ?? g.height / newH)
  const tmp = applyRows(g.data, g.width, wy, newH)
  return { data: applyCols(tmp, g.width, newH, wx, newW), width: newW, height: newH }
}

/** Horizontal-only resize with PIL's BICUBIC (the TTA views). */
export function bicubicResizeX(g: Gray, newW: number): Gray {
  const wx = bicubicWeights(g.width, newW)
  return { data: applyCols(g.data, g.width, g.height, wx, newW), width: newW, height: g.height }
}

/** numpy's default (linear) percentile of an already sorted array. */
export function percentileSorted(s: ArrayLike<number>, q: number): number {
  const pos = (q / 100) * (s.length - 1)
  const lo = Math.floor(pos)
  const hi = Math.min(lo + 1, s.length - 1)
  return s[lo] + (s[hi] - s[lo]) * (pos - lo)
}

export interface Rect {
  x: number
  y: number
  w: number
  h: number
}

/** An RGB image (3 bytes per pixel, alpha already composited onto white). */
export interface Rgb {
  data: Uint8Array
  width: number
  height: number
}

/** RGBA to RGB with alpha composited onto white (same rounding as rgbaToGray). */
export function rgbaToRgb(rgba: ArrayLike<number>, width: number, height: number): Rgb {
  const n = width * height
  const out = new Uint8Array(n * 3)
  for (let i = 0, p = 0, q = 0; i < n; i++, p += 4, q += 3) {
    const a = rgba[p + 3]
    for (let c = 0; c < 3; c++) {
      const v = rgba[p + c]
      out[q + c] = a === 255 ? v : Math.floor((v * a + 255 * (255 - a)) / 255 + 0.5)
    }
  }
  return { data: out, width, height }
}

export function rgbToGray(rgb: Rgb): Gray {
  const n = rgb.width * rgb.height
  const out = new Uint8Array(n)
  const d = rgb.data
  for (let i = 0, q = 0; i < n; i++, q += 3) out[i] = (d[q] * 19595 + d[q + 1] * 38470 + d[q + 2] * 7471 + 0x8000) >>> 16
  return { data: out, width: rgb.width, height: rgb.height }
}

/**
 * The `rect` of an image (`channels` bytes per pixel) rotated by `angle`
 * degrees (counter-clockwise on screen) about the image centre, sampled the
 * way PIL's Image.rotate(BICUBIC, fillcolor=white) does: pixel centres, the
 * a = -1 cubic, edge-clamped neighbours, each channel truncated to uint8.
 * Only the crop is computed, so deskewing costs nothing beyond the line crops.
 */
function rotatedCrop(data: Uint8Array, W: number, H: number, channels: number, angle: number, rect: Rect): Uint8Array {
  const out = new Uint8Array(rect.w * rect.h * channels)
  const a = (angle * Math.PI) / 180
  const c = Math.cos(a)
  const s = Math.sin(a)
  const cx = W / 2
  const cy = H / 2
  const stride = W * channels
  const row = (y: number) => (y < 0 ? 0 : y >= H ? H - 1 : y) * stride
  const col = (x: number) => (x < 0 ? 0 : x >= W ? W - 1 : x) * channels
  const cub = (v1: number, v2: number, v3: number, v4: number, d: number) =>
    v2 + d * (-v1 + v3 + d * (2 * (v1 - v2) + v3 - v4 + d * (-v1 + v2 - v3 + v4)))
  for (let j = 0; j < rect.h; j++) {
    const Y = rect.y + j + 0.5 - cy
    for (let i = 0; i < rect.w; i++) {
      const X = rect.x + i + 0.5 - cx
      let xin = c * X - s * Y + cx
      let yin = s * X + c * Y + cy
      const o = (j * rect.w + i) * channels
      if (xin < 0 || xin >= W || yin < 0 || yin >= H) {
        out.fill(255, o, o + channels)
        continue
      }
      xin -= 0.5
      yin -= 0.5
      const x = Math.floor(xin)
      const y = Math.floor(yin)
      const dx = xin - x
      const dy = yin - y
      const x0 = col(x - 1)
      const x1 = col(x)
      const x2 = col(x + 1)
      const x3 = col(x + 2)
      const r0 = row(y - 1)
      const r1 = row(y)
      const r2 = row(y + 1)
      const r3 = row(y + 2)
      for (let ch = 0; ch < channels; ch++) {
        const v0 = cub(data[r0 + x0 + ch], data[r0 + x1 + ch], data[r0 + x2 + ch], data[r0 + x3 + ch], dx)
        const v1 = cub(data[r1 + x0 + ch], data[r1 + x1 + ch], data[r1 + x2 + ch], data[r1 + x3 + ch], dx)
        const v2 = cub(data[r2 + x0 + ch], data[r2 + x1 + ch], data[r2 + x2 + ch], data[r2 + x3 + ch], dx)
        const v3 = cub(data[r3 + x0 + ch], data[r3 + x1 + ch], data[r3 + x2 + ch], data[r3 + x3 + ch], dx)
        const v = cub(v0, v1, v2, v3, dy)
        out[o + ch] = v <= 0 ? 0 : v >= 255 ? 255 : Math.trunc(v)
      }
    }
  }
  return out
}

/** Rotated crop of a gray image (white outside). */
export function cropRotated(g: Gray, angle: number, rect: Rect): Gray {
  return { data: rotatedCrop(g.data, g.width, g.height, 1, angle, rect), width: rect.w, height: rect.h }
}

/** Rotated crop of an RGB image, converted to gray after rotating, as the reference does. */
export function cropRotatedRgb(img: Rgb, angle: number, rect: Rect): Gray {
  const rgb = rotatedCrop(img.data, img.width, img.height, 3, angle, rect)
  return rgbToGray({ data: rgb, width: rect.w, height: rect.h })
}

/** A plain (unrotated) crop of a gray image. */
export function crop(g: Gray, rect: Rect): Gray {
  const out = new Uint8Array(rect.w * rect.h)
  for (let j = 0; j < rect.h; j++) {
    const p = (rect.y + j) * g.width + rect.x
    out.set(g.data.subarray(p, p + rect.w), j * rect.w)
  }
  return { data: out, width: rect.w, height: rect.h }
}

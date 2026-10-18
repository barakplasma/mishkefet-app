/**
 * Per-line tilt and curve (spec/straighten.md, reference.straighten_line).
 *
 * Handwriting on one page does not share one angle, so after the page is
 * deskewed each line crop gets its own fix: fit a quadratic through the ink's
 * vertical centre per column, shift every column so that centreline is level,
 * and keep only a band around it (cutting away neighbouring lines' ink the
 * axis-aligned crop caught). Lines that are already straight come back as the
 * same object.
 */

import { otsuThreshold, roundU8, type Gray } from './image.ts'
import { contrastNormalize } from './preprocess.ts'

const MIN_DEV = 0.15 // fraction of the height the centreline must wander to act
const BAND = 2.5 // keep rows within this many ink standard deviations of the line

/**
 * Weighted least-squares quadratic (numpy polyfit(x, y, 2, w=sqrt(wt))).
 * Solved on centred, scaled x so the normal equations stay well conditioned
 * for crops thousands of pixels wide; the fitted function is the same.
 */
function fitQuadratic(xs: number[], ys: number[], wt: number[]): (x: number) => number {
  let lo = Infinity
  let hi = -Infinity
  for (const x of xs) {
    lo = Math.min(lo, x)
    hi = Math.max(hi, x)
  }
  const c = (lo + hi) / 2
  const sc = Math.max((hi - lo) / 2, 1)
  // normal equations M a = v for basis [1, u, u^2]
  const M = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ]
  const v = [0, 0, 0]
  for (let i = 0; i < xs.length; i++) {
    const u = (xs[i] - c) / sc
    const b = [1, u, u * u]
    for (let r = 0; r < 3; r++) {
      v[r] += wt[i] * b[r] * ys[i]
      for (let k = 0; k < 3; k++) M[r][k] += wt[i] * b[r] * b[k]
    }
  }
  // Gaussian elimination with partial pivoting
  for (let col = 0; col < 3; col++) {
    let piv = col
    for (let r = col + 1; r < 3; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r
    ;[M[col], M[piv]] = [M[piv], M[col]]
    ;[v[col], v[piv]] = [v[piv], v[col]]
    const d = M[col][col]
    if (d === 0) continue
    for (let r = col + 1; r < 3; r++) {
      const f = M[r][col] / d
      for (let k = col; k < 3; k++) M[r][k] -= f * M[col][k]
      v[r] -= f * v[col]
    }
  }
  const a = [0, 0, 0]
  for (let r = 2; r >= 0; r--) {
    let s = v[r]
    for (let k = r + 1; k < 3; k++) s -= M[r][k] * a[k]
    a[r] = M[r][r] ? s / M[r][r] : 0
  }
  return (x: number) => {
    const u = (x - c) / sc
    return a[0] + u * (a[1] + u * a[2])
  }
}

/**
 * (f, sigma): the ink centreline per column (a weighted quadratic, refitted
 * without columns far from it, at most twice) and the ink's spread around it.
 * Null when there is too little ink to tell.
 */
export function lineCenterline(gray: Gray): { f: Float64Array; sigma: number } | null {
  const { width: w, height: h } = gray
  const flat = contrastNormalize(gray).data
  // ink weight: how far below the crop's own Otsu split a pixel is
  const t = otsuThreshold(flat)
  const wt = new Float64Array(w * h)
  const tw = Math.max(t, 1)
  for (let i = 0; i < wt.length; i++) {
    const v = (t - flat[i]) / tw
    wt[i] = v < 0 ? 0 : v > 1 ? 1 : v
  }
  const col = new Float64Array(w)
  const mom = new Float64Array(w)
  for (let y = 0; y < h; y++) {
    const o = y * w
    for (let x = 0; x < w; x++) {
      col[x] += wt[o + x]
      mom[x] += wt[o + x] * y
    }
  }
  let xs: number[] = []
  let m: number[] = []
  let wk: number[] = []
  for (let x = 0; x < w; x++) {
    if (col[x] > 0.5) {
      xs.push(x)
      m.push(mom[x] / col[x])
      wk.push(col[x])
    }
  }
  if (xs.length < Math.max(8, w * 0.15)) return null
  let fit: (x: number) => number = () => 0
  for (let iter = 0; iter < 2; iter++) {
    fit = fitQuadratic(xs, m, wk)
    const res = xs.map((x, i) => m[i] - fit(x))
    let num = 0
    let den = 0
    for (let i = 0; i < res.length; i++) {
      num += wk[i] * res[i] * res[i]
      den += wk[i]
    }
    const rms = Math.sqrt(num / den)
    const lim = Math.max(0.18 * h, 2.5 * rms + 1)
    const good = res.map((r) => Math.abs(r) < lim)
    if (good.every(Boolean)) break
    xs = xs.filter((_, i) => good[i])
    m = m.filter((_, i) => good[i])
    wk = wk.filter((_, i) => good[i])
    if (xs.length < 8) return null
  }
  const f = new Float64Array(w)
  for (let x = 0; x < w; x++) f[x] = fit(x)
  let num = 0
  let den = 0
  for (let y = 0; y < h; y++) {
    const o = y * w
    for (let x = 0; x < w; x++) {
      const d = y - f[x]
      num += wt[o + x] * d * d
      den += wt[o + x]
    }
  }
  return { f, sigma: Math.sqrt(num / Math.max(den, 1e-6)) }
}

/** Level a tilted or curved line and keep the band around it; straight lines are returned as is. */
export function straightenLine(gray: Gray): Gray {
  const { width: w, height: h, data } = gray
  const fit = lineCenterline(gray)
  if (!fit) return gray
  const { f, sigma } = fit
  let mean = 0
  for (let x = 0; x < w; x++) mean += f[x]
  mean /= w
  const dev = new Float64Array(w)
  let maxDev = 0
  for (let x = 0; x < w; x++) {
    dev[x] = f[x] - mean
    maxDev = Math.max(maxDev, Math.abs(dev[x]))
  }
  if (maxDev < MIN_DEV * h) return gray
  const pad = Math.ceil(maxDev)
  const H = h + 2 * pad
  const c = mean + pad
  const band = Math.max(BAND * sigma, 0.22 * h)
  const top = Math.trunc(Math.max(0, c - band))
  const bot = Math.trunc(Math.min(H, c + band + 1))
  const outH = Math.max(0, bot - top)
  const out = new Uint8Array(outH * w)
  for (let y = top; y < bot; y++) {
    const o = (y - top) * w
    for (let x = 0; x < w; x++) {
      const src = y - pad + dev[x]
      let v = 255
      if (src >= 0 && src <= h - 1) {
        const i0 = Math.floor(src)
        const i1 = Math.min(i0 + 1, h - 1)
        const a = src - i0
        v = data[i0 * w + x] * (1 - a) + data[i1 * w + x] * a
      }
      out[o + x] = roundU8(v)
    }
  }
  return { data: out, width: w, height: outH }
}

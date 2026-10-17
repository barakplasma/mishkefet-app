/**
 * Training-canvas strokes to a grayscale line crop (spec/strokes.md), plus
 * the deterministic augmentation (mulberry32) and sample hashing (FNV-1a) both
 * apps share. Rasterization is analytic (distance to each segment), so no
 * canvas is involved and web and iOS draw the same pixels.
 */

import type { Stroke } from './api.ts'
import type { Gray } from './image.ts'

const H = 128
const INK = 0.62
const PAD_X = 12
const RADIUS = 1.2
const PX_PER_CHAR = 44.2
const PAPER = 255
const INK_LEVEL = 30

export interface RenderParams {
  slant: number
  rotate: number
  radius: number
  density: number
  ink: number
}

/** The PRNG both apps use for augmentation: floats in [0, 1). */
export function mulberry32(seed: number): () => number {
  let state = seed >>> 0
  return () => {
    state = (state + 0x6d2b79f5) >>> 0
    let t = Math.imul(state ^ (state >>> 15), state | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** FNV-1a 32 of the UTF-8 bytes of `s`. */
export function fnv1a32(s: string): number {
  let h = 0x811c9dc5
  for (const b of new TextEncoder().encode(s)) {
    h ^= b
    h = Math.imul(h, 0x01000193) >>> 0
  }
  return h >>> 0
}

/** Seeds of a sample's training views: 0 (plain), then AUG augmented ones. */
export function viewSeeds(sampleId: string, aug: number): number[] {
  const base = (fnv1a32(sampleId) & 0xffffff) * 16
  const out = [0]
  for (let k = 1; k <= aug; k++) out.push(base + k)
  return out
}

/** A sample is held out for evaluation when fnv1a32(id) % modulo == 0. */
export function isHeldOut(sampleId: string, modulo = 8): boolean {
  return fnv1a32(sampleId) % modulo === 0
}

/** Rendering variation for one view. Seed 0 is the plain render. */
export function augmentParams(seed: number): RenderParams {
  if (seed === 0) return { slant: 0, rotate: 0, radius: RADIUS, density: 1, ink: INK }
  const r = mulberry32(seed)
  const slant = (r() * 2 - 1) * 0.25
  const rotate = (r() * 2 - 1) * 2
  const radius = 0.9 + r() * 1.1
  const density = 0.85 + r() * 0.33
  const ink = 0.5 + r() * 0.3
  return { slant, rotate, radius, density, ink }
}

/** Strokes (canvas units, y down) to a 128-px-high crop, or null when there is no ink. */
export function renderStrokes(strokes: readonly Stroke[], nChars: number, p: RenderParams = augmentParams(0)): Gray | null {
  let minX = Infinity
  let maxX = -Infinity
  let minY = Infinity
  let maxY = -Infinity
  for (const s of strokes) {
    for (const [x, y] of s) {
      if (x < minX) minX = x
      if (x > maxX) maxX = x
      if (y < minY) minY = y
      if (y > maxY) maxY = y
    }
  }
  if (minX === Infinity) return null
  const cx = (minX + maxX) / 2
  const cy = (minY + maxY) / 2
  const a = (p.rotate * Math.PI) / 180
  const ca = Math.cos(a)
  const sa = Math.sin(a)
  const tstrokes: Array<Array<[number, number]>> = []
  let x0 = Infinity
  let x1 = -Infinity
  let y0 = Infinity
  let y1 = -Infinity
  for (const s of strokes) {
    if (!s.length) continue
    const out: Array<[number, number]> = []
    for (const [px, py] of s) {
      const x = px + p.slant * (cy - py)
      const dx = x - cx
      const dy = py - cy
      const tx = cx + dx * ca - dy * sa
      const ty = cy + dx * sa + dy * ca
      out.push([tx, ty])
      if (tx < x0) x0 = tx
      if (tx > x1) x1 = tx
      if (ty < y0) y0 = ty
      if (ty > y1) y1 = ty
    }
    tstrokes.push(out)
  }
  const inkH = Math.max(y1 - y0, 1e-6)
  let s = (H * p.ink) / inkH
  // A single dot or a flat dash has no height to scale by; draw it small.
  if (y1 - y0 < 1e-3) s = 1
  const top = (H - (y1 - y0) * s) / 2
  const width = (x1 - x0) * s + 2 * PAD_X
  let fx = 1
  if (nChars > 0) fx = Math.min(Math.max((PX_PER_CHAR * nChars) / width, 0.5), 2)
  fx *= p.density
  const W = Math.max(16, Math.ceil((x1 - x0) * s * fx + 2 * PAD_X))

  const cov = new Float32Array(H * W)
  for (const stroke of tstrokes) {
    const px = stroke.map(([x, y]) => [(x - x0) * s * fx + PAD_X, (y - y0) * s + top] as const)
    if (px.length === 1) capsule(cov, W, px[0][0], px[0][1], px[0][0], px[0][1], p.radius)
    for (let i = 0; i + 1 < px.length; i++) capsule(cov, W, px[i][0], px[i][1], px[i + 1][0], px[i + 1][1], p.radius)
  }
  const f = Math.fround
  const data = new Uint8Array(H * W)
  for (let i = 0; i < data.length; i++) data[i] = Math.floor(f(f(PAPER - f(cov[i] * (PAPER - INK_LEVEL))) + 0.5))
  return { data, width: W, height: H }
}

/** cov = max(cov, clamp(r + 0.5 - distance to segment, 0, 1)) at pixel centres. */
function capsule(cov: Float32Array, w: number, ax: number, ay: number, bx: number, by: number, r: number): void {
  const reach = r + 1
  const xa = Math.max(Math.floor(Math.min(ax, bx) - reach), 0)
  const xb = Math.min(Math.ceil(Math.max(ax, bx) + reach), w - 1)
  const ya = Math.max(Math.floor(Math.min(ay, by) - reach), 0)
  const yb = Math.min(Math.ceil(Math.max(ay, by) + reach), H - 1)
  if (xa > xb || ya > yb) return
  const dx = bx - ax
  const dy = by - ay
  const ll = dx * dx + dy * dy
  for (let j = ya; j <= yb; j++) {
    const py = j + 0.5
    for (let i = xa; i <= xb; i++) {
      const px = i + 0.5
      let t = 0
      if (ll > 0) {
        t = ((px - ax) * dx + (py - ay) * dy) / ll
        t = t < 0 ? 0 : t > 1 ? 1 : t
      }
      const d = Math.hypot(px - (ax + t * dx), py - (ay + t * dy))
      let c = r + 0.5 - d
      c = c < 0 ? 0 : c > 1 ? 1 : c
      const k = j * w + i
      const cf = Math.fround(c)
      if (cf > cov[k]) cov[k] = cf
    }
  }
}

/**
 * The CTC head (spec/model.md): "MKH1" file format, and logits + log-softmax
 * applied to backbone features. The reference does this in float32 numpy;
 * here the dot products accumulate in double and are rounded to float32, which
 * is at least as accurate and well within the read tolerance.
 */

import type { HeadData } from './api.ts'
import type { LogProbs } from './decoder.ts'

export function parseHead(bytes: Uint8Array | ArrayBuffer): HeadData {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes)
  const buf = u8.slice().buffer
  const dv = new DataView(buf)
  const magic = String.fromCharCode(...new Uint8Array(buf, 0, 4))
  if (magic !== 'MKH1') throw new Error(`not a head file (${JSON.stringify(magic)})`)
  const classes = dv.getUint32(4, true)
  const dim = dv.getUint32(8, true)
  if (buf.byteLength < 12 + 4 * (classes * dim + classes)) throw new Error('head file is truncated')
  const weight = new Float32Array(buf.slice(12, 12 + 4 * classes * dim))
  const bias = new Float32Array(buf.slice(12 + 4 * classes * dim, 12 + 4 * (classes * dim + classes)))
  return { classes, dim, weight, bias }
}

export function serializeHead(h: HeadData): Uint8Array {
  const out = new Uint8Array(12 + 4 * (h.classes * h.dim + h.classes))
  const dv = new DataView(out.buffer)
  out.set([0x4d, 0x4b, 0x48, 0x31]) // "MKH1"
  dv.setUint32(4, h.classes, true)
  dv.setUint32(8, h.dim, true)
  out.set(new Uint8Array(h.weight.buffer, h.weight.byteOffset, h.weight.byteLength), 12)
  out.set(new Uint8Array(h.bias.buffer, h.bias.byteOffset, h.bias.byteLength), 12 + 4 * h.classes * h.dim)
  return out
}

/** logits = features W^T + b, frames x classes (float32 values). */
export function headLogits(features: Float32Array, frames: number, h: HeadData): Float32Array {
  const { classes: C, dim: D, weight: W, bias: b } = h
  if (features.length < frames * D) throw new Error(`features: expected ${frames} x ${D}`)
  const out = new Float32Array(frames * C)
  for (let t = 0; t < frames; t++) {
    const f = t * D
    const o = t * C
    let k = 0
    for (; k + 3 < C; k += 4) {
      const w0 = k * D
      const w1 = w0 + D
      const w2 = w1 + D
      const w3 = w2 + D
      let s0 = 0
      let s1 = 0
      let s2 = 0
      let s3 = 0
      for (let d = 0; d < D; d++) {
        const x = features[f + d]
        s0 += x * W[w0 + d]
        s1 += x * W[w1 + d]
        s2 += x * W[w2 + d]
        s3 += x * W[w3 + d]
      }
      out[o + k] = s0 + b[k]
      out[o + k + 1] = s1 + b[k + 1]
      out[o + k + 2] = s2 + b[k + 2]
      out[o + k + 3] = s3 + b[k + 3]
    }
    for (; k < C; k++) {
      const w0 = k * D
      let s = 0
      for (let d = 0; d < D; d++) s += features[f + d] * W[w0 + d]
      out[o + k] = s + b[k]
    }
  }
  return out
}

/** In-place float32 log-softmax over each row of a frames x classes array. */
export function logSoftmax32(z: Float32Array, frames: number, classes: number): Float32Array {
  const f = Math.fround
  for (let t = 0; t < frames; t++) {
    const o = t * classes
    let m = -Infinity
    for (let k = 0; k < classes; k++) if (z[o + k] > m) m = z[o + k]
    let s = 0
    for (let k = 0; k < classes; k++) s += f(Math.exp(f(z[o + k] - m)))
    const ls = f(Math.log(f(s)))
    for (let k = 0; k < classes; k++) z[o + k] = f(f(z[o + k] - m) - ls)
  }
  return z
}

export function applyHead(features: Float32Array, frames: number, h: HeadData): LogProbs {
  return { data: logSoftmax32(headLogits(features, frames, h), frames, h.classes), frames, classes: h.classes }
}

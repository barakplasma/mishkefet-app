/**
 * Head-only fine-tuning (spec/adapter.md): full-batch Adam on cached backbone
 * features, CTC per sample normalized by label length, L2 toward the general
 * head. The backbone is frozen, so each step is two matmuls over every cached
 * frame (logits and the weight gradient) plus a small CTC per sample.
 *
 * The trainer is a step function rather than a loop so the worker can yield
 * between steps (progress messages, cancellation).
 */

import type { HeadData, TrainView } from './api.ts'
import { ctcLossGrad } from './ctc.ts'
import type { MatMul } from './matmul.ts'

export interface TrainConfig {
  lr: number
  l2: number
  beta1?: number
  beta2?: number
  eps?: number
  /**
   * Runs the two big products on ONNX Runtime (WebGPU or SIMD WASM). If it
   * ever fails, the trainer falls back to the JS loops for good.
   */
  matmul?: MatMul
}

export interface Trainer {
  /** One Adam step; resolves to the objective at the parameters before the step. */
  step(): Promise<number>
  /** Steps taken so far. */
  readonly steps: number
  /** The current parameters as a head (copies). */
  head(): HeadData
}

/** Z (N x C) = F (N x D) W^T + b, two frames by four classes at a time. */
function logits(F: Float32Array, N: number, D: number, W: Float32Array, b: Float32Array, C: number, Z: Float32Array): void {
  let t = 0
  for (; t + 1 < N; t += 2) {
    const f0 = t * D
    const f1 = f0 + D
    let k = 0
    for (; k + 3 < C; k += 4) {
      const w0 = k * D
      const w1 = w0 + D
      const w2 = w1 + D
      const w3 = w2 + D
      let a0 = 0
      let a1 = 0
      let a2 = 0
      let a3 = 0
      let c0 = 0
      let c1 = 0
      let c2 = 0
      let c3 = 0
      for (let d = 0; d < D; d++) {
        const x = F[f0 + d]
        const y = F[f1 + d]
        const u0 = W[w0 + d]
        const u1 = W[w1 + d]
        const u2 = W[w2 + d]
        const u3 = W[w3 + d]
        a0 += x * u0
        a1 += x * u1
        a2 += x * u2
        a3 += x * u3
        c0 += y * u0
        c1 += y * u1
        c2 += y * u2
        c3 += y * u3
      }
      const o0 = t * C + k
      const o1 = o0 + C
      Z[o0] = a0 + b[k]
      Z[o0 + 1] = a1 + b[k + 1]
      Z[o0 + 2] = a2 + b[k + 2]
      Z[o0 + 3] = a3 + b[k + 3]
      Z[o1] = c0 + b[k]
      Z[o1 + 1] = c1 + b[k + 1]
      Z[o1 + 2] = c2 + b[k + 2]
      Z[o1 + 3] = c3 + b[k + 3]
    }
    for (; k < C; k++) {
      const w0 = k * D
      let a = 0
      let c = 0
      for (let d = 0; d < D; d++) {
        a += F[f0 + d] * W[w0 + d]
        c += F[f1 + d] * W[w0 + d]
      }
      Z[t * C + k] = a + b[k]
      Z[(t + 1) * C + k] = c + b[k]
    }
  }
  for (; t < N; t++) {
    for (let k = 0; k < C; k++) {
      let a = 0
      for (let d = 0; d < D; d++) a += F[t * D + d] * W[k * D + d]
      Z[t * C + k] = a + b[k]
    }
  }
}

/** G (C x D) += dZ^T (C x N) F (N x D), four frames by two classes at a time. */
function gradW(dZ: Float32Array, F: Float32Array, N: number, D: number, C: number, G: Float64Array): void {
  let t = 0
  for (; t + 3 < N; t += 4) {
    const f0 = t * D
    const f1 = f0 + D
    const f2 = f1 + D
    const f3 = f2 + D
    let k = 0
    for (; k + 1 < C; k += 2) {
      const a0 = dZ[t * C + k]
      const a1 = dZ[(t + 1) * C + k]
      const a2 = dZ[(t + 2) * C + k]
      const a3 = dZ[(t + 3) * C + k]
      const b0 = dZ[t * C + k + 1]
      const b1 = dZ[(t + 1) * C + k + 1]
      const b2 = dZ[(t + 2) * C + k + 1]
      const b3 = dZ[(t + 3) * C + k + 1]
      const g0 = k * D
      const g1 = g0 + D
      for (let d = 0; d < D; d++) {
        const x0 = F[f0 + d]
        const x1 = F[f1 + d]
        const x2 = F[f2 + d]
        const x3 = F[f3 + d]
        G[g0 + d] += a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3
        G[g1 + d] += b0 * x0 + b1 * x1 + b2 * x2 + b3 * x3
      }
    }
    for (; k < C; k++) {
      const a0 = dZ[t * C + k]
      const a1 = dZ[(t + 1) * C + k]
      const a2 = dZ[(t + 2) * C + k]
      const a3 = dZ[(t + 3) * C + k]
      const g0 = k * D
      for (let d = 0; d < D; d++) G[g0 + d] += a0 * F[f0 + d] + a1 * F[f1 + d] + a2 * F[f2 + d] + a3 * F[f3 + d]
    }
  }
  for (; t < N; t++) {
    for (let k = 0; k < C; k++) {
      const a = dZ[t * C + k]
      for (let d = 0; d < D; d++) G[k * D + d] += a * F[t * D + d]
    }
  }
}

export function createTrainer(base: HeadData, views: readonly TrainView[], cfg: TrainConfig): Trainer {
  const { classes: C, dim: D } = base
  const { lr, l2 } = cfg
  const beta1 = cfg.beta1 ?? 0.9
  const beta2 = cfg.beta2 ?? 0.999
  const eps = cfg.eps ?? 1e-8

  // All views' frames in one contiguous block, for cache-friendly matmuls.
  const usable = views.filter((v) => v.frames > 0 && v.label.length > 0)
  const N = usable.reduce((a, v) => a + v.frames, 0)
  const F = new Float32Array(N * D)
  const starts: number[] = []
  let at = 0
  for (const v of usable) {
    if (v.features.length < v.frames * D) throw new Error(`training view: expected ${v.frames} x ${D} features`)
    F.set(v.features.subarray(0, v.frames * D), at * D)
    starts.push(at)
    at += v.frames
  }

  const W0 = base.weight
  const b0 = base.bias
  const W = Float32Array.from(W0)
  const b = Float32Array.from(b0)
  const mW = new Float32Array(C * D)
  const vW = new Float32Array(C * D)
  const mb = new Float32Array(C)
  const vb = new Float32Array(C)
  const Z = new Float32Array(N * C)
  const dZ = new Float32Array(N * C)
  const gW = new Float64Array(C * D)
  const gb = new Float64Array(C)
  let steps = 0
  let matmul = cfg.matmul

  async function computeLogits(): Promise<void> {
    if (matmul) {
      try {
        const Wt = new Float32Array(D * C)
        for (let k = 0; k < C; k++) for (let d = 0; d < D; d++) Wt[d * C + k] = W[k * D + d]
        const z = await matmul(F, Wt, N, D, C)
        for (let t = 0; t < N; t++) for (let k = 0; k < C; k++) Z[t * C + k] = z[t * C + k] + b[k]
        return
      } catch (e) {
        console.warn('[mishkefet] runtime matmul failed, training on the CPU', e)
        matmul = undefined
      }
    }
    logits(F, N, D, W, b, C, Z)
  }

  async function computeGradW(): Promise<void> {
    if (matmul) {
      try {
        const dZt = new Float32Array(C * N)
        for (let t = 0; t < N; t++) for (let k = 0; k < C; k++) dZt[k * N + t] = dZ[t * C + k]
        gW.set(await matmul(dZt, F, C, N, D))
        return
      } catch (e) {
        console.warn('[mishkefet] runtime matmul failed, training on the CPU', e)
        matmul = undefined
      }
    }
    gW.fill(0)
    gradW(dZ, F, N, D, C, gW)
  }

  async function lossAndGrads(): Promise<number> {
    await computeLogits()
    let total = 0
    let n = 0
    for (let i = 0; i < usable.length; i++) {
      const v = usable[i]
      const o = starts[i] * C
      const z = Z.subarray(o, o + v.frames * C)
      const g = dZ.subarray(o, o + v.frames * C)
      const r = ctcLossGrad(z, v.frames, C, v.label, g)
      if (!r.feasible) continue
      const scale = 1 / Math.max(1, v.label.length)
      total += r.loss * scale
      n++
      for (let j = 0; j < g.length; j++) g[j] *= scale
    }
    n = Math.max(n, 1)
    await computeGradW()
    gb.fill(0)
    for (let t = 0; t < N; t++) for (let k = 0; k < C; k++) gb[k] += dZ[t * C + k]
    total /= n
    let reg = 0
    for (let i = 0; i < W.length; i++) {
      const d = W[i] - W0[i]
      reg += d * d
      gW[i] = gW[i] / n + 2 * l2 * d
    }
    for (let k = 0; k < C; k++) {
      const d = b[k] - b0[k]
      reg += d * d
      gb[k] = gb[k] / n + 2 * l2 * d
    }
    return total + l2 * reg
  }

  function adam(p: Float32Array, g: Float64Array, m: Float32Array, v: Float32Array, c1: number, c2: number): void {
    for (let i = 0; i < p.length; i++) {
      const gi = g[i]
      m[i] = beta1 * m[i] + (1 - beta1) * gi
      v[i] = beta2 * v[i] + (1 - beta2) * gi * gi
      p[i] = p[i] - (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + eps)
    }
  }

  return {
    async step() {
      const loss = await lossAndGrads()
      steps++
      const c1 = 1 - beta1 ** steps
      const c2 = 1 - beta2 ** steps
      adam(W, gW, mW, vW, c1, c2)
      adam(b, gb, mb, vb, c1, c2)
      return loss
    },
    get steps() {
      return steps
    },
    head() {
      return { classes: C, dim: D, weight: Float32Array.from(W), bias: Float32Array.from(b) }
    },
  }
}

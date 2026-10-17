import { describe, expect, it } from 'vitest'
import type { TrainView } from '../../src/engine/api.ts'
import { createTrainer } from '../../src/engine/trainer.ts'
import { golden } from './fixtures.ts'

interface Run {
  rank: number
  steps: number
  lr: number
  l2: number
  losses: number[]
  W: number[]
  b: number[]
}
interface Vectors {
  D: number
  C: number
  W0: number[]
  b0: number[]
  samples: Array<{ features: number[]; T: number; label: number[] }>
  runs: Run[]
}

const v = golden<Vectors>('adapter.json')

describe('head trainer (golden/adapter.json, rank 0)', () => {
  const run = v.runs.find((r) => r.rank === 0)!
  it(`${run.steps} Adam steps: losses within 1e-3 relative, weights within 1e-4`, async () => {
    const base = { classes: v.C, dim: v.D, weight: Float32Array.from(v.W0), bias: Float32Array.from(v.b0) }
    const views: TrainView[] = v.samples.map((s) => ({ features: Float32Array.from(s.features), frames: s.T, label: s.label }))
    const t = createTrainer(base, views, { lr: run.lr, l2: run.l2 })
    const losses: number[] = []
    for (let i = 0; i < run.steps; i++) losses.push(await t.step())
    expect(t.steps).toBe(run.steps)
    losses.forEach((l, i) => expect(Math.abs(l - run.losses[i]) / Math.abs(run.losses[i])).toBeLessThan(1e-3))
    const h = t.head()
    h.weight.forEach((w, i) => expect(Math.abs(w - run.W[i])).toBeLessThan(1e-4))
    h.bias.forEach((b, i) => expect(Math.abs(b - run.b[i])).toBeLessThan(1e-4))
    // the base head is left untouched
    expect([...base.weight]).toEqual([...Float32Array.from(v.W0)])
  })
})

describe('head trainer with an injected matmul', () => {
  it('gives the same run as the built-in loops', async () => {
    const naive = async (a: Float32Array, b: Float32Array, m: number, k: number, n: number) => {
      const c = new Float32Array(m * n)
      for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) {
        let s = 0
        for (let p = 0; p < k; p++) s += a[i * k + p] * b[p * n + j]
        c[i * n + j] = s
      }
      return c
    }
    const run = v.runs.find((r) => r.rank === 0)!
    const base = { classes: v.C, dim: v.D, weight: Float32Array.from(v.W0), bias: Float32Array.from(v.b0) }
    const views: TrainView[] = v.samples.map((s) => ({ features: Float32Array.from(s.features), frames: s.T, label: s.label }))
    const t = createTrainer(base, views, { lr: run.lr, l2: run.l2, matmul: naive })
    for (let i = 0; i < run.steps; i++) expect(Math.abs((await t.step()) - run.losses[i]) / run.losses[i]).toBeLessThan(1e-3)
    t.head().weight.forEach((w, i) => expect(Math.abs(w - run.W[i])).toBeLessThan(1e-4))
  })
  it('falls back to the loops when the matmul throws', async () => {
    const run = v.runs.find((r) => r.rank === 0)!
    const base = { classes: v.C, dim: v.D, weight: Float32Array.from(v.W0), bias: Float32Array.from(v.b0) }
    const views: TrainView[] = v.samples.map((s) => ({ features: Float32Array.from(s.features), frames: s.T, label: s.label }))
    const t = createTrainer(base, views, { lr: run.lr, l2: run.l2, matmul: async () => Promise.reject(new Error('no gpu')) })
    expect(Math.abs((await t.step()) - run.losses[0]) / run.losses[0]).toBeLessThan(1e-3)
  })
})

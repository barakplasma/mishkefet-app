import { describe, expect, it } from 'vitest'
import { ctcLossGrad } from '../../src/engine/ctc.ts'
import { golden } from './fixtures.ts'

interface Case {
  T: number
  C: number
  label: number[]
  logits: number[]
  loss: number
  grad: number[]
}

function rel(a: number, b: number): number {
  return Math.abs(a - b) / Math.max(1, Math.abs(b))
}

describe('CTC loss and gradient (golden/ctc.json, 1e-4 relative)', () => {
  for (const c of golden<Case[]>('ctc.json')) {
    it(`T=${c.T} C=${c.C} label=${c.label.join(',')}`, () => {
      const grad = new Float32Array(c.T * c.C)
      const r = ctcLossGrad(Float32Array.from(c.logits), c.T, c.C, c.label, grad)
      expect(rel(r.loss, c.loss)).toBeLessThan(1e-4)
      expect(r.feasible).toBe(c.loss !== 0)
      let worst = 0
      for (let i = 0; i < grad.length; i++) worst = Math.max(worst, rel(grad[i], c.grad[i]))
      expect(worst).toBeLessThan(1e-4)
    })
  }
})

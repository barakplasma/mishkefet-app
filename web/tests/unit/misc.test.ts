import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { labelsToText, textToLabels } from '../../src/engine/charset.ts'
import { MODEL_FILES } from '../../src/engine/constants.ts'
import { applyHead, parseHead, serializeHead } from '../../src/engine/head.ts'
import { gaussianKernel, padIndex, resize, rgbaToGray, roundHalfEven } from '../../src/engine/image.ts'
import { cer, characterAccuracy, editDistance } from '../../src/engine/metrics.ts'
import { charset } from './fixtures.ts'
import { HAVE_MODEL, MODEL_DIR, SPEC } from './paths.ts'

describe('charset', () => {
  it('has 198 classes, blank at 0 and space at 38, and round-trips text', () => {
    const cs = charset()
    expect(cs.chars.length).toBe(198)
    expect(cs.chars[0]).toBe('')
    expect(cs.chars[38]).toBe(' ')
    const labels = textToLabels(cs, 'שלום עולם')
    expect(labels).toContain(38)
    expect(labelsToText(cs, labels)).toBe('שלום עולם')
    expect(textToLabels(cs, '\u{1F600}א')).toEqual([1])
  })
  it('matches the pinned charset.json in the model files list', () => {
    const f = MODEL_FILES.find((m) => m.key === 'charset')!
    expect(readFileSync(join(SPEC, 'charset.json')).length).toBe(f.bytes)
  })
})

describe('image primitives', () => {
  it('rgbaToGray is PIL L and composites alpha on white', () => {
    const g = rgbaToGray(Uint8ClampedArray.from([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 0, 0, 0, 0, 10, 20, 30, 128]), 5, 1)
    expect([...g.data.slice(0, 4)]).toEqual([76, 150, 29, 255])
    // (10,20,30) half over white: 132, 137, 142 -> L
    expect(g.data[4]).toBe((132 * 19595 + 137 * 38470 + 142 * 7471 + 0x8000) >>> 16)
  })
  it('round half to even like Python', () => {
    expect([0.5, 1.5, 2.5, 3.5, 2.4999, 2.5001].map(roundHalfEven)).toEqual([0, 2, 2, 4, 2, 3])
  })
  it('reflect101 and replicate padding', () => {
    expect([...padIndex(4, 3, 'reflect101')]).toEqual([3, 2, 1, 0, 1, 2, 3, 2, 1, 0])
    expect([...padIndex(4, 2, 'replicate')]).toEqual([0, 0, 0, 1, 2, 3, 3, 3])
    expect([...padIndex(1, 2, 'reflect101')]).toEqual([0, 0, 0, 0, 0])
  })
  it('OpenCV fixed small Gaussian kernels', () => {
    expect([...gaussianKernel(5, 0)]).toEqual([0.0625, 0.25, 0.375, 0.25, 0.0625])
    const k = gaussianKernel(15, 0)
    expect(k.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
  })
  it('area resize averages boxes, bilinear keeps constants', () => {
    const g = { data: Uint8Array.from([0, 100, 200, 50]), width: 4, height: 1 }
    expect([...resize(g, 2, 1, 'area').data]).toEqual([50, 125])
    const c = resize({ data: new Uint8Array(6).fill(77), width: 3, height: 2 }, 7, 5, 'linear')
    expect([...new Set(c.data)]).toEqual([77])
  })
})

describe('metrics', () => {
  it('edit distance, CER and character accuracy over code points', () => {
    expect(editDistance('שלום', 'שלם')).toBe(1)
    expect(editDistance('', 'abc')).toBe(3)
    expect(cer('abcd', 'abxd')).toBe(0.25)
    expect(characterAccuracy([{ target: 'abcd', read: 'abcd' }, { target: 'ab', read: '' }])).toBeCloseTo(1 - 2 / 6, 12)
  })
})

describe('head', () => {
  it('serializes and parses MKH1', () => {
    const h = { classes: 3, dim: 2, weight: Float32Array.from([1, 2, 3, 4, 5, 6]), bias: Float32Array.from([0.5, -1, 2]) }
    const back = parseHead(serializeHead(h))
    expect(back.classes).toBe(3)
    expect(back.dim).toBe(2)
    expect([...back.weight]).toEqual([...h.weight])
    expect([...back.bias]).toEqual([...h.bias])
    expect(() => parseHead(new Uint8Array(16))).toThrow()
  })
  it('log-softmax rows sum to one', () => {
    const h = { classes: 5, dim: 3, weight: Float32Array.from({ length: 15 }, (_, i) => Math.sin(i)), bias: new Float32Array(5) }
    const lp = applyHead(Float32Array.from([1, 2, 3, -1, 0, 0.5]), 2, h)
    for (let t = 0; t < 2; t++) {
      let s = 0
      for (let k = 0; k < 5; k++) s += Math.exp(lp.data[t * 5 + k])
      expect(s).toBeCloseTo(1, 5)
    }
  })
  it.skipIf(!HAVE_MODEL)('parses the shipped head.bin (198 x 512)', () => {
    const h = parseHead(readFileSync(join(MODEL_DIR, 'head.bin')))
    expect(h.classes).toBe(198)
    expect(h.dim).toBe(512)
    expect(h.weight.length).toBe(198 * 512)
    expect(h.bias.every(Number.isFinite)).toBe(true)
  })
})

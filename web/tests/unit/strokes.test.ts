import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { augmentParams, fnv1a32, isHeldOut, mulberry32, renderStrokes, viewSeeds } from '../../src/engine/strokes.ts'
import { golden } from './fixtures.ts'
import { loadPngChannel } from './images.ts'
import { SPEC } from './paths.ts'

interface Vectors {
  mulberry32_seed42: number[]
  sets: Array<{
    word: string
    strokes: Array<Array<[number, number]>>
    renders: Array<{ seed: number; params: Record<string, number>; height: number; width: number; sum: number; ink_pixels: number }>
  }>
}
const v = golden<Vectors>('strokes.json')

describe('mulberry32', () => {
  it('matches the reference sequence for seed 42', () => {
    const r = mulberry32(42)
    for (const want of v.mulberry32_seed42) expect(r()).toBe(want)
  })
})

describe('stroke renders (golden/strokes.json: same width, pixel sum within 0.5%)', () => {
  v.sets.forEach((set, i) => {
    for (const r of set.renders) {
      it(`${set.word} seed ${r.seed}`, () => {
        const p = augmentParams(r.seed)
        for (const [k, val] of Object.entries(r.params)) expect(p[k as keyof typeof p]).toBeCloseTo(val, 12)
        const img = renderStrokes(set.strokes, Array.from(set.word).length, p)!
        expect(img.height).toBe(r.height)
        expect(img.width).toBe(r.width)
        let sum = 0
        let ink = 0
        for (const px of img.data) {
          sum += px
          if (px < 128) ink++
        }
        expect(Math.abs(sum - r.sum) / r.sum).toBeLessThan(0.005)
        expect(Math.abs(ink - r.ink_pixels)).toBeLessThanOrEqual(Math.max(2, r.ink_pixels * 0.01))
        if (r.seed === 0) {
          const want = loadPngChannel(join(SPEC, 'golden/strokes', `${i}.png`))
          expect(want.width).toBe(img.width)
          let off = 0
          for (let k = 0; k < want.data.length; k++) if (Math.abs(want.data[k] - img.data[k]) > 1) off++
          expect(off).toBeLessThanOrEqual(want.data.length * 0.001)
        }
      })
    }
  })
  it('returns null for no ink and draws a lone dot small', () => {
    expect(renderStrokes([], 3)).toBeNull()
    expect(renderStrokes([[]], 3)).toBeNull()
    const dot = renderStrokes([[[10, 10]]], 1)!
    expect(dot.height).toBe(128)
    expect(dot.width).toBe(24)
    expect(Math.min(...dot.data)).toBeLessThan(128)
  })
})

describe('sample hashing', () => {
  it('fnv1a32 over UTF-8', () => {
    expect(fnv1a32('')).toBe(0x811c9dc5)
    expect(fnv1a32('a')).toBe(0xe40c292c)
    expect(fnv1a32('foobar')).toBe(0xbf9cf968)
  })
  it('view seeds and held-out rule', () => {
    const id = 'sample-1'
    const base = (fnv1a32(id) & 0xffffff) * 16
    expect(viewSeeds(id, 1)).toEqual([0, base + 1])
    expect(viewSeeds(id, 4)).toEqual([0, base + 1, base + 2, base + 3, base + 4])
    expect(isHeldOut(id)).toBe(fnv1a32(id) % 8 === 0)
  })
})

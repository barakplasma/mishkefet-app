import { existsSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { straightenLine } from '../../src/engine/straighten.ts'
import { golden, stem } from './fixtures.ts'
import { loadGray, loadPngChannel } from './images.ts'
import { SPEC } from './paths.ts'

interface Case {
  file: string
  in: [number, number]
  out: [number, number]
  changed: boolean
  sum: number
}

describe('straighten (golden/straighten.json: same shape and changed flag, pixel sum within 1%)', () => {
  for (const c of golden<Case[]>('straighten.json')) {
    it(`${stem(c.file)}${c.changed ? '' : ' (unchanged)'}`, async () => {
      const g = await loadGray(join(SPEC, 'fixtures', c.file))
      expect([g.width, g.height]).toEqual(c.in)
      const s = straightenLine(g)
      expect(s !== g).toBe(c.changed)
      expect([s.width, s.height]).toEqual(c.out)
      let sum = 0
      for (const v of s.data) sum += v
      expect(Math.abs(sum - c.sum) / c.sum).toBeLessThan(0.01)
      const png = join(SPEC, 'golden/straighten', `${stem(c.file)}.png`)
      if (c.changed && existsSync(png)) {
        const want = loadPngChannel(png)
        let off = 0
        for (let i = 0; i < want.data.length; i++) if (Math.abs(want.data[i] - s.data[i]) > 2) off++
        expect(off / want.data.length).toBeLessThan(0.01)
      }
    })
  }
  it('leaves blank and tiny crops alone', () => {
    const blank = { data: new Uint8Array(40 * 200).fill(255), width: 200, height: 40 }
    expect(straightenLine(blank)).toBe(blank)
    const tiny = { data: new Uint8Array(4 * 5).fill(0), width: 5, height: 4 }
    expect(straightenLine(tiny)).toBe(tiny)
  })
  it('levels a synthetic slanted stroke', () => {
    // graded ink on lightly textured paper (a pure two-level image has no Otsu split: t = 0, no ink weight)
    const w = 300
    const h = 60
    const data = new Uint8Array(w * h)
    for (let i = 0; i < data.length; i++) data[i] = 235 + ((i * 7919) % 17)
    for (let x = 10; x < 290; x++) {
      const yc = Math.round(15 + (x * 30) / w)
      for (let d = -3; d <= 3; d++) data[(yc + d) * w + x] = [180, 110, 50, 20, 50, 110, 180][d + 3]
    }
    const g = { data, width: w, height: h }
    const s = straightenLine(g)
    expect(s).not.toBe(g)
    expect(s.width).toBe(w)
    // after straightening, the ink sits on (about) the same rows at both ends
    const rows = (x: number) => {
      const ys: number[] = []
      for (let y = 0; y < s.height; y++) if (s.data[y * w + x] < 100) ys.push(y)
      return ys.reduce((a, b) => a + b, 0) / ys.length
    }
    expect(Math.abs(rows(20) - rows(280))).toBeLessThan(1.5)
  })
})

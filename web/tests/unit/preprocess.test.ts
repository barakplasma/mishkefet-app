import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { preprocess, ttaView } from '../../src/engine/preprocess.ts'
import { golden, stem } from './fixtures.ts'
import { loadGray } from './images.ts'
import { SPEC } from './paths.ts'

const lines = golden<Array<{ file: string; width: number; height: number }>>('lines.json')

describe('preprocess (golden/preprocess/*.u8, within 3/255)', () => {
  for (const line of lines) {
    it(stem(line.file), async () => {
      const g = await loadGray(join(SPEC, 'fixtures', line.file))
      const x = preprocess(g)
      expect(x.width).toBe(line.width)
      expect(x.data.length).toBe(64 * line.width)
      const want = readFileSync(join(SPEC, 'golden/preprocess', `${stem(line.file)}.u8`))
      let worst = 0
      for (let i = 0; i < want.length; i++) worst = Math.max(worst, Math.abs(Math.floor(x.data[i] * 255 + 0.5) - want[i]))
      expect(worst).toBeLessThanOrEqual(3)
    })
  }
})

describe('tta views', () => {
  it('resize horizontally to max(8, trunc(w * s))', async () => {
    const g = await loadGray(join(SPEC, 'fixtures', lines[0].file))
    expect(ttaView(g, 1)).toBe(g)
    const v = ttaView(g, 0.82)
    expect(v.width).toBe(Math.trunc(g.width * 0.82))
    expect(v.height).toBe(g.height)
    expect(ttaView({ data: new Uint8Array(5 * 2), width: 5, height: 2 }, 0.82).width).toBe(8)
  })
  it('keeps a constant image constant', () => {
    const v = ttaView({ data: new Uint8Array(40 * 3).fill(200), width: 40, height: 3 }, 1.22)
    expect([...new Set(v.data)]).toEqual([200])
  })
})

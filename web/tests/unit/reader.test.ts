import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import type { HeadData, LineBox } from '../../src/engine/api.ts'
import { CancelledError, readLine, readPage, readStrokes, transcript, type Engine } from '../../src/engine/reader.ts'
import { charset, golden } from './fixtures.ts'
import { loadGray, loadRgba } from './images.ts'
import { SPEC } from './paths.ts'

/** A backbone that "sees" one fixed class per frame, so reads are predictable without the model. */
function stubEngine(): Engine & { calls: number[] } {
  const C = 198
  const D = 512
  // feature d = 1 at frame t picks class 1 + (t % 3) via an identity-like head
  const weight = new Float32Array(C * D)
  for (let k = 1; k < 4; k++) weight[k * D + k] = 20
  const head: HeadData = { classes: C, dim: D, weight, bias: new Float32Array(C) }
  const calls: number[] = []
  return {
    calls,
    charset: charset(),
    lm: { order: 6, logprob: () => -1 },
    words: { logprob: () => -14, bonus: () => 0 },
    baseHead: head,
    backbone: {
      async features(image, width) {
        expect(image.length).toBe(64 * width)
        calls.push(width)
        const frames = width >> 2
        const values = new Float32Array(frames * D)
        for (let t = 0; t < frames; t++) values[t * D + 1 + (t % 3)] = 1
        return { values, frames }
      },
    },
  }
}

describe('reader with a stub backbone', () => {
  it('reads a line with TTA 1 and 3 (three views at the TTA widths)', async () => {
    const e = stubEngine()
    const g = await loadGray(join(SPEC, 'fixtures/lines/p1-02.png'))
    const r1 = await readLine(e, g, { tta: 1 })
    expect(r1.text.startsWith('אבג')).toBe(true)
    expect(r1.confidence).toBeGreaterThan(0)
    e.calls.length = 0
    const r3 = await readLine(e, g, { tta: 3 })
    expect(e.calls.length).toBe(3)
    expect(r3.confidence).toBe(r1.confidence)
  })

  it('segments a page, reports boxes in the original grid, then every line', async () => {
    const e = stubEngine()
    const page = await loadRgba(join(SPEC, 'fixtures/pages/page1.jpg'))
    const want = golden<Array<{ boxes: LineBox[] }>>('pages.json')[0]
    let boxes: LineBox[] = []
    const seen: number[] = []
    const res = await readPage(e, { rgba: page.data, width: page.width, height: page.height }, { tta: 1, mode: 'greedy' }, {
      onSegmented: (_s, b) => (boxes = b),
      onLine: (i, total) => {
        expect(total).toBe(want.boxes.length)
        seen.push(i)
      },
    })
    expect(res.seg.angle).toBe(-1)
    expect(boxes.length).toBe(want.boxes.length)
    expect(seen).toEqual(boxes.map((_, i) => i))
    expect(res.lines.map((l) => l.box)).toEqual(boxes)
    for (const b of boxes) {
      expect(b.x).toBeGreaterThanOrEqual(0)
      expect(b.y).toBeGreaterThanOrEqual(0)
      expect(b.x + b.w).toBeLessThanOrEqual(page.width)
      expect(b.y + b.h).toBeLessThanOrEqual(page.height)
    }
    expect(transcript(res.lines).length).toBeLessThanOrEqual(res.lines.length)
  })

  it('stops between lines when cancelled', async () => {
    const e = stubEngine()
    const page = await loadRgba(join(SPEC, 'fixtures/pages/page2.jpg'))
    let n = 0
    await expect(
      readPage(e, { rgba: page.data, width: page.width, height: page.height }, { tta: 1, mode: 'greedy' }, { onLine: () => n++, shouldStop: () => n >= 2 }),
    ).rejects.toBeInstanceOf(CancelledError)
    expect(n).toBe(2)
  })

  it('reads strokes from their plain render, and nothing from no ink', async () => {
    const e = stubEngine()
    const set = golden<{ sets: Array<{ word: string; strokes: Array<Array<[number, number]>> }> }>('strokes.json').sets[0]
    expect((await readStrokes(e, set.strokes, set.word)).length).toBeGreaterThan(0)
    expect(await readStrokes(e, [], 'x')).toBe('')
  })
})

import { join } from 'node:path'
import { beforeAll, describe, expect, it } from 'vitest'
import type { Gray } from '../../src/engine/image.ts'
import { cvLine, lineKernel, morph, rectKernel } from '../../src/engine/morph.ts'
import { binarize, cropRect, segmentPage, toOriginal, type Box } from '../../src/engine/segmenter.ts'
import { golden, stem } from './fixtures.ts'
import { loadGray, loadPngChannel } from './images.ts'
import { SPEC } from './paths.ts'

interface Page {
  file: string
  scale: number
  angle: number
  pitch: number
  boxes: Box[]
}

function iou(a: Box, b: Box): number {
  const w = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x)
  const h = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y)
  if (w <= 0 || h <= 0) return 0
  const i = w * h
  return i / (a.w * a.h + b.w * b.h - i)
}

for (const page of golden<Page[]>('pages.json')) {
  describe(`segmentation of ${stem(page.file)}`, () => {
    let gray: Gray
    beforeAll(async () => {
      gray = await loadGray(join(SPEC, 'fixtures', page.file))
    })

    it('binary mask agrees with golden/pages/*.binary.png on >= 99.9% of pixels', () => {
      const { binary, scale } = binarize(gray)
      expect(scale).toBe(page.scale)
      const want = loadPngChannel(join(SPEC, 'golden/pages', `${stem(page.file)}.binary.png`))
      expect([binary.width, binary.height]).toEqual([want.width, want.height])
      let agree = 0
      for (let i = 0; i < want.data.length; i++) if (want.data[i] === binary.data[i]) agree++
      expect(agree / want.data.length).toBeGreaterThanOrEqual(0.999)
    })

    it('same angle and pitch; >= 90% of golden boxes matched at IoU >= 0.9', () => {
      const t0 = performance.now()
      const seg = segmentPage(gray)
      console.info(`${stem(page.file)}: segmented in ${Math.round(performance.now() - t0)} ms, ${seg.boxes.length} lines`)
      expect(seg.angle).toBe(page.angle)
      expect(seg.pitch).toBe(page.pitch)
      const matched = page.boxes.filter((g) => seg.boxes.some((b) => iou(g, b) >= 0.9)).length
      expect(matched / page.boxes.length).toBeGreaterThanOrEqual(0.9)
      // reading order: the golden order should be reproduced box for box
      const exact = page.boxes.filter((g, i) => seg.boxes[i] && iou(g, seg.boxes[i]) >= 0.9).length
      expect(exact / page.boxes.length).toBeGreaterThanOrEqual(0.9)
    })
  })
}

describe('geometry helpers', () => {
  it('maps a deskewed box back onto the original image', () => {
    const b = { x: 100, y: 200, w: 300, h: 40 }
    expect(toOriginal(b, 0, 1000, 800)).toEqual(b)
    const r = toOriginal(b, 2, 1000, 800)
    // the rotated box's bound contains the box's centre mapped back and is a little larger
    expect(r.w).toBeGreaterThan(b.w)
    expect(r.h).toBeGreaterThan(b.h)
    const a = (2 * Math.PI) / 180
    const X = 250 - 500
    const Y = 220 - 400
    const cx = Math.cos(a) * X - Math.sin(a) * Y + 500
    const cy = Math.sin(a) * X + Math.cos(a) * Y + 400
    expect(cx).toBeGreaterThan(r.x)
    expect(cx).toBeLessThan(r.x + r.w)
    expect(cy).toBeGreaterThan(r.y)
    expect(cy).toBeLessThan(r.y + r.h)
  })
  it('pads and clamps crop rectangles', () => {
    expect(cropRect({ x: 10, y: 10, w: 100, h: 50 }, 1000, 1000)).toEqual({ x: 8, y: 4, w: 104, h: 62 })
    expect(cropRect({ x: 0, y: 0, w: 100, h: 50 }, 102, 55)).toEqual({ x: 0, y: 0, w: 102, h: 55 })
  })
})

describe('morphology', () => {
  it('draws lines like OpenCV LineIterator', () => {
    expect(cvLine(0, 0, 4, 2)).toEqual([
      [0, 0],
      [1, 0],
      [2, 1],
      [3, 1],
      [4, 2],
    ])
    const k = lineKernel(40, 2)
    expect(k.rows).toBe(2)
    expect(lineKernel(40, -2).runs[1]![0]).toBe(0)
  })
  it('erodes and dilates with anchor (cols // 2, rows // 2), outside never counting', () => {
    const w = 7
    const a = new Uint8Array(w)
    a[3] = 255
    const d = morph(a, w, 1, rectKernel(3, 1), 'dilate')
    expect([...d]).toEqual([0, 0, 255, 255, 255, 0, 0])
    const e = morph(new Uint8Array(w).fill(255), w, 1, rectKernel(5, 1), 'erode')
    expect([...e]).toEqual(new Array(w).fill(255))
    // 2x2: offsets -1..0 for both ops
    const b = new Uint8Array(9)
    b[4] = 255
    expect([...morph(b, 3, 3, rectKernel(2, 2), 'dilate')]).toEqual([0, 0, 0, 0, 255, 255, 0, 255, 255])
  })
})

import { join } from 'node:path'
import { beforeAll, describe, expect, it } from 'vitest'
import { expandedSize, rgbaToRgb, rgbToGray, rot90, type Rgb } from '../../src/engine/image.ts'
import { cvLine, lineKernel, morph, rectKernel } from '../../src/engine/morph.ts'
import { binarize, cropRect, estimateSkewWide, rotateNearestExpand, segmentPage, type Box } from '../../src/engine/segmenter.ts'
import { golden, stem } from './fixtures.ts'
import { loadPngChannel, loadRgba } from './images.ts'
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

// The plain segmentation of each fixture page as it is (no paper crop, no turning).
for (const page of golden<Page[]>('pages.json')) {
  describe(`segmentation of ${stem(page.file)}`, () => {
    let rgb: Rgb
    beforeAll(async () => {
      const im = await loadRgba(join(SPEC, 'fixtures', page.file))
      rgb = rgbaToRgb(im.data, im.width, im.height)
    })

    it('binary mask agrees with golden/pages/*.binary.png on >= 99.9% of pixels', () => {
      const { binary, scale } = binarize(rgbToGray(rgb))
      expect(scale).toBe(page.scale)
      const want = loadPngChannel(join(SPEC, 'golden/pages', `${stem(page.file)}.binary.png`))
      expect([binary.width, binary.height]).toEqual([want.width, want.height])
      let agree = 0
      for (let i = 0; i < want.data.length; i++) if (want.data[i] === binary.data[i]) agree++
      expect(agree / want.data.length).toBeGreaterThanOrEqual(0.999)
    })

    it('same angle and pitch; >= 90% of golden boxes matched at IoU >= 0.9, in order', () => {
      const t0 = performance.now()
      const seg = segmentPage(rgb)
      console.info(`${stem(page.file)}: segmented in ${Math.round(performance.now() - t0)} ms, ${seg.boxes.length} lines`)
      expect(seg.angle).toBe(page.angle)
      expect(seg.pitch).toBe(page.pitch)
      const matched = page.boxes.filter((g) => seg.boxes.some((b) => iou(g, b) >= 0.9)).length
      expect(matched / Math.max(1, page.boxes.length)).toBeGreaterThanOrEqual(0.9)
      const exact = page.boxes.filter((g, i) => seg.boxes[i] && iou(g, seg.boxes[i]) >= 0.9).length
      expect(exact / Math.max(1, page.boxes.length)).toBeGreaterThanOrEqual(0.9)
      // the boxes refer to the deskewed page, which holds them all
      for (const b of seg.boxes) {
        expect(b.x + b.w).toBeLessThanOrEqual(seg.deskewed.width)
        expect(b.y + b.h).toBeLessThanOrEqual(seg.deskewed.height)
      }
    })
  })
}

describe('geometry helpers', () => {
  it('pads and clamps crop rectangles', () => {
    expect(cropRect({ x: 10, y: 10, w: 100, h: 50 }, 1000, 1000)).toEqual({ x: 8, y: 4, w: 104, h: 62 })
    expect(cropRect({ x: 0, y: 0, w: 100, h: 50 }, 102, 55)).toEqual({ x: 0, y: 0, w: 102, h: 55 })
  })
  it('expanded canvas holds the rotated page; nearest rotation keeps the centre', () => {
    expect(expandedSize(100, 50, 0)).toEqual({ width: 100, height: 50 })
    // cos(90 deg) is 6e-17, so ceil/floor widen it by 2, exactly as the reference's expanded_size
    expect(expandedSize(100, 50, 90)).toEqual({ width: 52, height: 100 })
    const e = expandedSize(1000, 600, 20)
    const half = 500 * Math.cos(Math.PI / 9) + 300 * Math.sin(Math.PI / 9)
    expect(e.width).toBe(Math.ceil(half) - Math.floor(-half))
    // matches reference.rotate_nearest_expand on the same 21 x 11 mask
    const m = { data: new Uint8Array(21 * 11), width: 21, height: 11 }
    m.data[5 * 21 + 10] = 255
    m.data[2 * 21 + 3] = 255
    const r = rotateNearestExpand(m, 30)
    expect([r.width, r.height]).toEqual([24, 22])
    const on: Array<[number, number]> = []
    r.data.forEach((v, i) => v && on.push([Math.floor(i / r.width), i % r.width]))
    expect(on).toEqual([
      [11, 11],
      [12, 4],
    ])
  })
  it('a page without ink stays level in the wide skew search', () => {
    const blank = { data: new Uint8Array(400 * 300), width: 400, height: 300 }
    expect(estimateSkewWide(blank)).toEqual({ angle: 0, score: 0 })
  })
  it('rot90 turns counter-clockwise as displayed', () => {
    // 3 wide x 2 tall: [a b c / d e f] -> 2 wide x 3 tall: [c f / b e / a d]
    const img = { data: Uint8Array.of(1, 2, 3, 4, 5, 6), width: 3, height: 2 }
    const r = rot90(img, 1)
    expect([r.width, r.height]).toEqual([2, 3])
    expect([...r.data]).toEqual([3, 6, 2, 5, 1, 4])
    expect([...rot90(img, 2).data]).toEqual([6, 5, 4, 3, 2, 1])
    expect([...rot90(img, 3).data]).toEqual([4, 1, 5, 2, 6, 3])
    expect(rot90(rot90(img, 1), 3).data).toEqual(img.data)
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

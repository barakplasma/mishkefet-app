import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { applyHead } from '../../src/engine/head.ts'
import { confidence } from '../../src/engine/decoder.ts'
import { rgbaToRgb, rgbToGray, rot90, otsuThreshold, type Rgb } from '../../src/engine/image.ts'
import { cropToPaper, findPaper, orientPage } from '../../src/engine/orientation.ts'
import { grayFeatures } from '../../src/engine/reader.ts'
import { binarize, estimateSkewWide, segmentPage, type Box } from '../../src/engine/segmenter.ts'
import { golden, HAVE_BACKBONE, modelEngine, stem } from './fixtures.ts'
import { loadRgba } from './images.ts'
import { HAVE_MODEL, SPEC } from './paths.ts'

interface Page {
  file: string
  read: { paper: [number, number, number, number] | null; turns: number; angle: number; deskewed: [number, number]; boxes: Box[]; lines: string[] }
}
const pages = golden<Page[]>('pages.json')

async function pageRgb(file: string): Promise<Rgb> {
  const im = await loadRgba(join(SPEC, 'fixtures', file))
  return rgbaToRgb(im.data, im.width, im.height)
}

describe('otsu', () => {
  it('splits two clusters and takes the first maximum', () => {
    const v = new Uint8Array(100)
    v.fill(20, 0, 60)
    v.fill(200, 60)
    const t = otsuThreshold(v)
    expect(t).toBeGreaterThanOrEqual(20)
    expect(t).toBeLessThan(200)
    expect(otsuThreshold(new Uint8Array(10).fill(7))).toBe(0)
  })
})

describe('paper crop and wide deskew (golden pages.json "read")', () => {
  for (const page of pages) {
    it(`${stem(page.file)}: paper ${page.read.paper ? 'crop' : 'none'}, turned page deskews to ${page.read.angle} degrees`, async () => {
      const rgb = await pageRgb(page.file)
      const paper = findPaper(rgb)
      if (!page.read.paper) expect(paper).toBeNull()
      else {
        expect(paper).not.toBeNull()
        const [x, y, w, h] = page.read.paper
        for (const [a, b] of [
          [paper!.x, x],
          [paper!.y, y],
          [paper!.w, w],
          [paper!.h, h],
        ])
          expect(Math.abs(a - b)).toBeLessThanOrEqual(2)
      }
      // the golden turns applied to the (golden) paper crop: the wide search finds the golden angle
      const crop = page.read.paper ? cropToPaper(rgb, { x: page.read.paper[0], y: page.read.paper[1], w: page.read.paper[2], h: page.read.paper[3] }) : rgb
      const upright = rot90(crop, page.read.turns, 3)
      const { binary } = binarize(rgbToGray(upright))
      expect(estimateSkewWide(binary).angle).toBe(page.read.angle)
      const seg = segmentPage(upright)
      expect(seg.angle).toBe(page.read.angle)
      expect([seg.deskewed.width, seg.deskewed.height]).toEqual(page.read.deskewed)
      expect(Math.abs(seg.boxes.length - page.read.boxes.length)).toBeLessThanOrEqual(3)
    })
  }
})

describe.skipIf(!HAVE_MODEL || !HAVE_BACKBONE)('quarter turns with the model vote (golden read.turns)', () => {
  for (const page of pages) {
    it(`${stem(page.file)}: ${page.read.turns} turn(s)`, { timeout: 300_000 }, async () => {
      const engine = await modelEngine()
      const score = async (g: Parameters<typeof grayFeatures>[1]) => {
        const f = await grayFeatures(engine, g)
        return confidence(applyHead(f.values, f.frames, engine.baseHead), true)
      }
      const rgb = await pageRgb(page.file)
      const t0 = performance.now()
      const { turns, seg } = await orientPage(cropToPaper(rgb, findPaper(rgb)), score)
      console.info(`${stem(page.file)}: oriented (${turns} turns) and segmented in ${Math.round(performance.now() - t0)} ms`)
      expect(turns).toBe(page.read.turns)
      expect(seg.angle).toBe(page.read.angle)
      expect([seg.deskewed.width, seg.deskewed.height]).toEqual(page.read.deskewed)
    })
  }
})

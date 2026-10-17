import { describe, expect, it } from 'vitest'
import { labelsToText } from '../../src/engine/charset.ts'
import { beamSearch, confidence, greedy } from '../../src/engine/decoder.ts'
import { charset, golden, goldenLogProbs, languageModels, stem } from './fixtures.ts'
import { HAVE_MODEL } from './paths.ts'

interface Line {
  file: string
  frames: number
  greedy: string
  beam: string
  emitting_confidence: number
}
const lines = golden<Line[]>('lines.json')
const cs = charset()

describe('greedy + confidence on golden log-probs', () => {
  for (const line of lines) {
    it(stem(line.file), () => {
      const lp = goldenLogProbs(line.file)
      expect(lp.frames).toBe(line.frames)
      expect(labelsToText(cs, greedy(lp))).toBe(line.greedy)
      expect(confidence(lp, true)).toBeCloseTo(line.emitting_confidence, 12)
    })
  }
})

describe.skipIf(!HAVE_MODEL)('beam search on golden log-probs (identical text)', () => {
  for (const line of lines) {
    it(stem(line.file), () => {
      const { lm, words } = languageModels()
      const lp = goldenLogProbs(line.file)
      expect(labelsToText(cs, beamSearch(lp, lm, words))).toBe(line.beam)
    })
  }
})

describe('decoder edge cases', () => {
  it('handles zero frames', () => {
    const lp = { data: new Float32Array(0), frames: 0, classes: 198 }
    expect(greedy(lp)).toEqual([])
    expect(confidence(lp, false)).toBe(0)
    expect(confidence(lp, true)).toBe(0)
  })
})

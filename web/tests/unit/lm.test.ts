import { describe, expect, it } from 'vitest'
import { golden, languageModels } from './fixtures.ts'
import { HAVE_MODEL } from './paths.ts'

interface LmVectors {
  char: Array<{ context: number[]; char: number; logprob: number }>
  word: Array<{ word: number[]; text: string; bonus: number }>
}
const v = golden<LmVectors>('lm.json')

describe.skipIf(!HAVE_MODEL)('language models (golden/lm.json, within 1e-4)', () => {
  it(`char LM: ${v.char.length} cases`, () => {
    const { lm } = languageModels()
    for (const c of v.char) expect(Math.abs(lm.logprob(c.context, c.char) - c.logprob)).toBeLessThan(1e-4)
  })
  it(`word bonus: ${v.word.length} words`, () => {
    const { words } = languageModels()
    for (const w of v.word) expect(Math.abs(words.bonus(w.word) - w.bonus), w.text).toBeLessThan(1e-4)
  })
  it('char LM answers the same twice (cache)', () => {
    const { lm } = languageModels()
    const c = v.char[0]
    expect(lm.logprob(c.context, c.char)).toBe(lm.logprob(c.context, c.char))
  })
})

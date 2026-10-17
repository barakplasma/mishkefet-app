import { describe, expect, it } from 'vitest'
import { pairLines, wordDiff } from '../../src/lib/text.ts'

const line = (text: string, y: number) => ({ text, confidence: 0.9, box: { x: 0, y, w: 100, h: 20 } })

describe('compare', () => {
  it('pairs lines by box when the readings dropped different blank lines', () => {
    const before = [line('אחד', 0), line('שתיים שלוש', 40), line('ארבע', 80)]
    const after = [line('שתיים שלש', 40), line('ארבע', 80), line('חמש', 120)]
    expect(pairLines(before, after)).toEqual([
      ['שתיים שלוש', 'שתיים שלש'],
      ['ארבע', 'ארבע'],
      ['', 'חמש'],
      ['אחד', ''],
    ])
  })

  it('marks only the words that differ', () => {
    const t = wordDiff('היום יום א מנחם', 'קום יום אל מנחם', 'after')
    expect(t.map((x) => x.changed)).toEqual([true, false, true, false])
  })
})

/**
 * Which word to ask for next (spec/training.md).
 *
 * Round 1 makes sure every letter, final forms included, is written at least
 * once. After that, words are weighted by how often this hand's letters get
 * misread, so the rounds spend their time where the model is weakest. Words
 * from the user's own scans join the pool when the model was unsure of the
 * line they came from; the user never sees that judgment, it only steers
 * which words to practise.
 */

import { editOps } from './text.ts'

export const LETTERS = 'אבגדהוזחטיכךלמםנןסעפףצץקרשת'

export interface Attempt {
  word: string
  readAs: string
}

export interface WordPicker {
  next(): string
}

/** Per-letter misread rate with a mild prior: (errors + 1) / (seen + 4). */
export function letterErrorRates(attempts: Attempt[]): Map<string, number> {
  const seen = new Map<string, number>()
  const err = new Map<string, number>()
  for (const a of attempts) {
    const wrong = new Set(editOps(a.word, a.readAs).wrongTargetIndices)
    ;[...a.word].forEach((c, i) => {
      seen.set(c, (seen.get(c) ?? 0) + 1)
      if (wrong.has(i)) err.set(c, (err.get(c) ?? 0) + 1)
    })
  }
  const out = new Map<string, number>()
  for (const c of LETTERS) out.set(c, ((err.get(c) ?? 0) + 1) / ((seen.get(c) ?? 0) + 4))
  return out
}

export function makePicker(pool: string[], attempts: Attempt[], scanWords: string[], random: () => number = Math.random): WordPicker {
  const common = pool.filter((w) => w.length >= 2 && w.length <= 7).slice(0, 2000)
  const recent = new Set(attempts.slice(-120).map((a) => a.word))
  const rates = letterErrorRates(attempts)
  const firstRound = attempts.length < 40
  const covered = new Set<string>(firstRound ? attempts.flatMap((a) => [...a.word]) : LETTERS)
  const extra = [...new Set(scanWords.filter((w) => /^[א-ת]{2,7}$/.test(w)))]

  const weightOf = (w: string, rank: number): number => {
    const letters = [...w]
    const errors = letters.reduce((s, c) => s + (rates.get(c) ?? 0.25), 0) / letters.length
    const freq = 1 / Math.sqrt(1 + rank / 50)
    const len = w.length >= 3 && w.length <= 6 ? 1 : 0.5
    return errors * errors * freq * len
  }

  return {
    next(): string {
      const missing = [...LETTERS].filter((c) => !covered.has(c))
      let candidates: Array<[string, number]>
      if (missing.length) {
        // cover the letters this round has not had yet, common words first
        candidates = common
          .map((w, i): [string, number] => [w, [...new Set(w)].filter((c) => missing.includes(c)).length * (1 / Math.sqrt(1 + i / 30))])
          .filter(([w, s]) => s > 0 && !recent.has(w))
      } else {
        candidates = [
          ...common.map((w, i): [string, number] => [w, weightOf(w, i)]),
          ...extra.map((w): [string, number] => [w, weightOf(w, 100) * 2]),
        ].filter(([w]) => !recent.has(w))
      }
      if (!candidates.length) candidates = common.map((w): [string, number] => [w, 1])
      const total = candidates.reduce((s, [, x]) => s + x, 0)
      let r = random() * total
      let pick = candidates[candidates.length - 1][0]
      for (const [w, x] of candidates) {
        r -= x
        if (r <= 0) {
          pick = w
          break
        }
      }
      recent.add(pick)
      for (const c of pick) covered.add(c)
      return pick
    },
  }
}

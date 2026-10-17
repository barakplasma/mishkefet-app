/**
 * CTC decoding (spec/decoder.md): greedy, confidence and the prefix beam
 * search with the character and word LMs. All score arithmetic is in double.
 *
 * Prefixes are strings whose char codes are class indices: a Map keyed by
 * them keeps insertion order (the tie-break the reference relies on) and
 * extending one is a cheap concatenation.
 */

import { BLANK, SPACE } from './constants.ts'
import type { CharLM, WordLM } from './lm.ts'

/** Log-probabilities of one view: frames x classes, row-major. */
export interface LogProbs {
  data: Float32Array
  frames: number
  classes: number
}

export const DECODE = { beamWidth: 12, topK: 8, lmWeight: 0.4, lengthBonus: 0.6, wordWeight: 0.2 } as const

function lse(a: number, b: number): number {
  if (a === -Infinity) return b
  if (b === -Infinity) return a
  return a > b ? a + Math.log1p(Math.exp(b - a)) : b + Math.log1p(Math.exp(a - b))
}

function argmaxRow(lp: LogProbs, t: number): number {
  const o = t * lp.classes
  let best = 0
  let v = lp.data[o]
  for (let k = 1; k < lp.classes; k++) {
    if (lp.data[o + k] > v) {
      v = lp.data[o + k]
      best = k
    }
  }
  return best
}

/** Argmax per frame (ties to the lower index), collapse repeats, drop blanks. */
export function greedy(lp: LogProbs): number[] {
  const out: number[] = []
  let prev = -1
  for (let t = 0; t < lp.frames; t++) {
    const idx = argmaxRow(lp, t)
    if (idx !== prev && idx !== BLANK) out.push(idx)
    prev = idx
  }
  return out
}

/**
 * Mean of exp(max logprob) over all frames, or (emittingOnly) over frames whose
 * argmax is a non-blank that differs from the previous frame's.
 */
export function confidence(lp: LogProbs, emittingOnly: boolean): number {
  if (lp.frames === 0) return 0
  let sum = 0
  let n = 0
  let prev = -1
  for (let t = 0; t < lp.frames; t++) {
    const best = argmaxRow(lp, t)
    const p = Math.exp(lp.data[t * lp.classes + best])
    if (!emittingOnly || (best !== BLANK && (t === 0 || best !== prev))) {
      sum += p
      n++
    }
    prev = best
  }
  return n ? sum / n : 0
}

/** Indices of the k largest values of row t, largest first, ties to the lower index. */
function topK(lp: LogProbs, t: number, k: number, out: Int32Array): number {
  const o = t * lp.classes
  let n = 0
  for (let c = 0; c < lp.classes; c++) {
    const v = lp.data[o + c]
    if (n === k && v <= lp.data[o + out[n - 1]]) continue
    // insert after every value >= v (equal values keep the lower index first)
    let i = n < k ? n : k - 1
    while (i > 0 && lp.data[o + out[i - 1]] < v) {
      out[i] = out[i - 1]
      i--
    }
    out[i] = c
    if (n < k) n++
  }
  return n
}

/** The characters after the last space (the whole prefix when there is none). */
function lastWord(prefix: string): string {
  const i = prefix.lastIndexOf(String.fromCharCode(SPACE))
  return i < 0 ? prefix : prefix.slice(i + 1)
}

function codes(s: string): number[] {
  const out = new Array<number>(s.length)
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i)
  return out
}

interface Entry {
  pb: number
  pnb: number
}

/** CTC prefix beam search with LMs, mirroring reference.beam_search exactly. */
export function beamSearch(lp: LogProbs, lm: CharLM, words: WordLM, opts = DECODE): number[] {
  const { beamWidth, topK: k, lmWeight, lengthBonus, wordWeight } = opts
  let beams: Array<[string, Entry]> = [['', { pb: 0, pnb: -Infinity }]]
  const cands = new Int32Array(Math.min(k, lp.classes))
  const get = (m: Map<string, Entry>, key: string): Entry => {
    let e = m.get(key)
    if (!e) {
      e = { pb: -Infinity, pnb: -Infinity }
      m.set(key, e)
    }
    return e
  }
  for (let t = 0; t < lp.frames; t++) {
    const nc = topK(lp, t, cands.length, cands)
    const o = t * lp.classes
    const next = new Map<string, Entry>()
    for (const [prefix, { pb, pnb }] of beams) {
      const ptot = lse(pb, pnb)
      let ctx: number[] | null = null
      for (let ci = 0; ci < nc; ci++) {
        const c = cands[ci]
        const p = lp.data[o + c]
        if (c === BLANK) {
          const e = get(next, prefix)
          e.pb = lse(e.pb, ptot + p)
          continue
        }
        let src: number
        if (prefix.length && prefix.charCodeAt(prefix.length - 1) === c) {
          const e = get(next, prefix)
          e.pnb = lse(e.pnb, pnb + p)
          src = pb
        } else {
          src = ptot
        }
        if (src === -Infinity) continue
        ctx ??= codes(prefix)
        let bonus = lmWeight * lm.logprob(ctx, c) + lengthBonus
        if (wordWeight && c === SPACE) {
          const closed = lastWord(prefix)
          if (closed.length) bonus += wordWeight * words.bonus(codes(closed))
        }
        const e = get(next, prefix + String.fromCharCode(c))
        e.pnb = lse(e.pnb, src + p + bonus)
      }
    }
    const ranked = [...next].map((kv) => ({ kv, s: lse(kv[1].pb, kv[1].pnb) }))
    ranked.sort((a, b) => (a.s > b.s ? -1 : a.s < b.s ? 1 : 0)) // stable
    beams = ranked.slice(0, beamWidth).map((r) => r.kv)
    if (!beams.length) beams = [['', { pb: 0, pnb: -Infinity }]]
  }
  let best = beams[0][0]
  let bestScore = -Infinity
  let first = true
  for (const [prefix, e] of beams) {
    let s = lse(e.pb, e.pnb)
    if (wordWeight) {
      const tail = lastWord(prefix)
      if (tail.length) s += wordWeight * words.bonus(codes(tail))
    }
    if (first || s > bestScore) {
      best = prefix
      bestScore = s
      first = false
    }
  }
  return codes(best)
}

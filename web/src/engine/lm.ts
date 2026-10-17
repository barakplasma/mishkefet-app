/**
 * The two language-model binaries (spec/lm-format.md), read in place from
 * their ArrayBuffers: a stupid-backoff character 6-gram trie and a word
 * unigram prior. Characters are charset indices.
 */

import type { Charset } from './charset.ts'

const ABSENT = 65535

/** A fresh, 4-byte aligned copy (typed-array views need aligned offsets). */
function ownBuffer(bytes: Uint8Array | ArrayBuffer): ArrayBuffer {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes)
  return u8.slice().buffer
}

function magic(buf: ArrayBuffer, want: string): void {
  const got = String.fromCharCode(...new Uint8Array(buf, 0, 4))
  if (got !== want) throw new Error(`expected ${want} header, got ${JSON.stringify(got)}`)
}

export interface CharLM {
  order: number
  /** Log-probability of class `c` after `context` (class indices, oldest first). */
  logprob(context: ArrayLike<number>, c: number): number
}

export function parseCharLM(bytes: Uint8Array | ArrayBuffer): CharLM {
  const buf = ownBuffer(bytes)
  magic(buf, 'MKL1')
  const dv = new DataView(buf)
  const order = dv.getUint32(4, true)
  const logBackoff = dv.getFloat32(8, true)
  const logFloor = dv.getFloat32(12, true)
  const scale = dv.getFloat32(16, true)
  const count: number[] = []
  for (let k = 0; k < order; k++) count.push(dv.getUint32(24 + 4 * k, true))
  let off = 24 + 4 * order
  const child: Uint32Array[] = []
  for (let k = 0; k < order - 1; k++) {
    child.push(new Uint32Array(buf, off, count[k] + 1))
    off += 4 * (count[k] + 1)
  }
  const q: Uint16Array[] = []
  for (let k = 0; k < order; k++) {
    q.push(new Uint16Array(buf, off, count[k]))
    off += 2 * count[k]
  }
  const ch: Uint8Array[] = []
  for (let k = 0; k < order; k++) {
    ch.push(new Uint8Array(buf, off, count[k]))
    off += count[k]
  }
  if (off > buf.byteLength) throw new Error('charlm.bin is truncated')

  /** q of the n-gram gram[from..to) + [c], or -1 when the node does not exist. */
  function find(gram: ArrayLike<number>, from: number, to: number, c: number): number {
    let lo = 0
    let hi = count[0]
    let node = -1
    const len = to - from + 1
    for (let depth = 0; depth < len; depth++) {
      const sym = depth < len - 1 ? gram[from + depth] : c
      if (depth) {
        lo = child[depth - 1][node]
        hi = child[depth - 1][node + 1]
      }
      const arr = ch[depth]
      // lower bound
      let a = lo
      let b = hi
      while (a < b) {
        const m = (a + b) >>> 1
        if (arr[m] < sym) a = m + 1
        else b = m
      }
      if (a >= hi || arr[a] !== sym) return -1
      node = a
    }
    return q[len - 1][node]
  }

  // The decoder asks the same (context, c) questions over and over.
  const cache = new Map<string, number>()

  function logprob(context: ArrayLike<number>, c: number): number {
    const n = context.length
    const start = Math.max(0, n - (order - 1))
    let key = String.fromCharCode(c)
    for (let i = start; i < n; i++) key += String.fromCharCode(context[i])
    const hit = cache.get(key)
    if (hit !== undefined) return hit
    let penalty = 0
    let result = NaN
    for (let L = n - start; L >= 0; L--) {
      const v = find(context, n - L, n, c)
      if (v >= 0 && v !== ABSENT) {
        result = penalty - v / scale
        break
      }
      penalty += logBackoff
    }
    if (Number.isNaN(result)) result = penalty + logFloor
    if (cache.size > 200_000) cache.clear()
    cache.set(key, result)
    return result
  }

  return { order, logprob }
}

export interface WordLM {
  logprob(word: ArrayLike<number>): number
  /** max(0, logprob(word) - oov): what a closed word adds to a beam. */
  bonus(word: ArrayLike<number>): number
}

const PREFIX_LETTERS = 'בהוכלמש'

export function parseWordLM(bytes: Uint8Array | ArrayBuffer, charset: Charset): WordLM {
  const buf = ownBuffer(bytes)
  magic(buf, 'MKW1')
  const dv = new DataView(buf)
  const n = dv.getUint32(4, true)
  const oov = dv.getFloat32(8, true)
  const discount = dv.getFloat32(12, true)
  const scale = dv.getFloat32(16, true)
  let off = 24
  const offsets = new Uint32Array(buf, off, n + 1)
  off += 4 * (n + 1)
  const q = new Uint16Array(buf, off, n)
  off += 2 * n
  const blob = new Uint8Array(buf, off)
  const prefixes = new Set<number>()
  for (const ch of PREFIX_LETTERS) {
    const i = charset.index.get(ch)
    if (i !== undefined) prefixes.add(i)
  }

  /** Bytewise compare of entry i against word[from..]: unsigned, then shorter first. */
  function cmp(i: number, word: ArrayLike<number>, from: number): number {
    const a = offsets[i]
    const la = offsets[i + 1] - a
    const lb = word.length - from
    const m = Math.min(la, lb)
    for (let k = 0; k < m; k++) {
      const d = blob[a + k] - word[from + k]
      if (d) return d
    }
    return la - lb
  }

  function get(word: ArrayLike<number>, from: number): number | null {
    let lo = 0
    let hi = n
    while (lo < hi) {
      const mid = (lo + hi) >>> 1
      if (cmp(mid, word, from) < 0) lo = mid + 1
      else hi = mid
    }
    if (lo < n && cmp(lo, word, from) === 0) return -q[lo] / scale
    return null
  }

  function logprob(word: ArrayLike<number>): number {
    if (!word.length) return 0
    const hit = get(word, 0)
    if (hit !== null) return hit
    if (word.length > 2 && prefixes.has(word[0])) {
      const rest = get(word, 1)
      if (rest !== null) return rest + discount
    }
    return oov
  }

  const cache = new Map<string, number>()
  function bonus(word: ArrayLike<number>): number {
    let key = ''
    for (let i = 0; i < word.length; i++) key += String.fromCharCode(word[i])
    const hit = cache.get(key)
    if (hit !== undefined) return hit
    const b = Math.max(0, logprob(word) - oov)
    if (cache.size > 100_000) cache.clear()
    cache.set(key, b)
    return b
  }

  return { logprob, bonus }
}

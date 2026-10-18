/** Small text utilities: alignment, word diff, Hebrew dates, titles. */

import type { LineResult } from '../engine/api.ts'

/** Levenshtein alignment of target against a reading: which target characters were not read right. */
export function editOps(target: string, read: string): { distance: number; wrongTargetIndices: number[] } {
  const a = [...target]
  const b = [...read]
  const d: number[][] = Array.from({ length: a.length + 1 }, (_, i) => [i, ...Array<number>(b.length).fill(0)])
  for (let j = 1; j <= b.length; j++) d[0][j] = j
  for (let i = 1; i <= a.length; i++)
    for (let j = 1; j <= b.length; j++)
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1))
  const wrong: number[] = []
  let i = a.length
  let j = b.length
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && d[i][j] === d[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1)) {
      if (a[i - 1] !== b[j - 1]) wrong.push(i - 1)
      i--
      j--
    } else if (i > 0 && d[i][j] === d[i - 1][j] + 1) {
      wrong.push(i - 1)
      i--
    } else {
      j--
    }
  }
  return { distance: d[a.length][b.length], wrongTargetIndices: wrong.reverse() }
}

export type DiffToken = { text: string; changed: boolean }

/** Word-level diff of `after` against `before`: tokens of `side`, marked when not in the common subsequence. */
export function wordDiff(before: string, after: string, side: 'before' | 'after'): DiffToken[] {
  const a = before.split(/\s+/).filter(Boolean)
  const b = after.split(/\s+/).filter(Boolean)
  const L: number[][] = Array.from({ length: a.length + 1 }, () => Array<number>(b.length + 1).fill(0))
  for (let i = a.length - 1; i >= 0; i--)
    for (let j = b.length - 1; j >= 0; j--) L[i][j] = a[i] === b[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1])
  const keepA = new Set<number>()
  const keepB = new Set<number>()
  let i = 0
  let j = 0
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) {
      keepA.add(i++)
      keepB.add(j++)
    } else if (L[i + 1][j] >= L[i][j + 1]) i++
    else j++
  }
  return side === 'before' ? a.map((t, k) => ({ text: t, changed: !keepA.has(k) })) : b.map((t, k) => ({ text: t, changed: !keepB.has(k) }))
}

const boxKey = (l: LineResult) => (l.box ? `${l.box.x},${l.box.y},${l.box.w},${l.box.h}` : '')

/**
 * Pairs each new line with the old line at the same place on the page. A re-read
 * segments the same image the same way, so boxes match exactly; matching by index
 * would not, because each reading drops its own blank lines.
 */
export function pairLines(before: LineResult[], after: LineResult[]): Array<[string, string]> {
  const old = new Map(before.map((l) => [boxKey(l), l.text]))
  const used = new Set<string>()
  const pairs: Array<[string, string]> = after.map((l, i) => {
    const k = boxKey(l)
    const hit = k && old.has(k) ? k : null
    if (hit) used.add(hit)
    // without boxes (a single-line scan) fall back to position
    return [hit ? old.get(hit)! : k ? '' : (before[i]?.text ?? ''), l.text]
  })
  // old lines the new reading left blank still count as changes
  for (const l of before) if (boxKey(l) && !used.has(boxKey(l)) && after.some((a) => a.box)) pairs.push([l.text, ''])
  return pairs
}

/** Lines worth keeping in a page's transcript (spec/decoder.md, Page). */
export function keepLine(l: LineResult): boolean {
  const t = l.text.trim()
  if (!t) return false
  return t.replace(/\s+/g, '').length > 3 || l.confidence >= 0.9
}

const MONTHS = ['בינואר', 'בפברואר', 'במרץ', 'באפריל', 'במאי', 'ביוני', 'ביולי', 'באוגוסט', 'בספטמבר', 'באוקטובר', 'בנובמבר', 'בדצמבר']

export function dateLabel(ms: number, now = Date.now()): string {
  const d = new Date(ms)
  const today = new Date(now)
  const startOf = (x: Date) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime()
  const days = Math.round((startOf(today) - startOf(d)) / 86400000)
  const hm = `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`
  if (days === 0) return `היום, ${hm}`
  if (days === 1) return `אתמול, ${hm}`
  const year = d.getFullYear() !== today.getFullYear() ? ` ${d.getFullYear()}` : ''
  return `${d.getDate()} ${MONTHS[d.getMonth()]}${year}`
}

/** A default title: the first words of the first line, else the date. */
export function defaultTitle(lines: string[], ms: number): string {
  const first = lines.find((l) => l.trim())?.trim().split(/\s+/).slice(0, 4).join(' ')
  return first || `סריקה ${dateLabel(ms)}`
}

export function linesLabel(n: number): string {
  return n === 1 ? 'שורה אחת' : `${n} שורות`
}

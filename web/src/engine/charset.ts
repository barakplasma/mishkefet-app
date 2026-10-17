/**
 * The 198 output classes (charset.json: index 0 is the CTC blank, 38 is space).
 *
 * Label sequences are carried around as arrays of class indices; the decoder
 * also uses strings whose char codes are class indices (all < 198), which makes
 * prefixes cheap Map keys.
 */

export interface Charset {
  chars: readonly string[]
  index: ReadonlyMap<string, number>
}

export function parseCharset(json: string | readonly string[]): Charset {
  const chars: string[] = typeof json === 'string' ? JSON.parse(json) : [...json]
  if (!Array.isArray(chars) || chars.some((c) => typeof c !== 'string')) throw new Error('charset.json is not a list of strings')
  const index = new Map<string, number>()
  chars.forEach((c, i) => {
    if (i > 0 && !index.has(c)) index.set(c, i)
  })
  return { chars, index }
}

/** Class indices to text. */
export function labelsToText(cs: Charset, labels: ArrayLike<number>): string {
  let s = ''
  for (let i = 0; i < labels.length; i++) s += cs.chars[labels[i]] ?? ''
  return s
}

/** Text to class indices; characters outside the charset are dropped. */
export function textToLabels(cs: Charset, text: string): number[] {
  const out: number[] = []
  for (const ch of text) {
    const i = cs.index.get(ch)
    if (i !== undefined) out.push(i)
  }
  return out
}

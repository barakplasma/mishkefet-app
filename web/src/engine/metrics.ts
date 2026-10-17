/** Edit distance and the accuracy numbers built on it, over Unicode code points. */

export function editDistance(a: string, b: string): number {
  const x = Array.from(a)
  const y = Array.from(b)
  if (!x.length) return y.length
  if (!y.length) return x.length
  let prev = new Int32Array(y.length + 1)
  let cur = new Int32Array(y.length + 1)
  for (let j = 0; j <= y.length; j++) prev[j] = j
  for (let i = 1; i <= x.length; i++) {
    cur[0] = i
    for (let j = 1; j <= y.length; j++) {
      const sub = prev[j - 1] + (x[i - 1] === y[j - 1] ? 0 : 1)
      cur[j] = Math.min(sub, prev[j] + 1, cur[j - 1] + 1)
    }
    ;[prev, cur] = [cur, prev]
  }
  return prev[y.length]
}

/** Character error rate of `hyp` against `ref` (edit distance / reference length). */
export function cer(ref: string, hyp: string): number {
  const n = Array.from(ref).length
  return n ? editDistance(ref, hyp) / n : hyp.length ? 1 : 0
}

/** 1 - sum(edit distance) / sum(target length), the held-out score of a training round. */
export function characterAccuracy(pairs: ReadonlyArray<{ target: string; read: string }>): number {
  let edits = 0
  let total = 0
  for (const p of pairs) {
    edits += editDistance(p.target, p.read)
    total += Array.from(p.target).length
  }
  return total ? 1 - edits / total : 0
}

/**
 * CTC negative log-likelihood and its gradient with respect to the logits
 * (spec/adapter.md, adapter.ctc_loss_grad). Double precision; -1e30 stands
 * for log zero, as in the reference, so the arithmetic matches it exactly.
 */

const NEG = -1e30

function lse2(a: number, b: number): number {
  const m = a > b ? a : b
  if (m <= NEG / 2) return NEG
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m))
}

export interface CtcResult {
  /** -log p(label | logits), or 0 when the label cannot fit in T frames. */
  loss: number
  /** False when the alignment is infeasible (the sample then contributes nothing). */
  feasible: boolean
}

/**
 * `logits` is T x C row-major (pre-softmax). The gradient d loss / d logits is
 * written to `grad` (T x C, float32 like the reference), zeros when infeasible.
 */
export function ctcLossGrad(logits: ArrayLike<number>, T: number, C: number, label: readonly number[], grad: Float32Array, blank = 0): CtcResult {
  // log-softmax in double
  const lp = new Float64Array(T * C)
  for (let t = 0; t < T; t++) {
    const o = t * C
    let m = -Infinity
    for (let k = 0; k < C; k++) if (logits[o + k] > m) m = logits[o + k]
    let s = 0
    for (let k = 0; k < C; k++) s += Math.exp(logits[o + k] - m)
    const ls = Math.log(s)
    for (let k = 0; k < C; k++) lp[o + k] = logits[o + k] - m - ls
  }
  const ext = [blank]
  for (const c of label) ext.push(c, blank)
  const S = ext.length
  // skip[s]: a transition from s-2 to s is allowed
  const skip = new Uint8Array(S)
  for (let s = 2; s < S; s++) skip[s] = ext[s] !== blank && ext[s] !== ext[s - 2] ? 1 : 0

  const alpha = new Float64Array(T * S).fill(NEG)
  alpha[0] = lp[ext[0]]
  if (S > 1) alpha[1] = lp[ext[1]]
  for (let t = 1; t < T; t++) {
    const p = (t - 1) * S
    const o = t * S
    for (let s = 0; s < S; s++) {
      let a = alpha[p + s]
      if (s >= 1) a = lse2(a, alpha[p + s - 1])
      if (s >= 2 && skip[s]) a = lse2(a, alpha[p + s - 2])
      alpha[o + s] = a + lp[t * C + ext[s]]
    }
  }

  // beta[t][s]: log prob of emitting the rest after state s at t, excluding frame t
  const beta = new Float64Array(T * S).fill(NEG)
  beta[(T - 1) * S + S - 1] = 0
  if (S > 1) beta[(T - 1) * S + S - 2] = 0
  const nxt = new Float64Array(S)
  for (let t = T - 2; t >= 0; t--) {
    const n = (t + 1) * S
    for (let s = 0; s < S; s++) nxt[s] = beta[n + s] + lp[(t + 1) * C + ext[s]]
    const o = t * S
    for (let s = 0; s < S; s++) {
      let b = nxt[s]
      if (s + 1 < S) b = lse2(b, nxt[s + 1])
      if (s + 2 < S && skip[s + 2]) b = lse2(b, nxt[s + 2])
      beta[o + s] = b
    }
  }

  let logz = alpha[(T - 1) * S + S - 1]
  if (S > 1) logz = lse2(logz, alpha[(T - 1) * S + S - 2])
  if (logz <= NEG / 2) {
    grad.fill(0, 0, T * C)
    return { loss: 0, feasible: false }
  }
  const occ = new Float64Array(C)
  for (let t = 0; t < T; t++) {
    occ.fill(0)
    for (let s = 0; s < S; s++) occ[ext[s]] += Math.exp(alpha[t * S + s] + beta[t * S + s] - logz)
    const o = t * C
    for (let k = 0; k < C; k++) grad[o + k] = Math.exp(lp[o + k]) - occ[k]
  }
  return { loss: -logz, feasible: true }
}

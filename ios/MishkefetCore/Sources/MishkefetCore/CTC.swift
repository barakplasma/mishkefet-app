import Accelerate
import Foundation

/// CTC negative log-likelihood and its gradient w.r.t. the logits (adapter.py `ctc_loss_grad`).
enum CTC {
    static let logZero = -1e30

    /// `logits` (frames x classes, float32). Returns nil when the label cannot be aligned
    /// in `frames` (the sample then contributes nothing, like zero_infinity).
    static func lossAndGradient(_ logits: UnsafeBufferPointer<Float>, frames T: Int, classes C: Int,
                                label: [Int], forceLogDomain: Bool = false) -> (loss: Double, grad: [Float])? {
        // A label fits when T covers every character plus a blank between repeats.
        let repeats = zip(label, label.dropFirst()).filter { $0 == $1 }.count
        guard T > 0, T >= label.count + repeats else { return nil }
        // softmax in double
        var p = [Double](repeating: 0, count: T * C)
        var logNorm = [Double](repeating: 0, count: T)  // max + log(sum exp(z - max))
        var n = Int32(C)
        p.withUnsafeMutableBufferPointer { pp in
            for t in 0..<T {
                let row = pp.baseAddress! + t * C
                let z = logits.baseAddress! + t * C
                vDSP_vspdp(z, 1, row, 1, vDSP_Length(C))
                var m = 0.0
                vDSP_maxvD(row, 1, &m, vDSP_Length(C))
                var negM = -m
                vDSP_vsaddD(row, 1, &negM, row, 1, vDSP_Length(C))
                vvexp(row, row, &n)
                var s = 0.0
                vDSP_sveD(row, 1, &s, vDSP_Length(C))
                var inv = 1 / s
                vDSP_vsmulD(row, 1, &inv, row, 1, vDSP_Length(C))
                logNorm[t] = m + log(s)
            }
        }
        var ext = [Charset.blank]
        for c in label { ext += [c, Charset.blank] }
        let S = ext.count
        let skip = (0..<S).map { s in s >= 2 && ext[s] != Charset.blank && ext[s] != ext[s - 2] }

        if !forceLogDomain, let fast = scaled(p, T: T, C: C, ext: ext, skip: skip) {
            return fast
        }
        return logDomain(logits, logNorm: logNorm, T: T, C: C, ext: ext, skip: skip)
    }

    /// Forward-backward with per-frame normalization (Rabiner scaling). Nil if any
    /// scale or the final mass underflows, or anything turns non-finite.
    private static func scaled(_ p: [Double], T: Int, C: Int, ext: [Int], skip: [Bool]) -> (loss: Double, grad: [Float])? {
        let S = ext.count
        var alpha = [Double](repeating: 0, count: T * S)
        var beta = [Double](repeating: 0, count: T * S)
        var c = [Double](repeating: 0, count: T)
        alpha[0] = p[ext[0]]
        if S > 1 { alpha[1] = p[ext[1]] }
        for t in 0..<T {
            let a = t * S
            if t > 0 {
                let prev = (t - 1) * S
                for s in 0..<S {
                    var v = alpha[prev + s]
                    if s >= 1 { v += alpha[prev + s - 1] }
                    if skip[s] { v += alpha[prev + s - 2] }
                    alpha[a + s] = v * p[t * C + ext[s]]
                }
            }
            var sum = 0.0
            for s in 0..<S { sum += alpha[a + s] }
            guard sum > 0, sum.isFinite else { return nil }
            c[t] = sum
            for s in 0..<S { alpha[a + s] /= sum }
        }
        let last = (T - 1) * S
        let final = alpha[last + S - 1] + (S > 1 ? alpha[last + S - 2] : 0)
        guard final > 0 else { return nil }
        beta[last + S - 1] = 1
        if S > 1 { beta[last + S - 2] = 1 }
        if T >= 2 {
            for t in stride(from: T - 2, through: 0, by: -1) {
                let b = t * S, nb = (t + 1) * S
                let inv = 1 / c[t + 1]
                for s in 0..<S {
                    var v = beta[nb + s] * p[(t + 1) * C + ext[s]]
                    if s + 1 < S { v += beta[nb + s + 1] * p[(t + 1) * C + ext[s + 1]] }
                    if s + 2 < S && skip[s + 2] { v += beta[nb + s + 2] * p[(t + 1) * C + ext[s + 2]] }
                    beta[b + s] = v * inv
                }
            }
        }
        var logZ = log(final)
        for t in 0..<T { logZ += log(c[t]) }
        // dz = p - occupancy, occupancy = alpha * beta / final per (t, ext[s])
        var g = p
        for t in 0..<T {
            for s in 0..<S {
                let occ = alpha[t * S + s] * beta[t * S + s] / final
                guard occ.isFinite else { return nil }
                g[t * C + ext[s]] -= occ
            }
        }
        var grad = [Float](repeating: 0, count: T * C)
        vDSP_vdpsp(g, 1, &grad, 1, vDSP_Length(T * C))
        guard logZ.isFinite else { return nil }
        return (-logZ, grad)
    }

    @inline(__always) private static func lse2(_ a: Double, _ b: Double) -> Double {
        let m = max(a, b)
        return m <= logZero / 2 ? logZero : m + log(exp(a - m) + exp(b - m))
    }

    /// The reference recursion in log space (slow path for extreme inputs).
    private static func logDomain(_ logits: UnsafeBufferPointer<Float>, logNorm: [Double],
                                  T: Int, C: Int, ext: [Int], skip: [Bool]) -> (loss: Double, grad: [Float])? {
        let S = ext.count
        func lp(_ t: Int, _ k: Int) -> Double { Double(logits[t * C + k]) - logNorm[t] }
        var alpha = [Double](repeating: logZero, count: T * S)
        var beta = [Double](repeating: logZero, count: T * S)
        alpha[0] = lp(0, ext[0])
        if S > 1 { alpha[1] = lp(0, ext[1]) }
        if T > 1 {
            for t in 1..<T {
                for s in 0..<S {
                    var v = alpha[(t - 1) * S + s]
                    if s >= 1 { v = lse2(v, alpha[(t - 1) * S + s - 1]) }
                    if skip[s] { v = lse2(v, alpha[(t - 1) * S + s - 2]) }
                    alpha[t * S + s] = v + lp(t, ext[s])
                }
            }
        }
        beta[(T - 1) * S + S - 1] = 0
        if S > 1 { beta[(T - 1) * S + S - 2] = 0 }
        if T >= 2 {
            for t in stride(from: T - 2, through: 0, by: -1) {
                func nxt(_ s: Int) -> Double { beta[(t + 1) * S + s] + lp(t + 1, ext[s]) }
                for s in 0..<S {
                    var v = nxt(s)
                    if s + 1 < S { v = lse2(v, nxt(s + 1)) }
                    if s + 2 < S && skip[s + 2] { v = lse2(v, nxt(s + 2)) }
                    beta[t * S + s] = v
                }
            }
        }
        var logZ = alpha[(T - 1) * S + S - 1]
        if S > 1 { logZ = lse2(logZ, alpha[(T - 1) * S + S - 2]) }
        if logZ <= logZero / 2 { return nil }
        var grad = [Float](repeating: 0, count: T * C)
        for t in 0..<T {
            var row = (0..<C).map { exp(lp(t, $0)) }
            for s in 0..<S { row[ext[s]] -= exp(alpha[t * S + s] + beta[t * S + s] - logZ) }
            for k in 0..<C { grad[t * C + k] = Float(row[k]) }
        }
        return (-logZ, grad)
    }
}

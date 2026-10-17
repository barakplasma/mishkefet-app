import Accelerate
import Foundation

/// On-device fine-tuning settings (spec/adapt.json).
public struct AdaptConfig: Codable, Sendable, Equatable {
    public var steps: Int
    public var lr: Double
    public var beta1: Double
    public var beta2: Double
    public var eps: Double
    /// Pull toward the general head: l2 * (|W - W0|^2 + |b - b0|^2).
    public var l2: Double
    /// Augmented renders per sample, besides the plain one (iOS ships 4).
    public var augmentedViews: Int
    /// New samples that trigger a round.
    public var roundSize: Int
    /// Idle time after the last stroke before a canvas sample is taken.
    public var inactivityMs: Int
    /// Train on at most the latest this many non-held-out samples.
    public var maxTrainSamples: Int
    /// A sample is held out when fnv1a32(id) % heldOutModulo == 0.
    public var heldOutModulo: Int
    /// Accept a round when held-out accuracy drops by at most this much.
    public var acceptIfAccuracyDropAtMost: Double

    public init(steps: Int, lr: Double, beta1: Double = 0.9, beta2: Double = 0.999, eps: Double = 1e-8,
                l2: Double, augmentedViews: Int = 4, roundSize: Int = 40, inactivityMs: Int = 1200,
                maxTrainSamples: Int = 240, heldOutModulo: Int = 8, acceptIfAccuracyDropAtMost: Double = 0) {
        self.steps = steps
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.l2 = l2
        self.augmentedViews = augmentedViews
        self.roundSize = roundSize
        self.inactivityMs = inactivityMs
        self.maxTrainSamples = maxTrainSamples
        self.heldOutModulo = heldOutModulo
        self.acceptIfAccuracyDropAtMost = acceptIfAccuracyDropAtMost
    }

    public static let shipped = AdaptConfig(steps: 100, lr: 1e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8, l2: 0.1,
                                            augmentedViews: 4, roundSize: 40, inactivityMs: 1200,
                                            maxTrainSamples: 240, heldOutModulo: 8, acceptIfAccuracyDropAtMost: 0)

    public func isHeldOut(sampleID: String) -> Bool {
        heldOutModulo > 0 && fnv1a32(sampleID) % UInt32(heldOutModulo) == 0
    }
}

/// One training view: cached backbone features and its label.
public struct TrainingSample: Sendable {
    /// Row-major (frames x dim).
    public let features: [Float]
    public let frames: Int
    /// Charset indices.
    public let label: [Int]

    public init(features: [Float], frames: Int, label: [Int]) {
        self.features = features
        self.frames = frames
        self.label = label
    }
}

/// Full-batch Adam on the CTC head over cached features (spec/adapter.md, adapter.py `train`).
public enum HeadTrainer {
    /// `progress(step, loss)` is called after each step with that step's loss (before its update).
    public static func train(base: Head, samples: [TrainingSample], config: AdaptConfig,
                             progress: ((Int, Double) -> Void)?) -> Head {
        let C = base.classes, D = base.dim
        let used = samples.filter { $0.frames > 0 && $0.features.count == $0.frames * D }
        let offsets = used.reduce(into: [0]) { $0.append($0.last! + $1.frames) }
        let N = offsets.last!
        guard N > 0, config.steps > 0 else { return base }
        var F = [Float](repeating: 0, count: N * D)
        for (i, s) in used.enumerated() {
            F.withUnsafeMutableBufferPointer { _ = memcpy($0.baseAddress! + offsets[i] * D, s.features, s.features.count * 4) }
        }

        var head = base
        let W0 = base.weight, b0 = base.bias
        var mW = [Float](repeating: 0, count: C * D), vW = mW
        var mb = [Float](repeating: 0, count: C), vb = mb
        var dZ = [Float](repeating: 0, count: N * C)
        var gW = [Float](repeating: 0, count: C * D)
        var gb = [Float](repeating: 0, count: C)
        var losses = [Double](repeating: 0, count: used.count)
        var feasible = [Bool](repeating: false, count: used.count)

        for step in 1...config.steps {
            // Z = F W^T + b
            let Z = head.logits(F, frames: N)
            // Per-sample CTC, scaled by 1 / len(label).
            Z.withUnsafeBufferPointer { z in
                dZ.withUnsafeMutableBufferPointer { dz in
                    losses.withUnsafeMutableBufferPointer { lossOut in
                        feasible.withUnsafeMutableBufferPointer { ok in
                            DispatchQueue.concurrentPerform(iterations: used.count) { i in
                                let T = used[i].frames, o = offsets[i]
                                let rows = UnsafeBufferPointer(rebasing: z[(o * C)..<((o + T) * C)])
                                let dst = dz.baseAddress! + o * C
                                guard let r = CTC.lossAndGradient(rows, frames: T, classes: C, label: used[i].label) else {
                                    ok[i] = false
                                    dst.update(repeating: 0, count: T * C)
                                    return
                                }
                                var scale = 1 / Float(max(1, used[i].label.count))
                                vDSP_vsmul(r.grad, 1, &scale, dst, 1, vDSP_Length(T * C))
                                lossOut[i] = r.loss * Double(scale)
                                ok[i] = true
                            }
                        }
                    }
                }
            }
            let n = max(1, feasible.filter { $0 }.count)
            var loss = zip(losses, feasible).reduce(0.0) { $0 + ($1.1 ? $1.0 : 0) } / Double(n)

            // gW = dZ^T F / n,  gb = colsum(dZ) / n
            gW = Matrix.multiplyTransposedLeft(dZ, rows: N, cols: C, F, cols: D)
            var invN = 1 / Float(n)
            vDSP_vsmul(gW, 1, &invN, &gW, 1, vDSP_Length(C * D))
            dZ.withUnsafeBufferPointer { dz in
                for k in 0..<C {
                    var s: Float = 0
                    vDSP_sve(dz.baseAddress! + k, vDSP_Stride(C), &s, vDSP_Length(N))
                    gb[k] = s / Float(n)
                }
            }

            // L2 toward the general head
            let reg = l2Pull(head.weight, W0, &gW, l2: config.l2) + l2Pull(head.bias, b0, &gb, l2: config.l2)
            loss += config.l2 * reg

            adam(&head.weight, gW, &mW, &vW, step: step, config)
            adam(&head.bias, gb, &mb, &vb, step: step, config)
            progress?(step, loss)
        }
        return head
    }

    /// Adds 2 l2 (x - x0) to `g`; returns |x - x0|^2.
    private static func l2Pull(_ x: [Float], _ x0: [Float], _ g: inout [Float], l2: Double) -> Double {
        var d = [Float](repeating: 0, count: x.count)
        vDSP_vsub(x0, 1, x, 1, &d, 1, vDSP_Length(x.count))  // x - x0
        var sq: Float = 0
        vDSP_svesq(d, 1, &sq, vDSP_Length(d.count))
        var twoL2 = 2 * Float(l2)
        g.withUnsafeMutableBufferPointer { gp in
            vDSP_vsma(d, 1, &twoL2, gp.baseAddress!, 1, gp.baseAddress!, 1, vDSP_Length(d.count))
        }
        return Double(sq)
    }

    /// Bias-corrected Adam, float32 parameters and moments.
    private static func adam(_ x: inout [Float], _ g: [Float], _ m: inout [Float], _ v: inout [Float],
                             step: Int, _ c: AdaptConfig) {
        let n = vDSP_Length(x.count)
        var b1 = Float(c.beta1), b2 = Float(c.beta2), ob1 = Float(1 - c.beta1), ob2 = Float(1 - c.beta2)
        var ic1 = Float(1 / (1 - pow(c.beta1, Double(step)))), ic2 = Float(1 / (1 - pow(c.beta2, Double(step))))
        var eps = Float(c.eps), negLr = -Float(c.lr)
        var g2 = [Float](repeating: 0, count: x.count), mh = g2, vh = g2
        // m = b1 m + (1 - b1) g;  v = b2 v + (1 - b2) g^2
        vDSP_vsmul(m, 1, &b1, &m, 1, n)
        vDSP_vsma(g, 1, &ob1, m, 1, &m, 1, n)
        vDSP_vsq(g, 1, &g2, 1, n)
        vDSP_vsmul(v, 1, &b2, &v, 1, n)
        vDSP_vsma(g2, 1, &ob2, v, 1, &v, 1, n)
        // x -= lr * (m / c1) / (sqrt(v / c2) + eps)
        vDSP_vsmul(m, 1, &ic1, &mh, 1, n)
        vDSP_vsmul(v, 1, &ic2, &vh, 1, n)
        var count = Int32(x.count)
        vvsqrtf(&vh, vh, &count)
        vDSP_vsadd(vh, 1, &eps, &vh, 1, n)
        vDSP_vdiv(vh, 1, mh, 1, &mh, 1, n)
        vDSP_vsma(mh, 1, &negLr, x, 1, &x, 1, n)
    }
}

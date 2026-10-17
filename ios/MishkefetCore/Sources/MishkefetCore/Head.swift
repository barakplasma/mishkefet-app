import Accelerate
import Foundation

/// The CTC head (`head.bin`, "MKH1"): logits = features * weight^T + bias.
public struct Head: Sendable, Equatable {
    public let classes: Int
    public let dim: Int
    /// Row-major [classes][dim].
    public var weight: [Float]
    public var bias: [Float]

    public init(classes: Int, dim: Int, weight: [Float], bias: [Float]) {
        precondition(weight.count == classes * dim && bias.count == classes, "head shape mismatch")
        self.classes = classes
        self.dim = dim
        self.weight = weight
        self.bias = bias
    }

    /// Reads "MKH1" u32 classes, u32 dim, f32 weight[classes][dim], f32 bias[classes].
    /// Anything after the bias (an optional adapter) is ignored.
    public init(contentsOf url: URL) throws {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard data.count >= 12, data.prefix(4) == Data("MKH1".utf8) else {
            throw MishkefetError.badFile(url.lastPathComponent)
        }
        let classes = Int(data.u32(at: 4)), dim = Int(data.u32(at: 8))
        guard classes > 0, dim > 0, data.count >= 12 + 4 * (classes * dim + classes) else {
            throw MishkefetError.badFile(url.lastPathComponent)
        }
        self.init(classes: classes, dim: dim,
                  weight: data.f32Array(at: 12, count: classes * dim),
                  bias: data.f32Array(at: 12 + 4 * classes * dim, count: classes))
    }

    public func write(to url: URL) throws {
        var data = Data("MKH1".utf8)
        data.appendLE(UInt32(classes))
        data.appendLE(UInt32(dim))
        weight.withUnsafeBytes { data.append(contentsOf: $0) }
        bias.withUnsafeBytes { data.append(contentsOf: $0) }
        try data.write(to: url, options: .atomic)
    }

    /// Logits (frames x classes) for row-major features (frames x dim), float32.
    func logits(_ features: [Float], frames: Int) -> [Float] {
        var z = Matrix.multiplyTransposed(features, rows: frames, weight, rows: classes, inner: dim)
        z.withUnsafeMutableBufferPointer { zp in
            for t in 0..<frames {
                vDSP_vadd(zp.baseAddress! + t * classes, 1, bias, 1, zp.baseAddress! + t * classes, 1,
                          vDSP_Length(classes))
            }
        }
        return z
    }

    /// Log-softmax of the logits, float32 (frames x classes).
    public func logprobs(features: [Float], frames: Int) -> [Float] {
        var z = logits(features, frames: frames)
        LogSoftmax.inPlace(&z, rows: frames, cols: classes)
        return z
    }
}

enum Matrix {
    /// A (m x k) times B^T, B being (n x k); all row-major. Returns (m x n).
    static func multiplyTransposed(_ a: [Float], rows m: Int, _ b: [Float], rows n: Int, inner k: Int) -> [Float] {
        var out = [Float](repeating: 0, count: m * n)
        guard m > 0, n > 0, k > 0 else { return out }
        var bt = [Float](repeating: 0, count: k * n)
        vDSP_mtrans(b, 1, &bt, 1, vDSP_Length(k), vDSP_Length(n))
        vDSP_mmul(a, 1, bt, 1, &out, 1, vDSP_Length(m), vDSP_Length(n), vDSP_Length(k))
        return out
    }

    /// A^T (A being k x m) times B (k x n); all row-major. Returns (m x n).
    static func multiplyTransposedLeft(_ a: [Float], rows k: Int, cols m: Int, _ b: [Float], cols n: Int) -> [Float] {
        var out = [Float](repeating: 0, count: m * n)
        guard m > 0, n > 0, k > 0 else { return out }
        var at = [Float](repeating: 0, count: m * k)
        vDSP_mtrans(a, 1, &at, 1, vDSP_Length(m), vDSP_Length(k))
        vDSP_mmul(at, 1, b, 1, &out, 1, vDSP_Length(m), vDSP_Length(n), vDSP_Length(k))
        return out
    }
}

enum LogSoftmax {
    /// Row-wise `z - max - log(sum(exp(z - max)))` in float32.
    static func inPlace(_ z: inout [Float], rows: Int, cols: Int) {
        var e = [Float](repeating: 0, count: cols)
        var n = Int32(cols)
        z.withUnsafeMutableBufferPointer { zp in
            for t in 0..<rows {
                let row = zp.baseAddress! + t * cols
                var m: Float = 0
                vDSP_maxv(row, 1, &m, vDSP_Length(cols))
                var neg = -m
                vDSP_vsadd(row, 1, &neg, row, 1, vDSP_Length(cols))
                vvexpf(&e, row, &n)
                var s: Float = 0
                vDSP_sve(e, 1, &s, vDSP_Length(cols))
                var shift = -logf(s)
                vDSP_vsadd(row, 1, &shift, row, 1, vDSP_Length(cols))
            }
        }
    }
}

extension Data {
    func u32(at offset: Int) -> UInt32 {
        withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self)) }
    }

    func f32(at offset: Int) -> Float {
        Float(bitPattern: u32(at: offset))
    }

    func f32Array(at offset: Int, count: Int) -> [Float] {
        [Float](unsafeUninitializedCapacity: count) { buf, n in
            withUnsafeBytes { raw in
                _ = memcpy(buf.baseAddress!, raw.baseAddress! + offset, count * 4)
            }
            n = count
        }
    }

    func u32Array(at offset: Int, count: Int) -> [UInt32] {
        [UInt32](unsafeUninitializedCapacity: count) { buf, n in
            withUnsafeBytes { raw in _ = memcpy(buf.baseAddress!, raw.baseAddress! + offset, count * 4) }
            n = count
        }
    }

    func u16Array(at offset: Int, count: Int) -> [UInt16] {
        [UInt16](unsafeUninitializedCapacity: count) { buf, n in
            withUnsafeBytes { raw in _ = memcpy(buf.baseAddress!, raw.baseAddress! + offset, count * 2) }
            n = count
        }
    }

    func u8Array(at offset: Int, count: Int) -> [UInt8] {
        [UInt8](self[(startIndex + offset)..<(startIndex + offset + count)])
    }

    mutating func appendLE(_ v: UInt32) {
        var le = v.littleEndian
        Swift.withUnsafeBytes(of: &le) { append(contentsOf: $0) }
    }
}

import Accelerate
import Foundation

/// Image primitives shared by preprocessing and segmentation (reference.py, top section).
enum ImageOps {
    /// Round half away from zero (for x >= -0.5), saturate to uint8.
    @inline(__always) static func roundU8(_ x: Double) -> UInt8 {
        let v = (x + 0.5).rounded(.down)
        return v <= 0 ? 0 : (v >= 255 ? 255 : UInt8(v))
    }

    /// Python's `round` (half to even).
    @inline(__always) static func roundHalfEven(_ x: Double) -> Int {
        Int(x.rounded(.toNearestOrEven))
    }

    /// OpenCV getGaussianKernel, including its fixed small-kernel tables for sigma <= 0.
    static func gaussianKernel(size: Int, sigma: Double) -> [Double] {
        var sigma = sigma
        if sigma <= 0 {
            switch size {
            case 1: return [1]
            case 3: return [0.25, 0.5, 0.25]
            case 5: return [0.0625, 0.25, 0.375, 0.25, 0.0625]
            case 7: return [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125]
            default: sigma = 0.3 * ((Double(size) - 1) * 0.5 - 1) + 0.8
            }
        }
        let c = Double(size - 1) / 2
        let k = (0..<size).map { i -> Double in
            let x = Double(i) - c
            return exp(-(x * x) / (2 * sigma * sigma))
        }
        let s = k.reduce(0, +)
        return k.map { $0 / s }
    }

    /// OpenCV's kernel size for GaussianBlur(ksize=(0,0)) on 8-bit input.
    static func gaussianSize(sigma: Double) -> Int {
        Int((sigma * 6 + 1 + 0.5).rounded(.down)) | 1
    }

    enum Border { case reflect101, replicate }

    /// Source indices for a signal of length n padded by r on each side.
    static func padIndex(_ n: Int, _ r: Int, _ border: Border) -> [Int] {
        (-r..<(n + r)).map { i in
            if n == 1 { return 0 }
            switch border {
            case .replicate: return min(max(i, 0), n - 1)
            case .reflect101:
                let period = 2 * (n - 1)
                var m = i % period
                if m < 0 { m += period }
                return m >= n ? period - m : m
            }
        }
    }

    /// Separable correlation in double, rows then columns (`convolve_sep`).
    static func convolveSeparable(_ src: [Double], width w: Int, height h: Int,
                                  kx: [Double], ky: [Double], border: Border = .reflect101) -> [Double] {
        var a = src
        let rx = kx.count / 2, ry = ky.count / 2
        if rx > 0 {
            a = correlateRows(a, width: w, height: h, kernel: kx, border: border)
        }
        if ry > 0 {
            var t = [Double](repeating: 0, count: w * h)
            vDSP_mtransD(a, 1, &t, 1, vDSP_Length(w), vDSP_Length(h))
            t = correlateRows(t, width: h, height: w, kernel: ky, border: border)
            vDSP_mtransD(t, 1, &a, 1, vDSP_Length(h), vDSP_Length(w))
        }
        return a
    }

    /// Each row correlated with `kernel` (odd length), padded per `border`.
    private static func correlateRows(_ a: [Double], width w: Int, height h: Int,
                                      kernel: [Double], border: Border) -> [Double] {
        let r = kernel.count / 2
        let idx = padIndex(w, r, border)
        let n = w + 2 * r
        var out = [Double](repeating: 0, count: w * h)
        a.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                let s = src.baseAddress!, d = dst.baseAddress!
                kernel.withUnsafeBufferPointer { k in
                    // Rows are independent; split them across cores.
                    let chunks = min(h, 8)
                    DispatchQueue.concurrentPerform(iterations: chunks) { c in
                        var padded = [Double](repeating: 0, count: n)
                        let y0 = c * h / chunks, y1 = (c + 1) * h / chunks
                        for y in y0..<y1 {
                            let row = s + y * w
                            padded.withUnsafeMutableBufferPointer { pd in
                                (pd.baseAddress! + r).update(from: row, count: w)
                                for i in 0..<r {
                                    pd[i] = row[idx[i]]
                                    pd[n - 1 - i] = row[idx[n - 1 - i]]
                                }
                            }
                            vDSP_convD(padded, 1, k.baseAddress!, 1, d + y * w, 1, vDSP_Length(w), vDSP_Length(kernel.count))
                        }
                    }
                }
            }
        }
        return out
    }

    /// Gaussian blur of an 8-bit image with OpenCV's kernel size, exact (reflect101), as uint8.
    static func blur(_ g: GrayImage, sigma: Double) -> [UInt8] {
        let k = gaussianKernel(size: gaussianSize(sigma: sigma), sigma: sigma)
        return roundU8(convolveSeparable(g.doubles(), width: g.width, height: g.height, kx: k, ky: k))
    }

    /// `roundU8` over an array, vectorized.
    static func roundU8(_ x: [Double]) -> [UInt8] {
        var t = x
        var half = 0.5, lo = 0.0, hi = 255.0, n = Int32(x.count)
        vDSP_vsaddD(t, 1, &half, &t, 1, vDSP_Length(x.count))
        vvfloor(&t, t, &n)
        vDSP_vclipD(t, 1, &lo, &hi, &t, 1, vDSP_Length(x.count))
        var out = [UInt8](repeating: 0, count: x.count)
        vDSP_vfixu8D(t, 1, &out, 1, vDSP_Length(x.count))
        return out
    }

    // MARK: - Resizing

    /// Sparse resampling weights: for each destination index, (first source, weights).
    struct Weights {
        var start: [Int]
        var values: [[Double]]
    }

    /// Box filter: dst i covers src [i*scale, (i+1)*scale), weights normalized.
    static func areaWeights(src n: Int, dst m: Int, scale: Double) -> Weights {
        var start = [Int](repeating: 0, count: m), values = [[Double]](repeating: [], count: m)
        for i in 0..<m {
            let a = Double(i) * scale, b = min(Double(i + 1) * scale, Double(n))
            let j0 = Int(a.rounded(.down)), j1 = min(Int(b.rounded(.up)), n)
            var w: [Double] = []
            if j0 < j1 {
                for j in j0..<j1 { w.append(min(b, Double(j + 1)) - max(a, Double(j))) }
            }
            let s = w.reduce(0, +)
            start[i] = j0
            values[i] = s > 0 ? w.map { $0 / s } : w
        }
        return Weights(start: start, values: values)
    }

    /// Bilinear with half-pixel centres, edges clamped.
    static func linearWeights(src n: Int, dst m: Int, scale: Double) -> Weights {
        var start = [Int](repeating: 0, count: m), values = [[Double]](repeating: [], count: m)
        for i in 0..<m {
            var f = (Double(i) + 0.5) * scale - 0.5
            var j = Int(f.rounded(.down))
            f -= Double(j)
            if j < 0 { j = 0; f = 0 }
            if j >= n - 1 { j = n - 1; f = 0 }
            start[i] = j
            values[i] = f != 0 ? [1 - f, f] : [1 - f]
        }
        return Weights(start: start, values: values)
    }

    /// PIL BICUBIC (a = -0.5, support widened when shrinking).
    static func bicubicWeights(src n: Int, dst m: Int) -> Weights {
        let scale = Double(n) / Double(m)
        let fscale = max(scale, 1)
        let support = 2 * fscale
        func cubic(_ x: Double) -> Double {
            let x = abs(x)
            if x < 1 { return ((1.5 * x - 2.5) * x) * x + 1 }
            if x < 2 { return ((-0.5 * x + 2.5) * x - 4) * x + 2 }
            return 0
        }
        var start = [Int](repeating: 0, count: m), values = [[Double]](repeating: [], count: m)
        for i in 0..<m {
            let centre = (Double(i) + 0.5) * scale
            let lo = max(Int(centre - support + 0.5), 0)
            let hi = min(Int(centre + support + 0.5), n)
            var w: [Double] = []
            if lo < hi {
                for j in lo..<hi { w.append(cubic((Double(j) - centre + 0.5) / fscale)) }
            }
            let s = w.reduce(0, +)
            start[i] = lo
            values[i] = s != 0 ? w.map { $0 / s } : w
        }
        return Weights(start: start, values: values)
    }

    /// `round_u8(wy @ img @ wx^T)` in double: columns resampled first, then rows.
    static func resample(_ g: GrayImage, wx: Weights, wy: Weights) -> GrayImage {
        let w = g.width, nw = wx.start.count, nh = wy.start.count
        let src = g.doubles()
        var tmp = [Double](repeating: 0, count: nh * w)
        src.withUnsafeBufferPointer { s in
            tmp.withUnsafeMutableBufferPointer { t in
                for i in 0..<nh {
                    let row = t.baseAddress! + i * w
                    for (k, wt) in wy.values[i].enumerated() {
                        var wt = wt
                        vDSP_vsmaD(s.baseAddress! + (wy.start[i] + k) * w, 1, &wt, row, 1, row, 1, vDSP_Length(w))
                    }
                }
            }
        }
        // Flatten the column weights once so the inner loop is pointer arithmetic.
        let taps = wx.values.map(\.count)
        let offsets = taps.reduce(into: [0]) { $0.append($0.last! + $1) }
        let flatW = wx.values.flatMap { $0 }
        var acc = [Double](repeating: 0, count: nh * nw)
        tmp.withUnsafeBufferPointer { t in
            acc.withUnsafeMutableBufferPointer { o in
                flatW.withUnsafeBufferPointer { fw in
                    wx.start.withUnsafeBufferPointer { st in
                        offsets.withUnsafeBufferPointer { off in
                            for i in 0..<nh {
                                let row = t.baseAddress! + i * w
                                for x in 0..<nw {
                                    var a = 0.0
                                    let s = st[x], o0 = off[x], o1 = off[x + 1]
                                    for k in o0..<o1 { a += fw[k] * row[s + k - o0] }
                                    o[i * nw + x] = a
                                }
                            }
                        }
                    }
                }
            }
        }
        return GrayImage(width: nw, height: nh, pixels: roundU8(acc))
    }

    static func identityWeights(_ n: Int) -> Weights {
        Weights(start: Array(0..<n), values: Array(repeating: [1], count: n))
    }

    /// Area (shrinking) resize; `sx`, `sy` override the source-per-destination scale.
    static func resizeArea(_ g: GrayImage, width nw: Int, height nh: Int, sx: Double? = nil, sy: Double? = nil) -> GrayImage {
        resample(g, wx: areaWeights(src: g.width, dst: nw, scale: sx ?? Double(g.width) / Double(nw)),
                 wy: areaWeights(src: g.height, dst: nh, scale: sy ?? Double(g.height) / Double(nh)))
    }

    static func resizeLinear(_ g: GrayImage, width nw: Int, height nh: Int) -> GrayImage {
        resample(g, wx: linearWeights(src: g.width, dst: nw, scale: Double(g.width) / Double(nw)),
                 wy: linearWeights(src: g.height, dst: nh, scale: Double(g.height) / Double(nh)))
    }

    /// Horizontal-only PIL BICUBIC resize.
    static func bicubicResizeX(_ g: GrayImage, width nw: Int) -> GrayImage {
        resample(g, wx: bicubicWeights(src: g.width, dst: nw), wy: identityWeights(g.height))
    }

    /// numpy's default (linear) percentile of an ascending array.
    static func percentile(sorted s: [Double], _ q: Double) -> Double {
        let pos = q / 100 * Double(s.count - 1)
        let lo = Int(pos.rounded(.down))
        let hi = min(lo + 1, s.count - 1)
        return s[lo] + (s[hi] - s[lo]) * (pos - Double(lo))
    }
}

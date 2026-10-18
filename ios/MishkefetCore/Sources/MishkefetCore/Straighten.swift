import Foundation

/// Flattens a tilted or curved line crop (spec/straighten.md, reference `straighten_line`).
public enum Straighten {
    static let minDeviation = 0.15
    static let band = 2.5

    /// The straightened crop, or the input unchanged (`changed == false`) when its ink
    /// centreline is level enough or there is too little ink to tell.
    public static func line(_ gray: GrayImage) -> (image: GrayImage, changed: Bool) {
        let h = gray.height, w = gray.width
        guard w > 0, h > 0, let (f, sigma) = centerline(gray) else { return (gray, false) }
        let mean = f.reduce(0, +) / Double(w)
        let dev = f.map { $0 - mean }
        let maxDev = dev.map(abs).max() ?? 0
        if maxDev < minDeviation * Double(h) { return (gray, false) }
        let pad = Int(maxDev.rounded(.up))
        let H = h + 2 * pad
        var out = [Double](repeating: 255, count: H * w)
        gray.pixels.withUnsafeBufferPointer { g in
            out.withUnsafeMutableBufferPointer { o in
                for x in 0..<w {
                    for y in 0..<H {
                        let src = Double(y - pad) + dev[x]
                        guard src >= 0, src <= Double(h - 1) else { continue }
                        let i0 = Int(src.rounded(.down)), i1 = min(i0 + 1, h - 1)
                        let a = src - Double(i0)
                        o[y * w + x] = Double(g[i0 * w + x]) * (1 - a) + Double(g[i1 * w + x]) * a
                    }
                }
            }
        }
        let c = mean + Double(pad)
        let b = max(band * sigma, 0.22 * Double(h))
        let top = Int(max(0, c - b)), bot = Int(min(Double(H), c + b + 1))
        guard bot > top else { return (gray, false) }
        let rows = Array(out[(top * w)..<(bot * w)])
        return (GrayImage(width: w, height: bot - top, pixels: ImageOps.roundU8(rows)), true)
    }

    /// (f, sigma): a weighted quadratic through the ink's vertical centre per column,
    /// refitted up to twice without outlying columns; sigma is the ink's spread around it.
    static func centerline(_ gray: GrayImage) -> ([Double], Double)? {
        let h = gray.height, w = gray.width
        let flat = Preprocess.contrastNormalize(gray)
        let t = Double(Orientation.otsu(flat.pixels))
        let denom = max(t, 1)
        var wt = [Double](repeating: 0, count: w * h)
        var col = [Double](repeating: 0, count: w), moment = col
        for y in 0..<h {
            for x in 0..<w {
                let v = min(max((t - Double(flat.pixels[y * w + x])) / denom, 0), 1)
                wt[y * w + x] = v
                col[x] += v
                moment[x] += Double(y) * v
            }
        }
        var xs: [Double] = [], m: [Double] = [], wk: [Double] = []
        for x in 0..<w where col[x] > 0.5 {
            xs.append(Double(x))
            m.append(moment[x] / col[x])
            wk.append(col[x])
        }
        if Double(xs.count) < max(8, Double(w) * 0.15) { return nil }
        var fit: Quadratic?
        for _ in 0..<2 {
            guard let q = Quadratic(xs: xs, ys: m, weights: wk) else { return nil }
            fit = q
            let res = zip(xs, m).map { $1 - q($0) }
            let wsum = wk.reduce(0, +)
            let rms = (zip(res, wk).reduce(0) { $0 + $1.0 * $1.0 * $1.1 } / wsum).squareRoot()
            let limit = max(0.18 * Double(h), 2.5 * rms + 1)
            let good = res.map { abs($0) < limit }
            if !good.contains(false) { break }
            xs = zip(xs, good).filter(\.1).map(\.0)
            m = zip(m, good).filter(\.1).map(\.0)
            wk = zip(wk, good).filter(\.1).map(\.0)
            if xs.count < 8 { return nil }
        }
        guard let q = fit else { return nil }
        let f = (0..<w).map { q(Double($0)) }
        var num = 0.0, total = 0.0
        for y in 0..<h {
            for x in 0..<w {
                let d = Double(y) - f[x]
                num += wt[y * w + x] * d * d
                total += wt[y * w + x]
            }
        }
        return (f, (num / max(total, 1e-6)).squareRoot())
    }
}

/// Weighted least-squares quadratic (numpy `polyfit(x, y, 2, w=sqrt(weights))`), solved in
/// a centred and scaled variable so the 3x3 normal equations stay well conditioned.
struct Quadratic {
    private let c: (Double, Double, Double)
    private let centre: Double
    private let scale: Double

    init?(xs: [Double], ys: [Double], weights: [Double]) {
        guard let lo = xs.min(), let hi = xs.max(), xs.count >= 3 else { return nil }
        centre = (lo + hi) / 2
        scale = max((hi - lo) / 2, 1)
        var a = [[Double]](repeating: [0, 0, 0], count: 3)
        var b = [0.0, 0.0, 0.0]
        for (i, x) in xs.enumerated() {
            let t = (x - centre) / scale
            let phi = [1, t, t * t]
            for r in 0..<3 {
                b[r] += weights[i] * phi[r] * ys[i]
                for k in 0..<3 { a[r][k] += weights[i] * phi[r] * phi[k] }
            }
        }
        guard let s = Self.solve(a, b) else { return nil }
        c = (s[0], s[1], s[2])
    }

    func callAsFunction(_ x: Double) -> Double {
        let t = (x - centre) / scale
        return c.0 + t * (c.1 + t * c.2)
    }

    /// Gaussian elimination with partial pivoting.
    private static func solve(_ a: [[Double]], _ b: [Double]) -> [Double]? {
        var m = a, v = b
        for col in 0..<3 {
            var p = col
            for r in (col + 1)..<3 where abs(m[r][col]) > abs(m[p][col]) { p = r }
            guard abs(m[p][col]) > 1e-12 else { return nil }
            m.swapAt(col, p)
            v.swapAt(col, p)
            for r in (col + 1)..<3 {
                let f = m[r][col] / m[col][col]
                for k in col..<3 { m[r][k] -= f * m[col][k] }
                v[r] -= f * v[col]
            }
        }
        var x = [0.0, 0.0, 0.0]
        for r in stride(from: 2, through: 0, by: -1) {
            var s = v[r]
            for k in (r + 1)..<3 { s -= m[r][k] * x[k] }
            x[r] = s / m[r][r]
        }
        return x
    }
}

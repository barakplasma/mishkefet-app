import Accelerate
import Foundation

/// Getting a page upright (spec/orientation.md): the paper crop for screenshots,
/// quarter turns, and the stripe test that picks the turn candidates.
public enum Orientation {
    /// Otsu's threshold over uint8 values: the first argmax of the between-class variance,
    /// computed in the reference's order (sequential cumulative sums, 0/0 as 0, x/0 as huge).
    static func otsu(_ values: [UInt8]) -> Int {
        var hist = [Double](repeating: 0, count: 256)
        for v in values { hist[Int(v)] += 1 }
        let total = max(Double(values.count), 1)
        var omega = [Double](repeating: 0, count: 256), mu = omega
        var o = 0.0, m = 0.0
        for i in 0..<256 {
            let p = hist[i] / total
            o += p
            m += p * Double(i)
            omega[i] = o
            mu[i] = m
        }
        let muT = mu[255]
        var best = 0, bestValue = -Double.infinity
        for i in 0..<256 {
            let d = muT * omega[i] - mu[i]
            var sb = (d * d) / (omega[i] * (1 - omega[i]))
            if sb.isNaN { sb = 0 } else if sb == .infinity { sb = .greatestFiniteMagnitude }
            if sb > bestValue {
                best = i
                bestValue = sb
            }
        }
        return best
    }

    /// The paper's rectangle (x, y, w, h) when it sits inside near-black surroundings
    /// (a screenshot's app bars, a photo on a black screen), else nil.
    public static func findPaper(_ gray: GrayImage, work: Int = 256) -> LineBox? {
        let W = gray.width, H = gray.height
        guard W > 0, H > 0 else { return nil }
        let s = Double(work) / Double(max(H, W))
        let w = max(1, ImageOps.roundHalfEven(Double(W) * s)), h = max(1, ImageOps.roundHalfEven(Double(H) * s))
        let small = ImageOps.resizeArea(gray, width: w, height: h, sx: Double(W) / Double(w), sy: Double(H) / Double(h))
        let t = otsu(small.pixels)
        var bright = small.pixels.map { Int($0) > t }
        let k = max(3, ImageOps.roundHalfEven(Double(min(w, h)) * 0.03)) | 1
        // close, then open; outside the image counts as dark in every pass
        bright = BoolMorph.erode(BoolMorph.dilate(bright, w, h, k), w, h, k)
        bright = BoolMorph.dilate(BoolMorph.erode(bright, w, h, 3), w, h, 3)
        let mask = GrayImage(width: w, height: h, pixels: bright.map { $0 ? 255 : 0 })
        let comps = Segmenter.components(mask, eightConnected: false)
        guard var best = comps.first else { return nil }
        for c in comps.dropFirst() where c.area > best.area { best = c }
        let n = Double(w * h)
        guard Double(best.area) >= 0.2 * n else { return nil }
        let b = best.box
        let x0 = b.x, y0 = b.y, x1 = b.x + b.w, y1 = b.y + b.h
        guard Double(b.w * b.h) <= 0.92 * n else { return nil }
        var outsideSum = 0, outsideCount = 0
        for y in 0..<h {
            for x in 0..<w where !(y >= y0 && y < y1 && x >= x0 && x < x1) {
                outsideSum += Int(small.pixels[y * w + x])
                outsideCount += 1
            }
        }
        guard outsideCount > 0, Double(outsideSum) / Double(outsideCount) <= 25,
              Double(outsideCount) / n >= 0.25 else { return nil }
        let m = 0.015
        let X0 = max(0, Int((Double(x0) / Double(w) + m) * Double(W)))
        let Y0 = max(0, Int((Double(y0) / Double(h) + m) * Double(H)))
        let X1 = min(W, Int((Double(x1) / Double(w) - m) * Double(W)))
        let Y1 = min(H, Int((Double(y1) / Double(h) - m) * Double(H)))
        guard X1 - X0 >= 16, Y1 - Y0 >= 16 else { return nil }
        return LineBox(x: X0, y: Y0, w: X1 - X0, h: Y1 - Y0)
    }

    /// Quarter-turn candidates from the stripe test: the sharper of upright and
    /// turned once, unless they are within 3x of each other (then both).
    static func candidates(_ binary: GrayImage) -> [Int] {
        let s0 = Segmenter.estimateSkewWide(binary).score
        let s1 = Segmenter.estimateSkewWide(binary.rot90(1)).score
        if max(s0, s1) >= 3 * min(s0, s1) { return [s0 >= s1 ? 0 : 1] }
        return [0, 1]
    }
}

/// Square binary erosion and dilation with everything outside the image dark (scipy border_value=0).
enum BoolMorph {
    private static func pass(_ a: [Bool], _ w: Int, _ h: Int, _ k: Int, horizontal: Bool, erode: Bool) -> [Bool] {
        let r = k / 2
        var out = [Bool](repeating: false, count: a.count)
        let (lines, len) = horizontal ? (h, w) : (w, h)
        var prefix = [Int](repeating: 0, count: len + 1)
        for l in 0..<lines {
            func at(_ i: Int) -> Int { horizontal ? l * w + i : i * w + l }
            for i in 0..<len { prefix[i + 1] = prefix[i] + (a[at(i)] ? 1 : 0) }
            for i in 0..<len {
                let lo = i - r, hi = i + r
                let count = prefix[min(hi, len - 1) + 1] - prefix[max(lo, 0)]
                out[at(i)] = erode ? (lo >= 0 && hi < len && count == k) : count > 0
            }
        }
        return out
    }

    static func erode(_ a: [Bool], _ w: Int, _ h: Int, _ k: Int) -> [Bool] {
        pass(pass(a, w, h, k, horizontal: true, erode: true), w, h, k, horizontal: false, erode: true)
    }

    static func dilate(_ a: [Bool], _ w: Int, _ h: Int, _ k: Int) -> [Bool] {
        pass(pass(a, w, h, k, horizontal: true, erode: false), w, h, k, horizontal: false, erode: false)
    }
}

extension GrayImage {
    /// `k` quarter turns counter-clockwise as displayed (numpy rot90):
    /// k = 1 gives `out[y][x] = in[x][W-1-y]`, size H x W.
    public func rot90(_ k: Int) -> GrayImage {
        let k = ((k % 4) + 4) % 4
        if k == 0 { return self }
        if k == 2 { return GrayImage(width: width, height: height, pixels: pixels.reversed()) }
        // transpose, then reverse the row order (k = 1) or each row (k = 3)
        let t = transposed()  // t is width rows of height pixels: t[x][y] = in[y][x]
        let tw = t.width, th = t.height
        var out = [UInt8](repeating: 0, count: pixels.count)
        t.pixels.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                for y in 0..<th {
                    if k == 1 {
                        (dst.baseAddress! + y * tw).update(from: src.baseAddress! + (th - 1 - y) * tw, count: tw)
                    } else {
                        let s = src.baseAddress! + y * tw, d = dst.baseAddress! + y * tw
                        for x in 0..<tw { d[x] = s[tw - 1 - x] }
                    }
                }
            }
        }
        return GrayImage(width: tw, height: th, pixels: out)
    }
}

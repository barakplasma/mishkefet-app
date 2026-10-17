import Accelerate
import Foundation

/// OpenCV-convention binary morphology with kernels whose rows are single runs
/// (spec/segmenter.md). Pixels outside the image never count.
enum Morphology {
    /// A 0/1 kernel of `rows x cols`, anchor (cols / 2, rows / 2), one run [c0, c1] per row.
    struct Kernel {
        var rows: Int
        var cols: Int
        var runs: [(row: Int, c0: Int, c1: Int)]

        static func rect(width: Int, height: Int) -> Kernel {
            Kernel(rows: height, cols: width, runs: (0..<height).map { ($0, 0, width - 1) })
        }

        /// Cells of an OpenCV LineIterator line of `length` at `angle` degrees.
        static func line(length: Int, angle: Double) -> Kernel {
            let drop = ImageOps.roundHalfEven(tan(angle * .pi / 180) * Double(length - 1))
            let rows = abs(drop) + 1
            let y0 = drop >= 0 ? 0 : rows - 1
            var lo = [Int](repeating: Int.max, count: rows), hi = [Int](repeating: -1, count: rows)
            for (x, y) in cvLine(0, y0, length - 1, y0 + drop) {
                lo[y] = min(lo[y], x)
                hi[y] = max(hi[y], x)
            }
            let runs = (0..<rows).filter { hi[$0] >= 0 }.map { (row: $0, c0: lo[$0], c1: hi[$0]) }
            return Kernel(rows: rows, cols: length, runs: runs)
        }
    }

    /// OpenCV LineIterator (connectivity 8) pixel sequence.
    static func cvLine(_ x0: Int, _ y0: Int, _ x1: Int, _ y1: Int) -> [(Int, Int)] {
        var dx = x1 - x0, dy = y1 - y0
        let s = dx < 0 ? -1 : 0
        dx = (dx ^ s) - s
        let sy = dy < 0 ? -1 : 0
        dy = (dy ^ sy) - sy
        let stepx = s != 0 ? -1 : 1, stepy = sy != 0 ? -1 : 1
        let swap = dy > dx
        if swap { (dx, dy) = (dy, dx) }
        var err = dx - (dy + dy)
        let plusDelta = dx + dx, minusDelta = -(dy + dy)
        var x = x0, y = y0
        var pts: [(Int, Int)] = []
        for _ in 0..<(dx + 1) {
            pts.append((x, y))
            let mask = err < 0 ? -1 : 0
            err += minusDelta + (plusDelta & mask)
            if swap {
                y += stepy
                if mask != 0 { x += stepx }
            } else {
                x += stepx
                if mask != 0 { y += stepy }
            }
        }
        return pts
    }

    @inline(__always) private static func op(_ erode: Bool, _ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>,
                                             _ c: UnsafeMutablePointer<Float>, _ n: Int) {
        if erode {
            vDSP_vmin(a, 1, b, 1, c, 1, vDSP_Length(n))
        } else {
            vDSP_vmax(a, 1, b, 1, c, 1, vDSP_Length(n))
        }
    }

    /// `out[i] = op(p[i ..< i + length])` for i in 0..<n, as a sparse table: windows of
    /// 2^j by doubling, then two overlapping power-of-two windows (min/max are idempotent).
    private static func runExtreme(_ p: [Float], count n: Int, length: Int, erode: Bool) -> [Float] {
        if length == 1 { return p }
        var cur = p, next = p
        var k = 1
        while 2 * k <= length {
            let valid = p.count - 2 * k + 1
            cur.withUnsafeBufferPointer { c in
                next.withUnsafeMutableBufferPointer { nx in
                    op(erode, c.baseAddress!, c.baseAddress! + k, nx.baseAddress!, valid)
                }
            }
            swap(&cur, &next)
            k *= 2
        }
        if k == length { return cur }
        var out = [Float](repeating: 0, count: n)
        cur.withUnsafeBufferPointer { c in
            out.withUnsafeMutableBufferPointer { o in
                op(erode, c.baseAddress!, c.baseAddress! + (length - k), o.baseAddress!, n)
            }
        }
        return out
    }

    /// Erode (min) or dilate (max) with `kernel`.
    static func apply(_ a: GrayImage, _ kernel: Kernel, erode: Bool) -> GrayImage {
        let w = a.width, h = a.height
        guard w > 0, h > 0 else { return a }
        let neutral: Float = erode ? 255 : 0
        let ax = kernel.cols / 2, ay = kernel.rows / 2
        // Rows padded by `pad` neutral columns on each side, plus a neutral row at the end,
        // so every window we read stays inside the buffer and sees only its row's pixels.
        let pad = kernel.cols, ext = w + 2 * pad
        var plane = [Float](repeating: neutral, count: (h + 1) * ext)
        a.pixels.withUnsafeBufferPointer { src in
            plane.withUnsafeMutableBufferPointer { pl in
                for y in 0..<h {
                    vDSP_vfltu8(src.baseAddress! + y * w, 1, pl.baseAddress! + y * ext + pad, 1, vDSP_Length(w))
                }
            }
        }
        var cache: [Int: [Float]] = [:]
        for run in Set(kernel.runs.map { $0.c1 - $0.c0 + 1 }) {
            cache[run] = runExtreme(plane, count: h * ext, length: run, erode: erode)
        }
        var out = [Float](repeating: neutral, count: h * ext)
        out.withUnsafeMutableBufferPointer { o in
            for r in kernel.runs {
                let dx = r.c0 - ax, dy = r.row - ay
                let y0 = max(0, -dy), y1 = min(h, h - dy)
                guard y1 > y0 else { continue }
                let count = (y1 - y0 - 1) * ext + w
                cache[r.c1 - r.c0 + 1]!.withUnsafeBufferPointer { pl in
                    let dst = o.baseAddress! + y0 * ext + pad
                    op(erode, dst, pl.baseAddress! + (y0 + dy) * ext + pad + dx, dst, count)
                }
            }
        }
        var pixels = [UInt8](repeating: 0, count: w * h)
        out.withUnsafeBufferPointer { o in
            pixels.withUnsafeMutableBufferPointer { px in
                for y in 0..<h {
                    vDSP_vfixu8(o.baseAddress! + y * ext + pad, 1, px.baseAddress! + y * w, 1, vDSP_Length(w))
                }
            }
        }
        return GrayImage(width: w, height: h, pixels: pixels)
    }

    /// The same with the kernel applied along columns (`morph_t`): `kernel` is given in
    /// its row-run form and runs vertically.
    static func applyVertical(_ a: GrayImage, _ kernel: Kernel, erode: Bool) -> GrayImage {
        apply(a.transposed(), kernel, erode: erode).transposed()
    }

    static func erode(_ a: GrayImage, _ k: Kernel) -> GrayImage { apply(a, k, erode: true) }
    static func dilate(_ a: GrayImage, _ k: Kernel) -> GrayImage { apply(a, k, erode: false) }

    static func open(_ a: GrayImage, _ k: Kernel, vertical: Bool = false) -> GrayImage {
        vertical ? applyVertical(applyVertical(a, k, erode: true), k, erode: false)
            : dilate(erode(a, k), k)
    }

    static func maximum(_ a: GrayImage, _ b: GrayImage) -> GrayImage {
        combine(a, b) { x, y, n in vDSP_vmax(x, 1, y, 1, x, 1, n) }
    }

    /// Saturating a - b.
    static func subtract(_ a: GrayImage, _ b: GrayImage) -> GrayImage {
        combine(a, b) { x, y, n in
            vDSP_vmin(x, 1, y, 1, y, 1, n)
            vDSP_vsub(y, 1, x, 1, x, 1, n)  // x - min(x, y)
        }
    }

    /// `body(a, b, n)` on float copies, the result read back from `a`.
    private static func combine(_ a: GrayImage, _ b: GrayImage,
                                _ body: (UnsafeMutablePointer<Float>, UnsafeMutablePointer<Float>, vDSP_Length) -> Void) -> GrayImage {
        var fa = a.floats(), fb = b.floats()
        fa.withUnsafeMutableBufferPointer { x in
            fb.withUnsafeMutableBufferPointer { y in body(x.baseAddress!, y.baseAddress!, vDSP_Length(x.count)) }
        }
        return GrayImage(width: a.width, height: a.height, floats: fa)
    }
}

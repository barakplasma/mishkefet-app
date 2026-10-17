import Accelerate
import Foundation

/// Page to line boxes (spec/segmenter.md, reference `segment_page`). Every
/// threshold scales with the measured line pitch.
public enum Segmenter {
    public struct Result: Sendable {
        public let angle: Double
        public let pitch: Double
        /// Scale of the binarized page relative to the input (1 unless the page is over 2000 px).
        public let scale: Double
        public let boxes: [LineBox]
        /// The deskewed gray page the boxes refer to.
        public let page: GrayImage
        public let crops: [GrayImage]
    }

    static let maxSide = 2000
    static let ruleAngles: [Double] = [-2, -1, -0.4, 0, 0.4, 1, 2]

    typealias K = Morphology.Kernel

    public static func segment(_ gray: GrayImage) -> Result {
        var (binary, scale) = binarize(gray)
        let angle = estimateSkew(binary)
        var work = gray
        if abs(angle) > 1e-3 {
            binary = rotateNearest(binary, angle: angle)
            work = rotateBicubic(gray, angle: angle)
        }
        var pitch = estimatePitch(binary)
        binary = removeRules(binary, pitch: pitch)
        pitch = estimatePitch(binary)
        var boxes = findLines(binary, pitch: pitch)
        boxes = splitTall(binary, boxes, pitch: pitch)
        boxes = mergeOverlapping(boxes, pitch: pitch)
        boxes = readingOrder(boxes, pitch: pitch)
        let inv = 1 / scale
        let full = boxes.map {
            LineBox(x: Int(Double($0.x) * inv), y: Int(Double($0.y) * inv),
                    w: Int(Double($0.w) * inv), h: Int(Double($0.h) * inv))
        }
        return Result(angle: angle, pitch: pitch, scale: scale, boxes: full, page: work,
                      crops: full.map { crop(work, $0) })
    }

    /// The box with `0.12 h` of vertical and `0.05 h` of horizontal padding, clamped.
    static func crop(_ img: GrayImage, _ b: LineBox) -> GrayImage {
        let py = Int(Double(b.h) * 0.12), px = Int(Double(b.h) * 0.05)
        let y0 = max(b.y - py, 0), y1 = min(b.y + b.h + py, img.height)
        let x0 = max(b.x - px, 0), x1 = min(b.x + b.w + px, img.width)
        return img.cropped(x: x0, y: y0, w: max(0, x1 - x0), h: max(0, y1 - y0))
    }

    // MARK: - Binarization

    /// Background-flattened adaptive threshold, ink = 255; also the scale applied.
    public static func binarize(_ input: GrayImage) -> (GrayImage, Double) {
        var gray = input
        let scale = min(1, Double(maxSide) / Double(max(gray.width, gray.height)))
        if scale < 1 {
            gray = ImageOps.resizeArea(gray, width: ImageOps.roundHalfEven(Double(gray.width) * scale),
                                       height: ImageOps.roundHalfEven(Double(gray.height) * scale),
                                       sx: 1 / scale, sy: 1 / scale)
        }
        let w = gray.width, h = gray.height
        let n = w * h
        // flat = trunc(clamp(gray / max(bg, 1) * 128, 0, 255)) in float32 (scalar: IEEE division)
        let bg = ImageOps.blur(gray, sigma: Double(max(h, w)) / 30)
        var flat = GrayImage(width: w, height: h)
        gray.pixels.withUnsafeBufferPointer { g in
            bg.withUnsafeBufferPointer { b in
                flat.pixels.withUnsafeMutableBufferPointer { f in
                    for i in 0..<n {
                        let v = Float(g[i]) / max(Float(b[i]), 1) * 128
                        f[i] = v <= 0 ? 0 : (v >= 255 ? 255 : UInt8(v))
                    }
                }
            }
        }
        let block = max(15, (min(h, w) / 40) | 1)
        let k = ImageOps.gaussianKernel(size: block, sigma: 0)
        let mean = ImageOps.roundU8(ImageOps.convolveSeparable(flat.doubles(), width: w, height: h, kx: k, ky: k,
                                                               border: .replicate))
        var binary = GrayImage(width: w, height: h)
        flat.pixels.withUnsafeBufferPointer { f in
            mean.withUnsafeBufferPointer { m in
                binary.pixels.withUnsafeMutableBufferPointer { b in
                    for i in 0..<n where Int(f[i]) - Int(m[i]) <= -12 { b[i] = 255 }
                }
            }
        }
        return (Morphology.open(binary, K.rect(width: 2, height: 2)), scale)
    }

    // MARK: - Skew

    /// Inverse of the rotation about the centre: destination (x, y) to source.
    private static func inverseRotation(width w: Int, height h: Int, angle: Double) -> [Double] {
        let a = angle * .pi / 180
        let c = cos(a), s = sin(a)
        let cx = Double(w) / 2, cy = Double(h) / 2
        let t0 = (1 - c) * cx - s * cy, t1 = s * cx + (1 - c) * cy
        let det = c * c + s * s
        let i00 = c / det, i01 = -s / det, i10 = s / det, i11 = c / det
        return [i00, i01, -(i00 * t0 + i01 * t1), i10, i11, -(i10 * t0 + i11 * t1)]
    }

    /// warpAffine(INTER_NEAREST) about the centre, 0 outside.
    static func rotateNearest(_ img: GrayImage, angle: Double) -> GrayImage {
        let w = img.width, h = img.height
        let m = inverseRotation(width: w, height: h, angle: angle)
        var out = [UInt8](repeating: 0, count: w * h)
        var ramp = [Double](repeating: 0, count: w)
        var zero = 0.0, one = 1.0
        vDSP_vrampD(&zero, &one, &ramp, 1, vDSP_Length(w))
        var tmp = [Double](repeating: 0, count: w)
        var sx = [Int32](repeating: 0, count: w), sy = sx
        // floor(((a * x + b * y) + c) + 0.5) for every x of a row, vectorized in the
        // reference's evaluation order.
        func coordinates(_ a: Double, _ b: Double, _ c: Double, y: Int, into dst: inout [Int32]) {
            var a = a, by = b * Double(y), c = c, half = 0.5, n = Int32(w)
            vDSP_vsmulD(ramp, 1, &a, &tmp, 1, vDSP_Length(w))
            vDSP_vsaddD(tmp, 1, &by, &tmp, 1, vDSP_Length(w))
            vDSP_vsaddD(tmp, 1, &c, &tmp, 1, vDSP_Length(w))
            vDSP_vsaddD(tmp, 1, &half, &tmp, 1, vDSP_Length(w))
            vvfloor(&tmp, tmp, &n)
            vDSP_vfix32D(tmp, 1, &dst, 1, vDSP_Length(w))
        }
        img.pixels.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                for y in 0..<h {
                    coordinates(m[0], m[1], m[2], y: y, into: &sx)
                    coordinates(m[3], m[4], m[5], y: y, into: &sy)
                    sx.withUnsafeBufferPointer { px in
                        sy.withUnsafeBufferPointer { py in
                            let row = dst.baseAddress! + y * w
                            for x in 0..<w {
                                let u = Int(px[x]), v = Int(py[x])
                                if u >= 0 && u < w && v >= 0 && v < h { row[x] = src[v * w + u] }
                            }
                        }
                    }
                }
            }
        }
        return GrayImage(width: w, height: h, pixels: out)
    }

    /// PIL `rotate(angle, BICUBIC, fillcolor=255)` on a gray image (the a = -1 cubic of
    /// PIL's affine transform, truncated to uint8).
    static func rotateBicubic(_ img: GrayImage, angle: Double) -> GrayImage {
        let w = img.width, h = img.height
        func r15(_ v: Double) -> Double { (v * 1e15).rounded() / 1e15 }
        let a = -(angle.truncatingRemainder(dividingBy: 360) + (angle < 0 ? 360 : 0)) * .pi / 180
        var m = [r15(cos(a)), r15(sin(a)), 0, r15(-sin(a)), r15(cos(a)), 0]
        let cx = Double(w) / 2, cy = Double(h) / 2
        m[2] = m[0] * -cx + m[1] * -cy + m[2] + cx
        m[5] = m[3] * -cx + m[4] * -cy + m[5] + cy
        var out = [UInt8](repeating: 255, count: w * h)
        img.pixels.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                let rows = min(h, 8)
                DispatchQueue.concurrentPerform(iterations: rows) { chunk in
                    for y in (chunk * h / rows)..<((chunk + 1) * h / rows) {
                        let fy = Double(y) + 0.5
                        for x in 0..<w {
                            let fx = Double(x) + 0.5
                            var xin = m[0] * fx + m[1] * fy + m[2]
                            var yin = m[3] * fx + m[4] * fy + m[5]
                            if xin < 0 || xin >= Double(w) || yin < 0 || yin >= Double(h) { continue }
                            xin -= 0.5
                            yin -= 0.5
                            var ix = Int(xin.rounded(.down)), iy = Int(yin.rounded(.down))
                            let dx = xin - Double(ix), dy = yin - Double(iy)
                            ix -= 1
                            iy -= 1
                            let x0 = min(max(ix, 0), w - 1), x1 = min(max(ix + 1, 0), w - 1)
                            let x2 = min(max(ix + 2, 0), w - 1), x3 = min(max(ix + 3, 0), w - 1)
                            // The first row clamps; rows outside the image repeat the previous value.
                            let s = src.baseAddress!
                            let v1 = cubicRow(s + min(max(iy, 0), h - 1) * w, x0, x1, x2, x3, dx)
                            let v2 = iy + 1 >= 0 && iy + 1 < h ? cubicRow(s + (iy + 1) * w, x0, x1, x2, x3, dx) : v1
                            let v3 = iy + 2 >= 0 && iy + 2 < h ? cubicRow(s + (iy + 2) * w, x0, x1, x2, x3, dx) : v2
                            let v4 = iy + 3 >= 0 && iy + 3 < h ? cubicRow(s + (iy + 3) * w, x0, x1, x2, x3, dx) : v3
                            let value = cubic(v1, v2, v3, v4, dy)
                            dst[y * w + x] = value <= 0 ? 0 : (value >= 255 ? 255 : UInt8(value))
                        }
                    }
                }
            }
        }
        return GrayImage(width: w, height: h, pixels: out)
    }

    @inline(__always) private static func cubic(_ v1: Double, _ v2: Double, _ v3: Double, _ v4: Double,
                                                _ d: Double) -> Double {
        let p1 = v2, p2 = -v1 + v3, p3 = 2 * (v1 - v2) + v3 - v4, p4 = -v1 + v2 - v3 + v4
        return p1 + d * (p2 + d * (p3 + d * p4))
    }

    @inline(__always) private static func cubicRow(_ p: UnsafePointer<UInt8>, _ x0: Int, _ x1: Int, _ x2: Int,
                                                   _ x3: Int, _ d: Double) -> Double {
        cubic(Double(p[x0]), Double(p[x1]), Double(p[x2]), Double(p[x3]), d)
    }

    /// Angle in [-6, 6] (0.5 steps) maximizing the variance of the row-profile derivative.
    public static func estimateSkew(_ binary: GrayImage) -> Double {
        let small = ImageOps.resizeArea(binary, width: ImageOps.roundHalfEven(Double(binary.width) * 0.5),
                                        height: ImageOps.roundHalfEven(Double(binary.height) * 0.5), sx: 2, sy: 2)
        var bestAngle = 0.0, best = -1.0
        for i in 0..<25 {
            let angle = -6 + Double(i) * 0.5
            let prof = rotateNearest(small, angle: angle).rowSums()
            guard prof.count > 1 else { continue }
            let diff = (1..<prof.count).map { prof[$0] - prof[$0 - 1] }
            let mean = diff.reduce(0, +) / Double(diff.count)
            let score = diff.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(diff.count)
            if score > best {
                bestAngle = angle
                best = score
            }
        }
        return bestAngle
    }

    /// First autocorrelation peak of the row profile.
    public static func estimatePitch(_ binary: GrayImage) -> Double {
        var prof = binary.rowSums()
        let n = prof.count
        guard n > 0 else { return 30 }
        let mean = prof.reduce(0, +) / Double(n)
        prof = prof.map { $0 - mean }
        if !prof.contains(where: { $0 != 0 }) { return 30 }
        // up to half the height, not hebocr's 0.15: a few big lines would otherwise shatter (spec/segmenter.md)
        let lo = max(8, Int(Double(n) * 0.004)), hi = min(n - 1, n / 2)
        if hi <= lo + 2 { return 30 }
        let win: [Double] = (lo..<hi).map { lag in
            var s = 0.0
            prof.withUnsafeBufferPointer { p in
                vDSP_dotprD(p.baseAddress!, 1, p.baseAddress! + lag, 1, &s, vDSP_Length(n - lag))
            }
            return s
        }
        var k: Int?
        for i in 1..<(win.count - 1) where win[i] > win[i - 1] && win[i] >= win[i + 1] && win[i] > 0 {
            k = i
            break
        }
        if k == nil {
            var best = 0
            for i in 1..<win.count where win[i] > win[best] { best = i }
            k = best
        }
        return Double(lo + k!)
    }

    // MARK: - Rules

    /// Removes long thin horizontal and vertical strokes (ruled lines, margins).
    static func removeRules(_ binary: GrayImage, pitch: Double) -> GrayImage {
        let length = max(40, Int(pitch * 2)), thickness = max(3, Int(pitch * 0.28))
        let bt = binary.transposed()
        var horiz = GrayImage(width: binary.width, height: binary.height)
        var vertT = GrayImage(width: bt.width, height: bt.height)
        for angle in ruleAngles {
            let k = K.line(length: length, angle: angle)
            horiz = Morphology.maximum(horiz, Morphology.open(binary, k))
            vertT = Morphology.maximum(vertT, Morphology.open(bt, k))
        }
        let vert = vertT.transposed()
        let thin = K.rect(width: thickness, height: 1)
        let dil = K.rect(width: 3, height: 3)
        var out = binary
        let hRules = Morphology.subtract(horiz, Morphology.open(horiz, thin, vertical: true))
        out = Morphology.subtract(out, Morphology.dilate(hRules, dil))
        let vRules = Morphology.subtract(vert, Morphology.open(vert, thin))
        out = Morphology.subtract(out, Morphology.dilate(vRules, dil))
        return out
    }

    // MARK: - Lines

    struct Box: Equatable {
        var x, y, w, h: Int
    }

    /// Smear horizontally, then keep 8-connected components of line size.
    static func findLines(_ binary: GrayImage, pitch: Double) -> [Box] {
        let sx = max(8, Int(pitch * 0.55)), sy = max(1, Int(pitch * 0.05))
        var sm = Morphology.dilate(binary, K.rect(width: sx, height: 1))
        if sy > 1 { sm = Morphology.applyVertical(sm, K.rect(width: sy, height: 1), erode: false) }
        sm = Morphology.erode(Morphology.dilate(sm, K.rect(width: sx, height: 1)), K.rect(width: sx, height: 1))
        return components(sm).compactMap { c in
            let w = Double(c.box.w), h = Double(c.box.h)
            if w < pitch * 0.8 || h < pitch * 0.25 || h > pitch * 12 || Double(c.area) < pitch * pitch * 0.12 {
                return nil
            }
            return c.box
        }
    }

    /// 8-connected components of nonzero pixels, numbered in raster order of their first pixel.
    static func components(_ img: GrayImage) -> [(box: Box, area: Int)] {
        let w = img.width, h = img.height
        var label = [Int32](repeating: 0, count: w * h)
        var out: [(box: Box, area: Int)] = []
        var stack: [Int] = []
        img.pixels.withUnsafeBufferPointer { px in
            label.withUnsafeMutableBufferPointer { lab in
                for start in 0..<(w * h) where px[start] != 0 && lab[start] == 0 {
                    let id = Int32(out.count + 1)
                    var x0 = Int.max, y0 = Int.max, x1 = -1, y1 = -1, area = 0
                    lab[start] = id
                    stack.append(start)
                    while let p = stack.popLast() {
                        let x = p % w, y = p / w
                        area += 1
                        x0 = min(x0, x); x1 = max(x1, x); y0 = min(y0, y); y1 = max(y1, y)
                        for ny in max(y - 1, 0)...min(y + 1, h - 1) {
                            for nx in max(x - 1, 0)...min(x + 1, w - 1) {
                                let q = ny * w + nx
                                if px[q] != 0 && lab[q] == 0 {
                                    lab[q] = id
                                    stack.append(q)
                                }
                            }
                        }
                    }
                    out.append((Box(x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1), area))
                }
            }
        }
        return out
    }

    /// Splits boxes about n pitches tall at the profile minima near k/n of the height.
    static func splitTall(_ binary: GrayImage, _ boxes: [Box], pitch: Double) -> [Box] {
        let k5 = ImageOps.gaussianKernel(size: 5, sigma: 0)
        var out: [Box] = []
        for b in boxes {
            let n = ImageOps.roundHalfEven(Double(b.h) / pitch)
            if n <= 1 {
                out.append(b)
                continue
            }
            let roi = binary.cropped(x: b.x, y: b.y, w: b.w, h: b.h)
            let prof = roi.rowSums()
            let idx = ImageOps.padIndex(b.h, 2, .reflect101)
            let smooth = (0..<b.h).map { y in
                var s = 0.0
                for j in 0..<5 { s += k5[j] * prof[idx[y + j]] }
                return s
            }
            var cuts = [0]
            for k in 1..<n {
                let centre = Int(Double(k * b.h) / Double(n))
                let lo = max(cuts[cuts.count - 1] + 2, centre - Int(pitch * 0.35))
                let hi = min(b.h - 2, centre + Int(pitch * 0.35))
                if hi > lo {
                    var best = lo
                    for i in lo..<hi where smooth[i] < smooth[best] { best = i }
                    cuts.append(best)
                }
            }
            cuts.append(b.h)
            for (top, bottom) in zip(cuts, cuts.dropFirst()) where bottom > top {
                var r0 = Int.max, r1 = -1, c0 = Int.max, c1 = -1
                for y in top..<bottom {
                    for x in 0..<b.w where roi.pixels[y * b.w + x] != 0 {
                        r0 = min(r0, y); r1 = max(r1, y); c0 = min(c0, x); c1 = max(c1, x)
                    }
                }
                if r1 < 0 { continue }
                out.append(Box(x: b.x + c0, y: b.y + r0, w: c1 - c0 + 1, h: r1 - r0 + 1))
            }
        }
        return out
    }

    /// Merges boxes that overlap vertically by half the smaller height and are horizontally close.
    static func mergeOverlapping(_ boxes: [Box], pitch: Double) -> [Box] {
        var merged: [Box] = []
        let sorted = boxes.enumerated().sorted { a, b in
            (a.element.y, a.element.x, a.offset) < (b.element.y, b.element.x, b.offset)
        }.map(\.element)
        for b in sorted {
            var hit = false
            for (i, o) in merged.enumerated() {
                let v = min(b.y + b.h, o.y + o.h) - max(b.y, o.y)
                let gap = max(b.x, o.x) - min(b.x + b.w, o.x + o.w)
                if Double(v) > Double(min(b.h, o.h)) * 0.5 && Double(gap) < pitch * 1.2 {
                    let x0 = min(b.x, o.x), y0 = min(b.y, o.y)
                    let x1 = max(b.x + b.w, o.x + o.w), y1 = max(b.y + b.h, o.y + o.h)
                    merged[i] = Box(x: x0, y: y0, w: x1 - x0, h: y1 - y0)
                    hit = true
                    break
                }
            }
            if !hit { merged.append(b) }
        }
        return merged
    }

    /// Rows top to bottom; inside a row, right to left.
    static func readingOrder(_ boxes: [Box], pitch: Double) -> [Box] {
        guard !boxes.isEmpty else { return [] }
        func cy(_ b: Box) -> Double { Double(b.y) + Double(b.h) / 2 }
        let ordered = boxes.enumerated().sorted { (cy($0.element), $0.offset) < (cy($1.element), $1.offset) }
            .map(\.element)
        var rows: [[Box]] = [[ordered[0]]]
        for b in ordered.dropFirst() {
            let row = rows[rows.count - 1]
            let rc = row.reduce(0) { $0 + cy($1) } / Double(row.count)
            if abs(cy(b) - rc) < pitch * 0.45 {
                rows[rows.count - 1].append(b)
            } else {
                rows.append([b])
            }
        }
        return rows.flatMap { row in
            row.enumerated().sorted { (-($0.element.x + $0.element.w), $0.offset) < (-($1.element.x + $1.element.w), $1.offset) }
                .map(\.element)
        }
    }
}

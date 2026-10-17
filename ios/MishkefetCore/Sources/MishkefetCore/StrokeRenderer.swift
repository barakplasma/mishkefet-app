import CoreGraphics
import Foundation

/// The deterministic PRNG both apps use for augmentation.
public struct Mulberry32: Sendable {
    private var state: UInt32

    public init(seed: UInt32) {
        state = seed
    }

    /// A value in [0, 1).
    public mutating func next() -> Double {
        state = state &+ 0x6D2B_79F5
        var t = state
        t = (t ^ (t >> 15)) &* (t | 1)
        t ^= t &+ ((t ^ (t >> 7)) &* (t | 61))
        return Double(t ^ (t >> 14)) / 4_294_967_296
    }
}

/// FNV-1a 32 of the UTF-8 bytes.
public func fnv1a32(_ s: String) -> UInt32 {
    var h: UInt32 = 0x811C_9DC5
    for b in s.utf8 {
        h ^= UInt32(b)
        h = h &* 0x0100_0193
    }
    return h
}

/// How one training view is drawn (spec/strokes.md).
public struct RenderParams: Codable, Sendable, Equatable {
    public var slant: Double
    /// Degrees.
    public var rotate: Double
    public var radius: Double
    public var density: Double
    public var ink: Double

    public init(slant: Double, rotate: Double, radius: Double, density: Double, ink: Double) {
        self.slant = slant
        self.rotate = rotate
        self.radius = radius
        self.density = density
        self.ink = ink
    }

    public static let plain = RenderParams(slant: 0, rotate: 0, radius: 1.2, density: 1, ink: 0.62)

    /// Random variation for one training view; seed 0 is `plain`.
    public static func augmented(seed: UInt32) -> RenderParams {
        if seed == 0 { return .plain }
        var r = Mulberry32(seed: seed)
        let slant = (r.next() * 2 - 1) * 0.25
        let rotate = (r.next() * 2 - 1) * 2
        let radius = 0.9 + r.next() * 1.1
        let density = 0.85 + r.next() * 0.33
        let ink = 0.5 + r.next() * 0.3
        return RenderParams(slant: slant, rotate: rotate, radius: radius, density: density, ink: ink)
    }

    /// Seeds of a sample's training views: 0 (plain), then
    /// `(fnv1a32(id) & 0xFFFFFF) * 16 + k` for k = 1...augmented.
    public static func trainingSeeds(sampleID: String, augmented: Int) -> [UInt32] {
        let base = (fnv1a32(sampleID) & 0xFF_FFFF) * 16
        return [0] + (0..<max(0, augmented)).map { base + UInt32($0 + 1) }
    }
}

/// Training-canvas strokes to a grayscale line crop (dark ink on white, height 128).
public enum StrokeRenderer {
    public static let height = 128
    static let padX = 12.0
    static let pxPerChar = 44.2
    static let paper: Float = 255
    static let inkLevel: Float = 30

    /// Strokes in canvas units (y down); nil if there are no points.
    public static func render(_ strokes: [[CGPoint]], nChars: Int, params: RenderParams = .plain) -> GrayImage? {
        let pts = strokes.flatMap { $0 }
        guard !pts.isEmpty else { return nil }
        let xs = pts.map { Double($0.x) }, ys = pts.map { Double($0.y) }
        let cx = (xs.min()! + xs.max()!) / 2, cy = (ys.min()! + ys.max()!) / 2
        let a = params.rotate * .pi / 180
        let ca = cos(a), sa = sin(a)
        let transformed: [[(Double, Double)]] = strokes.filter { !$0.isEmpty }.map { s in
            s.map { p in
                let x = Double(p.x) + params.slant * (cy - Double(p.y))
                let dx = x - cx, dy = Double(p.y) - cy
                return (cx + dx * ca - dy * sa, cy + dx * sa + dy * ca)
            }
        }
        let flat = transformed.flatMap { $0 }
        let x0 = flat.map(\.0).min()!, x1 = flat.map(\.0).max()!
        let y0 = flat.map(\.1).min()!, y1 = flat.map(\.1).max()!
        let H = Double(height)
        var s = H * params.ink / max(y1 - y0, 1e-6)
        if y1 - y0 < 1e-3 { s = 1 }
        let top = (H - (y1 - y0) * s) / 2
        let width = (x1 - x0) * s + 2 * padX
        var fx = 1.0
        if nChars > 0 { fx = min(max(pxPerChar * Double(nChars) / width, 0.5), 2) }
        fx *= params.density
        let W = max(16, Int(((x1 - x0) * s * fx + 2 * padX).rounded(.up)))

        var cov = [Float](repeating: 0, count: height * W)
        for stroke in transformed {
            let px = stroke.map { ((($0.0 - x0) * s * fx + padX), (($0.1 - y0) * s + top)) }
            if px.count == 1 {
                capsule(&cov, W, px[0], px[0], params.radius)
            } else {
                for i in 1..<px.count { capsule(&cov, W, px[i - 1], px[i], params.radius) }
            }
        }
        let pixels = cov.map { c -> UInt8 in
            let g = paper - c * (paper - inkLevel)
            return UInt8((g + 0.5).rounded(.down))
        }
        return GrayImage(width: W, height: height, pixels: pixels)
    }

    /// cov = max(cov, clamp(r + 0.5 - distance to segment, 0, 1)) at pixel centres.
    private static func capsule(_ cov: inout [Float], _ w: Int, _ a: (Double, Double), _ b: (Double, Double), _ r: Double) {
        let h = height
        let reach = r + 1
        let xa = max(Int((min(a.0, b.0) - reach).rounded(.down)), 0)
        let xb = min(Int((max(a.0, b.0) + reach).rounded(.up)), w - 1)
        let ya = max(Int((min(a.1, b.1) - reach).rounded(.down)), 0)
        let yb = min(Int((max(a.1, b.1) + reach).rounded(.up)), h - 1)
        guard xa <= xb, ya <= yb else { return }
        let dx = b.0 - a.0, dy = b.1 - a.1
        let ll = dx * dx + dy * dy
        for j in ya...yb {
            let py = Double(j) + 0.5
            for i in xa...xb {
                let px = Double(i) + 0.5
                let t = ll > 0 ? min(max(((px - a.0) * dx + (py - a.1) * dy) / ll, 0), 1) : 0
                let d = hypot(px - (a.0 + t * dx), py - (a.1 + t * dy))
                let c = Float(min(max(r + 0.5 - d, 0), 1))
                if c > cov[j * w + i] { cov[j * w + i] = c }
            }
        }
    }
}

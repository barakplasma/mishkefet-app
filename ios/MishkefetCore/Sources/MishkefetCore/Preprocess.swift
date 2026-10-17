import Accelerate
import Foundation

/// Line crop to model input (spec/preprocess.md).
public enum Preprocess {
    public static let lineHeight = 64
    public static let minWidth = 16
    public static let maxWidth = 2560
    public static let ttaScales: [Double] = [1.0, 0.82, 1.22]

    /// Contrast normalization at the crop's own resolution.
    static func contrastNormalize(_ g: GrayImage) -> GrayImage {
        let blur = ImageOps.blur(g, sigma: max(Double(g.height) / 8, 3))
        let n = g.pixels.count
        var flat = [Float](repeating: 0, count: n)
        g.pixels.withUnsafeBufferPointer { gp in
            blur.withUnsafeBufferPointer { bp in
                flat.withUnsafeMutableBufferPointer { f in
                    for i in 0..<n { f[i] = Float(gp[i]) / max(Float(bp[i]), 1e-3) }
                }
            }
        }
        var sorted = [Double](repeating: 0, count: n)
        vDSP_vspdp(flat, 1, &sorted, 1, vDSP_Length(n))
        vDSP_vsortD(&sorted, vDSP_Length(n), 1)
        let lo = ImageOps.percentile(sorted: sorted, 2), hi = ImageOps.percentile(sorted: sorted, 98)
        if hi - lo < 1e-6 { return GrayImage(width: g.width, height: g.height, fill: 255) }
        var out = [UInt8](repeating: 0, count: n)
        let range = hi - lo
        flat.withUnsafeBufferPointer { f in
            out.withUnsafeMutableBufferPointer { o in
                for i in 0..<n {
                    let v = (Double(f[i]) - lo) / range * 255
                    o[i] = v <= 0 ? 0 : (v >= 255 ? 255 : UInt8(v))
                }
            }
        }
        return GrayImage(width: g.width, height: g.height, pixels: out)
    }

    /// Model input width for a crop of this size.
    public static func outputWidth(width: Int, height: Int) -> Int {
        let scale = Double(lineHeight) / Double(max(height, 1))
        return max(minWidth, min(ImageOps.roundHalfEven(Double(width) * scale), maxWidth))
    }

    /// uint8 line crop to float32 (64, W) row-major, ink near 1.
    public static func run(_ gray: GrayImage) -> (values: [Float], width: Int) {
        let g = contrastNormalize(gray)
        let scale = Double(lineHeight) / Double(max(g.height, 1))
        let nw = outputWidth(width: g.width, height: g.height)
        let r = scale < 1 ? ImageOps.resizeArea(g, width: nw, height: lineHeight)
            : ImageOps.resizeLinear(g, width: nw, height: lineHeight)
        // 1 - g / 255 in float32 (scalar for IEEE division)
        var x = [Float](repeating: 0, count: r.pixels.count)
        r.pixels.withUnsafeBufferPointer { p in
            x.withUnsafeMutableBufferPointer { o in
                for i in 0..<o.count { o[i] = 1 - Float(p[i]) / 255 }
            }
        }
        return (x, nw)
    }

    /// A TTA view: horizontal-only bicubic resize by `scale` (identity at 1.0).
    public static func ttaView(_ gray: GrayImage, scale: Double) -> GrayImage {
        if scale == 1 { return gray }
        return ImageOps.bicubicResizeX(gray, width: max(8, Int(Double(gray.width) * scale)))
    }
}

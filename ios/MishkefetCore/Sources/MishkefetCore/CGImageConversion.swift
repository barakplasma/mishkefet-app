import CoreGraphics
import Foundation

extension GrayImage {
    /// PIL "L" conversion: alpha composited onto white, then BT.601 luma
    /// `(R*19595 + G*38470 + B*7471 + 0x8000) >> 16`. Pixels are drawn in the image's
    /// own color space (no color management), so gray and RGB files keep their stored
    /// values. EXIF orientation is the caller's job.
    public init(cgImage image: CGImage) {
        let w = image.width, h = image.height
        let rect = CGRect(x: 0, y: 0, width: w, height: h)
        if let space = image.colorSpace, space.model == .monochrome,
           let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w,
                               space: space, bitmapInfo: CGImageAlphaInfo.none.rawValue) {
            ctx.setFillColor(gray: 1, alpha: 1)
            ctx.fill(rect)
            ctx.interpolationQuality = .none
            ctx.draw(image, in: rect)
            self = GrayImage(width: w, height: h, pixels: Self.bytes(of: ctx, width: w, height: h))
            return
        }
        self = RGBPlanes(cgImage: image).gray()
    }

    private static func bytes(of ctx: CGContext, width keep: Int, height: Int) -> [UInt8] {
        let src = ctx.data!.assumingMemoryBound(to: UInt8.self)
        let stride = ctx.bytesPerRow
        var out = [UInt8](repeating: 0, count: keep * height)
        out.withUnsafeMutableBufferPointer { dst in
            for y in 0..<height {
                dst.baseAddress!.advanced(by: y * keep).update(from: src + y * stride, count: keep)
            }
        }
        return out
    }

    /// A DeviceGray CGImage of these pixels (for display and debugging).
    public func makeCGImage() -> CGImage? {
        guard width > 0, height > 0 else { return nil }
        let data = Data(pixels) as CFData
        guard let provider = CGDataProvider(data: data) else { return nil }
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: width,
                       space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGBitmapInfo(rawValue: 0),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }
}

/// An RGB image as three uint8 planes (alpha composited on white). Kept for colour pages
/// so rotations can run per channel before the gray conversion, as the reference does.
struct RGBPlanes: Sendable {
    let width: Int
    let height: Int
    var r: GrayImage
    var g: GrayImage
    var b: GrayImage

    init(r: GrayImage, g: GrayImage, b: GrayImage) {
        width = r.width
        height = r.height
        self.r = r
        self.g = g
        self.b = b
    }

    /// Drawn in the image's own RGB space when it has one (no conversion), else sRGB.
    init(cgImage image: CGImage) {
        let w = image.width, h = image.height
        func rgbContext(_ space: CGColorSpace) -> CGContext? {
            CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                      space: space, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        }
        let own = image.colorSpace.flatMap { $0.model == .rgb ? rgbContext($0) : nil }
        guard let ctx = own ?? rgbContext(CGColorSpace(name: CGColorSpace.sRGB)!) else {
            let white = GrayImage(width: w, height: h, fill: 255)
            self.init(r: white, g: white, b: white)
            return
        }
        let rect = CGRect(x: 0, y: 0, width: w, height: h)
        ctx.setFillColor(red: 1, green: 1, blue: 1, alpha: 1)
        ctx.fill(rect)
        ctx.interpolationQuality = .none
        ctx.draw(image, in: rect)
        var planes = [[UInt8]](repeating: [UInt8](repeating: 0, count: w * h), count: 3)
        let stride = ctx.bytesPerRow
        let src = ctx.data!.assumingMemoryBound(to: UInt8.self)
        for c in 0..<3 {
            planes[c].withUnsafeMutableBufferPointer { dst in
                for y in 0..<h {
                    let row = src + y * stride
                    for x in 0..<w { dst[y * w + x] = row[4 * x + c] }
                }
            }
        }
        self.init(r: GrayImage(width: w, height: h, pixels: planes[0]),
                  g: GrayImage(width: w, height: h, pixels: planes[1]),
                  b: GrayImage(width: w, height: h, pixels: planes[2]))
    }

    /// BT.601 luma with PIL's fixed-point rounding.
    func gray() -> GrayImage {
        var out = [UInt8](repeating: 0, count: width * height)
        r.pixels.withUnsafeBufferPointer { rp in
            g.pixels.withUnsafeBufferPointer { gp in
                b.pixels.withUnsafeBufferPointer { bp in
                    out.withUnsafeMutableBufferPointer { dst in
                        for i in 0..<dst.count {
                            let rr: UInt32 = UInt32(rp[i]) * 19595
                            let gg: UInt32 = UInt32(gp[i]) * 38470
                            let bb: UInt32 = UInt32(bp[i]) * 7471
                            dst[i] = UInt8(truncatingIfNeeded: (rr + gg + bb + 0x8000) >> 16)
                        }
                    }
                }
            }
        }
        return GrayImage(width: width, height: height, pixels: out)
    }

    func map(_ f: (GrayImage) -> GrayImage) -> RGBPlanes {
        RGBPlanes(r: f(r), g: f(g), b: f(b))
    }
}

/// What segmentation works on: the gray page, plus its RGB planes for colour pages.
struct PageImage: Sendable {
    var gray: GrayImage
    var rgb: RGBPlanes?

    init(gray: GrayImage, rgb: RGBPlanes? = nil) {
        self.gray = gray
        self.rgb = rgb
    }

    init(cgImage image: CGImage) {
        if image.colorSpace?.model == .monochrome {
            self.init(gray: GrayImage(cgImage: image))
        } else {
            let rgb = RGBPlanes(cgImage: image)
            self.init(gray: rgb.gray(), rgb: rgb)
        }
    }

    func rot90(_ k: Int) -> PageImage {
        PageImage(gray: gray.rot90(k), rgb: rgb?.map { $0.rot90(k) })
    }

    func cropped(_ b: LineBox) -> PageImage {
        PageImage(gray: gray.cropped(x: b.x, y: b.y, w: b.w, h: b.h),
                  rgb: rgb?.map { $0.cropped(x: b.x, y: b.y, w: b.w, h: b.h) })
    }

    /// PIL bicubic rotation, per RGB channel when there is colour, then to gray.
    func rotated(_ angle: Double, expand: Bool) -> GrayImage {
        if let rgb { return rgb.map { Segmenter.rotateBicubic($0, angle: angle, expand: expand) }.gray() }
        return Segmenter.rotateBicubic(gray, angle: angle, expand: expand)
    }
}

enum Rotation {
    /// `image` turned `turns` quarter turns and then rotated by `angle` degrees (both
    /// counter-clockwise on screen), centre to centre on a `width x height` canvas, white
    /// outside. One CoreGraphics draw; this is the page that gets stored and shown.
    static func render(_ image: CGImage, turns: Int, angle: Double, width w: Int, height h: Int) -> CGImage? {
        let space = image.colorSpace.flatMap { $0.model == .rgb ? $0 : nil }
            ?? CGColorSpace(name: CGColorSpace.sRGB)!
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: space, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
        ctx.setFillColor(red: 1, green: 1, blue: 1, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        ctx.interpolationQuality = abs(angle) > 1e-3 ? .high : .none
        // The bitmap context is y-up, so a positive CG rotation is counter-clockwise on screen.
        ctx.translateBy(x: CGFloat(w) / 2, y: CGFloat(h) / 2)
        ctx.rotate(by: CGFloat((angle + 90 * Double(turns)) * .pi / 180))
        ctx.translateBy(x: -CGFloat(image.width) / 2, y: -CGFloat(image.height) / 2)
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        return ctx.makeImage()
    }
}

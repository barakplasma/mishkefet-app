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
        func rgbContext(_ space: CGColorSpace) -> CGContext? {
            CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                      space: space, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        }
        // The image's own RGB space when it has one (no conversion), else sRGB.
        let own = image.colorSpace.flatMap { $0.model == .rgb ? rgbContext($0) : nil }
        guard let ctx = own ?? rgbContext(CGColorSpace(name: CGColorSpace.sRGB)!) else {
            self = GrayImage(width: w, height: h, fill: 255)
            return
        }
        ctx.setFillColor(red: 1, green: 1, blue: 1, alpha: 1)
        ctx.fill(rect)
        ctx.interpolationQuality = .none
        ctx.draw(image, in: rect)
        var out = [UInt8](repeating: 0, count: w * h)
        let stride = ctx.bytesPerRow
        let src = ctx.data!.assumingMemoryBound(to: UInt8.self)
        out.withUnsafeMutableBufferPointer { dst in
            for y in 0..<h {
                let row = src + y * stride
                for x in 0..<w {
                    let r: UInt32 = UInt32(row[4 * x]) * 19595
                    let g: UInt32 = UInt32(row[4 * x + 1]) * 38470
                    let b: UInt32 = UInt32(row[4 * x + 2]) * 7471
                    dst[y * w + x] = UInt8(truncatingIfNeeded: (r + g + b + 0x8000) >> 16)
                }
            }
        }
        self = GrayImage(width: w, height: h, pixels: out)
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

enum Rotation {
    /// `image` rotated by `angle` degrees about its centre (positive is counter-clockwise
    /// on screen), same size, white outside. CoreGraphics resampling; for display.
    static func rotatedCGImage(_ image: CGImage, angle: Double) -> CGImage? {
        let w = image.width, h = image.height
        let space = image.colorSpace.flatMap { $0.model == .rgb ? $0 : nil }
            ?? CGColorSpace(name: CGColorSpace.sRGB)!
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: space, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
        ctx.setFillColor(red: 1, green: 1, blue: 1, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        ctx.interpolationQuality = .high
        // The bitmap context is y-up, so a positive CG rotation is counter-clockwise on screen.
        ctx.translateBy(x: CGFloat(w) / 2, y: CGFloat(h) / 2)
        ctx.rotate(by: CGFloat(angle * .pi / 180))
        ctx.translateBy(x: -CGFloat(w) / 2, y: -CGFloat(h) / 2)
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        return ctx.makeImage()
    }
}

import CoreImage
import CoreImage.CIFilterBuiltins
import ImageIO
import MishkefetCore
import PDFKit
import UIKit
import UniformTypeIdentifiers
import Vision

/// A picture to read: upright (EXIF orientation applied), on white, at most `Images.maxSide`.
struct Picture: @unchecked Sendable {
    var image: CGImage
    var name: String
    var width: Int { image.width }
    var height: Int { image.height }
}

enum Images {
    /// Pages are read at up to this long side; the segmenter works at 2000 anyway.
    static let maxSide: CGFloat = 3200
    static let thumbSide: CGFloat = 160

    /// Applies `imageOrientation`, flattens onto white, and caps the long side.
    static func normalized(_ image: UIImage, maxSide: CGFloat = maxSide) -> CGImage? {
        let size = image.size
        guard size.width > 0, size.height > 0 else { return nil }
        let s = min(1, maxSide / max(size.width * image.scale, size.height * image.scale)) * image.scale
        let target = CGSize(width: max(1, (size.width * s).rounded()), height: max(1, (size.height * s).rounded()))
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        let out = UIGraphicsImageRenderer(size: target, format: format).image { ctx in
            UIColor.white.setFill()
            ctx.fill(CGRect(origin: .zero, size: target))
            image.draw(in: CGRect(origin: .zero, size: target))
        }
        return out.cgImage
    }

    static func picture(data: Data, name: String) -> [Picture] {
        if data.starts(with: Array("%PDF".utf8)), let doc = PDFDocument(data: data) { return pdfPages(doc, name: name) }
        guard let img = UIImage(data: data), let cg = normalized(img) else { return [] }
        return [Picture(image: cg, name: name)]
    }

    static func pictures(fileURL url: URL) -> [Picture] {
        let name = url.deletingPathExtension().lastPathComponent
        if url.pathExtension.lowercased() == "pdf", let doc = PDFDocument(url: url) { return pdfPages(doc, name: name) }
        guard let data = try? Data(contentsOf: url) else { return [] }
        return picture(data: data, name: name)
    }

    static func pdfPages(_ doc: PDFDocument, name: String) -> [Picture] {
        var out: [Picture] = []
        for i in 0..<min(doc.pageCount, 20) {
            guard let page = doc.page(at: i) else { continue }
            let bounds = page.bounds(for: .mediaBox)
            let scale = min(maxSide / max(bounds.width, bounds.height), 4)
            let img = page.thumbnail(of: CGSize(width: bounds.width * scale, height: bounds.height * scale), for: .mediaBox)
            if let cg = normalized(img) {
                out.append(Picture(image: cg, name: doc.pageCount > 1 ? "\(name) \(i + 1)" : name))
            }
        }
        return out
    }

    static func jpeg(_ image: CGImage, quality: CGFloat = 0.92) -> Data? {
        UIImage(cgImage: image).jpegData(compressionQuality: quality)
    }

    static func thumbnail(_ image: CGImage) -> Data? {
        let s = thumbSide / CGFloat(max(image.width, image.height))
        let size = CGSize(width: max(1, (CGFloat(image.width) * s).rounded()), height: max(1, (CGFloat(image.height) * s).rounded()))
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        return UIGraphicsImageRenderer(size: size, format: format).image { _ in
            UIImage(cgImage: image).draw(in: CGRect(origin: .zero, size: size))
        }.jpegData(compressionQuality: 0.8)
    }

    static func load(_ url: URL) -> CGImage? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(src, 0, nil)
    }

    /// A line's pixels with the segmenter's padding (0.12 h vertical, 0.05 h horizontal), clamped.
    static func lineRect(_ b: LineBox, width: Int, height: Int) -> CGRect {
        let py = Int(Double(b.h) * 0.12), px = Int(Double(b.h) * 0.05)
        let x0 = max(0, b.x - px), y0 = max(0, b.y - py)
        let x1 = min(width, b.x + b.w + px), y1 = min(height, b.y + b.h + py)
        return CGRect(x: x0, y: y0, width: max(1, x1 - x0), height: max(1, y1 - y0))
    }

    static func cropLine(_ image: CGImage, box: LineBox) -> CGImage? {
        image.cropping(to: lineRect(box, width: image.width, height: image.height))
    }

    static func cropLine(_ gray: GrayImage, box: LineBox) -> GrayImage {
        let r = lineRect(box, width: gray.width, height: gray.height)
        let x = min(Int(r.minX), max(0, gray.width - 1)), y = min(Int(r.minY), max(0, gray.height - 1))
        let w = min(Int(r.width), gray.width - x), h = min(Int(r.height), gray.height - y)
        return gray.cropped(x: x, y: y, w: max(0, w), h: max(0, h))
    }

    static func png(_ image: CGImage) -> Data? {
        UIImage(cgImage: image).pngData()
    }

    // MARK: - Camera

    /// Crops `rect` (fractions of the image) out of an upright picture.
    static func crop(_ image: CGImage, fraction r: CGRect) -> CGImage? {
        let W = CGFloat(image.width), H = CGFloat(image.height)
        let rect = CGRect(x: r.minX * W, y: r.minY * H, width: r.width * W, height: r.height * H).integral
            .intersection(CGRect(x: 0, y: 0, width: W, height: H))
        guard rect.width >= 8, rect.height >= 8 else { return nil }
        return image.cropping(to: rect)
    }

    /// Finds the sheet of paper and flattens it; nil when no confident quadrilateral is found.
    static func documentCorrected(_ image: CGImage) -> CGImage? {
        let handler = VNImageRequestHandler(cgImage: image, orientation: .up)
        var quad: VNRectangleObservation?
        let doc = VNDetectDocumentSegmentationRequest()
        if (try? handler.perform([doc])) != nil, let r = doc.results?.first, r.confidence >= 0.6 {
            quad = r
        } else {
            let rect = VNDetectRectanglesRequest()
            rect.minimumAspectRatio = 0.3
            rect.maximumAspectRatio = 1
            rect.minimumSize = 0.4
            rect.minimumConfidence = 0.7
            rect.maximumObservations = 1
            if (try? handler.perform([rect])) != nil { quad = rect.results?.first }
        }
        guard let q = quad else { return nil }
        // Reject tiny or degenerate quads.
        let area = abs((q.topRight.x - q.topLeft.x) * (q.topLeft.y - q.bottomLeft.y))
        guard area > 0.15 else { return nil }
        let W = CGFloat(image.width), H = CGFloat(image.height)
        // Vision's normalized coordinates have a bottom-left origin, as Core Image does.
        func pt(_ p: CGPoint) -> CGPoint { CGPoint(x: p.x * W, y: p.y * H) }
        let f = CIFilter.perspectiveCorrection()
        f.inputImage = CIImage(cgImage: image)
        f.topLeft = pt(q.topLeft)
        f.topRight = pt(q.topRight)
        f.bottomLeft = pt(q.bottomLeft)
        f.bottomRight = pt(q.bottomRight)
        guard let out = f.outputImage else { return nil }
        let ctx = CIContext(options: [.useSoftwareRenderer: false])
        guard let cg = ctx.createCGImage(out, from: out.extent.integral) else { return nil }
        guard cg.width >= 64, cg.height >= 64 else { return nil }
        return cg
    }
}

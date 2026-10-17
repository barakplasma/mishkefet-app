import Accelerate
import CoreGraphics
import Foundation

/// An 8-bit grayscale image, row-major, 0 = black.
public struct GrayImage: Sendable, Equatable {
    public let width: Int
    public let height: Int
    public var pixels: [UInt8]

    public init(width: Int, height: Int, pixels: [UInt8]) {
        precondition(pixels.count == width * height, "pixel count does not match the size")
        self.width = width
        self.height = height
        self.pixels = pixels
    }

    public init(width: Int, height: Int, fill: UInt8 = 0) {
        self.init(width: width, height: height, pixels: [UInt8](repeating: fill, count: width * height))
    }

    @inline(__always) public subscript(x: Int, y: Int) -> UInt8 {
        get { pixels[y * width + x] }
        set { pixels[y * width + x] = newValue }
    }

    /// The rectangle `[x, x + w) x [y, y + h)`, which must lie inside the image.
    public func cropped(x: Int, y: Int, w: Int, h: Int) -> GrayImage {
        precondition(x >= 0 && y >= 0 && x + w <= width && y + h <= height, "crop outside the image")
        var out = [UInt8](repeating: 0, count: w * h)
        pixels.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                for r in 0..<h {
                    let s = (y + r) * width + x
                    dst.baseAddress!.advanced(by: r * w).update(from: src.baseAddress!.advanced(by: s), count: w)
                }
            }
        }
        return GrayImage(width: w, height: h, pixels: out)
    }

    func transposed() -> GrayImage {
        var f = floats(), t = [Float](repeating: 0, count: pixels.count)
        vDSP_mtrans(&f, 1, &t, 1, vDSP_Length(width), vDSP_Length(height))
        return GrayImage(width: height, height: width, floats: t)
    }

    func floats() -> [Float] {
        var out = [Float](repeating: 0, count: pixels.count)
        vDSP_vfltu8(pixels, 1, &out, 1, vDSP_Length(pixels.count))
        return out
    }

    func doubles() -> [Double] {
        var out = [Double](repeating: 0, count: pixels.count)
        vDSP_vfltu8D(pixels, 1, &out, 1, vDSP_Length(pixels.count))
        return out
    }

    /// From floats already in 0...255 (truncated toward zero).
    init(width: Int, height: Int, floats: [Float]) {
        var px = [UInt8](repeating: 0, count: floats.count)
        vDSP_vfixu8(floats, 1, &px, 1, vDSP_Length(floats.count))
        self.init(width: width, height: height, pixels: px)
    }

    /// Row sums, in double.
    func rowSums() -> [Double] {
        let f = floats()
        return f.withUnsafeBufferPointer { p in
            (0..<height).map { y in
                var s: Float = 0
                vDSP_sve(p.baseAddress! + y * width, 1, &s, vDSP_Length(width))
                return Double(s)
            }
        }
    }
}

/// A line's rectangle in page pixels (of the deskewed page).
public struct LineBox: Codable, Hashable, Sendable {
    public var x, y, w, h: Int

    public init(x: Int, y: Int, w: Int, h: Int) {
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    }
}

public struct LineResult: Codable, Hashable, Sendable {
    public var text: String
    /// Mean best-class probability over all frames of the scale-1.0 view.
    public var confidence: Double
    public var box: LineBox?

    public init(text: String, confidence: Double, box: LineBox? = nil) {
        self.text = text
        self.confidence = confidence
        self.box = box
    }

    /// The page transcript: line texts in reading order, blank lines dropped.
    public static func transcript(_ lines: [LineResult]) -> String {
        lines.map(\.text).filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            .joined(separator: "\n")
    }
}

public struct PageSegmentation: Sendable {
    /// Skew correction applied, in degrees (positive is counter-clockwise).
    public let angle: Double
    /// Measured line pitch, in pixels of the binarized page.
    public let pitch: Double
    /// Line boxes in reading order, in pixels of `deskewed`.
    public let boxes: [LineBox]
    /// Padded grayscale crops of the deskewed page, one per box.
    public let crops: [GrayImage]
    /// The page the boxes refer to: the input rotated by `angle` (the input itself when
    /// no rotation was needed). Nil only if the rotated image could not be drawn.
    public let deskewed: CGImage?

    public init(angle: Double, pitch: Double, boxes: [LineBox], crops: [GrayImage], deskewed: CGImage?) {
        self.angle = angle
        self.pitch = pitch
        self.boxes = boxes
        self.crops = crops
        self.deskewed = deskewed
    }
}

/// The shipped model files, all in one directory.
public struct ModelFiles: Sendable {
    public static let backboneName = "backbone.fp16.onnx"
    public static let headName = "head.bin"
    public static let charLMName = "charlm.bin"
    public static let wordsName = "words.bin"
    public static let charsetName = "charset.json"
    public static let allNames = [backboneName, headName, charLMName, wordsName, charsetName]

    public let directory: URL

    public init(directory: URL) {
        self.directory = directory
    }

    public var backbone: URL { directory.appendingPathComponent(Self.backboneName) }
    public var head: URL { directory.appendingPathComponent(Self.headName) }
    public var charLM: URL { directory.appendingPathComponent(Self.charLMName) }
    public var words: URL { directory.appendingPathComponent(Self.wordsName) }
    public var charset: URL { directory.appendingPathComponent(Self.charsetName) }

    /// File names that are missing from `directory`.
    public var missing: [String] {
        Self.allNames.filter { !FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path) }
    }
}

public enum MishkefetError: Error, LocalizedError, Equatable {
    case modelMissing(String)
    case badFile(String)
    case onnx(String)

    public var errorDescription: String? {
        switch self {
        case .modelMissing(let name): return "Model file missing or unreadable: \(name)"
        case .badFile(let what): return "Unexpected file format: \(what)"
        case .onnx(let message): return "ONNX Runtime error: \(message)"
        }
    }
}

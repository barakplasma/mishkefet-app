import CoreGraphics
import Foundation
import ImageIO
import XCTest
@testable import MishkefetCore

/// Paths to the shared fixtures in spec/ and to the model, valid on macOS and in
/// the iOS simulator (which can read host paths).
enum Fixtures {
    /// Repo root: four levels up from this file's directory.
    static let root: URL = {
        var u = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        for _ in 0..<4 { u = u.deletingLastPathComponent() }
        return u
    }()

    static let spec = root.appendingPathComponent("spec")
    static let golden = spec.appendingPathComponent("golden")
    static let fixtures = spec.appendingPathComponent("fixtures")

    static func goldenFile(_ name: String) -> URL { golden.appendingPathComponent(name) }

    static var hostHome: String {
        ProcessInfo.processInfo.environment["SIMULATOR_HOST_HOME"] ?? NSHomeDirectory()
    }

    /// The fetched model, from MISHKEFET_MODEL_DIR or the default cache.
    static var modelDirectory: URL {
        if let p = ProcessInfo.processInfo.environment["MISHKEFET_MODEL_DIR"], !p.isEmpty {
            return URL(fileURLWithPath: p)
        }
        return URL(fileURLWithPath: hostHome).appendingPathComponent(".cache/mishkefet/models/16380452")
    }

    /// Skips the test when the model files are not present.
    static func modelFiles() throws -> ModelFiles {
        let files = ModelFiles(directory: modelDirectory)
        if !files.missing.isEmpty {
            throw XCTSkip("model not found at \(modelDirectory.path) (missing \(files.missing)); "
                + "run tools/model/fetch-model.sh or set MISHKEFET_MODEL_DIR")
        }
        return files
    }

    private static let lock = NSLock()
    private static var cachedRecognizer: Recognizer?
    private static var cachedDecoder: Decoder?

    /// One prepared recognizer shared by the model tests (creating the session takes a while).
    static func recognizer() async throws -> Recognizer {
        let files = try modelFiles()
        if let r = lock.withLock({ cachedRecognizer }) { return r }
        let r = try Recognizer(files: files)
        try await r.prepare()
        lock.withLock { cachedRecognizer = r }
        return r
    }

    /// The beam decoder with both LMs (no backbone needed).
    static func decoder() throws -> Decoder {
        let files = try modelFiles()
        return try lock.withLock {
            if let d = cachedDecoder { return d }
            let charset = try Charset(contentsOf: files.charset)
            let d = Decoder(charLM: try CharLM(contentsOf: files.charLM),
                            words: try WordLM(contentsOf: files.words, charset: charset))
            cachedDecoder = d
            return d
        }
    }

    static var charset: Charset {
        get throws { try Charset(contentsOf: spec.appendingPathComponent("charset.json")) }
    }

    // MARK: - Images

    static func cgImage(_ url: URL) throws -> CGImage {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw NSError(domain: "Fixtures", code: 1, userInfo: [NSLocalizedDescriptionKey: "cannot read \(url.path)"])
        }
        return img
    }

    static func gray(_ relative: String) throws -> GrayImage {
        GrayImage(cgImage: try cgImage(fixtures.appendingPathComponent(relative)))
    }

    static func stem(_ file: String) -> String {
        (file as NSString).lastPathComponent.components(separatedBy: ".")[0]
    }

    // MARK: - Golden JSON

    struct Line: Decodable {
        let file: String
        let height: Int
        let width: Int
        let frames: Int
        let greedy: String
        let beam: String
        let read: String
        let confidence: Double
        let emitting_confidence: Double
    }

    struct Page: Decodable {
        let file: String
        let scale: Double
        let angle: Double
        let pitch: Double
        let boxes: [LineBox]
        let lines: [String]
    }

    static func decode<T: Decodable>(_ type: T.Type, _ name: String) throws -> T {
        try JSONDecoder().decode(T.self, from: Data(contentsOf: goldenFile(name)))
    }

    static func lines() throws -> [Line] { try decode([Line].self, "lines.json") }
    static func pages() throws -> [Page] { try decode([Page].self, "pages.json") }

    static func readFloat32LE(_ url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        var out = [Float](repeating: 0, count: data.count / 4)
        _ = out.withUnsafeMutableBytes { data.copyBytes(to: $0) }
        return out
    }

    static func iou(_ a: LineBox, _ b: LineBox) -> Double {
        let ix = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
        let iy = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
        let inter = Double(ix * iy)
        let union = Double(a.w * a.h + b.w * b.h) - inter
        return union > 0 ? inter / union : 0
    }

    static func ms(_ start: DispatchTime) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1e6
    }
}

import CMishkefetORT
import Foundation

/// The ONNX backbone: image (1, 1, 64, W) to features (1, W // 4, 512). Not thread-safe;
/// the Recognizer actor serializes calls.
final class Backbone: @unchecked Sendable {
    struct Options: Sendable {
        var intraOpThreads = 4
        var provider: Provider = .cpu
        var coreMLComputeUnits = "ALL"
        var coreMLCacheDirectory: URL?
        /// Widths change on every call, so a memory pattern buys nothing; the arena
        /// keeps activation buffers between lines.
        var arena = true
        var prepacking = true
        var memPattern = false

        enum Provider: Int32, Sendable { case cpu = 0, coreML = 1 }
    }

    static let dim = 512
    private let ort: OpaquePointer

    init(url: URL, options: Options = Options()) throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MishkefetError.modelMissing(url.lastPathComponent)
        }
        var err = [CChar](repeating: 0, count: 1024)
        let cacheDir = options.coreMLCacheDirectory?.path ?? ""
        if let dir = options.coreMLCacheDirectory {
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        let handle: OpaquePointer? = options.coreMLComputeUnits.withCString { units in
            cacheDir.withCString { cache in
                let o = MishkefetOrtOptions(intraOpThreads: Int32(options.intraOpThreads),
                                            provider: options.provider.rawValue,
                                            coreMLComputeUnits: units,
                                            coreMLCacheDirectory: cacheDir.isEmpty ? nil : cache,
                                            arena: options.arena ? 1 : 0,
                                            prepacking: options.prepacking ? 1 : 0,
                                            memPattern: options.memPattern ? 1 : 0)
                return mishkefet_ort_create(url.path, o, &err, err.count)
            }
        }
        guard let handle else {
            // A file that exists but won't load (truncated download) is reported like a missing one.
            NSLog("MishkefetCore: ONNX Runtime could not load the backbone: %@", String(cString: err))
            throw MishkefetError.modelMissing(url.lastPathComponent)
        }
        ort = handle
    }

    deinit {
        mishkefet_ort_destroy(ort)
    }

    /// Features (frames x 512) for a preprocessed (64 x width) image.
    func run(_ image: [Float], width: Int) throws -> (values: [Float], frames: Int) {
        precondition(image.count == 64 * width)
        let capacity = (width / 4 + 1) * Self.dim
        var out = [Float](repeating: 0, count: capacity)
        var frames: Int64 = 0, dim: Int64 = 0
        var err = [CChar](repeating: 0, count: 1024)
        let rc = image.withUnsafeBufferPointer { img in
            out.withUnsafeMutableBufferPointer { o in
                mishkefet_ort_run(ort, img.baseAddress!, Int64(width), o.baseAddress!, Int64(capacity),
                                  &frames, &dim, &err, err.count)
            }
        }
        guard rc == 0 else { throw MishkefetError.onnx(String(cString: err)) }
        guard dim == Self.dim else { throw MishkefetError.onnx("unexpected feature size \(dim)") }
        out.removeLast(capacity - Int(frames) * Self.dim)
        return (out, Int(frames))
    }
}

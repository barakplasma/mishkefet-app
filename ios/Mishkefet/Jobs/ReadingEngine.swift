import CoreGraphics
import Foundation
import MishkefetCore

/// What the app needs from the recognizer. The real one wraps MishkefetCore's `Recognizer` over the
/// bundled model; DEBUG builds can swap in a fake for screenshots.
protocol ReadingEngine: AnyObject, Sendable {
    /// Segments a page and reads its lines. `onSegmented` gets the page the boxes refer to (deskewed)
    /// before any line is read; `onLine(index, count, line)` after each one.
    func readPage(_ page: CGImage, head: Head?,
                  onSegmented: @escaping @Sendable (CGImage, [LineBox]) -> Void,
                  onLine: @escaping @Sendable (Int, Int, LineResult) -> Void) async throws -> (page: CGImage, lines: [LineResult])
    /// Re-reads known line boxes of a stored page (keeps its segmentation, so readings line up).
    func readLines(of page: CGImage, boxes: [LineBox], head: Head?) async throws -> [LineResult]
    /// Reads a single-line picture.
    func readLine(_ image: CGImage, head: Head?) async throws -> LineResult
    /// Reads training-canvas strokes (plain render, TTA 1).
    func readStrokes(_ strokes: [[CGPoint]], nChars: Int, head: Head?) async throws -> String
    /// One training view: canvas strokes rendered with `seed`, or a scan-line crop.
    func trainingView(strokes: [[CGPoint]]?, lineImage: CGImage?, word: String, seed: UInt32) async throws -> TrainingSample?
    func train(samples: [TrainingSample], config: AdaptConfig, progress: @escaping @Sendable (Int, Int) -> Void) async throws -> Head
    /// Releases the ONNX session (it is recreated on demand).
    func unload() async
}

enum EngineError: Error {
    case modelMissing
}

/// MishkefetCore over the model bundled under `Model/`.
final class CoreReadingEngine: ReadingEngine, @unchecked Sendable {
    private let lock = NSLock()
    private var recognizer: Recognizer?
    private let directory: URL

    init() {
        directory = Bundle.main.url(forResource: "Model", withExtension: nil)
            ?? Bundle.main.bundleURL.appendingPathComponent("Model", isDirectory: true)
    }

    private func engine() throws -> Recognizer {
        lock.lock()
        defer { lock.unlock() }
        if let r = recognizer { return r }
        let threads = max(2, min(4, ProcessInfo.processInfo.activeProcessorCount - 2))
        let r = try Recognizer(files: ModelFiles(directory: directory), threads: threads)
        recognizer = r
        return r
    }

    func readPage(_ page: CGImage, head: Head?,
                  onSegmented: @escaping @Sendable (CGImage, [LineBox]) -> Void,
                  onLine: @escaping @Sendable (Int, Int, LineResult) -> Void) async throws -> (page: CGImage, lines: [LineResult]) {
        let r = try engine()
        try await r.prepare()
        let seg = try await r.segment(page: page)
        let shown = seg.deskewed ?? page
        onSegmented(shown, seg.boxes)
        var lines: [LineResult] = []
        for (i, crop) in seg.crops.enumerated() {
            try Task.checkCancellation()
            var line = try await r.readLine(crop, head: head, tta: 3)
            line.box = seg.boxes[i]
            lines.append(line)
            onLine(i, seg.crops.count, line)
        }
        return (shown, lines)
    }

    func readLines(of page: CGImage, boxes: [LineBox], head: Head?) async throws -> [LineResult] {
        let r = try engine()
        let gray = await r.gray(from: page)
        var out: [LineResult] = []
        for b in boxes {
            try Task.checkCancellation()
            var line = try await r.readLine(Images.cropLine(gray, box: b), head: head, tta: 3)
            line.box = b
            out.append(line)
        }
        return out
    }

    func readLine(_ image: CGImage, head: Head?) async throws -> LineResult {
        let r = try engine()
        let gray = await r.gray(from: image)
        var line = try await r.readLine(gray, head: head, tta: 3)
        line.box = LineBox(x: 0, y: 0, w: image.width, h: image.height)
        return line
    }

    func readStrokes(_ strokes: [[CGPoint]], nChars: Int, head: Head?) async throws -> String {
        guard let gray = StrokeRenderer.render(strokes, nChars: nChars) else { return "" }
        return try await engine().readLine(gray, head: head, tta: 1).text
    }

    func trainingView(strokes: [[CGPoint]]?, lineImage: CGImage?, word: String, seed: UInt32) async throws -> TrainingSample? {
        let r = try engine()
        let gray: GrayImage?
        if let strokes {
            gray = StrokeRenderer.render(strokes, nChars: word.count, params: .augmented(seed: seed))
        } else if let lineImage {
            gray = await r.gray(from: lineImage)
        } else {
            gray = nil
        }
        guard let gray else { return nil }
        let f = try await r.features(of: gray)
        let label = r.charset.encode(word)
        guard f.frames > 0, !label.isEmpty else { return nil }
        return TrainingSample(features: f.values, frames: f.frames, label: label)
    }

    func train(samples: [TrainingSample], config: AdaptConfig, progress: @escaping @Sendable (Int, Int) -> Void) async throws -> Head {
        let base = try engine().baseHead
        let steps = config.steps
        return await Task.detached(priority: .userInitiated) {
            HeadTrainer.train(base: base, samples: samples, config: config) { step, _ in progress(step, steps) }
        }.value
    }

    func unload() async {
        lock.withLock { recognizer = nil }
    }
}

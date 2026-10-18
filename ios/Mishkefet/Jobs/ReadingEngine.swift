import CoreGraphics
import Foundation
import MishkefetCore

/// What the app needs from the recognizer. The real one wraps MishkefetCore's `Recognizer` over the
/// bundled model; DEBUG builds can swap in a fake for screenshots.
protocol ReadingEngine: AnyObject, Sendable {
    /// Crops a page to the paper, turns it upright, segments it and reads its lines. `onSegmented` gets
    /// the page the boxes refer to (upright, deskewed, possibly cropped: the image to store) before any
    /// line is read; `onLine(index, count, line)` after each one. The returned `page` is that image.
    func readPage(_ page: CGImage, head: Head?,
                  onSegmented: @escaping @Sendable (CGImage, [LineBox]) -> Void,
                  onLine: @escaping @Sendable (Int, Int, LineResult) -> Void) async throws -> (page: CGImage, lines: [LineResult])
    /// Segments and reads a stored page as it is (no paper crop or turning), so the boxes stay in its frame.
    func readStoredPage(_ page: CGImage, head: Head?) async throws -> [LineResult]
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

extension ReadingEngine {
    func readStoredPage(_ page: CGImage, head: Head?) async throws -> [LineResult] {
        try await readPage(page, head: head, onSegmented: { _, _ in }, onLine: { _, _, _ in }).lines
    }
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
        let seg = try await r.preparePage(page).segmentation
        let shown = seg.deskewed ?? page
        onSegmented(shown, seg.boxes)
        let lines = try await r.readLines(seg, head: head, tta: 3, onLine: onLine)
        return (shown, lines)
    }

    func readStoredPage(_ page: CGImage, head: Head?) async throws -> [LineResult] {
        let r = try engine()
        let seg = try await r.segment(page: page)
        return try await r.readLines(seg, head: head, tta: 3, onLine: nil)
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

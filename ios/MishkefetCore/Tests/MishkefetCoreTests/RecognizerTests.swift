import CoreGraphics
import XCTest
@testable import MishkefetCore

/// End to end with the real fp16 backbone on ONNX Runtime (skipped without the model).
final class RecognizerTests: XCTestCase {
    /// Mean CER against lines.json `read` (TTA 3) at most 0.02.
    func testLineReadMatchesGolden() async throws {
        let r = try await Fixtures.recognizer()
        var cers: [Double] = []
        var identical = 0
        for line in try Fixtures.lines() {
            let g = try Fixtures.gray(line.file)
            let res = try await r.readLine(g, head: nil, tta: 3)
            let cer = Metrics.cer(hyp: res.text, ref: line.read)
            cers.append(cer)
            if res.text == line.read { identical += 1 }
            XCTAssertEqual(res.confidence, line.confidence, accuracy: 0.01, line.file)
            if cer > 0 { print("  \(line.file) cer \(String(format: "%.3f", cer))\n    got  \(res.text)\n    want \(line.read)") }
        }
        let mean = cers.reduce(0, +) / Double(cers.count)
        print(String(format: "line read: mean CER %.4f vs golden, %d/%d identical", mean, identical, cers.count))
        XCTAssertLessThanOrEqual(mean, 0.02)
    }

    /// Features shape and TTA-1 beam text against lines.json `beam`.
    func testFeaturesAndBeam() async throws {
        let r = try await Fixtures.recognizer()
        var cers: [Double] = []
        for line in try Fixtures.lines() {
            let g = try Fixtures.gray(line.file)
            let f = try await r.features(of: g)
            XCTAssertEqual(f.frames, line.frames, line.file)
            XCTAssertEqual(f.values.count, line.frames * 512)
            let lp = r.baseHead.logprobs(features: f.values, frames: f.frames)
            let golden = try Fixtures.readFloat32LE(Fixtures.goldenFile("logprobs/\(Fixtures.stem(line.file)).f32"))
            let maxDiff = zip(lp, golden).map { abs($0 - $1) }.max() ?? 0
            XCTAssertLessThan(maxDiff, 0.5, line.file)
            let one = try await r.readLine(g, head: nil, tta: 1)
            cers.append(Metrics.cer(hyp: one.text, ref: line.beam))
        }
        print(String(format: "tta 1 beam: mean CER %.4f vs golden", cers.reduce(0, +) / Double(cers.count)))
    }

    private func readPage2(_ img: CGImage, label: String) async throws -> Double {
        let r = try await Fixtures.recognizer()
        let page = try XCTUnwrap(try Fixtures.pages().first { $0.file.hasSuffix("page2.jpg") })
        let t0 = DispatchTime.now()
        let res = try await r.readPage(img, head: nil, tta: 3, onLine: nil)
        let ms = Fixtures.ms(t0)
        XCTAssertEqual(res.segmentation.angle, page.angle)
        XCTAssertNotNil(res.segmentation.deskewed)
        XCTAssertEqual(res.segmentation.boxes, page.boxes)
        XCTAssertEqual(res.lines.count, page.lines.count)
        XCTAssertEqual(res.lines.map(\.box), res.segmentation.boxes.map { Optional($0) })
        let n = min(res.lines.count, page.lines.count)
        let cers = (0..<n).map { Metrics.cer(hyp: res.lines[$0].text, ref: page.lines[$0]) }
        for i in 0..<n where cers[i] > 0 {
            print("  line \(i) cer \(String(format: "%.3f", cers[i]))\n    got  \(res.lines[i].text)\n    want \(page.lines[i])")
        }
        let mean = cers.reduce(0, +) / Double(max(n, 1))
        print(String(format: "page2 read (%@): %d lines, %d identical, mean CER %.4f vs golden, %.0f ms",
                     label, res.lines.count, cers.filter { $0 == 0 }.count, mean, ms))
        return mean
    }

    /// Page 2 from PIL's own JPEG decode (Tests/Fixtures/page2-pil-gray.png, the reference's
    /// `to_gray` of page2.jpg saved losslessly): the engine alone, mean CER at most 0.05.
    func testPageReadFromReferenceGray() async throws {
        let url = Fixtures.root.appendingPathComponent("ios/MishkefetCore/Tests/Fixtures/page2-pil-gray.png")
        let mean = try await readPage2(try Fixtures.cgImage(url), label: "reference gray")
        XCTAssertLessThanOrEqual(mean, 0.05)
    }

    /// Page 2 decoded by ImageIO, as the app does. Apple's JPEG decoder differs from libjpeg
    /// by up to 2 levels on ~2% of pixels, which flips TTA choices on a few short, noisy lines
    /// of this out-of-domain page (measured 0.051), hence the looser bound.
    func testPageReadFromJPEG() async throws {
        let img = try Fixtures.cgImage(Fixtures.fixtures.appendingPathComponent("pages/page2.jpg"))
        let mean = try await readPage2(img, label: "ImageIO JPEG")
        XCTAssertLessThanOrEqual(mean, 0.08)
    }

    func testMissingModel() {
        let files = ModelFiles(directory: URL(fileURLWithPath: "/nonexistent"))
        XCTAssertEqual(files.missing.count, 5)
        XCTAssertThrowsError(try Recognizer(files: files))
    }

    func testFineTunedHeadIsUsed() async throws {
        let r = try await Fixtures.recognizer()
        var head = r.baseHead
        // Make the blank win everywhere: the line reads empty.
        for k in 0..<head.classes { head.bias[k] = k == 0 ? 1000 : -1000 }
        let g = try Fixtures.gray(try Fixtures.lines()[0].file)
        let res = try await r.readLine(g, head: head, tta: 3)
        XCTAssertEqual(res.text, "")
    }
}

/// Timings (fp16 backbone, CPU EP, 4 threads). Run with `-c release` for representative numbers.
final class PerformanceTests: XCTestCase {
    func testLineAndPageTimings() async throws {
        let r = try await Fixtures.recognizer()
        let lines = try Fixtures.lines()
        let grays = try lines.map { try Fixtures.gray($0.file) }
        _ = try await r.readLine(grays[0], head: nil, tta: 3)  // warm-up
        var t0 = DispatchTime.now()
        for g in grays { _ = try await r.readLine(g, head: nil, tta: 3) }
        let perLine = Fixtures.ms(t0) / Double(grays.count)
        let meanWidth = Double(lines.map(\.width).reduce(0, +)) / Double(lines.count)

        t0 = DispatchTime.now()
        for g in grays { _ = Preprocess.run(g) }
        let pre = Fixtures.ms(t0) / Double(grays.count)
        t0 = DispatchTime.now()
        for g in grays { _ = try await r.features(of: g) }
        let backbone = Fixtures.ms(t0) / Double(grays.count) - pre

        let lps = try lines.map { try Fixtures.readFloat32LE(Fixtures.goldenFile("logprobs/\(Fixtures.stem($0.file)).f32")) }
        t0 = DispatchTime.now()
        for (lp, line) in zip(lps, lines) { _ = await r.decode(logprobs: lp, frames: line.frames, beam: true) }
        let beam = Fixtures.ms(t0) / Double(lines.count)

        let page = try Fixtures.cgImage(Fixtures.fixtures.appendingPathComponent("pages/page1.jpg"))
        t0 = DispatchTime.now()
        let seg = try await r.segment(page: page)
        let segMs = Fixtures.ms(t0)
        print(String(format: "perf: %.0f ms per line (tta 3, mean model width %.0f); per view: preprocess %.1f ms, "
            + "backbone %.0f ms, beam %.1f ms; page segmentation %.0f ms (%d lines, 1099x1600)",
                     perLine, meanWidth, pre, backbone, beam, segMs, seg.boxes.count))
    }
}

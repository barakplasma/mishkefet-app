import XCTest
@testable import MishkefetCore

final class DecoderTests: XCTestCase {
    private func goldenLogprobs(_ line: Fixtures.Line) throws -> [Float] {
        let lp = try Fixtures.readFloat32LE(Fixtures.goldenFile("logprobs/\(Fixtures.stem(line.file)).f32"))
        XCTAssertEqual(lp.count, line.frames * 198)
        return lp
    }

    /// Identical greedy and beam text on the golden log-probs.
    func testGreedyAndBeamMatchGolden() throws {
        let decoder = try Fixtures.decoder()
        let charset = try Fixtures.charset
        var mismatches = 0
        for line in try Fixtures.lines() {
            let lp = try goldenLogprobs(line)
            let greedy = charset.decode(Decoder.greedy(lp, frames: line.frames, classes: 198))
            let beam = charset.decode(decoder.beamSearch(lp, frames: line.frames, classes: 198))
            XCTAssertEqual(greedy, line.greedy, "greedy \(line.file)")
            XCTAssertEqual(beam, line.beam, "beam \(line.file)")
            if beam != line.beam { mismatches += 1 }
        }
        print("decoder: \(mismatches) beam mismatches")
    }

    func testEmittingConfidence() throws {
        for line in try Fixtures.lines() {
            let lp = try goldenLogprobs(line)
            let c = Decoder.confidence(lp, frames: line.frames, classes: 198, emittingOnly: true)
            XCTAssertEqual(c, line.emitting_confidence, accuracy: 1e-9, line.file)
        }
    }

    func testRecognizerDecodeUsesBeam() async throws {
        let files = try Fixtures.modelFiles()
        let r = try Recognizer(files: files)
        let line = try Fixtures.lines()[1]
        let lp = try goldenLogprobs(line)
        let beam = await r.decode(logprobs: lp, frames: line.frames, beam: true)
        let greedy = await r.decode(logprobs: lp, frames: line.frames, beam: false)
        XCTAssertEqual(beam, line.beam)
        XCTAssertEqual(greedy, line.greedy)
    }

    func testTopKTiesToLowerIndex() {
        let row: [Float] = [0.5, 1, 1, 0.2, 1, 0.9]
        let k = row.withUnsafeBufferPointer { Decoder.topK($0.baseAddress!, classes: row.count, k: 4) }
        XCTAssertEqual(k, [1, 2, 4, 5])
    }

    func testLse() {
        XCTAssertEqual(Decoder.lse(-.infinity, 2), 2)
        XCTAssertEqual(Decoder.lse(3, -.infinity), 3)
        XCTAssertEqual(Decoder.lse(log(0.25), log(0.5)), log(0.75), accuracy: 1e-15)
    }

    func testLastWord() {
        XCTAssertEqual(Array(Decoder.lastWord([1, 2, 38, 4, 5])), [4, 5])
        XCTAssertEqual(Array(Decoder.lastWord([1, 2])), [1, 2])
        XCTAssertEqual(Array(Decoder.lastWord([1, 38])), [])
    }
}

final class LanguageModelTests: XCTestCase {
    private struct CharCase: Decodable {
        let context: [Int]
        let char: Int
        let logprob: Double
    }

    private struct WordCase: Decodable {
        let word: [Int]
        let text: String
        let bonus: Double
    }

    private struct Vectors: Decodable {
        let char: [CharCase]
        let word: [WordCase]
    }

    func testMatchesGolden() throws {
        let d = try Fixtures.decoder()
        let v = try Fixtures.decode(Vectors.self, "lm.json")
        XCTAssertFalse(v.char.isEmpty)
        for c in v.char {
            let got = d.charLM!.logprob(c.context.map(UInt8.init)[...], UInt8(c.char))
            XCTAssertEqual(got, c.logprob, accuracy: 1e-4, "\(c.context) \(c.char)")
        }
        let charset = try Fixtures.charset
        for w in v.word {
            XCTAssertEqual(charset.encode(w.text), w.word)
            XCTAssertEqual(d.words!.bonus(w.word.map(UInt8.init)[...]), w.bonus, accuracy: 1e-4, w.text)
        }
    }
}

final class CharsetTests: XCTestCase {
    func testEncodeDecode() throws {
        let c = try Fixtures.charset
        XCTAssertEqual(c.count, 198)
        XCTAssertEqual(c.symbols[Charset.space], " ")
        XCTAssertEqual(c.encode("של ום"), [26, 13, 38, 6, 15])
        XCTAssertEqual(c.encode("a\u{05B8}b"), [70, 71])  // niqqud is not in the charset
        XCTAssertEqual(c.decode([26, 13, 0, 38]), "של ")
        XCTAssertEqual(try Charset.load(Fixtures.spec.appendingPathComponent("charset.json")).count, 198)
    }
}

final class MetricsTests: XCTestCase {
    func testEditDistanceAndAccuracy() {
        XCTAssertEqual(Metrics.editDistance(Array("kitten"), Array("sitting")), 3)
        XCTAssertEqual(Metrics.editDistance([], Array("abc")), 3)
        XCTAssertEqual(Metrics.characterAccuracy(pairs: [(hyp: "abc", ref: "abd"), (hyp: "", ref: "x")]), 0.5)
        XCTAssertEqual(Metrics.characterAccuracy(pairs: []), 1)
    }
}

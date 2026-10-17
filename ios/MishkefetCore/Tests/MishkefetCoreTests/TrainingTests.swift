import XCTest
@testable import MishkefetCore

final class CTCTests: XCTestCase {
    private struct Case: Decodable {
        let T: Int
        let C: Int
        let label: [Int]
        let logits: [Float]
        let loss: Double
        let grad: [Float]
    }

    private func check(forceLogDomain: Bool) throws {
        let cases = try Fixtures.decode([Case].self, "ctc.json")
        XCTAssertEqual(cases.count, 5)
        for c in cases {
            let r = c.logits.withUnsafeBufferPointer {
                CTC.lossAndGradient($0, frames: c.T, classes: c.C, label: c.label, forceLogDomain: forceLogDomain)
            }
            guard let r else {
                // infeasible: the reference reports loss 0 and a zero gradient
                XCTAssertEqual(c.loss, 0, "T \(c.T) label \(c.label)")
                XCTAssertTrue(c.grad.allSatisfy { $0 == 0 })
                continue
            }
            XCTAssertEqual(r.loss, c.loss, accuracy: 1e-4 * abs(c.loss), "loss T \(c.T)")
            for (a, b) in zip(r.grad, c.grad) {
                XCTAssertEqual(a, b, accuracy: 1e-4 * max(abs(b), 1e-2), "grad T \(c.T)")
            }
        }
    }

    func testMatchesGolden() throws { try check(forceLogDomain: false) }
    func testLogDomainMatchesGolden() throws { try check(forceLogDomain: true) }
}

final class HeadTrainerTests: XCTestCase {
    private struct Sample: Decodable {
        let features: [Float]
        let T: Int
        let label: [Int]
    }

    private struct Run: Decodable {
        let rank: Int
        let steps: Int
        let lr: Double
        let l2: Double
        let losses: [Double]
        let W: [Float]
        let b: [Float]
    }

    private struct Vectors: Decodable {
        let D: Int
        let C: Int
        let W0: [Float]
        let b0: [Float]
        let samples: [Sample]
        let runs: [Run]
    }

    /// The rank-0 (head only) run of golden/adapter.json.
    func testMatchesGolden() throws {
        let v = try Fixtures.decode(Vectors.self, "adapter.json")
        let run = try XCTUnwrap(v.runs.first { $0.rank == 0 })
        let base = Head(classes: v.C, dim: v.D, weight: v.W0, bias: v.b0)
        let samples = v.samples.map { TrainingSample(features: $0.features, frames: $0.T, label: $0.label) }
        let config = AdaptConfig(steps: run.steps, lr: run.lr, l2: run.l2)
        var losses: [Double] = []
        let head = HeadTrainer.train(base: base, samples: samples, config: config) { _, l in losses.append(l) }
        XCTAssertEqual(losses.count, run.losses.count)
        for (a, b) in zip(losses, run.losses) { XCTAssertEqual(a, b, accuracy: 1e-3 * abs(b)) }
        for (a, b) in zip(head.weight, run.W) { XCTAssertEqual(a, b, accuracy: 1e-4) }
        for (a, b) in zip(head.bias, run.b) { XCTAssertEqual(a, b, accuracy: 1e-4) }
    }

    func testShippedConfig() throws {
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: Fixtures.spec.appendingPathComponent("adapt.json")))
            as! [String: Any]
        let s = AdaptConfig.shipped
        XCTAssertEqual(s.steps, json["steps"] as? Int)
        XCTAssertEqual(s.lr, json["lr"] as? Double)
        XCTAssertEqual(s.beta1, json["beta1"] as? Double)
        XCTAssertEqual(s.beta2, json["beta2"] as? Double)
        XCTAssertEqual(s.eps, json["eps"] as? Double)
        XCTAssertEqual(s.l2, json["l2"] as? Double)
        XCTAssertEqual(s.roundSize, json["round_size"] as? Int)
        XCTAssertEqual(s.inactivityMs, json["inactivity_ms"] as? Int)
        XCTAssertEqual(s.augmentedViews, (json["aug_views"] as? [String: Int])?["ios"])
        XCTAssertEqual(s.maxTrainSamples, json["max_train_samples"] as? Int)
        XCTAssertEqual(s.heldOutModulo, json["held_out_modulo"] as? Int)
        XCTAssertEqual(s.acceptIfAccuracyDropAtMost, json["accept_if_accuracy_drop_at_most"] as? Double)
    }

    func testHeadRoundTrip() throws {
        let h = Head(classes: 3, dim: 2, weight: [1, 2, 3, 4, 5, 6], bias: [-1, 0, 1])
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("head-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }
        try h.write(to: url)
        XCTAssertEqual(try Head(contentsOf: url), h)
        XCTAssertEqual(try Data(contentsOf: url).count, 12 + 4 * 9)
    }

    /// 40 samples x 5 views x ~60 frames, 100 steps, real head size.
    func testTrainingSpeed() throws {
        var rng = Mulberry32(seed: 7)
        let C = 198, D = 512
        let base = Head(classes: C, dim: D, weight: (0..<(C * D)).map { _ in Float(rng.next() - 0.5) * 0.1 },
                        bias: (0..<C).map { _ in Float(rng.next() - 0.5) * 0.1 })
        let samples = (0..<200).map { _ -> TrainingSample in
            let T = 50 + Int(rng.next() * 20)
            let label = (0..<(4 + Int(rng.next() * 6))).map { _ in 1 + Int(rng.next() * 37) }
            return TrainingSample(features: (0..<(T * D)).map { _ in Float(rng.next() * 2 - 1) }, frames: T, label: label)
        }
        let t0 = DispatchTime.now()
        var first = 0.0, last = 0.0
        _ = HeadTrainer.train(base: base, samples: samples, config: .shipped) { step, loss in
            if step == 1 { first = loss }
            last = loss
        }
        let s = Fixtures.ms(t0) / 1000
        print(String(format: "head training: 200 views x ~60 frames x 100 steps in %.2f s (loss %.3f -> %.3f)", s, first, last))
        XCTAssertLessThan(last, first)
    }
}

final class StrokeTests: XCTestCase {
    private struct Render: Decodable {
        let seed: UInt32
        let params: RenderParams
        let height: Int
        let width: Int
        let sum: Int
        let ink_pixels: Int
    }

    private struct StrokeSet: Decodable {
        let word: String
        let strokes: [[[Double]]]
        let renders: [Render]
    }

    private struct Vectors: Decodable {
        let mulberry32_seed42: [Double]
        let sets: [StrokeSet]
    }

    func testMulberry32() throws {
        let v = try Fixtures.decode(Vectors.self, "strokes.json")
        var r = Mulberry32(seed: 42)
        for x in v.mulberry32_seed42 { XCTAssertEqual(r.next(), x) }
    }

    func testFnv1a32() {
        XCTAssertEqual(fnv1a32(""), 0x811C_9DC5)
        XCTAssertEqual(fnv1a32("a"), 0xE40C_292C)
        XCTAssertEqual(fnv1a32("foobar"), 0xBF9C_F968)
        let base: UInt32 = (0xE40C_292C & 0xFF_FFFF) * 16
        XCTAssertEqual(RenderParams.trainingSeeds(sampleID: "a", augmented: 2), [0, base + 1, base + 2])
    }

    func testRendersMatchGolden() throws {
        let v = try Fixtures.decode(Vectors.self, "strokes.json")
        for set in v.sets {
            let strokes = set.strokes.map { $0.map { CGPoint(x: $0[0], y: $0[1]) } }
            for r in set.renders {
                let p = RenderParams.augmented(seed: r.seed)
                XCTAssertEqual(p.slant, r.params.slant, accuracy: 1e-12)
                XCTAssertEqual(p.rotate, r.params.rotate, accuracy: 1e-12)
                XCTAssertEqual(p.radius, r.params.radius, accuracy: 1e-12)
                XCTAssertEqual(p.density, r.params.density, accuracy: 1e-12)
                XCTAssertEqual(p.ink, r.params.ink, accuracy: 1e-12)
                let img = try XCTUnwrap(StrokeRenderer.render(strokes, nChars: set.word.count, params: p))
                XCTAssertEqual(img.height, r.height)
                XCTAssertEqual(img.width, r.width, "\(set.word) seed \(r.seed)")
                let sum = img.pixels.reduce(0) { $0 + Int($1) }
                XCTAssertEqual(Double(sum), Double(r.sum), accuracy: 0.005 * Double(r.sum), "\(set.word) seed \(r.seed)")
                XCTAssertEqual(sum, r.sum, "exact sum \(set.word) seed \(r.seed)")
            }
        }
        XCTAssertNil(StrokeRenderer.render([], nChars: 1))
        XCTAssertNil(StrokeRenderer.render([[]], nChars: 1))
    }

    func testSinglePointStroke() throws {
        let img = try XCTUnwrap(StrokeRenderer.render([[CGPoint(x: 5, y: 5)]], nChars: 1))
        XCTAssertEqual(img.height, 128)
        XCTAssertTrue(img.pixels.contains { $0 < 128 })
    }
}

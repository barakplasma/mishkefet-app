import CoreGraphics
import Foundation
import MishkefetCore

/// A training sample as plain values, for work off the main actor.
struct SampleInput: Sendable {
    var id: String
    var word: String
    var strokes: [[CGPoint]]?
    var lineImageURL: URL?
    var heldOut: Bool

    @MainActor
    init(_ s: Sample) {
        id = s.id
        word = s.word
        strokes = s.strokes
        lineImageURL = s.lineImageURL
        heldOut = s.heldOut
    }

    init(id: String, word: String, strokes: [[CGPoint]]?, lineImageURL: URL? = nil, heldOut: Bool = false) {
        self.id = id
        self.word = word
        self.strokes = strokes
        self.lineImageURL = lineImageURL
        self.heldOut = heldOut
    }
}

enum RefitStage: Int, Sendable {
    case collect, tune, test, done
}

struct RefitResult: Equatable, Sendable {
    var round: Int
    var accepted: Bool
    var before: Double
    var after: Double
}

/// One fine-tuning round (spec/adapter.md): features for every training view (cached), train the
/// head, measure the previous and the new head on the held-out canvas samples, keep the new head
/// only if it is not worse.
enum Refit {
    static func run(engine: ReadingEngine, round: Int, samples: [SampleInput], previous: Head?,
                    config: AdaptConfig = .shipped,
                    progress: @escaping @Sendable (RefitStage, Double) -> Void) async throws -> (result: RefitResult, head: Head?) {
        let train = Array(samples.filter { !$0.heldOut }.suffix(config.maxTrainSamples))
        let held = samples.filter { $0.heldOut && $0.strokes != nil }

        progress(.collect, 0)
        var views: [TrainingSample] = []
        for (i, s) in train.enumerated() {
            try Task.checkCancellation()
            views += try await trainingViews(of: s, engine: engine, config: config)
            progress(.collect, Double(i + 1) / Double(max(1, train.count)))
        }
        guard !views.isEmpty else { throw CancellationError() }

        progress(.tune, 0)
        let head = try await engine.train(samples: views, config: config) { step, steps in
            progress(.tune, Double(step) / Double(max(1, steps)))
        }

        progress(.test, 0)
        var before = 1.0, after = 1.0
        if !held.isEmpty {
            var pairsBefore: [(hyp: String, ref: String)] = []
            var pairsAfter: [(hyp: String, ref: String)] = []
            for (i, s) in held.enumerated() {
                try Task.checkCancellation()
                guard let strokes = s.strokes else { continue }
                let b = try await engine.readStrokes(strokes, nChars: s.word.count, head: previous)
                let a = try await engine.readStrokes(strokes, nChars: s.word.count, head: head)
                pairsBefore.append((b.trimmingCharacters(in: .whitespaces), s.word))
                pairsAfter.append((a.trimmingCharacters(in: .whitespaces), s.word))
                progress(.test, Double(i + 1) / Double(held.count))
            }
            before = Metrics.characterAccuracy(pairs: pairsBefore)
            after = Metrics.characterAccuracy(pairs: pairsAfter)
        }
        let accepted = after >= before - config.acceptIfAccuracyDropAtMost
        progress(.done, 1)
        return (RefitResult(round: round, accepted: accepted, before: before, after: after), accepted ? head : nil)
    }

    /// Plain + augmented renders of a canvas sample, or the one crop of a corrected scan line.
    static func trainingViews(of s: SampleInput, engine: ReadingEngine, config: AdaptConfig) async throws -> [TrainingSample] {
        let seeds: [UInt32] = s.strokes != nil
            ? RenderParams.trainingSeeds(sampleID: s.id, augmented: config.augmentedViews)
            : [0]
        var out: [TrainingSample] = []
        var lineImage: CGImage?
        if s.strokes == nil {
            guard let url = s.lineImageURL, let img = Images.load(url) else { return [] }
            lineImage = img
        }
        for seed in seeds {
            if let cached = FeatureCache.read(sampleID: s.id, seed: seed) {
                out.append(cached)
                continue
            }
            if let v = try await engine.trainingView(strokes: s.strokes, lineImage: lineImage, word: s.word, seed: seed) {
                FeatureCache.write(v, sampleID: s.id, seed: seed)
                out.append(v)
            }
        }
        return out
    }
}

import CoreGraphics
import Foundation
import MishkefetCore
import SwiftData

enum ScanKind: String, Codable {
    case line, page
}

/// Which model produced a reading: the general one ("base") or the fine-tuned head of a round ("r3").
enum ModelID {
    static let base = "base"
    static func round(_ n: Int) -> String { "r\(n)" }
    static func roundNumber(of id: String) -> Int? {
        guard id.hasPrefix("r") else { return nil }
        return Int(id.dropFirst())
    }
}

/// One reading of a scan by one model.
struct Reading: Codable, Hashable {
    var model: String
    var lines: [LineResult]
    var createdAt: Date
    /// The user corrected some lines by hand.
    var edited: Bool = false
}

@Model
final class Scan {
    @Attribute(.unique) var id: UUID
    var title: String
    var createdAt: Date
    var kindRaw: String
    /// File names inside `Storage.scansDirectory`. Page scans keep the deskewed page the boxes refer to.
    var imageFileName: String
    var thumbFileName: String
    var width: Int
    var height: Int
    /// JSON-encoded `[Reading]`.
    var readingsData: Data
    /// Index into `readings` of the one shown.
    var current: Int
    /// Re-read after fine-tuning and the new reading kept.
    var reread: Bool

    init(id: UUID = UUID(), title: String, createdAt: Date = Date(), kind: ScanKind, imageFileName: String,
         thumbFileName: String, width: Int, height: Int, readings: [Reading], current: Int = 0, reread: Bool = false) {
        self.id = id
        self.title = title
        self.createdAt = createdAt
        self.kindRaw = kind.rawValue
        self.imageFileName = imageFileName
        self.thumbFileName = thumbFileName
        self.width = width
        self.height = height
        self.readingsData = (try? JSONEncoder().encode(readings)) ?? Data()
        self.current = current
        self.reread = reread
    }

    var kind: ScanKind {
        get { ScanKind(rawValue: kindRaw) ?? .page }
        set { kindRaw = newValue.rawValue }
    }

    var readings: [Reading] {
        get { (try? JSONDecoder().decode([Reading].self, from: readingsData)) ?? [] }
        set { readingsData = (try? JSONEncoder().encode(newValue)) ?? Data() }
    }

    var reading: Reading? {
        let all = readings
        guard !all.isEmpty else { return nil }
        return all[min(max(0, current), all.count - 1)]
    }

    var lines: [LineResult] { reading?.lines ?? [] }

    var preview: String {
        lines.first(where: { !$0.text.trimmingCharacters(in: .whitespaces).isEmpty })?.text ?? ""
    }

    /// Mean confidence of the shown reading. Only orders the compare picker; never displayed.
    var meanConfidence: Double {
        let ls = lines
        return ls.isEmpty ? 1 : ls.reduce(0) { $0 + $1.confidence } / Double(ls.count)
    }

    var imageURL: URL { Storage.scansDirectory.appendingPathComponent(imageFileName) }
    var thumbURL: URL { Storage.scansDirectory.appendingPathComponent(thumbFileName) }
}

/// A training sample: a word written on the canvas, or a scan line the user corrected.
@Model
final class Sample {
    @Attribute(.unique) var id: String
    var createdAt: Date
    var round: Int
    var word: String
    /// JSON-encoded strokes, `[[[x, y]]]` in canvas points; nil for a scan line.
    var strokesData: Data?
    /// A corrected scan line's crop (PNG inside `Storage.samplesDirectory`).
    var lineImageFileName: String?
    var readAs: String
    var correct: Bool
    var heldOut: Bool

    init(id: String = UUID().uuidString, createdAt: Date = Date(), round: Int, word: String,
         strokes: [[CGPoint]]? = nil, lineImageFileName: String? = nil, readAs: String = "", correct: Bool = false,
         heldOut: Bool? = nil) {
        self.id = id
        self.createdAt = createdAt
        self.round = round
        self.word = word
        self.strokesData = strokes.map(StrokeCoding.encode)
        self.lineImageFileName = lineImageFileName
        self.readAs = readAs
        self.correct = correct
        // Only canvas samples are held out (they are what a round is measured on).
        self.heldOut = heldOut ?? (strokes != nil && AdaptConfig.shipped.isHeldOut(sampleID: id))
    }

    var strokes: [[CGPoint]]? {
        strokesData.flatMap(StrokeCoding.decode)
    }

    var lineImageURL: URL? {
        lineImageFileName.map { Storage.samplesDirectory.appendingPathComponent($0) }
    }
}

enum StrokeCoding {
    static func encode(_ strokes: [[CGPoint]]) -> Data {
        let arr = strokes.map { $0.map { [($0.x * 10).rounded() / 10, ($0.y * 10).rounded() / 10] } }
        return (try? JSONEncoder().encode(arr)) ?? Data()
    }

    static func decode(_ data: Data) -> [[CGPoint]]? {
        guard let arr = try? JSONDecoder().decode([[[Double]]].self, from: data) else { return nil }
        return arr.map { $0.compactMap { $0.count >= 2 ? CGPoint(x: $0[0], y: $0[1]) : nil } }
    }
}

enum Storage {
    private static func dir(_ name: String, base: FileManager.SearchPathDirectory = .applicationSupportDirectory) -> URL {
        var root = FileManager.default.urls(for: base, in: .userDomainMask)[0]
        // DEBUG demo launches (-seedDemo) keep their files apart from the real ones.
        if LaunchOptions.current.seedDemo { root = root.appendingPathComponent("Demo", isDirectory: true) }
        let d = root.appendingPathComponent(name, isDirectory: true)
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }

    /// Scan images and thumbnails (kept in backups: the user's own pages).
    static let scansDirectory = dir("Scans")
    /// Corrected scan-line crops used as training samples.
    static let samplesDirectory = dir("Samples")
    /// Accepted fine-tuned heads, head-r<round>.bin.
    static let headsDirectory = dir("Heads")
    /// Backbone features per training view; regenerable.
    static let featuresDirectory = dir("Features", base: .cachesDirectory)

    static func headURL(round: Int) -> URL {
        headsDirectory.appendingPathComponent("head-r\(round).bin")
    }

    static func remove(_ url: URL?) {
        guard let url else { return }
        try? FileManager.default.removeItem(at: url)
    }

    static func removeAll(in dir: URL) {
        for f in (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? [] {
            try? FileManager.default.removeItem(at: f)
        }
    }
}

/// The app's settings (UserDefaults).
struct Settings: Equatable {
    var seenIntro = false
    /// Rounds completed (accepted or not).
    var rounds = 0
    /// Round of the head in use, or 0 for the general model.
    var tunedRound = 0
    /// The user's choice for new scans; defaults to the fine-tuned model when one exists.
    var preferTuned = true

    static func load(_ d: UserDefaults = .standard) -> Settings {
        Settings(seenIntro: d.bool(forKey: "seenIntro"),
                 rounds: d.integer(forKey: "rounds"),
                 tunedRound: d.integer(forKey: "tunedRound"),
                 preferTuned: d.object(forKey: "preferTuned") as? Bool ?? true)
    }

    func save(_ d: UserDefaults = .standard) {
        d.set(seenIntro, forKey: "seenIntro")
        d.set(rounds, forKey: "rounds")
        d.set(tunedRound, forKey: "tunedRound")
        d.set(preferTuned, forKey: "preferTuned")
    }
}

/// Backbone features of one training view, cached on disk so later rounds only pay for new samples.
enum FeatureCache {
    static func url(sampleID: String, seed: UInt32) -> URL {
        Storage.featuresDirectory.appendingPathComponent("\(sampleID)-\(seed).feat")
    }

    /// Layout: frames (Int32), label count (Int32), labels (Int32 each), features (Float32), little-endian.
    static func read(sampleID: String, seed: UInt32) -> TrainingSample? {
        guard let data = try? Data(contentsOf: url(sampleID: sampleID, seed: seed)), data.count >= 8 else { return nil }
        return data.withUnsafeBytes { raw -> TrainingSample? in
            let frames = Int(raw.loadUnaligned(fromByteOffset: 0, as: Int32.self))
            let n = Int(raw.loadUnaligned(fromByteOffset: 4, as: Int32.self))
            guard frames >= 0, n >= 0, data.count >= 8 + n * 4 else { return nil }
            let label = (0..<n).map { Int(raw.loadUnaligned(fromByteOffset: 8 + $0 * 4, as: Int32.self)) }
            let start = 8 + n * 4
            let count = (data.count - start) / 4
            var features = [Float](repeating: 0, count: count)
            features.withUnsafeMutableBytes { dst in
                dst.copyMemory(from: UnsafeRawBufferPointer(rebasing: raw[start..<(start + count * 4)]))
            }
            return TrainingSample(features: features, frames: frames, label: label)
        }
    }

    static func write(_ s: TrainingSample, sampleID: String, seed: UInt32) {
        var d = Data()
        func i32(_ v: Int) { withUnsafeBytes(of: Int32(v).littleEndian) { d.append(contentsOf: $0) } }
        i32(s.frames)
        i32(s.label.count)
        s.label.forEach(i32)
        s.features.withUnsafeBytes { d.append(contentsOf: $0) }
        try? d.write(to: url(sampleID: sampleID, seed: seed), options: .atomic)
    }
}

/// What each accepted round measured (kept beside the head files).
struct RoundRecord: Codable, Equatable {
    var round: Int
    var createdAt: Date
    var accBefore: Double
    var accAfter: Double
    var samples: Int
}

enum HeadStore {
    private static var indexURL: URL { Storage.headsDirectory.appendingPathComponent("rounds.json") }

    static func records() -> [RoundRecord] {
        guard let d = try? Data(contentsOf: indexURL) else { return [] }
        return (try? JSONDecoder().decode([RoundRecord].self, from: d)) ?? []
    }

    static func save(_ head: Head, record: RoundRecord) throws {
        try head.write(to: Storage.headURL(round: record.round))
        var all = records().filter { $0.round != record.round }
        all.append(record)
        try JSONEncoder().encode(all).write(to: indexURL, options: .atomic)
    }

    static func head(round: Int) -> Head? {
        guard round > 0 else { return nil }
        return try? Head(contentsOf: Storage.headURL(round: round))
    }

    static func removeAll() {
        Storage.removeAll(in: Storage.headsDirectory)
    }
}

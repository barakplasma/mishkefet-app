import CoreGraphics
import MishkefetCore
import SwiftData
import XCTest
@testable import Mishkefet

final class HebrewDateTests: XCTestCase {
    private var cal: Calendar = {
        var c = Calendar(identifier: .gregorian)
        c.timeZone = TimeZone(identifier: "Asia/Jerusalem")!
        return c
    }()

    private func date(_ y: Int, _ m: Int, _ d: Int, _ h: Int = 12, _ min: Int = 0) -> Date {
        cal.date(from: DateComponents(year: y, month: m, day: d, hour: h, minute: min))!
    }

    func testTodayYesterdayAndOlder() {
        let now = date(2026, 10, 17, 10, 0)
        XCTAssertEqual(HebrewDate.label(date(2026, 10, 17, 9, 12), now: now, calendar: cal), "היום, 09:12")
        XCTAssertEqual(HebrewDate.label(date(2026, 10, 16, 18, 3), now: now, calendar: cal), "אתמול, 18:03")
        XCTAssertEqual(HebrewDate.label(date(2026, 10, 6), now: now, calendar: cal), "6 באוקטובר")
        XCTAssertEqual(HebrewDate.label(date(2026, 5, 1), now: now, calendar: cal), "1 במאי")
        XCTAssertEqual(HebrewDate.label(date(2025, 12, 31), now: now, calendar: cal), "31 בדצמבר 2025")
    }

    func testYesterdayAcrossMonthBoundary() {
        let now = date(2026, 11, 1, 0, 30)
        XCTAssertEqual(HebrewDate.label(date(2026, 10, 31, 23, 59), now: now, calendar: cal), "אתמול, 23:59")
    }

    func testTitles() {
        let now = date(2026, 10, 17, 10, 0)
        XCTAssertEqual(Titles.scan(lines: ["", "  ", "לחם חלב ביצים ועגבניות טריות"], at: now, now: now, calendar: cal), "לחם חלב ביצים ועגבניות")
        XCTAssertEqual(Titles.scan(lines: ["שלום"], at: now, now: now, calendar: cal), "שלום")
        XCTAssertEqual(Titles.scan(lines: [], at: date(2026, 10, 17, 8, 5), now: now, calendar: cal), "סריקה היום, 08:05")
        XCTAssertEqual(Titles.lines(1), "שורה אחת")
        XCTAssertEqual(Titles.lines(5), "5 שורות")
        XCTAssertEqual(Titles.linesShort(1), "שורה")
    }
}

final class TextDiffTests: XCTestCase {
    func testEditOps() {
        XCTAssertEqual(TextDiff.editOps(target: "שלום", read: "שלום").distance, 0)
        let r = TextDiff.editOps(target: "שלום", read: "שלוס")
        XCTAssertEqual(r.distance, 1)
        XCTAssertEqual(r.wrongTargetIndices, [3])
        let missing = TextDiff.editOps(target: "בית", read: "בת")
        XCTAssertEqual(missing.distance, 1)
        XCTAssertEqual(missing.wrongTargetIndices, [1])
        XCTAssertEqual(TextDiff.editOps(target: "אב", read: "").wrongTargetIndices, [0, 1])
        XCTAssertEqual(TextDiff.editOps(target: "", read: "אב").distance, 2)
    }

    func testWordDiff() {
        let before = "לחם חלב וביצים טריות"
        let after = "לחם חלב וביצים טריים מאוד"
        let a = TextDiff.words(before: before, after: after, side: .after)
        XCTAssertEqual(a.map(\.text), ["לחם", "חלב", "וביצים", "טריים", "מאוד"])
        XCTAssertEqual(a.map(\.changed), [false, false, false, true, true])
        let b = TextDiff.words(before: before, after: after, side: .before)
        XCTAssertEqual(b.map(\.changed), [false, false, false, true])
        XCTAssertTrue(TextDiff.words(before: "", after: "", side: .after).isEmpty)
        XCTAssertEqual(TextDiff.words(before: "", after: "א ב", side: .after).map(\.changed), [true, true])
    }

    func testCopyText() {
        XCTAssertEqual(ReadingText.plain([" שורה ראשונה ", "שורה שנייה"]), "שורה ראשונה\nשורה שנייה")
    }
}

final class WordPickerTests: XCTestCase {
    private func seeded(_ seed: UInt64) -> () -> Double {
        var state = seed
        return {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Double(state >> 11) / Double(1 << 53)
        }
    }

    private let pool = ["של", "על", "את", "הוא", "עם", "גם", "היא", "בין", "ישראל", "צפון", "חוץ", "קפה", "טוב", "זמן", "סוף",
                        "דרך", "שמש", "גשם", "עץ", "פרח", "ספר", "מכתב", "ילד", "בית", "כלב", "חתול", "נמל", "ענן", "ארץ", "קוף",
                        "צבע", "פעם", "גן", "ים", "חג", "זאב", "לחם", "שלום", "עוף", "כף", "מלך", "בוקר", "נץ", "דג", "רץ"]

    func testFirstRoundCoversEveryLetterBeforeRepeatingLetters() {
        var p = WordPicker(pool: pool, attempts: [], scanWords: [], random: seeded(7))
        var covered = Set<Character>()
        var picked: [String] = []
        for _ in 0..<40 {
            let w = p.next()
            picked.append(w)
            covered.formUnion(w)
            if covered.isSuperset(of: WordPicker.letters.filter { l in pool.contains { $0.contains(l) } }) { break }
        }
        let available = Set(WordPicker.letters.filter { l in pool.contains { $0.contains(l) } })
        XCTAssertTrue(covered.isSuperset(of: available), "missing: \(available.subtracting(covered))")
        XCTAssertEqual(Set(picked).count, picked.count, "no word repeats while covering letters")
    }

    func testErrorRatesFavorMisreadLetters() {
        let attempts = (0..<10).map { _ in WordPicker.Attempt(word: "שלום", readAs: "שלוס") }
        let rates = WordPicker.letterErrorRates(attempts)
        XCTAssertEqual(rates["ם"]!, 11.0 / 14.0, accuracy: 1e-9)
        XCTAssertEqual(rates["ש"]!, 1.0 / 14.0, accuracy: 1e-9)
        XCTAssertEqual(rates["א"]!, 0.25, accuracy: 1e-9)
    }

    func testAvoidsRecentWords() {
        let attempts = pool.prefix(40).map { WordPicker.Attempt(word: $0, readAs: $0) }
        var p = WordPicker(pool: pool, attempts: Array(attempts), scanWords: [], random: seeded(3))
        for _ in 0..<5 {
            XCTAssertFalse(pool.prefix(40).contains(p.next()))
        }
    }

    func testBundledPoolLoads() {
        XCTAssertGreaterThan(WordPicker.pool.count, 1000)
    }
}

final class HeldOutTests: XCTestCase {
    func testFNV1a() {
        // Reference values of FNV-1a 32.
        XCTAssertEqual(fnv1a32(""), 0x811C9DC5)
        XCTAssertEqual(fnv1a32("a"), 0xE40C292C)
        XCTAssertEqual(fnv1a32("foobar"), 0xBF9CF968)
    }

    func testHeldOutRule() {
        let ids = (0..<4000).map { "sample-\($0)" }
        let held = ids.filter { AdaptConfig.shipped.isHeldOut(sampleID: $0) }
        for id in held { XCTAssertEqual(fnv1a32(id) % 8, 0) }
        XCTAssertEqual(Double(held.count) / Double(ids.count), 1.0 / 8.0, accuracy: 0.03)
    }

    func testAugmentSeeds() {
        let id = "sample-42"
        let base = (fnv1a32(id) & 0xFFFFFF) * 16
        XCTAssertEqual(RenderParams.trainingSeeds(sampleID: id, augmented: 4), [0, base + 1, base + 2, base + 3, base + 4])
    }

    func testOnlyCanvasSamplesAreHeldOut() {
        let id = (0..<200).map { "s\($0)" }.first { fnv1a32($0) % 8 == 0 }!
        XCTAssertTrue(Sample(id: id, round: 1, word: "את", strokes: [[CGPoint(x: 0, y: 0)]]).heldOut)
        XCTAssertFalse(Sample(id: id, round: 1, word: "את", lineImageFileName: "x.png").heldOut)
    }
}

@MainActor
final class PersistenceTests: XCTestCase {
    func testScanAndSampleRoundTrip() throws {
        let container = try ModelContainer(for: Scan.self, Sample.self,
                                           configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        let ctx = container.mainContext
        let date = Date(timeIntervalSince1970: 1_790_000_000)
        let lines = [LineResult(text: "שורה ראשונה", confidence: 0.9, box: LineBox(x: 1, y: 2, w: 30, h: 8)),
                     LineResult(text: "שורה שנייה", confidence: 0.4, box: LineBox(x: 1, y: 12, w: 28, h: 8))]
        let scan = Scan(title: "בדיקה", createdAt: date, kind: .page, imageFileName: "a.jpg", thumbFileName: "a-thumb.jpg",
                        width: 40, height: 30, readings: [Reading(model: ModelID.base, lines: lines, createdAt: date)])
        ctx.insert(scan)
        let strokes: [[CGPoint]] = [[CGPoint(x: 1.25, y: 2.5), CGPoint(x: 3, y: 4)], [CGPoint(x: 9, y: 9)]]
        ctx.insert(Sample(id: "s-1", round: 2, word: "שלום", strokes: strokes, readAs: "שלוס"))
        try ctx.save()

        let fetched = try XCTUnwrap(ctx.fetch(FetchDescriptor<Scan>()).first)
        XCTAssertEqual(fetched.title, "בדיקה")
        XCTAssertEqual(fetched.kind, .page)
        XCTAssertEqual(fetched.lines, lines)
        XCTAssertEqual(fetched.preview, "שורה ראשונה")
        XCTAssertEqual(fetched.meanConfidence, 0.65, accuracy: 1e-9)

        var readings = fetched.readings
        readings.append(Reading(model: ModelID.round(1), lines: [lines[0]], createdAt: date))
        fetched.readings = readings
        fetched.current = 1
        try ctx.save()
        let again = try XCTUnwrap(ctx.fetch(FetchDescriptor<Scan>()).first)
        XCTAssertEqual(again.readings.count, 2)
        XCTAssertEqual(again.reading?.model, "r1")
        XCTAssertEqual(ModelID.roundNumber(of: again.reading!.model), 1)
        XCTAssertNil(ModelID.roundNumber(of: ModelID.base))

        let sample = try XCTUnwrap(ctx.fetch(FetchDescriptor<Sample>()).first)
        XCTAssertEqual(sample.word, "שלום")
        XCTAssertEqual(sample.strokes?.count, 2)
        XCTAssertEqual(sample.strokes?[0][0].x ?? 0, 1.3, accuracy: 0.051)
        XCTAssertEqual(sample.strokes?[1][0], CGPoint(x: 9, y: 9))
    }

    func testFeatureCacheRoundTrip() {
        let s = TrainingSample(features: [0.5, -1, 2.25, 3, 4, 5], frames: 3, label: [4, 7, 38])
        FeatureCache.write(s, sampleID: "test-cache", seed: 17)
        let r = FeatureCache.read(sampleID: "test-cache", seed: 17)
        XCTAssertEqual(r?.features, s.features)
        XCTAssertEqual(r?.frames, 3)
        XCTAssertEqual(r?.label, [4, 7, 38])
        Storage.remove(FeatureCache.url(sampleID: "test-cache", seed: 17))
        XCTAssertNil(FeatureCache.read(sampleID: "test-cache", seed: 17))
    }

    func testHeadFileRoundTrip() throws {
        let head = Head(classes: 2, dim: 3, weight: [1, 2, 3, 4, 5, 6], bias: [0.5, -0.5])
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("head-test.bin")
        try head.write(to: url)
        XCTAssertEqual(try Head(contentsOf: url), head)
        try? FileManager.default.removeItem(at: url)
    }

    func testSettingsRoundTrip() {
        let d = UserDefaults(suiteName: "mishkefet-tests")!
        d.removePersistentDomain(forName: "mishkefet-tests")
        XCTAssertEqual(Settings.load(d), Settings())
        XCTAssertTrue(Settings.load(d).preferTuned)
        Settings(seenIntro: true, rounds: 3, tunedRound: 2, preferTuned: false).save(d)
        XCTAssertEqual(Settings.load(d), Settings(seenIntro: true, rounds: 3, tunedRound: 2, preferTuned: false))
        d.removePersistentDomain(forName: "mishkefet-tests")
    }
}

#if DEBUG
import Foundation
import MishkefetCore
import SwiftData
import UIKit

/// Sample content for screenshots and UI work. Only reachable through DEBUG launch arguments:
/// -seedDemo (in-memory store, demo settings), -fakeEngine, -screen <name>, -sheet <name>,
/// -readFile <path> (reads a host image with the real engine and logs the lines).
enum DemoContent {
    struct Page {
        var title: String
        var daysAgo: Int
        var hour: Int
        var minute: Int
        var lines: [String]
        var kind: ScanKind = .page
    }

    static let pages: [Page] = [
        Page(title: "רשימת קניות לשבת", daysAgo: 0, hour: 9, minute: 12,
             lines: ["לחם, חלב וביצים", "עגבניות, מלפפונים ובצל", "שמן זית וקפה טחון", "גבינה לבנה ויוגורט", "פרחים לשולחן"]),
        Page(title: "מכתב מסבתא", daysAgo: 1, hour: 18, minute: 3,
             lines: ["יקירתי, שמחתי לקבל את המכתב שלך", "הגינה פורחת השנה יותר מתמיד", "סבא שולח נשיקות ומחכה לביקור", "באהבה גדולה, סבתא"]),
        Page(title: "סיכום שיעור בהיסטוריה", daysAgo: 3, hour: 11, minute: 40,
             lines: ["העלייה הראשונה החלה בשנת 1882", "רוב העולים הגיעו ממזרח אירופה", "הוקמו מושבות חקלאיות ראשונות", "ראשון לציון, זכרון יעקב וגדרה"]),
        Page(title: "עוגת שמרים של אמא", daysAgo: 8, hour: 20, minute: 15,
             lines: ["קילו קמח, כוס סוכר וחבילת שמרים", "שתי ביצים וכוס חלב פושר", "ללוש היטב ולהניח לתפיחה שעה", "למלא בשוקולד ולאפות ארבעים דקות"]),
        Page(title: "להתקשר לרופא ביום ראשון", daysAgo: 12, hour: 8, minute: 30,
             lines: ["להתקשר לרופא ביום ראשון"], kind: .line),
    ]

    /// The tuned reading of the first page differs in a couple of words (for the compare screen).
    static let tunedFirst = ["לחם, חלב וביצים", "עגבניות, מלפפונים ובצל", "שמן זית וקפה טחון", "גבינה לבנה ויוגורט", "פרחים לשולחן"]
    static let baseFirst = ["לחם, חלב וביצים", "עגבניות, מלפפוגים ובצל", "שמן זית וקפה טחון", "גבינה לכנה ויוגורט", "פרחים לשולתן"]

    @MainActor
    static func apply(_ o: LaunchOptions, to model: AppModel) {
        var first: Scan?
        if o.seedDemo {
            Storage.removeAll(in: Storage.scansDirectory)
            let cal = Calendar.current
            for (i, p) in pages.enumerated() {
                let day = cal.date(byAdding: .day, value: -p.daysAgo, to: Date())!
                let date = cal.date(bySettingHour: p.hour, minute: p.minute, second: 0, of: day)!
                let lines = i == 0 ? tunedFirst : p.lines
                guard let page = render(lines: lines, kind: p.kind) else { continue }
                let id = UUID()
                let img = "\(id.uuidString).jpg", thumb = "\(id.uuidString)-thumb.jpg"
                try? Images.jpeg(page.image)?.write(to: Storage.scansDirectory.appendingPathComponent(img))
                try? Images.thumbnail(page.image)?.write(to: Storage.scansDirectory.appendingPathComponent(thumb))
                var readings: [Reading]
                if i == 0 {
                    readings = [
                        Reading(model: ModelID.base, lines: zip(baseFirst, page.boxes).map { LineResult(text: $0, confidence: 0.6, box: $1) }, createdAt: date),
                        Reading(model: ModelID.round(2), lines: zip(tunedFirst, page.boxes).map { LineResult(text: $0, confidence: 0.85, box: $1) }, createdAt: date),
                    ]
                } else {
                    readings = [Reading(model: ModelID.base,
                                        lines: zip(p.lines, page.boxes).map { LineResult(text: $0, confidence: 0.7 + Double(i) * 0.04, box: $1) },
                                        createdAt: date)]
                }
                let scan = Scan(id: id, title: p.title, createdAt: date, kind: p.kind, imageFileName: img, thumbFileName: thumb,
                                width: page.image.width, height: page.image.height, readings: readings,
                                current: readings.count - 1, reread: i == 0)
                model.context.insert(scan)
                if i == 0 { first = scan }
            }
            // A round in progress: 14 words written.
            let words = Array(WordPicker.pool.prefix(54))
            for (k, w) in words.enumerated() {
                let round = k < 40 ? 1 : 3
                let strokes = scribble(w, seed: k)
                FakeReadingEngine.words[FakeReadingEngine.key(StrokeCoding.decode(StrokeCoding.encode(strokes)) ?? strokes)] = w
                model.context.insert(Sample(createdAt: Date().addingTimeInterval(Double(k - 60) * 30), round: round, word: w,
                                            strokes: strokes, readAs: k % 4 == 0 ? String(w.dropLast()) : w,
                                            correct: k % 4 != 0))
            }
            try? model.context.save()
            // A pretend round-2 head for the fake engine (the real engine needs a real head).
            if o.fakeEngine { model.installDemoHead(Head(classes: 1, dim: 1, weight: [0], bias: [0]), round: 2) }
        }

        if let path = o.readFile {
            // End-to-end check of the real engine on a host image: logs MISHKEFET_LINE / MISHKEFET_DONE.
            let pics = Images.pictures(fileURL: URL(fileURLWithPath: path))
            model.readPictures(pics, kind: .page)
        }

        switch o.screen {
        case "list":
            model.screen = .list
        case "result":
            if let first { model.currentID = first.id; model.resultFrom = .list; model.screen = .result }
        case "camera":
            model.screen = .camera
        case "intro":
            model.screen = .intro
        case "train":
            model.settings.seenIntro = true
            model.screen = .train
        case "processing":
            if let page = render(lines: pages[2].lines, kind: .page) {
                model.readPictures([Picture(image: page.image, name: "")], kind: .page)
            }
        case "refit":
            model.startRefit()
            // Logs the outcome, for checking the real round from the console.
            Task { @MainActor in
                while model.refit?.stage ?? 3 < 3 { try? await Task.sleep(for: .milliseconds(250)) }
                if let r = model.refit?.result { print("MISHKEFET_REFIT accepted=\(r.accepted) before=\(r.before) after=\(r.after)") }
                if model.refit?.failed == true { print("MISHKEFET_REFIT failed") }
            }
        case "compare":
            if let first {
                let boxes = first.lines.map(\.box)
                model.currentID = first.id
                model.compare = CompareState(scanID: first.id, model: ModelID.round(2), beforeModel: ModelID.base,
                                             before: zip(baseFirst, boxes).map { LineResult(text: $0, confidence: 0.6, box: $1) },
                                             after: zip(tunedFirst, boxes).map { LineResult(text: $0, confidence: 0.85, box: $1) })
                model.screen = .compare
            }
        default:
            break
        }

        switch o.sheet {
        case "newScan": model.sheet = .newScan
        case "about": model.sheet = .about
        case "compare": model.sheet = .comparePick
        default: break
        }
    }

    // MARK: - Generated pages

    struct RenderedPage {
        var image: CGImage
        var boxes: [LineBox]
    }

    /// A ruled page written in Amatic SC (or one line for a line scan), with the line boxes.
    static func render(lines: [String], kind: ScanKind) -> RenderedPage? {
        let width: CGFloat = kind == .line ? 1400 : 1200
        let pitch: CGFloat = 150
        let top: CGFloat = kind == .line ? 40 : 160
        let height: CGFloat = kind == .line ? 220 : top + pitch * CGFloat(lines.count) + 140
        let font = UIFont(name: "AmaticSC-Bold", size: 104) ?? .systemFont(ofSize: 80)
        let ink = UIColor(red: 0.12, green: 0.16, blue: 0.38, alpha: 1)
        let para = NSMutableParagraphStyle()
        para.baseWritingDirection = .rightToLeft
        para.alignment = .right
        let attrs: [NSAttributedString.Key: Any] = [.font: font, .foregroundColor: ink, .paragraphStyle: para]
        var boxes: [LineBox] = []
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        let img = UIGraphicsImageRenderer(size: CGSize(width: width, height: height), format: format).image { ctx in
            UIColor(red: 0.98, green: 0.97, blue: 0.93, alpha: 1).setFill()
            ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
            if kind == .page {
                UIColor(red: 0.62, green: 0.75, blue: 0.9, alpha: 0.6).setFill()
                var y = top + pitch - 22
                while y < height - 60 {
                    ctx.fill(CGRect(x: 60, y: y, width: width - 120, height: 2))
                    y += pitch
                }
            }
            for (i, line) in lines.enumerated() {
                let s = NSAttributedString(string: line, attributes: attrs)
                let size = s.boundingRect(with: CGSize(width: width - 160, height: pitch), options: [.usesLineFragmentOrigin], context: nil)
                let y = top + CGFloat(i) * pitch
                let x = width - 90 - size.width + CGFloat((i * 37) % 60)
                s.draw(in: CGRect(x: x, y: y, width: size.width + 4, height: size.height))
                boxes.append(LineBox(x: Int(x), y: Int(y + size.height * 0.18), w: Int(size.width), h: Int(size.height * 0.7)))
            }
        }
        guard let cg = img.cgImage else { return nil }
        return RenderedPage(image: cg, boxes: boxes)
    }

    /// Pseudo-handwriting strokes for a word (for the refit grid).
    static func scribble(_ word: String, seed: Int) -> [[CGPoint]] {
        var out: [[CGPoint]] = []
        var x: CGFloat = CGFloat(word.count) * 34 + 20
        for (k, _) in word.enumerated() {
            var s: [CGPoint] = []
            let h = 26 + CGFloat((seed + k * 7) % 14)
            for t in 0...8 {
                let a = CGFloat(t) / 8
                s.append(CGPoint(x: x - a * 24 + sin(a * 6 + CGFloat(seed)) * 4, y: 60 - h * sin(a * .pi) + CGFloat(k % 2) * 6))
            }
            out.append(s)
            x -= 32
        }
        return out
    }
}

/// Plays back the demo readings with design-like pacing; trains nothing.
final class FakeReadingEngine: ReadingEngine, @unchecked Sendable {
    func readPage(_ page: CGImage, head: Head?,
                  onSegmented: @escaping @Sendable (CGImage, [LineBox]) -> Void,
                  onLine: @escaping @Sendable (Int, Int, LineResult) -> Void) async throws -> (page: CGImage, lines: [LineResult]) {
        try await Task.sleep(for: .milliseconds(600))
        let texts = DemoContent.pages[2].lines
        let n = texts.count
        let pitch = page.height / (n + 3)
        let boxes = (0..<n).map { LineBox(x: page.width / 12, y: pitch * ($0 + 1) + pitch / 5, w: page.width * 5 / 6, h: pitch * 3 / 5) }
        onSegmented(page, boxes)
        var lines: [LineResult] = []
        for i in 0..<n {
            try await Task.sleep(for: .milliseconds(900))
            let r = LineResult(text: texts[i], confidence: 0.8, box: boxes[i])
            lines.append(r)
            onLine(i, n, r)
        }
        return (page, lines)
    }

    func readLines(of page: CGImage, boxes: [LineBox], head: Head?) async throws -> [LineResult] {
        try await Task.sleep(for: .milliseconds(1200))
        let texts = head == nil ? DemoContent.baseFirst : DemoContent.tunedFirst
        return boxes.enumerated().map { LineResult(text: texts[$0 % texts.count], confidence: 0.8, box: $1) }
    }

    func readLine(_ image: CGImage, head: Head?) async throws -> LineResult {
        try await Task.sleep(for: .milliseconds(800))
        return LineResult(text: DemoContent.pages[4].lines[0], confidence: 0.8, box: LineBox(x: 0, y: 0, w: image.width, h: image.height))
    }

    /// Words of the demo scribbles, so the fake can "read" them.
    nonisolated(unsafe) static var words: [String: String] = [:]

    static func key(_ strokes: [[CGPoint]]) -> String {
        strokes.prefix(2).map { $0.prefix(2).map { "\(Int($0.x * 10)),\(Int($0.y * 10))" }.joined(separator: ";") }.joined(separator: "|")
    }

    func readStrokes(_ strokes: [[CGPoint]], nChars: Int, head: Head?) async throws -> String {
        try await Task.sleep(for: .milliseconds(150))
        guard let word = Self.words[Self.key(strokes)] else { return "" }
        // The general (and the demo round 2) head misreads more of the demo words than the "tuned" one.
        let h = Int(fnv1a32(word) % 100)
        let tuned = head?.weight.first == 0.5
        let wrong = tuned ? h < 12 : h < 45
        return wrong ? String(word.dropLast()) + "ו" : word
    }

    func trainingView(strokes: [[CGPoint]]?, lineImage: CGImage?, word: String, seed: UInt32) async throws -> TrainingSample? {
        try await Task.sleep(for: .milliseconds(8))
        return TrainingSample(features: [0], frames: 1, label: [1])
    }

    func train(samples: [TrainingSample], config: AdaptConfig, progress: @escaping @Sendable (Int, Int) -> Void) async throws -> Head {
        for step in 1...config.steps {
            try await Task.sleep(for: .milliseconds(25))
            progress(step, config.steps)
        }
        return Head(classes: 1, dim: 1, weight: [0.5], bias: [0])
    }

    func unload() async {}
}
#endif

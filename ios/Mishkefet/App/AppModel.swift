import Foundation
import MishkefetCore
import Observation
import PhotosUI
import SwiftData
import SwiftUI
import UniformTypeIdentifiers

enum Screen: Int {
    case home, list, camera, intro, processing, train, result, refit, compare

    /// The design's DEPTH table.
    var depth: Int {
        switch self {
        case .home: return 0
        case .list, .camera, .intro: return 1
        case .processing, .train: return 2
        case .result, .refit: return 3
        case .compare: return 4
        }
    }

    /// Screens that scale in and out instead of sliding.
    var isModal: Bool { self == .processing || self == .refit }
}

enum SheetKind: Equatable {
    case newScan
    case comparePick
    case about
}

struct Toast: Equatable {
    var id = UUID()
    var text: String
}

enum ImportTypes {
    static let all: [UTType] = [.image, .jpeg, .png, .heic, .pdf]
}

/// The compare screen: a scan re-read with the newly fine-tuned head, against its shown reading.
struct CompareState: Equatable {
    var scanID: UUID
    var model: String
    var beforeModel: String
    var before: [LineResult]
    /// nil while the re-read is running.
    var after: [LineResult]?
}

/// The refit screen's state, driven by the real round.
struct RefitState: Equatable {
    var round: Int
    var stage = 0
    var pct = 0
    var tiles: [RefitTile] = []
    var held = 0
    var result: RefitResult?
    var failed = false
}

struct RefitTile: Equatable, Identifiable {
    var id: String
    var strokes: [[CGPoint]]?
    var word: String
}

@MainActor
@Observable
final class AppModel {
    // Navigation
    var screen: Screen = .home
    var resultFrom: Screen = .list
    var currentID: UUID?
    var sheet: SheetKind?
    var toast: Toast?
    var showPhotoPicker = false
    var showFileImporter = false
    var photoSelection: [PhotosPickerItem] = []

    // Settings and the fine-tuned model
    var settings: Settings {
        didSet { if settings != oldValue { settings.save(defaults) } }
    }
    private(set) var tunedHead: Head?
    /// Words written in the current round.
    private(set) var roundCount = 0

    // Result screen
    var rereadingID: UUID?

    // Training flow
    var compare: CompareState?
    var refit: RefitState?

    let jobs: JobQueue
    let engine: ReadingEngine
    let container: ModelContainer
    var context: ModelContext { container.mainContext }

    @ObservationIgnored private let defaults: UserDefaults
    @ObservationIgnored private var toastTask: Task<Void, Never>?
    @ObservationIgnored private var refitTask: Task<Void, Never>?
    @ObservationIgnored private var compareTask: Task<Void, Never>?

    static let roundSize = AdaptConfig.shipped.roundSize

    init(container: ModelContainer, engine: ReadingEngine, defaults: UserDefaults = .standard) {
        self.container = container
        self.engine = engine
        self.defaults = defaults
        self.jobs = JobQueue(engine: engine, context: container.mainContext)
        self.settings = Settings.load(defaults)
        reloadTunedHead()
        refreshRoundCount()
    }

    // MARK: Model choice

    var hasTuned: Bool { tunedHead != nil && settings.tunedRound > 0 }
    /// What new scans are read with: the fine-tuned model by default whenever one exists.
    var useTuned: Bool { hasTuned && settings.preferTuned }
    var tunedModelID: String? { hasTuned ? ModelID.round(settings.tunedRound) : nil }

    func setUseTuned(_ v: Bool) {
        settings.preferTuned = v
    }

    func head(for modelID: String) -> Head? {
        guard let n = ModelID.roundNumber(of: modelID) else { return nil }
        if n == settings.tunedRound, let tunedHead { return tunedHead }
        return HeadStore.head(round: n)
    }

    private func reloadTunedHead() {
        tunedHead = HeadStore.head(round: settings.tunedRound)
    }

    func refreshRoundCount() {
        let round = settings.rounds + 1
        let d = FetchDescriptor<Sample>(predicate: #Predicate { $0.round == round && $0.strokesData != nil })
        roundCount = (try? context.fetchCount(d)) ?? 0
    }

    // MARK: Navigation

    func go(_ s: Screen) {
        withAnimation(Mk.ease(s.isModal || screen.isModal ? 0.6 : 0.55)) {
            sheet = nil
            screen = s
        }
    }

    func open(_ id: UUID, from: Screen) {
        currentID = id
        resultFrom = from
        go(.result)
    }

    func backFromResult() {
        go(resultFrom == .home ? .home : .list)
    }

    func scan(_ id: UUID?) -> Scan? {
        guard let id else { return nil }
        var d = FetchDescriptor<Scan>(predicate: #Predicate { $0.id == id })
        d.fetchLimit = 1
        return try? context.fetch(d).first
    }

    func present(_ s: SheetKind) {
        withAnimation(Mk.ease(0.42)) { sheet = s }
    }

    func closeSheet() {
        withAnimation(Mk.ease(0.42)) { sheet = nil }
    }

    func startTraining() {
        go(settings.seenIntro ? .train : .intro)
    }

    func beginWriting() {
        settings.seenIntro = true
        go(.train)
    }

    // MARK: Reading

    func openCamera() {
        go(.camera)
    }

    func pickPhotos() {
        closeSheet()
        Task {
            try? await Task.sleep(for: .milliseconds(sheet == nil ? 0 : 350))
            showPhotoPicker = true
        }
    }

    func pickFiles() {
        closeSheet()
        Task {
            try? await Task.sleep(for: .milliseconds(350))
            showFileImporter = true
        }
    }

    /// Pictures chosen in the photo picker, read as pages.
    func importPhotos(_ items: [PhotosPickerItem]) {
        guard !items.isEmpty else { return }
        photoSelection = []
        Task {
            var pics: [Picture] = []
            for (i, item) in items.enumerated() {
                guard let data = try? await item.loadTransferable(type: Data.self) else { continue }
                let found = await Task.detached { Images.picture(data: data, name: items.count > 1 ? "תמונה \(i + 1)" : "") }.value
                pics += found
            }
            if pics.isEmpty {
                showToast("לא הצלחתי לפתוח את התמונה")
                return
            }
            readPictures(pics, kind: .page)
        }
    }

    func importFiles(_ urls: [URL]) {
        Task {
            var pics: [Picture] = []
            for url in urls {
                let scoped = url.startAccessingSecurityScopedResource()
                let found = await Task.detached { Images.pictures(fileURL: url) }.value
                if scoped { url.stopAccessingSecurityScopedResource() }
                if found.isEmpty { showToast("לא הצלחתי לפתוח את \(url.lastPathComponent)") }
                pics += found
            }
            readPictures(pics, kind: .page)
        }
    }

    func readPictures(_ pics: [Picture], kind: ScanKind) {
        guard !pics.isEmpty, !jobs.isBusy else { return }
        let modelID = useTuned ? (tunedModelID ?? ModelID.base) : ModelID.base
        let head = modelID == ModelID.base ? nil : tunedHead
        go(.processing)
        jobs.read(pics, kind: kind, head: head, modelID: modelID) { [weak self] outcome in
            guard let self else { return }
            switch outcome {
            case let .finished(ids):
                if ids.count > 1 {
                    self.currentID = ids.last
                    self.go(.list)
                } else if let id = ids.last {
                    self.open(id, from: .home)
                }
            case .cancelled:
                self.go(.home)
            case .failed:
                self.showToast("משהו השתבש בקריאה. נסו שוב.")
                self.go(.home)
            }
        }
    }

    func cancelReading() {
        jobs.cancel()
        go(.home)
    }

    /// Reads an existing scan with another model. Page scans keep their segmentation, so the lines line up.
    func reread(_ scan: Scan, modelID: String) async throws -> [LineResult] {
        guard let image = Images.load(scan.imageURL) else { throw CocoaError(.fileReadNoSuchFile) }
        let head = self.head(for: modelID)
        if scan.kind == .line {
            var r = try await engine.readLine(image, head: head)
            r.box = LineBox(x: 0, y: 0, w: scan.width, h: scan.height)
            return [r]
        }
        let boxes = scan.lines.compactMap(\.box)
        if boxes.isEmpty {
            let lines = try await engine.readStoredPage(image, head: head)
            return lines.filter { JobQueue.keeps($0, kind: scan.kind) }
        }
        return try await engine.readLines(of: image, boxes: boxes, head: head)
    }

    /// The result screen's model toggle: show the cached reading of that model, or read it now.
    func switchModel(_ scan: Scan, useTuned: Bool) {
        let target = useTuned ? (tunedModelID ?? ModelID.base) : ModelID.base
        guard target != scan.reading?.model, rereadingID == nil else { return }
        let readings = scan.readings
        if let i = readings.lastIndex(where: { $0.model == target }) {
            withAnimation(.easeInOut(duration: 0.25)) { scan.current = i }
            try? context.save()
            return
        }
        rereadingID = scan.id
        Task {
            defer { rereadingID = nil }
            do {
                let lines = try await reread(scan, modelID: target)
                var all = scan.readings
                all.append(Reading(model: target, lines: lines, createdAt: Date()))
                withAnimation(.easeInOut(duration: 0.25)) {
                    scan.readings = all
                    scan.current = all.count - 1
                }
                try? context.save()
            } catch {
                showToast("משהו השתבש בקריאה. נסו שוב.")
            }
        }
    }

    /// Saves line edits and the title; corrected lines become training samples (real paper ink).
    func saveEdits(_ scan: Scan, drafts: [String], title: String) {
        var readings = scan.readings
        guard !readings.isEmpty else { return }
        let ci = min(max(0, scan.current), readings.count - 1)
        let old = readings[ci].lines
        var changed: [Int] = []
        var lines = old
        for i in lines.indices where i < drafts.count {
            let t = drafts[i].trimmingCharacters(in: .whitespacesAndNewlines)
            if t != old[i].text.trimmingCharacters(in: .whitespacesAndNewlines) { changed.append(i) }
            lines[i].text = t
        }
        readings[ci].lines = lines
        if !changed.isEmpty { readings[ci].edited = true }
        scan.readings = readings
        let t = title.trimmingCharacters(in: .whitespacesAndNewlines)
        if !t.isEmpty { scan.title = t }

        var recorded = 0
        if !changed.isEmpty, let image = Images.load(scan.imageURL) {
            let round = settings.rounds + 1
            for i in changed {
                guard let box = lines[i].box, !lines[i].text.isEmpty,
                      let crop = Images.cropLine(image, box: box), let png = Images.png(crop) else { continue }
                let id = UUID().uuidString
                let name = "\(id).png"
                do {
                    try png.write(to: Storage.samplesDirectory.appendingPathComponent(name), options: .atomic)
                } catch { continue }
                context.insert(Sample(id: id, round: round, word: lines[i].text, lineImageFileName: name, heldOut: false))
                recorded += 1
            }
        }
        try? context.save()
        if recorded > 0 { showToast("התיקונים נשמרו, וישמשו ללימוד") }
    }

    func deleteScans(_ ids: Set<UUID>) {
        guard !ids.isEmpty else { return }
        for id in ids {
            if let s = scan(id) {
                Storage.remove(s.imageURL)
                Storage.remove(s.thumbURL)
                context.delete(s)
            }
        }
        try? context.save()
        showToast(ids.count == 1 ? "סריקה אחת נמחקה" : "\(ids.count) סריקות נמחקו")
    }

    func deleteCurrentScan() {
        guard let id = currentID, let s = scan(id) else { return }
        Storage.remove(s.imageURL)
        Storage.remove(s.thumbURL)
        context.delete(s)
        try? context.save()
        go(.list)
        showToast("הסריקה נמחקה")
    }

    // MARK: Training

    func makePicker() -> WordPicker {
        let samples = (try? context.fetch(FetchDescriptor<Sample>(sortBy: [SortDescriptor(\.createdAt)]))) ?? []
        let attempts = samples.filter { $0.strokesData != nil }.map { WordPicker.Attempt(word: $0.word, readAs: $0.readAs) }
        return WordPicker(pool: WordPicker.pool, attempts: attempts, scanWords: unsureScanWords())
    }

    /// Words of the user's scans from the lines the model was least sure of (never shown as such).
    private func unsureScanWords() -> [String] {
        let scans = (try? context.fetch(FetchDescriptor<Scan>())) ?? []
        let lines = scans.flatMap(\.lines).sorted { $0.confidence < $1.confidence }
        guard !lines.isEmpty else { return [] }
        return lines.prefix((lines.count + 2) / 3).flatMap { $0.text.split(whereSeparator: { $0.isWhitespace }).map(String.init) }
    }

    /// Reads the canvas with the fine-tuned head in use (or the general one).
    func readStrokes(_ strokes: [[CGPoint]], word: String) async -> String {
        let head = tunedHead
        do {
            return try await engine.readStrokes(strokes, nChars: word.count, head: head)
                .trimmingCharacters(in: .whitespacesAndNewlines)
        } catch {
            return ""
        }
    }

    func recordAttempt(word: String, strokes: [[CGPoint]], readAs: String) {
        let s = Sample(round: settings.rounds + 1, word: word, strokes: strokes, readAs: readAs, correct: readAs == word)
        context.insert(s)
        try? context.save()
        roundCount += 1
    }

    func roundComplete() {
        startRefit()
    }

    func startRefit() {
        let round = settings.rounds + 1
        let all = (try? context.fetch(FetchDescriptor<Sample>(sortBy: [SortDescriptor(\.createdAt)]))) ?? []
        let inputs = all.map(SampleInput.init)
        var state = RefitState(round: round)
        state.tiles = all.filter { $0.round == round && $0.strokesData != nil }.suffix(Self.roundSize)
            .map { RefitTile(id: $0.id, strokes: $0.strokes, word: $0.word) }
        state.held = all.filter { $0.heldOut && $0.strokesData != nil }.count
        refit = state
        go(.refit)

        let engine = self.engine
        let previous = tunedHead
        refitTask?.cancel()
        refitTask = Task { [weak self] in
            // Let the grid be seen before it dissolves into the ring.
            try? await Task.sleep(for: .milliseconds(1600))
            do {
                let (result, head) = try await Refit.run(engine: engine, round: round, samples: inputs, previous: previous) { stage, p in
                    Task { @MainActor in self?.refitProgress(stage, p) }
                }
                guard let self else { return }
                if let head {
                    let used = inputs.filter { !$0.heldOut }.suffix(AdaptConfig.shipped.maxTrainSamples).count
                    try HeadStore.save(head, record: RoundRecord(round: round, createdAt: Date(), accBefore: result.before,
                                                                 accAfter: result.after, samples: used))
                    self.tunedHead = head
                    self.settings.tunedRound = round
                    self.settings.preferTuned = true
                }
                self.settings.rounds = round
                self.refreshRoundCount()
                self.refit?.stage = 2
                self.refit?.pct = 100
                try? await Task.sleep(for: .milliseconds(900))
                withAnimation(Mk.ease(0.7)) {
                    self.refit?.result = result
                    self.refit?.stage = 3
                }
            } catch {
                guard let self else { return }
                withAnimation(Mk.ease(0.7)) {
                    self.refit?.failed = true
                    self.refit?.stage = 3
                }
            }
        }
    }

    private func refitProgress(_ stage: RefitStage, _ p: Double) {
        guard var r = refit, r.stage < 3 else { return }
        let idx: Int
        switch stage {
        case .collect: idx = p > 0.02 ? 1 : 0
        case .tune: idx = 1
        case .test: idx = 2
        case .done: idx = 2
        }
        let pct: Int
        switch stage {
        case .collect: pct = Int((p * 40).rounded())
        case .tune: pct = 40 + Int((p * 60).rounded())
        case .test, .done: pct = 100
        }
        let newStage = max(r.stage, idx)
        if newStage != r.stage {
            withAnimation(Mk.ease(0.9)) { refit?.stage = newStage }
            r.stage = newStage
        }
        refit?.pct = max(r.pct, pct)
    }

    func continueWriting() {
        refit = nil
        go(.train)
    }

    func openComparePicker() {
        present(.comparePick)
    }

    /// Re-reads the chosen scan with the new head and opens the compare screen.
    func compare(_ scan: Scan) {
        guard let modelID = tunedModelID, let before = scan.reading else { return }
        currentID = scan.id
        compare = CompareState(scanID: scan.id, model: modelID, beforeModel: before.model, before: before.lines, after: nil)
        go(.compare)
        compareTask?.cancel()
        compareTask = Task {
            do {
                let after = try await reread(scan, modelID: modelID)
                compare?.after = after
            } catch {
                showToast("משהו השתבש בקריאה. נסו שוב.")
                go(.list)
            }
        }
    }

    func keepCompare(new keepNew: Bool) {
        guard let c = compare else { go(.list); return }
        if keepNew, let after = c.after, let s = scan(c.scanID) {
            var all = s.readings
            all.append(Reading(model: c.model, lines: after, createdAt: Date()))
            s.readings = all
            s.current = all.count - 1
            s.reread = true
            try? context.save()
            refit = nil
            open(s.id, from: .list)
            showToast("הקריאה החדשה נשמרה")
        } else {
            refit = nil
            go(.list)
            showToast("הקריאה הקודמת נשארה")
        }
        compare = nil
    }

    func leaveCompare() {
        compareTask?.cancel()
        compare = nil
        refit = nil
        go(.list)
    }

    /// "Forget my handwriting": samples, cached features and every fine-tuned head. Scans stay.
    func forgetHandwriting() {
        try? context.delete(model: Sample.self)
        try? context.save()
        Storage.removeAll(in: Storage.samplesDirectory)
        Storage.removeAll(in: Storage.featuresDirectory)
        HeadStore.removeAll()
        settings.rounds = 0
        settings.tunedRound = 0
        tunedHead = nil
        refreshRoundCount()
        closeSheet()
        showToast("הדוגמאות וההתאמה נמחקו")
    }

    #if DEBUG
    /// Demo content: pretend a fine-tuned head exists.
    func installDemoHead(_ head: Head, round: Int) {
        tunedHead = head
        settings.rounds = round
        settings.tunedRound = round
        settings.preferTuned = true
        refreshRoundCount()
    }
    #endif

    // MARK: Toast

    func showToast(_ text: String) {
        toastTask?.cancel()
        withAnimation(Mk.ease(0.4)) { toast = Toast(text: text) }
        toastTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(2.4))
            guard !Task.isCancelled, let self else { return }
            withAnimation(.easeOut(duration: 0.25)) { self.toast = nil }
        }
    }

    // MARK: Lifecycle

    func enteredBackground() {
        // The ONNX session holds a few hundred MB; don't keep it while suspended.
        let refitRunning = (refit?.stage ?? 3) < 3
        if !jobs.isBusy && !refitRunning && rereadingID == nil { Task { await engine.unload() } }
    }

    func memoryWarning() {
        let refitRunning = (refit?.stage ?? 3) < 3
        if !jobs.isBusy && rereadingID == nil && !refitRunning { Task { await engine.unload() } }
    }
}

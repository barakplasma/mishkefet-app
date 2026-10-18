import Foundation
import MishkefetCore
import Observation
import SwiftData
import UIKit

/// Reads pictures into scans, one at a time, driving the processing screen.
@MainActor
@Observable
final class JobQueue {
    /// The job on the processing screen.
    struct Live {
        var image: UIImage
        var imageSize: CGSize
        var boxes: [LineBox] = []
        var done = 0
        var total = 0
        var lastText = ""
        var picture = 1
        var pictures = 1
        var kind: ScanKind
        var segmented = false
    }

    private(set) var live: Live?

    @ObservationIgnored let engine: ReadingEngine
    @ObservationIgnored let context: ModelContext
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var backgroundTask: UIBackgroundTaskIdentifier = .invalid

    enum Outcome {
        case finished([UUID])
        case cancelled
        case failed
    }

    init(engine: ReadingEngine, context: ModelContext) {
        self.engine = engine
        self.context = context
    }

    var isBusy: Bool { task != nil }

    /// Reads `pictures` with `head` (nil = the general model), saving one scan per picture.
    func read(_ pictures: [Picture], kind: ScanKind, head: Head?, modelID: String, completion: @escaping (Outcome) -> Void) {
        guard task == nil, !pictures.isEmpty else { return }
        beginBackgroundTask()
        task = Task { [weak self] in
            guard let self else { return }
            var ids: [UUID] = []
            var outcome: Outcome
            do {
                for (i, pic) in pictures.enumerated() {
                    try Task.checkCancellation()
                    ids.append(try await self.readOne(pic, kind: kind, head: head, modelID: modelID, index: i, count: pictures.count))
                }
                outcome = .finished(ids)
            } catch is CancellationError {
                outcome = ids.isEmpty ? .cancelled : .finished(ids)
            } catch {
                outcome = ids.isEmpty ? .failed : .finished(ids)
            }
            self.task = nil
            self.endBackgroundTask()
            if case .finished = outcome { try? await Task.sleep(for: .milliseconds(450)) }
            self.live = nil
            completion(outcome)
        }
    }

    func cancel() {
        task?.cancel()
    }

    private func readOne(_ pic: Picture, kind: ScanKind, head: Head?, modelID: String, index: Int, count: Int) async throws -> UUID {
        live = Live(image: UIImage(cgImage: pic.image), imageSize: CGSize(width: pic.width, height: pic.height),
                    picture: index + 1, pictures: count, kind: kind)
        let engine = self.engine
        var page = pic.image
        var lines: [LineResult]
        if kind == .line {
            let full = LineBox(x: 0, y: 0, w: pic.width, h: pic.height)
            live?.boxes = [full]
            live?.total = 1
            live?.segmented = true
            var r = try await engine.readLine(pic.image, head: head)
            r.box = full
            lines = [r]
            live?.done = 1
            live?.lastText = r.text
        } else {
            let result = try await engine.readPage(pic.image, head: head, onSegmented: { [weak self] shown, boxes in
                let ui = UIImage(cgImage: shown)
                Task { @MainActor in
                    guard let self, var live = self.live else { return }
                    live.image = ui
                    live.imageSize = CGSize(width: shown.width, height: shown.height)
                    live.boxes = boxes
                    live.total = boxes.count
                    live.segmented = true
                    self.live = live
                }
            }, onLine: { [weak self] i, total, line in
                Task { @MainActor in
                    guard let self, var live = self.live else { return }
                    live.done = max(live.done, i + 1)
                    live.total = total
                    if !line.text.trimmingCharacters(in: .whitespaces).isEmpty { live.lastText = line.text }
                    self.live = live
                }
            })
            page = result.page
            lines = result.lines
        }
        try Task.checkCancellation()

        let id = UUID()
        let now = Date()
        let kept = lines.filter { Self.keeps($0, kind: kind) }
        let imageName = "\(id.uuidString).jpg", thumbName = "\(id.uuidString)-thumb.jpg"
        let saved = await Task.detached(priority: .userInitiated) { [page] () -> Bool in
            guard let jpg = Images.jpeg(page), let thumb = Images.thumbnail(page) else { return false }
            do {
                try jpg.write(to: Storage.scansDirectory.appendingPathComponent(imageName), options: .atomic)
                try thumb.write(to: Storage.scansDirectory.appendingPathComponent(thumbName), options: .atomic)
                return true
            } catch {
                return false
            }
        }.value
        guard saved else { throw CocoaError(.fileWriteUnknown) }
        var title = Titles.scan(lines: kept.map(\.text), at: now)
        if kept.isEmpty, !pic.name.isEmpty { title = pic.name }
        let scan = Scan(id: id, title: title, createdAt: now, kind: kind, imageFileName: imageName, thumbFileName: thumbName,
                        width: page.width, height: page.height,
                        readings: [Reading(model: modelID, lines: kept, createdAt: now)])
        context.insert(scan)
        try? context.save()
        #if DEBUG
        for l in kept { print("MISHKEFET_LINE \(l.text)") }
        print("MISHKEFET_DONE \(kept.count)")
        #endif
        return id
    }

    /// Whether a read line goes into the stored reading: not blank, and on pages not a ruling or
    /// binding fragment (at most 3 characters besides spaces, read with all-frame confidence below 0.9).
    nonisolated static func keeps(_ line: LineResult, kind: ScanKind) -> Bool {
        let text = line.text.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.isEmpty { return false }
        if kind != .line, text.filter({ $0 != " " }).count <= 3, line.confidence < 0.9 { return false }
        return true
    }

    private func beginBackgroundTask() {
        guard backgroundTask == .invalid else { return }
        backgroundTask = UIApplication.shared.beginBackgroundTask(withName: "read") { [weak self] in
            MainActor.assumeIsolated {
                self?.cancel()
                self?.endBackgroundTask()
            }
        }
    }

    private func endBackgroundTask() {
        guard backgroundTask != .invalid else { return }
        UIApplication.shared.endBackgroundTask(backgroundTask)
        backgroundTask = .invalid
    }
}

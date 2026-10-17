import CoreGraphics
import Foundation

/// Mishkefet-v1 inference: page segmentation, line preprocessing, the ONNX backbone,
/// the CTC head (base or fine-tuned) and LM beam search. Mirrors reference.py `Reader`.
public actor Recognizer {
    public nonisolated let files: ModelFiles
    public nonisolated let baseHead: Head
    public nonisolated let charset: Charset
    private let threads: Int
    private var backbone: Backbone?
    private var decoder = Decoder()
    private var languageModelsLoaded = false

    /// Reads the head and charset (small); `prepare()` loads the rest.
    public init(files: ModelFiles, threads: Int = 4) throws {
        self.files = files
        self.threads = threads
        do {
            baseHead = try Head(contentsOf: files.head)
        } catch let e as MishkefetError {
            throw e
        } catch {
            throw MishkefetError.modelMissing(ModelFiles.headName)
        }
        do {
            charset = try Charset(contentsOf: files.charset)
        } catch let e as MishkefetError {
            throw e
        } catch {
            throw MishkefetError.modelMissing(ModelFiles.charsetName)
        }
    }

    /// Creates the ONNX session and loads the language models. Idempotent; the
    /// other methods call it when needed.
    public func prepare() throws {
        if !languageModelsLoaded { try loadLanguageModels() }
        if backbone == nil {
            var o = Backbone.Options()
            o.intraOpThreads = threads
            backbone = try Backbone(url: files.backbone, options: o)
        }
    }

    private func loadLanguageModels() throws {
        do {
            decoder.charLM = try CharLM(contentsOf: files.charLM)
        } catch {
            throw (error as? MishkefetError) ?? .modelMissing(ModelFiles.charLMName)
        }
        do {
            decoder.words = try WordLM(contentsOf: files.words, charset: charset)
        } catch {
            throw (error as? MishkefetError) ?? .modelMissing(ModelFiles.wordsName)
        }
        languageModelsLoaded = true
    }

    // MARK: - Images

    /// BT.601 gray (PIL-exact), alpha composited on white. The caller applies EXIF orientation first.
    public func gray(from image: CGImage) -> GrayImage {
        GrayImage(cgImage: image)
    }

    public func segment(page: CGImage) throws -> PageSegmentation {
        let r = Segmenter.segment(GrayImage(cgImage: page))
        let deskewed = abs(r.angle) > 1e-3 ? Rotation.rotatedCGImage(page, angle: r.angle) : page
        return PageSegmentation(angle: r.angle, pitch: r.pitch, boxes: r.boxes, crops: r.crops, deskewed: deskewed)
    }

    // MARK: - Lines

    /// Preprocess + backbone: features (frames x 512), row-major.
    public func features(of gray: GrayImage) throws -> (values: [Float], frames: Int) {
        try prepare()
        guard gray.width > 0, gray.height > 0 else { return ([], 0) }
        let x = Preprocess.run(gray)
        return try backbone!.run(x.values, width: x.width)
    }

    /// Log-probabilities (frames x classes) of one view.
    func logprobs(_ gray: GrayImage, head: Head?) throws -> (values: [Float], frames: Int) {
        let f = try features(of: gray)
        return ((head ?? baseHead).logprobs(features: f.values, frames: f.frames), f.frames)
    }

    /// Reads a line crop: best of the TTA views (1.0, 0.82, 1.22) by emitting-frame
    /// confidence; the stored confidence is the scale-1.0 view's all-frame confidence.
    public func readLine(_ gray: GrayImage, head: Head?, tta: Int = 3) throws -> LineResult {
        let classes = (head ?? baseHead).classes
        var bestText = "", bestScore = -Double.infinity, baseConfidence = 0.0
        for scale in Preprocess.ttaScales.prefix(max(1, tta)) {
            let lp = try logprobs(Preprocess.ttaView(gray, scale: scale), head: head)
            let text = decode(logprobs: lp.values, frames: lp.frames, beam: true)
            if scale == 1 {
                baseConfidence = Decoder.confidence(lp.values, frames: lp.frames, classes: classes, emittingOnly: false)
            }
            if tta <= 1 { return LineResult(text: text, confidence: baseConfidence) }
            let blank = text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            let score = blank ? -Double.infinity
                : Decoder.confidence(lp.values, frames: lp.frames, classes: classes, emittingOnly: true)
            if score > bestScore {
                bestScore = score
                bestText = text
            }
        }
        return LineResult(text: bestText, confidence: baseConfidence)
    }

    /// Segments the page and reads every line in reading order. `lines` has one entry per
    /// box (blank ones included; `LineResult.transcript` drops them). `onLine(index, count, line)`
    /// reports progress.
    public func readPage(_ page: CGImage, head: Head?, tta: Int = 3,
                         onLine: (@Sendable (Int, Int, LineResult) -> Void)?) throws
        -> (segmentation: PageSegmentation, lines: [LineResult]) {
        try prepare()
        let seg = try segment(page: page)
        var lines: [LineResult] = []
        for (i, crop) in seg.crops.enumerated() {
            try Task.checkCancellation()
            var r = try readLine(crop, head: head, tta: tta)
            r.box = seg.boxes[i]
            lines.append(r)
            onLine?(i, seg.crops.count, r)
        }
        return (seg, lines)
    }

    /// Decodes log-probabilities (frames x classes): LM beam search, or greedy.
    public func decode(logprobs: [Float], frames: Int, beam: Bool) -> String {
        let classes = frames > 0 ? logprobs.count / frames : baseHead.classes
        if beam && !languageModelsLoaded { try? loadLanguageModels() }
        let labels = beam ? decoder.beamSearch(logprobs, frames: frames, classes: classes)
            : Decoder.greedy(logprobs, frames: frames, classes: classes)
        return charset.decode(labels)
    }
}

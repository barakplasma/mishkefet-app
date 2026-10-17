@preconcurrency import AVFoundation
import SwiftUI
import UIKit

enum CameraMode: Hashable {
    case line, page

    var kind: ScanKind { self == .line ? .line : .page }

    /// The design's guide rectangle inside a viewfinder of `size`.
    func guide(in size: CGSize) -> CGSize {
        switch self {
        case .page: return CGSize(width: min(290, size.width - 40), height: min(400, size.height - 70))
        case .line: return CGSize(width: min(320, size.width - 30), height: 90)
        }
    }
}

struct CameraScreen: View {
    @Environment(AppModel.self) private var app
    @State private var camera = CameraController()
    @State private var mode: CameraMode = .page
    @State private var finder: CGSize = CGSize(width: 360, height: 520)
    @State private var shooting = false

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                CircleIconButton(glyph: .close, size: 18, lineWidth: 2.2, label: "סגירה", foreground: Mk.cream) { app.go(.home) }
                Spacer()
                Segmented(options: [(CameraMode.line, "שורה"), (CameraMode.page, "דף שלם")], selection: $mode, dark: true, label: "מצב צילום")
                Spacer()
                Color.clear.frame(width: 46, height: 46)
            }
            .padding(.horizontal, 10)
            .frame(height: 56)

            viewfinder
                .padding(.vertical, 4)

            if app.hasTuned {
                ModelToggle(useTuned: app.useTuned, dark: true) { app.setUseTuned($0) }
                    .padding(.top, 10)
            }

            HStack {
                Color.clear.frame(width: 52, height: 52)
                Spacer()
                Button(action: shoot) {
                    Circle()
                        .strokeBorder(Mk.cream, lineWidth: 3)
                        .overlay(Circle().fill(Mk.cream).padding(8))
                        .frame(width: 78, height: 78)
                        .opacity(camera.state == .running && !shooting ? 1 : 0.35)
                }
                .buttonStyle(PressStyle(scale: 0.92))
                .disabled(camera.state != .running || shooting)
                .accessibilityLabel("צילום")
                Spacer()
                Button { app.pickPhotos() } label: {
                    Icon(glyph: .gallery, size: 22)
                        .foregroundStyle(Mk.cream)
                        .frame(width: 52, height: 52)
                        .overlay(RoundedRectangle(cornerRadius: 14).strokeBorder(Mk.cream.opacity(0.4), lineWidth: 1.5))
                }
                .buttonStyle(PressStyle(scale: 0.94))
                .accessibilityLabel("גלריה")
            }
            .padding(.horizontal, 34)
            .padding(.bottom, 20)
            .frame(height: 130)
        }
        .foregroundStyle(Mk.cream)
        .onChange(of: app.screen) { _, s in
            if s == .camera { camera.start() } else { camera.stop(); shooting = false }
        }
        .onAppear { if app.screen == .camera { camera.start() } }
    }

    private var viewfinder: some View {
        GeometryReader { geo in
            let guide = mode.guide(in: geo.size)
            ZStack {
                Color(hex: 0x1f1f1e)
                if camera.state == .running || camera.state == .starting {
                    CameraPreview(session: camera.session)
                    GuideCorners()
                        .stroke(Mk.orange, style: StrokeStyle(lineWidth: 3, lineCap: .round))
                        .frame(width: guide.width, height: guide.height)
                        .animation(Mk.ease(0.45), value: mode)
                    VStack {
                        Text(mode == .page ? "יישרו את הדף בתוך המסגרת" : "שורה אחת בתוך המסגרת")
                            .font(Mk.rubik(13))
                            .padding(.horizontal, 14)
                            .padding(.vertical, 7)
                            .background(RoundedRectangle(cornerRadius: 14).fill(Color(hex: 0x141414, opacity: 0.7)))
                            .padding(.top, 18)
                        Spacer()
                    }
                } else if camera.state == .denied || camera.state == .unavailable {
                    // After "Don't Allow" stay silent (App Review 5.1.1(iv)): only offer the gallery.
                    Button { app.pickPhotos() } label: {
                        PillLabel(title: "בחירה מהגלריה", glyph: .gallery, foreground: Mk.cream, background: Mk.cream.opacity(0.12))
                    }
                    .buttonStyle(PressStyle())
                }
                if shooting {
                    Color.black.opacity(0.35)
                    Spinner(size: 28)
                }
            }
            .clipped()
            .onAppear { finder = geo.size }
            .onChange(of: geo.size) { _, s in finder = s }
        }
    }

    private func shoot() {
        guard !shooting else { return }
        shooting = true
        let mode = self.mode
        let size = finder
        let guide = mode.guide(in: size)
        camera.capture { image in
            guard let image else {
                shooting = false
                app.showToast("הצילום לא הצליח. נסו שוב.")
                return
            }
            Task {
                let pic = await Task.detached(priority: .userInitiated) {
                    Self.picture(from: image, mode: mode, finder: size, guide: guide)
                }.value
                shooting = false
                if let pic { app.readPictures([pic], kind: mode.kind) }
            }
        }
    }

    /// Page: the detected sheet, flattened; otherwise the guide crop. Line: the guide band.
    nonisolated static func picture(from image: UIImage, mode: CameraMode, finder: CGSize, guide: CGSize) -> Picture? {
        guard let upright = Images.normalized(image, maxSide: 4032) else { return nil }
        let iw = CGFloat(upright.width), ih = CGFloat(upright.height)
        // The preview is aspect-fill; map the centred guide into image fractions.
        let scale = max(finder.width / iw, finder.height / ih)
        let ox = (finder.width - iw * scale) / 2, oy = (finder.height - ih * scale) / 2
        let pad: CGFloat = mode == .line ? 12 : 0
        let gx = (finder.width - guide.width) / 2, gy = (finder.height - guide.height) / 2
        let x = max(0, (gx - pad - ox) / scale / iw)
        let y = max(0, (gy - pad - oy) / scale / ih)
        let w = min(1 - x, (guide.width + 2 * pad) / scale / iw)
        let h = min(1 - y, (guide.height + 2 * pad) / scale / ih)
        let fraction = CGRect(x: x, y: y, width: w, height: h)

        var out: CGImage?
        if mode == .page { out = Images.documentCorrected(upright) }
        if out == nil { out = Images.crop(upright, fraction: fraction) }
        guard let cg = out, let final = Images.normalized(UIImage(cgImage: cg)) else { return nil }
        return Picture(image: final, name: "")
    }
}

/// The four orange corners of the guide (26pt arms, 10pt radius).
struct GuideCorners: Shape {
    func path(in r: CGRect) -> Path {
        let a: CGFloat = 26, rad: CGFloat = 10
        var p = Path()
        // top-left
        p.move(to: CGPoint(x: r.minX, y: r.minY + a))
        p.addLine(to: CGPoint(x: r.minX, y: r.minY + rad))
        p.addQuadCurve(to: CGPoint(x: r.minX + rad, y: r.minY), control: CGPoint(x: r.minX, y: r.minY))
        p.addLine(to: CGPoint(x: r.minX + a, y: r.minY))
        // top-right
        p.move(to: CGPoint(x: r.maxX - a, y: r.minY))
        p.addLine(to: CGPoint(x: r.maxX - rad, y: r.minY))
        p.addQuadCurve(to: CGPoint(x: r.maxX, y: r.minY + rad), control: CGPoint(x: r.maxX, y: r.minY))
        p.addLine(to: CGPoint(x: r.maxX, y: r.minY + a))
        // bottom-right
        p.move(to: CGPoint(x: r.maxX, y: r.maxY - a))
        p.addLine(to: CGPoint(x: r.maxX, y: r.maxY - rad))
        p.addQuadCurve(to: CGPoint(x: r.maxX - rad, y: r.maxY), control: CGPoint(x: r.maxX, y: r.maxY))
        p.addLine(to: CGPoint(x: r.maxX - a, y: r.maxY))
        // bottom-left
        p.move(to: CGPoint(x: r.minX + a, y: r.maxY))
        p.addLine(to: CGPoint(x: r.minX + rad, y: r.maxY))
        p.addQuadCurve(to: CGPoint(x: r.minX, y: r.maxY - rad), control: CGPoint(x: r.minX, y: r.maxY))
        p.addLine(to: CGPoint(x: r.minX, y: r.maxY - a))
        return p
    }
}

/// The back camera behind an aspect-fill preview layer, and a photo output.
@MainActor
@Observable
final class CameraController: NSObject {
    enum State { case idle, starting, running, denied, unavailable }

    private(set) var state: State = .idle
    @ObservationIgnored nonisolated(unsafe) let session = AVCaptureSession()
    @ObservationIgnored private let output = AVCapturePhotoOutput()
    @ObservationIgnored private var configured = false
    @ObservationIgnored private var pending: ((UIImage?) -> Void)?
    @ObservationIgnored private let queue = DispatchQueue(label: "mishkefet.camera")
    @ObservationIgnored private var wanted = false

    func start() {
        wanted = true
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            run()
        case .notDetermined:
            state = .starting
            AVCaptureDevice.requestAccess(for: .video) { granted in
                Task { @MainActor in
                    if granted { self.run() } else { self.state = .denied }
                }
            }
        default:
            // Denied or restricted: no prompt, no pointer to Settings; the gallery stays available.
            state = .denied
        }
    }

    func stop() {
        wanted = false
        guard configured else { return }
        let session = self.session
        queue.async { if session.isRunning { session.stopRunning() } }
        if state == .running || state == .starting { state = .idle }
    }

    private func run() {
        guard wanted else { return }
        if !configured {
            guard configure() else { state = .unavailable; return }
        }
        state = .starting
        let session = self.session
        queue.async {
            if !session.isRunning { session.startRunning() }
            Task { @MainActor in
                if self.wanted { self.state = session.isRunning ? .running : .unavailable }
            }
        }
    }

    private func configure() -> Bool {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device) else { return false }
        session.beginConfiguration()
        session.sessionPreset = .photo
        guard session.canAddInput(input), session.canAddOutput(output) else {
            session.commitConfiguration()
            return false
        }
        session.addInput(input)
        session.addOutput(output)
        output.maxPhotoQualityPrioritization = .quality
        session.commitConfiguration()
        if device.isFocusModeSupported(.continuousAutoFocus) {
            try? device.lockForConfiguration()
            device.focusMode = .continuousAutoFocus
            device.unlockForConfiguration()
        }
        configured = true
        return true
    }

    func capture(_ done: @escaping (UIImage?) -> Void) {
        guard state == .running else { done(nil); return }
        pending = done
        let settings = AVCapturePhotoSettings()
        settings.photoQualityPrioritization = .balanced
        if let c = output.connection(with: .video), c.isVideoRotationAngleSupported(90) {
            c.videoRotationAngle = 90
        }
        output.capturePhoto(with: settings, delegate: self)
    }
}

extension CameraController: AVCapturePhotoCaptureDelegate {
    nonisolated func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        let image = photo.fileDataRepresentation().flatMap(UIImage.init(data:))
        Task { @MainActor in
            self.pending?(error == nil ? image : nil)
            self.pending = nil
        }
    }
}

private struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }

    func makeUIView(context: Context) -> PreviewView {
        let v = PreviewView()
        v.previewLayer.session = session
        v.previewLayer.videoGravity = .resizeAspectFill
        if let c = v.previewLayer.connection, c.isVideoRotationAngleSupported(90) { c.videoRotationAngle = 90 }
        return v
    }

    func updateUIView(_ v: PreviewView, context: Context) {
        if let c = v.previewLayer.connection, c.isVideoRotationAngleSupported(90), c.videoRotationAngle != 90 {
            c.videoRotationAngle = 90
        }
    }
}

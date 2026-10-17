import SwiftUI
import UIKit

/// The training canvas: finger or Apple Pencil, coalesced touches, 4.5pt round ink.
final class StrokeCanvasView: UIView {
    var onDown: (() -> Void)?
    var onUp: (() -> Void)?
    var isAccepting = true

    private(set) var strokes: [[CGPoint]] = []
    private var drawing = false
    private let ink = CAShapeLayer()

    var inkColor: UIColor = UIColor(hex: 0x262626) {
        didSet { ink.strokeColor = inkColor.cgColor }
    }
    private var recolored = false

    /// The theme's ink: charcoal on the light card, cream on the dark one.
    private var baseInk: UIColor {
        UIColor(hex: traitCollection.userInterfaceStyle == .dark ? 0xe2e8ce : 0x262626)
    }

    override init(frame: CGRect) {
        super.init(frame: frame)
        isMultipleTouchEnabled = false
        backgroundColor = .clear
        ink.fillColor = nil
        ink.lineWidth = 4.5
        ink.lineCap = .round
        ink.lineJoin = .round
        inkColor = baseInk
        ink.actions = ["path": NSNull(), "strokeColor": NSNull()]
        layer.addSublayer(ink)
        registerForTraitChanges([UITraitUserInterfaceStyle.self]) { (v: StrokeCanvasView, _: UITraitCollection) in
            if !v.recolored { v.inkColor = v.baseInk }
        }
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) is not used") }

    override func layoutSubviews() {
        super.layoutSubviews()
        ink.frame = bounds
    }

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard isAccepting, let t = touches.first else { return }
        drawing = true
        onDown?()
        strokes.append([round(t.location(in: self))])
        rebuild()
    }

    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard drawing, let t = touches.first, !strokes.isEmpty else { return }
        let all = event?.coalescedTouches(for: t) ?? [t]
        for c in all { strokes[strokes.count - 1].append(round(c.location(in: self))) }
        rebuild()
    }

    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        finish()
    }

    override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
        finish()
    }

    private func finish() {
        guard drawing else { return }
        drawing = false
        onUp?()
    }

    private func round(_ p: CGPoint) -> CGPoint {
        CGPoint(x: (p.x * 10).rounded() / 10, y: (p.y * 10).rounded() / 10)
    }

    private func rebuild() {
        let path = CGMutablePath()
        for s in strokes {
            guard let first = s.first else { continue }
            path.move(to: first)
            if s.count == 1 {
                path.addLine(to: CGPoint(x: first.x + 0.1, y: first.y))
            } else {
                for p in s.dropFirst() { path.addLine(to: p) }
            }
        }
        ink.path = path
    }

    func clear() {
        strokes = []
        drawing = false
        recolored = false
        inkColor = baseInk
        rebuild()
    }

    func recolor(_ color: UIColor) {
        let anim = CABasicAnimation(keyPath: "strokeColor")
        anim.fromValue = ink.strokeColor
        anim.toValue = color.cgColor
        anim.duration = 0.25
        ink.add(anim, forKey: "recolor")
        recolored = true
        inkColor = color
    }
}

/// Lets SwiftUI reach the canvas view (clear, recolor, read strokes).
@MainActor
final class CanvasHandle {
    weak var view: StrokeCanvasView?
    var strokes: [[CGPoint]] { view?.strokes ?? [] }
    func clear() { view?.clear() }
    func recolor(_ c: UIColor) { view?.recolor(c) }
    func setAccepting(_ v: Bool) { view?.isAccepting = v }
}

struct StrokeCanvas: UIViewRepresentable {
    let handle: CanvasHandle
    var onDown: () -> Void
    var onUp: () -> Void

    func makeUIView(context: Context) -> StrokeCanvasView {
        let v = StrokeCanvasView()
        handle.view = v
        v.onDown = onDown
        v.onUp = onUp
        return v
    }

    func updateUIView(_ v: StrokeCanvasView, context: Context) {
        handle.view = v
        v.onDown = onDown
        v.onUp = onUp
    }
}

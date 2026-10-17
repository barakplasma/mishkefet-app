import MishkefetCore
import SwiftUI

/// The binoculars sweep across the line being read; read lines turn sage.
struct ProcessingScreen: View {
    @Environment(AppModel.self) private var app

    private let frameW: CGFloat = 300
    private let frameH: CGFloat = 210

    var body: some View {
        let live = app.jobs.live
        ZStack(alignment: .bottom) {
            VStack(spacing: 28) {
                page(live)
                VStack(spacing: 6) {
                    Text(verbatim: "\(percent(live))%")
                        .font(Mk.karantina(80, bold: false))
                        .monospacedDigit()
                        .frame(height: 72)
                        .environment(\.layoutDirection, .leftToRight)
                        .contentTransition(.numericText())
                        .animation(.easeOut(duration: 0.3), value: percent(live))
                    Text("קוראת את כתב היד")
                        .font(Mk.rubik(21, .medium))
                    Text(verbatim: subtitle(live))
                        .font(Mk.rubik(15))
                        .foregroundStyle(Mk.sub)
                        .contentTransition(.opacity)
                }
                ZStack {
                    if let text = live?.lastText, !text.isEmpty {
                        Text(verbatim: text)
                            .id(text)
                            .transition(.asymmetric(insertion: .opacity.combined(with: .offset(y: 4)), removal: .opacity))
                    }
                }
                .font(Mk.rubik(15))
                .lineSpacing(7)
                .foregroundStyle(Mk.sub)
                .multilineTextAlignment(.center)
                .lineLimit(2)
                .frame(width: 300, height: 52, alignment: .top)
                .animation(.easeOut(duration: 0.5), value: live?.lastText)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .padding(.bottom, 40)

            VStack(spacing: 10) {
                DeviceBadge(text: "על המכשיר · שום דבר לא עולה לשרת")
                Button("ביטול") { app.cancelReading() }
                    .font(Mk.rubik(15))
                    .foregroundStyle(Mk.sub)
                    .frame(height: 44)
            }
            .padding(.bottom, 30)
        }
    }

    private func percent(_ live: JobQueue.Live?) -> Int {
        guard let live, live.total > 0 else { return 0 }
        return Int((Double(live.done) / Double(live.total) * 100).rounded())
    }

    private func subtitle(_ live: JobQueue.Live?) -> String {
        guard let live else { return "" }
        var s: String
        if live.total == 0 {
            s = "מחפשת שורות"
        } else if live.done >= live.total {
            s = "סיימתי"
        } else {
            s = "שורה \(live.done + 1) מתוך \(live.total)"
        }
        if live.pictures > 1 { s += " · תמונה \(live.picture) מתוך \(live.pictures)" }
        return s
    }

    private func page(_ live: JobQueue.Live?) -> some View {
        let size = live?.imageSize ?? CGSize(width: 1, height: 1)
        let s = min(frameW / max(1, size.width), frameH / max(1, size.height))
        let ox = (frameW - size.width * s) / 2, oy = (frameH - size.height * s) / 2
        let boxes = live?.boxes ?? []
        let done = live?.done ?? 0
        let total = live?.total ?? 0
        let cur = min(done, boxes.count - 1)
        let lens: (x: CGFloat, y: CGFloat, on: Bool) = {
            if cur >= 0, cur < boxes.count {
                let b = boxes[cur]
                return (ox + (CGFloat(b.x) + CGFloat(b.w) / 2) * s - 50, oy + (CGFloat(b.y) + CGFloat(b.h) / 2) * s - 26, done < total)
            }
            return (frameW / 2 - 50, frameH / 2 - 26, true)
        }()
        return ZStack(alignment: .topLeading) {
            if let image = live?.image {
                Image(uiImage: image)
                    .resizable()
                    .frame(width: size.width * s, height: size.height * s)
                    .opacity(0.9)
                    .offset(x: ox, y: oy)
            }
            ForEach(boxes.indices, id: \.self) { i in
                let b = boxes[i]
                RoundedRectangle(cornerRadius: 4)
                    .fill(i < done ? Mk.sage.opacity(0.55) : i == done ? Mk.orange.opacity(0.32) : Mk.charcoal.opacity(0.1))
                    .frame(width: max(4, CGFloat(b.w) * s), height: max(3, CGFloat(b.h) * s))
                    .offset(x: ox + CGFloat(b.x) * s, y: oy + CGFloat(b.y) * s)
                    .animation(.easeInOut(duration: 0.3), value: done)
            }
            BinocularsLens()
                .stroke(Mk.orange, style: StrokeStyle(lineWidth: 3, lineCap: .round))
                .frame(width: 100, height: 52)
                .shadow(color: Mk.orange.opacity(0.45), radius: 10)
                .offset(x: lens.x, y: lens.y)
                .opacity(lens.on ? 1 : 0)
                .animation(Mk.ease(0.9), value: cur)
                .animation(.easeInOut(duration: 0.5), value: lens.on)
        }
        .frame(width: frameW, height: frameH, alignment: .topLeading)
        .background(Mk.card)
        .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
        .environment(\.layoutDirection, .leftToRight)
        .accessibilityHidden(true)
    }
}

/// The processing lens: two r22 circles at (26, 26) and (74, 26) and a bridge, in a 100x52 box.
struct BinocularsLens: Shape {
    func path(in r: CGRect) -> Path {
        let sx = r.width / 100, sy = r.height / 52
        var p = Path()
        p.addEllipse(in: CGRect(x: 4, y: 4, width: 44, height: 44))
        p.addEllipse(in: CGRect(x: 52, y: 4, width: 44, height: 44))
        p.move(to: CGPoint(x: 45, y: 18))
        p.addLine(to: CGPoint(x: 55, y: 18))
        return p.applying(CGAffineTransform(translationX: r.minX, y: r.minY).scaledBy(x: sx, y: sy))
    }
}

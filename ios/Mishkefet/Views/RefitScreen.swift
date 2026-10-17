import SwiftUI

/// The round's fine-tuning, driven by the real work: the samples grid, the progress ring
/// (features, then training steps), the held-out test, then before and after.
struct RefitScreen: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        if let state = app.refit {
            RefitContent(state: state)
        } else {
            Color.clear
        }
    }
}

private struct RefitContent: View {
    @Environment(AppModel.self) private var app
    let state: RefitState
    @State private var glow = false

    var body: some View {
        let stage = state.stage
        let done = stage == 3
        let ringOn = stage == 1 || stage == 2
        VStack(spacing: 0) {
            Text(verbatim: "התאמה אישית · סבב \(state.round)")
                .font(Mk.rubik(14))
                .foregroundStyle(Mk.nightSub)
                .frame(maxWidth: .infinity)
                .padding(.top, 40)

            ZStack {
                tiles
                    .opacity(stage == 0 ? 1 : 0)
                    .scaleEffect(stage == 0 ? 1 : 0.35)
                    .animation(Mk.ease(1.1), value: stage)
                ring
                    .opacity(ringOn ? 1 : 0)
                    .scaleEffect(ringOn ? 1 : 0.7)
                    .animation(Mk.ease(0.9), value: ringOn)
                scores
                    .opacity(done ? 1 : 0)
                    .offset(y: done ? 0 : 16)
                    .animation(Mk.ease(0.7), value: done)
            }
            .frame(height: 300)
            .padding(.top, 12)

            VStack(alignment: .leading, spacing: 2) {
                ForEach(steps.indices, id: \.self) { i in
                    stepRow(i, steps[i].0, steps[i].1, stage: stage)
                }
            }
            .padding(.horizontal, 28)
            .padding(.top, 6)
            .frame(maxWidth: .infinity, alignment: .leading)

            Spacer(minLength: 0)

            VStack(spacing: 6) {
                if state.result?.accepted == true {
                    Button { app.openComparePicker() } label: {
                        CTALabel(title: "לנסות שוב על סריקה קודמת")
                    }
                    .buttonStyle(PressStyle())
                }
                Button("להמשיך לכתוב") { app.continueWriting() }
                    .font(Mk.rubik(15))
                    .foregroundStyle(Mk.nightSub)
                    .frame(height: 44)
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 30)
            .opacity(done ? 1 : 0)
            .allowsHitTesting(done)
            .animation(.easeInOut(duration: 0.5), value: done)
        }
        .foregroundStyle(Mk.cream)
        .background(Mk.night.ignoresSafeArea())
        .environment(\.colorScheme, .dark)
    }

    private var steps: [(String, String)] {
        let n = state.tiles.isEmpty ? AppModel.roundSize : state.tiles.count
        return [
            ("אוספת את \(n) הדוגמאות שלך", "כל מילה שכתבת, יחד עם מה שהיה צריך להיות כתוב בה."),
            ("מכווננת את המודל", "רק השכבה האחרונה זזה, כך שמה שמשקפת כבר יודעת לא נשכח."),
            ("בודקת על מילים שלא ראתה", state.held > 0
                ? "\(state.held) מהמילים שלך נשמרו בצד, כדי למדוד אם באמת השתפרה."
                : "מילים שנשמרו בצד, כדי למדוד אם באמת השתפרה."),
        ]
    }

    private func stepRow(_ i: Int, _ h: String, _ d: String, stage: Int) -> some View {
        HStack(alignment: .top, spacing: 14) {
            ZStack {
                if stage > i {
                    Circle().fill(Mk.sage)
                    Icon(glyph: .check, size: 12, lineWidth: 3.6).foregroundStyle(Mk.charcoal)
                } else {
                    Circle().strokeBorder(stage == i ? Mk.orange : Mk.cream.opacity(0.3), lineWidth: 2)
                        .modifier(Blink(on: stage == i))
                }
            }
            .frame(width: 22, height: 22)
            .padding(.top, 1)
            VStack(alignment: .leading, spacing: 3) {
                Text(verbatim: h).font(Mk.rubik(16, .semibold))
                Text(verbatim: d).font(Mk.rubik(14)).lineSpacing(3).foregroundStyle(Mk.nightSub)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(.vertical, 10)
        .opacity(stage >= i ? 1 : 0.35)
        .animation(.easeInOut(duration: 0.4), value: stage)
    }

    private var tiles: some View {
        let cols = Array(repeating: GridItem(.fixed(34), spacing: 6), count: 8)
        return LazyVGrid(columns: cols, spacing: 6) {
            ForEach(0..<AppModel.roundSize, id: \.self) { i in
                ZStack {
                    RoundedRectangle(cornerRadius: 7).fill(Mk.nightCard)
                    if i < state.tiles.count {
                        let t = state.tiles[i]
                        if let strokes = t.strokes, !strokes.isEmpty {
                            StrokeTile(strokes: strokes)
                                .stroke(Mk.cream, style: StrokeStyle(lineWidth: 1.4, lineCap: .round, lineJoin: .round))
                                .padding(3)
                        } else {
                            Text(verbatim: t.word).font(Mk.amatic(15)).foregroundStyle(Mk.cream).lineLimit(1)
                        }
                    }
                }
                .frame(width: 34, height: 26)
            }
        }
        .frame(width: 8 * 34 + 7 * 6)
        .environment(\.layoutDirection, .rightToLeft)
    }

    private var ring: some View {
        ZStack {
            Circle().fill(Mk.sage).opacity(0.25).padding(26)
                .scaleEffect(glow ? 1.045 : 1)
                .onAppear {
                    withAnimation(.easeInOut(duration: 1.3).repeatForever(autoreverses: true)) { glow = true }
                }
            Circle().stroke(Mk.cream.opacity(0.12), lineWidth: 6).padding(8)
            Circle()
                .trim(from: 0, to: CGFloat(state.pct) / 100)
                .stroke(Mk.orange, style: StrokeStyle(lineWidth: 6, lineCap: .round))
                .rotationEffect(.degrees(-90))
                .padding(8)
                .animation(.linear(duration: 0.12), value: state.pct)
            if state.stage >= 2 {
                Icon(glyph: .check, size: 56, lineWidth: 2.6).foregroundStyle(Mk.cream)
                    .transition(.scale.combined(with: .opacity))
            } else {
                Text(verbatim: "\(state.pct)%")
                    .font(Mk.karantina(72, bold: false))
                    .monospacedDigit()
            }
        }
        .frame(width: 200, height: 200)
        .environment(\.layoutDirection, .leftToRight)
    }

    private var scores: some View {
        VStack(spacing: 6) {
            if state.failed {
                Text("משהו השתבש בהתאמה. הדוגמאות שמורות, ונסה שוב בסבב הבא.")
                    .font(Mk.rubik(17))
                    .lineSpacing(5)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 30)
            } else if let r = state.result {
                Text("דיוק על כתב היד שלך")
                    .font(Mk.rubik(15))
                    .foregroundStyle(Mk.nightSub)
                HStack(alignment: .bottom, spacing: 26) {
                    score(r.before, label: "לפני", size: 56, color: Mk.cream.opacity(0.45))
                    score(r.after, label: "אחרי", size: 104, color: r.accepted ? Mk.orange : Mk.cream.opacity(0.45))
                }
                // Physical order: before on the left, after on the right.
                .environment(\.layoutDirection, .leftToRight)
                if !r.accepted {
                    Text("הפעם לא השתפרה, אז נשארת הגרסה הקודמת. עוד סבב בדרך כלל עוזר.")
                        .font(Mk.rubik(15))
                        .lineSpacing(5)
                        .foregroundStyle(Mk.cream.opacity(0.75))
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 30)
                        .padding(.top, 6)
                }
            }
        }
    }

    private func score(_ v: Double, label: String, size: CGFloat, color: Color) -> some View {
        VStack(spacing: 4) {
            Text(verbatim: "\(Int((v * 100).rounded()))%")
                .font(Mk.karantina(size))
                .foregroundStyle(color)
            Text(verbatim: label)
                .font(Mk.rubik(13))
                .foregroundStyle(Mk.nightSub)
        }
    }
}

/// A sample's strokes, aspect-fit with 8% padding.
private struct StrokeTile: Shape {
    var strokes: [[CGPoint]]

    func path(in rect: CGRect) -> Path {
        let pts = strokes.flatMap { $0 }
        guard let x0 = pts.map(\.x).min(), let x1 = pts.map(\.x).max(),
              let y0 = pts.map(\.y).min(), let y1 = pts.map(\.y).max() else { return Path() }
        let w = max(1, x1 - x0), h = max(1, y1 - y0)
        let pad = max(w, h) * 0.08
        let s = min(rect.width / (w + 2 * pad), rect.height / (h + 2 * pad))
        let ox = rect.minX + (rect.width - (w + 2 * pad) * s) / 2
        let oy = rect.minY + (rect.height - (h + 2 * pad) * s) / 2
        var p = Path()
        for stroke in strokes {
            guard let first = stroke.first else { continue }
            func map(_ q: CGPoint) -> CGPoint { CGPoint(x: ox + (q.x - x0 + pad) * s, y: oy + (q.y - y0 + pad) * s) }
            p.move(to: map(first))
            if stroke.count == 1 { p.addLine(to: map(CGPoint(x: first.x + 0.5, y: first.y))) }
            for q in stroke.dropFirst() { p.addLine(to: map(q)) }
        }
        return p
    }
}

private struct Blink: ViewModifier {
    var on: Bool
    @State private var dim = false

    func body(content: Content) -> some View {
        content
            .opacity(on && dim ? 0.25 : 1)
            .onAppear { start() }
            .onChange(of: on) { _, _ in start() }
    }

    private func start() {
        guard on else { dim = false; return }
        withAnimation(.easeInOut(duration: 0.55).repeatForever(autoreverses: true)) { dim = true }
    }
}

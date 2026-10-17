import MishkefetCore
import SwiftUI
import UIKit

struct TrainIntroScreen: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                CircleIconButton(glyph: .close, size: 18, lineWidth: 2.2, label: "סגירה") { app.go(.home) }
                Spacer()
            }
            .padding(.horizontal, 10)
            .frame(height: 56)

            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    Text("ללמד את משקפת לקרוא אותך")
                        .font(Mk.karantina(58))
                        .lineSpacing(-10)
                        .fixedSize(horizontal: false, vertical: true)
                    Text("כל כתב יד שונה. כמה דקות של כתיבה על המסך, והמודל מתכוונן לאיך שאתם כותבים.")
                        .font(Mk.rubik(16))
                        .lineSpacing(6)
                        .foregroundStyle(Mk.sub)
                    VStack(spacing: 2) {
                        step(1, "כתבו את המילה שמופיעה", "באצבע או בעט, כמו שאתם כותבים על נייר.")
                        Rectangle().fill(Mk.line).frame(height: 1)
                        step(2, "עצרו, והיא ממשיכה לבד", "ירוק אם קראה נכון, אדום אם לא. גם הטעויות נשמרות ללימוד.")
                        Rectangle().fill(Mk.line).frame(height: 1)
                        step(3, "כל \(AppModel.roundSize) מילים, התאמה", "המודל מתאמן מחדש על הדוגמאות שלכם, ואפשר לנסות שוב סריקה ישנה.")
                    }
                    .padding(.top, 22)
                }
                .padding(.horizontal, 26)
                .padding(.top, 8)
                .frame(maxWidth: .infinity, alignment: .leading)
            }

            VStack(spacing: 12) {
                HStack(spacing: 8) {
                    Circle().fill(Mk.sage).frame(width: 8, height: 8)
                    Text("הכל על המכשיר. הדוגמאות לא עולות לשום מקום.").font(Mk.rubik(13))
                }
                Button { app.beginWriting() } label: {
                    CTALabel(title: "להתחיל לכתוב")
                }
                .buttonStyle(PressStyle())
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 30)
        }
    }

    private func step(_ n: Int, _ h: String, _ d: String) -> some View {
        HStack(alignment: .top, spacing: 16) {
            Text(verbatim: "\(n)")
                .font(Mk.karantina(40))
                .foregroundStyle(Mk.orange)
                .frame(width: 22, height: 32)
            VStack(alignment: .leading, spacing: 2) {
                Text(verbatim: h).font(Mk.rubik(17, .semibold))
                Text(verbatim: d).font(Mk.rubik(14)).foregroundStyle(Mk.sub)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.vertical, 12)
    }
}

struct TrainScreen: View {
    @Environment(AppModel.self) private var app

    enum Phase { case idle, drawing, waiting, reading, ok, bad, out }

    @State private var phase: Phase = .idle
    @State private var word = ""
    @State private var count = 0
    @State private var readAs = ""
    @State private var lastOK = false
    @State private var empty = true
    @State private var picker: WordPicker?
    @State private var handle = CanvasHandle()
    @State private var waitTask: Task<Void, Never>?
    @State private var flowTask: Task<Void, Never>?

    private let size = AppModel.roundSize
    private let inactivity = Double(AdaptConfig.shipped.inactivityMs) / 1000

    var body: some View {
        let out = phase == .out
        VStack(spacing: 0) {
            HStack {
                CircleIconButton(glyph: .close, size: 18, lineWidth: 2.2, label: "סגירה") { close() }
                Spacer()
                Text(verbatim: "\(count) / \(size)")
                    .font(Mk.rubik(15, .medium))
                    .monospacedDigit()
                    .environment(\.layoutDirection, .leftToRight)
                Spacer()
                Button("דילוג") { skip() }
                    .font(Mk.rubik(14))
                    .foregroundStyle(Mk.sub)
                    .padding(.horizontal, 14)
                    .frame(height: 40)
            }
            .padding(.horizontal, 10)
            .frame(height: 56)

            HStack(spacing: 2) {
                ForEach(0..<size, id: \.self) { i in
                    Capsule().fill(i < count ? Mk.orange : Mk.line).frame(height: 4)
                }
            }
            .padding(.horizontal, 18)
            .animation(.easeInOut(duration: 0.3), value: count)

            VStack(spacing: 4) {
                Text("כתבו").font(Mk.rubik(14)).foregroundStyle(Mk.sub)
                Text(verbatim: word)
                    .font(Mk.rubik(46, .semibold))
                    .frame(height: 54)
            }
            .padding(.top, 28)
            .padding(.bottom, 18)
            .opacity(out ? 0 : 1)
            .offset(y: out ? -10 : 0)
            .animation(.easeOut(duration: 0.22), value: out)

            pad
                .padding(.horizontal, 16)
                .opacity(out ? 0 : 1)
                .offset(y: out ? -10 : 0)
                .animation(.easeOut(duration: 0.22), value: out)

            HStack {
                Text(verbatim: feedback)
                    .font(Mk.rubik(15, .medium))
                    .foregroundStyle(feedbackColor)
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
                Spacer()
                Button { clear() } label: { PillLabel(title: "ניקוי", small: true) }
                    .buttonStyle(PressStyle())
            }
            .padding(.horizontal, 18)
            .frame(height: 64)
            .padding(.bottom, 30)
        }
        .onChange(of: app.screen) { _, s in
            if s == .train { enter() } else { leave() }
        }
        .onAppear { if app.screen == .train { enter() } }
    }

    private var pad: some View {
        let ok = phase == .ok, bad = phase == .bad
        let showOK = ok || (phase == .out && lastOK)
        return ZStack(alignment: .top) {
            RoundedRectangle(cornerRadius: 30, style: .continuous).fill(Mk.card)
            GeometryReader { g in
                Path { p in
                    p.move(to: CGPoint(x: 24, y: g.size.height * 0.62))
                    p.addLine(to: CGPoint(x: g.size.width - 24, y: g.size.height * 0.62))
                }
                .stroke(Mk.line, style: StrokeStyle(lineWidth: 1.5, dash: [5, 4]))
            }
            if empty {
                Text("כתבו כאן")
                    .font(Mk.rubik(15))
                    .foregroundStyle(Mk.sub)
                    .opacity(0.65)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .allowsHitTesting(false)
            }
            StrokeCanvas(handle: handle, onDown: touchDown, onUp: touchUp)
                .environment(\.layoutDirection, .leftToRight)
            ZStack {
                Circle().fill(showOK ? Mk.green : Mk.red)
                Icon(glyph: showOK ? .check : .close, size: showOK ? 28 : 24, lineWidth: showOK ? 2.8 : 2.2)
                    .foregroundStyle(.white)
            }
            .frame(width: 60, height: 60)
            .scaleEffect(ok || bad ? 1 : 0.4)
            .opacity(ok || bad ? 1 : 0)
            .animation(.spring(response: 0.35, dampingFraction: 0.55), value: phase)
            .padding(.top, 18)
            .allowsHitTesting(false)
            // The wait bar grows from the reading start (right) while the inactivity timer runs.
            GeometryReader { g in
                Rectangle()
                    .fill(Mk.orange)
                    .frame(width: phase == .waiting ? g.size.width : 0, height: 4)
                    .animation(phase == .waiting ? .linear(duration: inactivity) : nil, value: phase)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomLeading)
            }
            .allowsHitTesting(false)
        }
        .clipShape(RoundedRectangle(cornerRadius: 30, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 30, style: .continuous)
            .strokeBorder(ok ? Mk.green : bad ? Mk.red : .clear, lineWidth: 2))
        .animation(.easeInOut(duration: 0.25), value: phase)
    }

    private var feedback: String {
        switch phase {
        case .ok: return "נכון · קראתי „\(readAs)\""
        case .bad: return readAs.isEmpty ? "לא הצלחתי לקרוא · נשמר ללימוד" : "קראתי „\(readAs)\" · נשמר ללימוד"
        case .waiting, .reading: return "ממתינה שתסיימו…"
        case .drawing: return ""
        case .idle, .out: return "עצרו כשתסיימו, ונמשיך לבד"
        }
    }

    private var feedbackColor: Color {
        switch phase {
        case .ok: return Mk.green
        case .bad: return Color(hex: 0xd81516)
        default: return Mk.sub
        }
    }

    // MARK: Flow

    private func enter() {
        leave()
        var p = app.makePicker()
        word = p.next()
        picker = p
        count = app.roundCount
        phase = .idle
        empty = true
        handle.clear()
        handle.setAccepting(true)
        if count >= size { app.roundComplete() }
    }

    private func leave() {
        waitTask?.cancel()
        flowTask?.cancel()
        waitTask = nil
        flowTask = nil
    }

    private func touchDown() {
        guard [.idle, .drawing, .waiting].contains(phase) else { return }
        waitTask?.cancel()
        empty = false
        phase = .drawing
    }

    private func touchUp() {
        guard phase == .drawing else { return }
        phase = .waiting
        waitTask = Task {
            try? await Task.sleep(for: .seconds(inactivity))
            guard !Task.isCancelled, phase == .waiting else { return }
            evaluate()
        }
    }

    private func evaluate() {
        let target = word
        let strokes = handle.strokes
        guard !strokes.isEmpty else { phase = .idle; return }
        phase = .reading
        handle.setAccepting(false)
        flowTask = Task {
            let read = await app.readStrokes(strokes, word: target)
            guard !Task.isCancelled else { return }
            let ok = read == target
            readAs = read
            lastOK = ok
            handle.recolor(UIColor(hex: ok ? 0x2f9a4f : 0xff1b1c))
            phase = ok ? .ok : .bad
            UINotificationFeedbackGenerator().notificationOccurred(ok ? .success : .error)
            app.recordAttempt(word: target, strokes: strokes, readAs: read)
            try? await Task.sleep(for: .milliseconds(1150))
            guard !Task.isCancelled else { return }
            phase = .out
            try? await Task.sleep(for: .milliseconds(250))
            guard !Task.isCancelled else { return }
            advance()
        }
    }

    private func advance() {
        handle.clear()
        handle.setAccepting(true)
        empty = true
        count += 1
        if count >= size {
            app.roundComplete()
            return
        }
        word = picker?.next() ?? word
        phase = .idle
    }

    private func clear() {
        waitTask?.cancel()
        guard [.idle, .drawing, .waiting].contains(phase) else { return }
        handle.clear()
        empty = true
        phase = .idle
    }

    private func skip() {
        guard [.idle, .drawing, .waiting].contains(phase) else { return }
        clear()
        word = picker?.next() ?? word
    }

    private func close() {
        leave()
        app.refreshRoundCount()
        app.go(.home)
    }
}

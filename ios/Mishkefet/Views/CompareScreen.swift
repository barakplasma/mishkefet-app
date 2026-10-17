import MishkefetCore
import SwiftUI

/// A scan re-read with the newly fine-tuned head, against the reading it had.
/// Lines are matched by position (a page re-read keeps the scan's segmentation).
struct CompareScreen: View {
    @Environment(AppModel.self) private var app
    @State private var side: TextDiff.Side = .after

    var body: some View {
        if let c = app.compare, let scan = app.scan(c.scanID) {
            content(c, title: scan.title)
                .onChange(of: c.scanID) { _, _ in side = .after }
        } else {
            Color.clear
        }
    }

    private func content(_ c: CompareState, title: String) -> some View {
        let rows = diffRows(c)
        let changed = rows.after.reduce(0) { $0 + $1.filter(\.changed).count }
        let total = rows.after.reduce(0) { $0 + $1.count }
        let shown = side == .after ? rows.after : rows.before
        return VStack(spacing: 0) {
            HStack {
                CircleIconButton(glyph: .chevronBack, lineWidth: 2, label: "חזרה") { app.leaveCompare() }
                Spacer()
                Segmented(options: [(TextDiff.Side.before, "לפני"), (TextDiff.Side.after, "אחרי")], selection: $side, label: "השוואה")
            }
            .padding(.leading, 8)
            .padding(.trailing, 12)
            .frame(height: 52)
            .padding(.top, 8)

            VStack(alignment: .leading, spacing: 8) {
                Text(verbatim: title)
                    .font(Mk.karantina(46))
                    .lineSpacing(-4)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
                HStack(spacing: 8) {
                    if c.after != nil {
                        Text(verbatim: changed > 0 ? "\(changed) מתוך \(total) מילים השתנו" : "אין שינויים")
                            .font(Mk.rubik(12.5, .medium))
                            .foregroundStyle(Mk.charcoal)
                            .padding(.horizontal, 9)
                            .padding(.vertical, 2)
                            .background(RoundedRectangle(cornerRadius: 8).fill(side == .after ? Mk.sage : Mk.red.opacity(0.16)))
                    } else {
                        Spinner()
                        Text("קוראת מחדש עם המודל המותאם").font(Mk.rubik(14)).foregroundStyle(Mk.sub)
                    }
                    Text(verbatim: modelLabel(c))
                        .font(Mk.rubik(14))
                        .foregroundStyle(Mk.sub)
                        .lineLimit(1)
                }
            }
            .padding(.horizontal, 22)
            .padding(.top, 2)
            .padding(.bottom, 14)

            ScrollView {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(shown.indices, id: \.self) { i in
                        VStack(alignment: .leading, spacing: 4) {
                            Text(verbatim: "שורה \(i + 1)").font(Mk.rubik(12.5)).foregroundStyle(Mk.sub)
                            DiffText(tokens: shown[i], side: side)
                        }
                        .padding(.vertical, 10)
                        .padding(.horizontal, 12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                .padding(.horizontal, 10)
                .padding(.bottom, 12)
            }

            VStack(spacing: 4) {
                Button { app.keepCompare(new: true) } label: {
                    CTALabel(title: "לשמור את הקריאה החדשה", height: 58)
                        .opacity(c.after == nil ? 0.5 : 1)
                }
                .buttonStyle(PressStyle())
                .disabled(c.after == nil)
                Button("להשאיר את הקודמת") { app.keepCompare(new: false) }
                    .font(Mk.rubik(14))
                    .foregroundStyle(Mk.sub)
                    .frame(height: 40)
            }
            .padding(.top, 10)
            .padding(.horizontal, 16)
            .padding(.bottom, 30)
        }
    }

    private func modelLabel(_ c: CompareState) -> String {
        if side == .after {
            return "מודל מותאם · סבב \(ModelID.roundNumber(of: c.model) ?? 0)"
        }
        if let n = ModelID.roundNumber(of: c.beforeModel) { return "מודל מותאם · סבב \(n)" }
        return "משקפת v1, לפני ההתאמה"
    }

    private func diffRows(_ c: CompareState) -> (before: [[TextDiff.Token]], after: [[TextDiff.Token]]) {
        guard let after = c.after else { return ([], []) }
        let n = max(c.before.count, after.count)
        var b: [[TextDiff.Token]] = [], a: [[TextDiff.Token]] = []
        for i in 0..<n {
            let x = i < c.before.count ? c.before[i].text : ""
            let y = i < after.count ? after[i].text : ""
            b.append(TextDiff.words(before: x, after: y, side: .before))
            a.append(TextDiff.words(before: x, after: y, side: .after))
        }
        return (b, a)
    }
}

/// A line's words; changed words are highlighted sage (after) or get a red wavy underline (before).
private struct DiffText: View {
    let tokens: [TextDiff.Token]
    let side: TextDiff.Side

    var body: some View {
        WrapLayout(spacing: 5, lineSpacing: 6) {
            ForEach(tokens.indices, id: \.self) { i in
                let t = tokens[i]
                Text(verbatim: t.text)
                    .font(Mk.rubik(18, relativeTo: .body))
                    .foregroundStyle(Mk.ink)
                    .padding(.horizontal, t.changed && side == .after ? 4 : 0)
                    .background {
                        if t.changed && side == .after {
                            RoundedRectangle(cornerRadius: 6).fill(Mk.sage.opacity(0.75))
                        }
                    }
                    .overlay(alignment: .bottom) {
                        if t.changed && side == .before {
                            WavyLine().stroke(Mk.red, lineWidth: 1.5).frame(height: 4).offset(y: 5)
                        }
                    }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private struct WavyLine: Shape {
    func path(in r: CGRect) -> Path {
        var p = Path()
        let step: CGFloat = 4
        p.move(to: CGPoint(x: r.minX, y: r.midY))
        var x = r.minX
        var up = true
        while x < r.maxX {
            let nx = min(r.maxX, x + step)
            p.addQuadCurve(to: CGPoint(x: nx, y: r.midY), control: CGPoint(x: (x + nx) / 2, y: up ? r.minY - 1 : r.maxY + 1))
            x = nx
            up.toggle()
        }
        return p
    }
}

/// Flows children in lines (respects the layout direction, so Hebrew words run right to left).
struct WrapLayout: Layout {
    var spacing: CGFloat = 5
    var lineSpacing: CGFloat = 6

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? .infinity
        let rows = arrange(width: width, subviews: subviews)
        let h = rows.reduce(0) { $0 + $1.height } + CGFloat(max(0, rows.count - 1)) * lineSpacing
        let w = rows.map(\.width).max() ?? 0
        return CGSize(width: proposal.width ?? w, height: h)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let rows = arrange(width: bounds.width, subviews: subviews)
        var y = bounds.minY
        for row in rows {
            var x = bounds.minX
            for i in row.indices {
                let s = subviews[i].sizeThatFits(.unspecified)
                subviews[i].place(at: CGPoint(x: x, y: y + (row.height - s.height) / 2), proposal: ProposedViewSize(s))
                x += s.width + spacing
            }
            y += row.height + lineSpacing
        }
    }

    private struct Row {
        var indices: [Int] = []
        var width: CGFloat = 0
        var height: CGFloat = 0
    }

    private func arrange(width: CGFloat, subviews: Subviews) -> [Row] {
        var rows: [Row] = []
        var cur = Row()
        for i in subviews.indices {
            let s = subviews[i].sizeThatFits(.unspecified)
            let add = cur.indices.isEmpty ? s.width : cur.width + spacing + s.width
            if add > width && !cur.indices.isEmpty {
                rows.append(cur)
                cur = Row()
            }
            cur.width = cur.indices.isEmpty ? s.width : cur.width + spacing + s.width
            cur.height = max(cur.height, s.height)
            cur.indices.append(i)
        }
        if !cur.indices.isEmpty { rows.append(cur) }
        return rows
    }
}

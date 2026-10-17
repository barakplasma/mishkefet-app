import MishkefetCore
import SwiftUI
import UIKit

struct ResultScreen: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        if let scan = app.scan(app.currentID) {
            ResultContent(scan: scan)
                .id(scan.id)
        } else {
            Color.clear
        }
    }
}

private struct ResultContent: View {
    @Environment(AppModel.self) private var app
    let scan: Scan

    @State private var active = 0
    @State private var showImage = true
    @State private var editing = false
    @State private var drafts: [String] = []
    @State private var titleDraft = ""
    @State private var copied = false
    @State private var copyTask: Task<Void, Never>?
    @State private var image: UIImage?

    var body: some View {
        let reading = scan.reading
        let lines = reading?.lines ?? []
        let isTuned = (reading?.model ?? ModelID.base) != ModelID.base
        VStack(spacing: 0) {
            header
            head(lines: lines, isTuned: isTuned)
            if showImage {
                ScanImageFrame(image: image, imageSize: CGSize(width: scan.width, height: scan.height),
                               boxes: lines.map(\.box), active: active, height: 170) { i in
                    withAnimation(.easeInOut(duration: 0.3)) { active = i }
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 8)
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 2) {
                        ForEach(Array(lines.enumerated()), id: \.offset) { i, line in
                            LineRow(index: i, text: line.text, isActive: i == active, editing: editing,
                                    draft: Binding(get: { i < drafts.count ? drafts[i] : line.text },
                                                   set: { if i < drafts.count { drafts[i] = $0 } }))
                                .id(i)
                                .onTapGesture { withAnimation(.easeInOut(duration: 0.3)) { active = i } }
                        }
                        if lines.isEmpty {
                            Text("לא מצאתי כאן כתב יד. נסו לצלם שוב, קרוב יותר ובאור טוב.")
                                .font(Mk.rubik(16))
                                .lineSpacing(6)
                                .foregroundStyle(Mk.sub)
                                .padding(.vertical, 40)
                                .padding(.horizontal, 12)
                        }
                    }
                    .padding(.horizontal, 10)
                    .padding(.bottom, 12)
                }
                .scrollDismissesKeyboard(.interactively)
                .onChange(of: active) { _, i in
                    withAnimation(Mk.ease(0.5)) { proxy.scrollTo(i, anchor: .center) }
                }
            }
            if !editing { bottomBar(lines: lines) }
        }
        .background(Mk.bg)
        .task(id: scan.imageFileName) {
            let url = scan.imageURL
            image = await Task.detached { UIImage(contentsOfFile: url.path) }.value
        }
        .onChange(of: app.screen) { _, s in
            if s != .result && editing { withAnimation { editing = false } }
        }
    }

    // MARK: Header

    private var header: some View {
        HStack {
            CircleIconButton(glyph: .chevronBack, lineWidth: 2, label: "חזרה") {
                if editing {
                    withAnimation(.easeInOut(duration: 0.25)) { editing = false }
                } else {
                    app.backFromResult()
                }
            }
            Spacer()
            HStack(spacing: 4) {
                if editing {
                    CircleIconButton(glyph: .trash, size: 22, label: "מחיקה") { app.deleteCurrentScan() }
                    CircleIconButton(glyph: .check, size: 22, lineWidth: 2.4, label: "שמירה",
                                     background: Mk.orange, foreground: Mk.charcoal) { saveEdit() }
                } else {
                    CircleIconButton(glyph: .gallery, size: 22, label: "תמונה", background: showImage ? Mk.card : .clear) {
                        withAnimation(Mk.ease(0.4)) { showImage.toggle() }
                    }
                    CircleIconButton(glyph: .pencil, size: 22, label: "עריכה") { startEdit() }
                }
            }
        }
        .padding(.leading, 8)
        .padding(.trailing, 10)
        .frame(height: 52)
        .padding(.top, 8)
    }

    private func head(lines: [LineResult], isTuned: Bool) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            if editing {
                TextField("כותרת", text: $titleDraft)
                    .font(Mk.karantina(46))
                    .foregroundStyle(Mk.ink)
                    .submitLabel(.done)
                    .padding(.bottom, 2)
                    .overlay(alignment: .bottom) { Rectangle().fill(Mk.orange).frame(height: 2) }
            } else {
                Text(verbatim: scan.title)
                    .font(Mk.karantina(46))
                    .foregroundStyle(Mk.ink)
                    .lineSpacing(-4)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            HStack(spacing: 8) {
                Text(verbatim: HebrewDate.label(scan.createdAt))
                    .font(Mk.rubik(14))
                    .foregroundStyle(Mk.sub)
                Chip(text: Titles.lines(lines.count))
                Chip(text: scan.kind == .page ? "דף" : "שורה")
                if isTuned { Chip(text: "מותאם לכתב היד שלך", sage: true) }
            }
            if app.hasTuned && !editing {
                HStack(spacing: 10) {
                    ModelToggle(useTuned: isTuned) { app.switchModel(scan, useTuned: $0) }
                    if app.rereadingID == scan.id {
                        HStack(spacing: 8) {
                            Spinner()
                            Text("קוראת מחדש").font(Mk.rubik(14)).foregroundStyle(Mk.sub)
                        }
                        .transition(.opacity)
                    }
                }
                .padding(.top, 4)
                .animation(.easeInOut(duration: 0.25), value: app.rereadingID)
            }
        }
        .padding(.horizontal, 22)
        .padding(.top, 2)
        .padding(.bottom, 12)
    }

    private func bottomBar(lines: [LineResult]) -> some View {
        HStack(spacing: 10) {
            Button { copy(lines) } label: {
                CTALabel(title: copied ? "הועתק" : "העתקת הטקסט", glyph: copied ? .check : .copy,
                         fill: copied ? Mk.sage : Mk.orange, height: 58)
                    .animation(.easeInOut(duration: 0.3), value: copied)
            }
            .buttonStyle(PressStyle())
            Button { share(lines) } label: {
                Icon(glyph: .share, size: 22, lineWidth: 1.9)
                    .foregroundStyle(Mk.ink)
                    .frame(width: 58, height: 58)
                    .background(Circle().fill(Mk.card))
            }
            .buttonStyle(PressStyle(scale: 0.94))
            .accessibilityLabel("שיתוף")
        }
        .padding(.top, 10)
        .padding(.horizontal, 16)
        .padding(.bottom, 30)
    }

    // MARK: Actions

    private func startEdit() {
        drafts = scan.lines.map(\.text)
        titleDraft = scan.title
        withAnimation(.easeInOut(duration: 0.25)) { editing = true }
    }

    private func saveEdit() {
        app.saveEdits(scan, drafts: drafts, title: titleDraft)
        withAnimation(.easeInOut(duration: 0.25)) { editing = false }
    }

    private func copy(_ lines: [LineResult]) {
        UIPasteboard.general.string = ReadingText.plain(lines.map(\.text))
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
        copied = true
        copyTask?.cancel()
        copyTask = Task {
            try? await Task.sleep(for: .seconds(1.6))
            if !Task.isCancelled { copied = false }
        }
    }

    private func share(_ lines: [LineResult]) {
        ShareSheet.present(items: [ReadingText.plain(lines.map(\.text))]) { _ in }
    }
}

private struct LineRow: View {
    let index: Int
    let text: String
    let isActive: Bool
    let editing: Bool
    @Binding var draft: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(verbatim: "שורה \(index + 1)")
                .font(Mk.rubik(12.5))
                .foregroundStyle(Mk.sub)
            if editing {
                TextField("", text: $draft, axis: .vertical)
                    .font(Mk.rubik(18, relativeTo: .body))
                    .lineSpacing(18 * 0.62 - 4)
                    .foregroundStyle(Mk.ink)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(RoundedRectangle(cornerRadius: 12).fill(Mk.card))
            } else {
                Text(verbatim: text)
                    .font(Mk.rubik(18, relativeTo: .body))
                    .lineSpacing(18 * 0.62 - 4)
                    .foregroundStyle(Mk.ink)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(.vertical, 10)
        .padding(.horizontal, 12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 20).fill(Mk.sage.opacity(isActive && !editing ? 0.42 : 0)))
        .contentShape(Rectangle())
        .accessibilityElement(children: .combine)
    }
}

import SwiftData
import SwiftUI
import UIKit

/// The design's bottom sheets: 44pt radius card inset 8pt, grabber, dimmed backdrop.
struct SheetHost: View {
    @Environment(AppModel.self) private var app
    @State private var drag: CGFloat = 0

    var body: some View {
        ZStack(alignment: .bottom) {
            if app.sheet != nil {
                Color(red: 20 / 255, green: 20 / 255, blue: 20 / 255).opacity(0.4)
                    .ignoresSafeArea()
                    .onTapGesture { app.closeSheet() }
                    .transition(.opacity)
                    .accessibilityLabel("סגירה")
                    .accessibilityAddTraits(.isButton)
            }
            if let sheet = app.sheet {
                VStack(alignment: .leading, spacing: 14) {
                    Capsule()
                        .fill(Mk.line)
                        .frame(width: 40, height: 5)
                        .frame(maxWidth: .infinity)
                        .padding(.bottom, 4)
                    content(sheet)
                }
                .padding(.top, 10)
                .padding(.horizontal, 22)
                .padding(.bottom, 26)
                .background(RoundedRectangle(cornerRadius: 44, style: .continuous).fill(Mk.bg))
                .shadow(color: .black.opacity(0.3), radius: 20, y: -10)
                .padding(8)
                .offset(y: max(0, drag))
                .gesture(DragGesture()
                    .onChanged { drag = $0.translation.height }
                    .onEnded { v in
                        if v.translation.height > 90 || v.predictedEndTranslation.height > 220 { app.closeSheet() }
                        withAnimation(Mk.ease(0.35)) { drag = 0 }
                    })
                .transition(.move(edge: .bottom))
                .zIndex(1)
            }
        }
        .ignoresSafeArea(.container, edges: .bottom)
    }

    @ViewBuilder
    private func content(_ sheet: SheetKind) -> some View {
        switch sheet {
        case .newScan: NewScanSheet()
        case .comparePick: ComparePickSheet()
        case .about: AboutSheet()
        }
    }
}

private struct SheetTitle: View {
    var text: String
    var body: some View {
        Text(verbatim: text)
            .font(Mk.karantina(40))
            .foregroundStyle(Mk.ink)
            .frame(height: 40, alignment: .center)
    }
}

private struct SheetRow<Trailing: View>: View {
    var title: String
    var danger = false
    var action: () -> Void
    @ViewBuilder var trailing: Trailing

    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                Text(verbatim: title).font(Mk.rubik(17, .medium))
                Spacer(minLength: 0)
                trailing
            }
            .foregroundStyle(danger ? Mk.red : Mk.ink)
            .padding(.horizontal, 20)
            .frame(minHeight: 60)
            .background(RoundedRectangle(cornerRadius: 20).fill(Mk.card))
            .contentShape(Rectangle())
        }
        .buttonStyle(PressStyle())
    }
}

private struct SubText: View {
    var text: String
    var mono = false
    var body: some View {
        Text(verbatim: text)
            .font(mono ? .system(size: 14, design: .monospaced) : Mk.rubik(14))
            .foregroundStyle(Mk.sub)
    }
}

/// "סריקה חדשה": camera, gallery or file, and which model reads it.
private struct NewScanSheet: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        SheetTitle(text: "סריקה חדשה")
        if app.hasTuned {
            HStack(spacing: 10) {
                Text("לקרוא עם").font(Mk.rubik(15)).foregroundStyle(Mk.sub)
                Spacer()
                ModelToggle(useTuned: app.useTuned) { app.setUseTuned($0) }
            }
        }
        SheetRow(title: "צילום", action: { app.openCamera() }) { SubText(text: "שורה או דף") }
        SheetRow(title: "בחירה מהגלריה", action: { app.pickPhotos() }) { SubText(text: "כמה תמונות בבת אחת") }
        SheetRow(title: "קובץ", action: { app.pickFiles() }) {
            SubText(text: ".jpg .png .heic .pdf", mono: true).environment(\.layoutDirection, .leftToRight)
        }
    }
}

/// "על איזו סריקה?": scans to re-read with the new head. Ordered internally by the lowest mean
/// confidence (the ones most likely to change); the confidence itself is never shown.
private struct ComparePickSheet: View {
    @Environment(AppModel.self) private var app
    @Query private var scans: [Scan]

    var body: some View {
        let ordered = Array(scans.sorted { $0.meanConfidence < $1.meanConfidence }.prefix(8))
        SheetTitle(text: "על איזו סריקה?")
        Text("נקרא אותה שוב עם המודל המותאם ונראה מה השתנה.")
            .font(Mk.rubik(15))
            .foregroundStyle(Mk.sub)
            .padding(.top, -6)
        ScrollView {
            VStack(spacing: 10) {
                ForEach(ordered) { s in
                    Button { app.closeSheet(); app.compare(s) } label: {
                        HStack(spacing: 12) {
                            ScanThumb(url: s.thumbURL, width: 34, height: 42, radius: 7)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(verbatim: s.title).font(Mk.rubik(16, .semibold)).lineLimit(1)
                                Text(verbatim: "\(HebrewDate.label(s.createdAt)) · \(Titles.lines(s.lines.count))")
                                    .font(Mk.rubik(13))
                                    .foregroundStyle(Mk.sub)
                            }
                            Spacer(minLength: 0)
                            Icon(glyph: .chevron, size: 20, lineWidth: 2).opacity(0.5)
                        }
                        .foregroundStyle(Mk.ink)
                        .padding(.leading, 14)
                        .padding(.trailing, 16)
                        .frame(height: 64)
                        .background(RoundedRectangle(cornerRadius: 20).fill(Mk.card))
                    }
                    .buttonStyle(PressStyle())
                }
                if ordered.isEmpty {
                    Text("עוד אין סריקות. צלמו דף, ואז אפשר להשוות.")
                        .font(Mk.rubik(15))
                        .foregroundStyle(Mk.sub)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
        .frame(maxHeight: 64 * 5 + 40)
        .fixedSize(horizontal: false, vertical: ordered.count < 5)
    }
}

private struct AboutSheet: View {
    @Environment(AppModel.self) private var app
    @Environment(\.openURL) private var openURL
    @State private var confirm = false

    var body: some View {
        SheetTitle(text: "משקפת")
        Text("קריאת כתב יד עברי שרצה כולה על המכשיר. התמונות, הקריאות והדוגמאות נשמרות רק כאן, ושום דבר לא עולה לשרת.")
            .font(Mk.rubik(15))
            .lineSpacing(5)
            .foregroundStyle(Mk.sub)
            .fixedSize(horizontal: false, vertical: true)
        Text(verbatim: modelLine)
            .font(Mk.rubik(14))
            .lineSpacing(5)
            .foregroundStyle(Mk.sub)
        if app.hasTuned || app.settings.rounds > 0 {
            if confirm {
                SheetRow(title: "למחוק את הדוגמאות ואת ההתאמה", danger: true, action: { app.forgetHandwriting() }) {
                    SubText(text: "אי אפשר לבטל")
                }
            } else {
                SheetRow(title: "לשכוח את כתב היד שלי", action: { withAnimation { confirm = true } }) {
                    SubText(text: "הסריקות נשארות")
                }
            }
        }
        SheetRow(title: "פרטיות", action: { open("https://itayinbarr.github.io/mishkefet-app/privacy.html") }) {
            Icon(glyph: .chevron, size: 20, lineWidth: 2).opacity(0.5)
        }
        SheetRow(title: "קוד פתוח", action: { open("https://github.com/itayinbarr/mishkefet-app") }) {
            Icon(glyph: .chevron, size: 20, lineWidth: 2).opacity(0.5)
        }
    }

    /// The Latin model name is isolated (LRI ... PDI) so the Hebrew line keeps its order.
    private var modelLine: String {
        var s = "\u{200F}המודל: \u{2066}Mishkefet-v1 (CC BY-NC-SA 4.0)\u{2069}"
        let n = app.settings.rounds
        if n == 1 { s += "\nסבב התאמה אחד" } else if n > 1 { s += "\n\u{200F}\(n) סבבי התאמה" }
        if app.hasTuned { s += " · בשימוש: סבב \(app.settings.tunedRound)" }
        return s
    }

    private func open(_ s: String) {
        if let url = URL(string: s) { openURL(url) }
    }
}

/// Presents UIActivityViewController from the top view controller.
enum ShareSheet {
    @MainActor
    static func present(items: [Any], completion: @escaping (Bool) -> Void) {
        guard let root = UIApplication.shared.connectedScenes
            .compactMap({ ($0 as? UIWindowScene)?.keyWindow?.rootViewController }).first else { return }
        var top = root
        while let p = top.presentedViewController { top = p }
        let vc = UIActivityViewController(activityItems: items, applicationActivities: nil)
        vc.completionWithItemsHandler = { _, done, _, _ in completion(done) }
        top.present(vc, animated: true)
    }
}

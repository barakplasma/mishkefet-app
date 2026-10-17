import SwiftData
import SwiftUI

struct ListScreen: View {
    @Environment(AppModel.self) private var app
    @Query(sort: \Scan.createdAt, order: .reverse) private var scans: [Scan]
    @State private var selecting = false
    @State private var picked: Set<UUID> = []

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                CircleIconButton(glyph: .chevronBack, lineWidth: 2, label: "חזרה") { app.go(.home) }
                Spacer()
                if !scans.isEmpty {
                    if selecting {
                        HStack(spacing: 6) {
                            Button {
                                app.deleteScans(picked)
                                withAnimation(Mk.ease(0.35)) { picked = []; selecting = false }
                            } label: {
                                PillLabel(title: picked.isEmpty ? "מחיקה" : "מחיקה (\(picked.count))", small: true,
                                          foreground: picked.isEmpty ? Mk.sub : Mk.red)
                            }
                            .buttonStyle(PressStyle())
                            .disabled(picked.isEmpty)
                            Button {
                                withAnimation(Mk.ease(0.35)) { picked = []; selecting = false }
                            } label: {
                                PillLabel(title: "ביטול", small: true)
                            }
                            .buttonStyle(PressStyle())
                        }
                    } else {
                        Button { withAnimation(Mk.ease(0.35)) { selecting = true } } label: {
                            PillLabel(title: "בחירה", small: true)
                        }
                        .buttonStyle(PressStyle())
                    }
                }
            }
            .padding(.leading, 8)
            .padding(.trailing, 12)
            .frame(height: 52)
            .padding(.top, 8)

            Text("סריקות")
                .font(Mk.karantina(54))
                .frame(height: 54)
                .padding(.horizontal, 22)
                .padding(.top, 6)
                .padding(.bottom, 12)

            if scans.isEmpty {
                VStack(alignment: .leading, spacing: 16) {
                    Text("עוד אין סריקות. צלמו דף או בחרו תמונה, והקריאה תישמר כאן, רק במכשיר הזה.")
                        .font(Mk.rubik(16))
                        .lineSpacing(6)
                        .foregroundStyle(Mk.sub)
                    Button { app.present(.newScan) } label: {
                        PillLabel(title: "סריקה חדשה", glyph: .upload)
                    }
                    .buttonStyle(PressStyle())
                }
                .padding(.horizontal, 22)
                .padding(.vertical, 40)
                Spacer(minLength: 0)
            } else {
                List {
                    ForEach(scans) { scan in
                        ScanRow(scan: scan, selecting: selecting, isPicked: picked.contains(scan.id))
                            .contentShape(Rectangle())
                            .onTapGesture { tap(scan) }
                            .listRowInsets(EdgeInsets())
                            .listRowSeparator(.hidden)
                            .listRowBackground(Mk.bg)
                            .swipeActions(edge: .trailing, allowsFullSwipe: true) {
                                if !selecting {
                                    Button(role: .destructive) { app.deleteScans([scan.id]) } label: {
                                        Label("מחיקה", systemImage: "trash")
                                    }
                                    .tint(Mk.red)
                                }
                            }
                    }
                    Color.clear.frame(height: 30)
                        .listRowInsets(EdgeInsets())
                        .listRowSeparator(.hidden)
                        .listRowBackground(Color.clear)
                }
                .listStyle(.plain)
                .scrollContentBackground(.hidden)
                .environment(\.defaultMinListRowHeight, 0)
            }
        }
        .background(Mk.bg)
        .onChange(of: app.screen) { _, s in
            if s != .list { selecting = false; picked = [] }
        }
    }

    private func tap(_ scan: Scan) {
        if selecting {
            withAnimation(.easeOut(duration: 0.2)) {
                if picked.contains(scan.id) { picked.remove(scan.id) } else { picked.insert(scan.id) }
            }
        } else {
            app.open(scan.id, from: .list)
        }
    }
}

struct ScanThumb: View {
    var url: URL
    var width: CGFloat = 46
    var height: CGFloat = 58
    var radius: CGFloat = 10
    @State private var image: UIImage?

    var body: some View {
        ZStack {
            Mk.card
            if let image {
                Image(uiImage: image).resizable().scaledToFill()
            }
        }
        .frame(width: width, height: height)
        .clipShape(RoundedRectangle(cornerRadius: radius, style: .continuous))
        .task(id: url) { image = UIImage(contentsOfFile: url.path) }
        .accessibilityHidden(true)
    }
}

struct ScanRow: View {
    var scan: Scan
    var selecting: Bool
    var isPicked: Bool

    var body: some View {
        let lines = scan.lines
        HStack(spacing: 14) {
            if selecting {
                ZStack {
                    Circle()
                        .strokeBorder(isPicked ? Mk.orange : Mk.line, lineWidth: 2)
                        .background(Circle().fill(isPicked ? Mk.orange : .clear))
                    if isPicked {
                        Icon(glyph: .check, size: 14, lineWidth: 3).foregroundStyle(Mk.charcoal)
                    }
                }
                .frame(width: 24, height: 24)
                .transition(.opacity)
            }
            ScanThumb(url: scan.thumbURL)
            VStack(alignment: .leading, spacing: 4) {
                HStack(alignment: .firstTextBaseline, spacing: 12) {
                    Text(verbatim: scan.title)
                        .font(Mk.rubik(17, .semibold))
                        .foregroundStyle(Mk.ink)
                        .lineLimit(1)
                    Spacer(minLength: 0)
                    Text(verbatim: Titles.linesShort(lines.count))
                        .font(Mk.rubik(13))
                        .foregroundStyle(Mk.sub)
                        .fixedSize()
                }
                Text(verbatim: scan.preview)
                    .font(Mk.rubik(15))
                    .foregroundStyle(Mk.sub)
                    .lineLimit(1)
                HStack(spacing: 8) {
                    Text(verbatim: HebrewDate.label(scan.createdAt))
                        .font(Mk.rubik(12.5))
                        .foregroundStyle(Mk.sub)
                    if scan.reread { Chip(text: "נקרא מחדש", sage: true) }
                }
            }
        }
        .padding(.vertical, 14)
        .padding(.horizontal, 22)
        .background(Mk.bg)
        .overlay(alignment: .bottom) {
            Rectangle().fill(Mk.line).frame(height: 1).padding(.horizontal, 22)
        }
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(isPicked ? .isSelected : [])
    }
}

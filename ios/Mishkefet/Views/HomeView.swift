import SwiftUI

struct HomeView: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        VStack(spacing: 0) {
            header
            Spacer(minLength: 0)
            VStack(spacing: 0) {
                Button { app.openCamera() } label: {
                    HomeOrb()
                }
                .buttonStyle(PressStyle(scale: 0.98))
                .accessibilityLabel("צילום")
                Text("לחצו לצילום כתב יד")
                    .font(Mk.rubik(16, .medium))
                    .foregroundStyle(Mk.sub)
                    .padding(.top, 16)
                Button { app.pickPhotos() } label: {
                    PillLabel(title: "בחירה מהגלריה", glyph: .gallery)
                }
                .buttonStyle(PressStyle())
                .padding(.top, 18)
            }
            Spacer(minLength: 0)
            trainCard
                .padding(.horizontal, 14)
                .padding(.bottom, 30)
        }
    }

    private var header: some View {
        HStack(spacing: 0) {
            HStack(spacing: 6) {
                CircleIconButton(glyph: .menu, label: "סריקות") { app.go(.list) }
                    .offset(y: 4)
                Button { app.present(.about) } label: {
                    HStack(spacing: 9) {
                        BinocularsLogo(width: 36, height: 24)
                            .padding(.top, 9)
                        Text("משקפת")
                            .font(Mk.karantina(46))
                            .foregroundStyle(Mk.ink)
                            .frame(height: 34)
                            .padding(.top, 4)
                    }
                }
                .buttonStyle(.plain)
                .accessibilityLabel("אודות")
            }
            Spacer()
            CircleIconButton(glyph: .upload, label: "סריקה חדשה") { app.present(.newScan) }
                .offset(y: 4)
        }
        .padding(.leading, 10)
        .padding(.trailing, 12)
        .frame(height: 62)
    }

    private var trainCard: some View {
        let size = AppModel.roundSize
        let count = app.roundCount
        let sub = count == 0
            ? "סבב \(app.settings.rounds + 1) · \(size) מילים עד ההתאמה הבאה"
            : "\(count) מתוך \(size) מילים עד ההתאמה הבאה"
        return Button { app.startTraining() } label: {
            HStack(spacing: 14) {
                Icon(glyph: .pencil, size: 22)
                    .foregroundStyle(Mk.charcoal)
                    .frame(width: 44, height: 44)
                    .background(Circle().fill(Mk.orange))
                VStack(alignment: .leading, spacing: 6) {
                    Text("ללמד את משקפת את כתב היד שלך")
                        .font(Mk.rubik(16, .semibold))
                        .foregroundStyle(Mk.ink)
                    Text(verbatim: sub)
                        .font(Mk.rubik(13))
                        .foregroundStyle(Mk.sub)
                    GeometryReader { g in
                        ZStack(alignment: .leading) {
                            Capsule().fill(Mk.line)
                            Capsule().fill(Mk.orange)
                                .frame(width: g.size.width * min(1, CGFloat(count) / CGFloat(size)))
                        }
                    }
                    .frame(height: 4)
                    .animation(Mk.ease(0.4), value: count)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(.vertical, 14)
            .padding(.horizontal, 16)
            .background(RoundedRectangle(cornerRadius: 24, style: .continuous).fill(Mk.card))
        }
        .buttonStyle(PressStyle())
    }
}

/// The big breathing orb: a thin orange ring around a card disc with the binoculars.
struct HomeOrb: View {
    @State private var breathe = false

    var body: some View {
        ZStack {
            Circle().strokeBorder(Mk.orange.opacity(0.35), lineWidth: 1.5)
            Circle()
                .fill(Mk.card)
                .padding(12)
                .overlay(BinocularsLogo(width: 124, height: 80, strokeUnits: 2.6, lensY: 13))
                .scaleEffect(breathe ? 1.045 : 1)
        }
        .frame(width: 264, height: 264)
        .contentShape(Circle())
        .onAppear {
            withAnimation(.easeInOut(duration: 1.6).repeatForever(autoreverses: true)) { breathe = true }
        }
    }
}

import MishkefetCore
import SwiftUI

/// 46pt round icon button (header buttons).
struct CircleIconButton: View {
    var glyph: Glyph
    var size: CGFloat = 24
    var lineWidth: CGFloat = 1.8
    var label: String
    var background: Color = .clear
    var foreground: Color = Mk.ink
    var action: () -> Void

    var body: some View {
        Button(action: action) {
            Icon(glyph: glyph, size: size, lineWidth: lineWidth)
                .foregroundStyle(foreground)
                .frame(width: 46, height: 46)
                .background(Circle().fill(background))
                .contentShape(Circle())
        }
        .buttonStyle(PressStyle(scale: 0.92))
        .accessibilityLabel(label)
    }
}

struct PressStyle: ButtonStyle {
    var scale: CGFloat = 0.97

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? scale : 1)
            .animation(.easeOut(duration: 0.15), value: configuration.isPressed)
    }
}

/// The card-colored capsule button (".pill").
struct PillLabel: View {
    var title: String
    var glyph: Glyph?
    var small = false
    var foreground: Color = Mk.ink
    var background: Color = Mk.card

    var body: some View {
        HStack(spacing: 8) {
            if let glyph { Icon(glyph: glyph, size: 20) }
            Text(title).font(Mk.rubik(small ? 14 : 15, .medium))
        }
        .foregroundStyle(foreground)
        .padding(.horizontal, small ? 16 : 18)
        .frame(height: small ? 40 : 44)
        .background(Capsule().fill(background))
    }
}

/// The orange call to action (".cta").
struct CTALabel: View {
    var title: String
    var glyph: Glyph?
    var fill: Color = Mk.orange
    var height: CGFloat = 60

    var body: some View {
        HStack(spacing: 10) {
            if let glyph { Icon(glyph: glyph, size: 22, lineWidth: 1.9) }
            Text(title).font(Mk.rubik(17, .semibold))
        }
        .foregroundStyle(Mk.charcoal)
        .frame(maxWidth: .infinity)
        .frame(height: height)
        .background(Capsule().fill(fill))
    }
}

/// A two-or-more option segmented control (".seg"); `dark` is the camera's variant.
struct Segmented<Value: Hashable>: View {
    var options: [(Value, String)]
    @Binding var selection: Value
    var dark = false
    var label: String

    var body: some View {
        HStack(spacing: 2) {
            ForEach(options.indices, id: \.self) { i in
                let (v, title) = options[i]
                let on = v == selection
                Button {
                    withAnimation(.easeInOut(duration: 0.25)) { selection = v }
                } label: {
                    Text(title)
                        .font(Mk.rubik(14, .medium))
                        .foregroundStyle(on ? (dark ? Mk.charcoal : Mk.bg) : (dark ? Mk.cream : Mk.ink))
                        .padding(.horizontal, 16)
                        .frame(height: dark ? 32 : 34)
                        .background(Capsule().fill(on ? (dark ? Mk.cream : Mk.ink) : .clear))
                        .contentShape(Capsule())
                }
                .buttonStyle(.plain)
                .accessibilityAddTraits(on ? .isSelected : [])
            }
        }
        .padding(3)
        .background(Capsule().fill(dark ? Mk.cream.opacity(0.12) : Mk.card))
        .accessibilityElement(children: .contain)
        .accessibilityLabel(label)
    }
}

/// "כללי / מותאם אישית": the general model or the user's fine-tuned one. Shown only once one exists.
struct ModelToggle: View {
    var useTuned: Bool
    var dark = false
    var onChange: (Bool) -> Void

    var body: some View {
        Segmented(options: [(false, "כללי"), (true, "מותאם אישית")],
                  selection: Binding(get: { useTuned }, set: { onChange($0) }),
                  dark: dark, label: "מודל")
    }
}

struct Chip: View {
    var text: String
    var sage = false

    var body: some View {
        Text(text)
            .font(Mk.rubik(12.5, sage ? .medium : .regular))
            .foregroundStyle(sage ? Mk.charcoal : Mk.ink)
            .padding(.horizontal, 9)
            .padding(.vertical, 2)
            .background(RoundedRectangle(cornerRadius: 8).fill(sage ? Mk.sage : Mk.card))
    }
}

/// "על המכשיר · שום דבר לא עולה לשרת" style badge with the sage dot.
struct DeviceBadge: View {
    var text: String
    var background: Color = Mk.card

    var body: some View {
        HStack(spacing: 8) {
            Circle().fill(Mk.sage).frame(width: 8, height: 8)
            Text(text).font(Mk.rubik(13))
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(RoundedRectangle(cornerRadius: 14).fill(background))
    }
}

/// A small spinner (".spinner").
struct Spinner: View {
    var size: CGFloat = 14
    @State private var spin = false

    var body: some View {
        ZStack {
            Circle().stroke(Mk.line, lineWidth: 2)
            Circle().trim(from: 0, to: 0.25).stroke(Mk.orange, style: StrokeStyle(lineWidth: 2, lineCap: .round))
                .rotationEffect(.degrees(spin ? 360 : 0))
        }
        .frame(width: size, height: size)
        .onAppear {
            withAnimation(.linear(duration: 0.8).repeatForever(autoreverses: false)) { spin = true }
        }
    }
}

/// Draws a scan image aspect-fit into a frame with its line boxes; `onTap(index)` for each box.
struct ScanImageFrame: View {
    var image: UIImage?
    var imageSize: CGSize
    var boxes: [LineBox?]
    var active: Int?
    var height: CGFloat
    var onTap: ((Int) -> Void)?

    var body: some View {
        GeometryReader { geo in
            let W = geo.size.width, H = geo.size.height
            let s = min(W / max(1, imageSize.width), H / max(1, imageSize.height))
            let ox = (W - imageSize.width * s) / 2, oy = (H - imageSize.height * s) / 2
            ZStack(alignment: .topLeading) {
                if let image {
                    Image(uiImage: image)
                        .resizable()
                        .frame(width: imageSize.width * s, height: imageSize.height * s)
                        .offset(x: ox, y: oy)
                }
                ForEach(boxes.indices, id: \.self) { i in
                    if let b = boxes[i] {
                        let on = i == active
                        RoundedRectangle(cornerRadius: 4)
                            .fill(on ? Mk.orange.opacity(0.16) : Color.clear)
                            .overlay(RoundedRectangle(cornerRadius: 4).strokeBorder(on ? Mk.orange : .clear, lineWidth: 2))
                            .frame(width: max(6, CGFloat(b.w) * s), height: max(5, CGFloat(b.h) * s))
                            .contentShape(Rectangle())
                            .offset(x: ox + CGFloat(b.x) * s, y: oy + CGFloat(b.y) * s)
                            .onTapGesture { onTap?(i) }
                            .animation(.easeInOut(duration: 0.25), value: on)
                    }
                }
            }
            .frame(width: W, height: H, alignment: .topLeading)
        }
        .frame(height: height)
        .background(Mk.card)
        .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
        // Image pixels are physical; never mirror them for RTL.
        .environment(\.layoutDirection, .leftToRight)
    }
}


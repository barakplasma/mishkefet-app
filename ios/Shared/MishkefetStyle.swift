import SwiftUI
import UIKit

// Design tokens: Ozen's THEMES table (the design prototype is built on them); light/dark follow the system.

extension Color {
    init(hex: UInt32, opacity: Double = 1) {
        self.init(.sRGB,
                  red: Double((hex >> 16) & 0xff) / 255,
                  green: Double((hex >> 8) & 0xff) / 255,
                  blue: Double(hex & 0xff) / 255,
                  opacity: opacity)
    }

    /// A color that resolves per trait collection (light / dark).
    init(light: UIColor, dark: UIColor) {
        self.init(uiColor: UIColor { $0.userInterfaceStyle == .dark ? dark : light })
    }
}

extension UIColor {
    convenience init(hex: UInt32, alpha: CGFloat = 1) {
        self.init(red: CGFloat((hex >> 16) & 0xff) / 255,
                  green: CGFloat((hex >> 8) & 0xff) / 255,
                  blue: CGFloat(hex & 0xff) / 255,
                  alpha: alpha)
    }
}

enum Mk {
    // Theme tokens
    static let bg = Color(light: UIColor(hex: 0xe2e8ce), dark: UIColor(hex: 0x262626))
    static let ink = Color(light: UIColor(hex: 0x262626), dark: UIColor(hex: 0xe2e8ce))
    static let sub = Color(light: UIColor(hex: 0x262626, alpha: 0.62), dark: UIColor(hex: 0xe2e8ce, alpha: 0.6))
    static let card = Color(light: UIColor(hex: 0xeef2e0), dark: UIColor(hex: 0x343532))
    static let line = Color(light: UIColor(hex: 0x262626, alpha: 0.1), dark: UIColor(hex: 0xe2e8ce, alpha: 0.12))

    // Accents (same in both themes)
    static let orange = Color(hex: 0xff7f11)
    static let red = Color(hex: 0xff1b1c)
    static let sage = Color(hex: 0xacbfa4)
    static let cream = Color(hex: 0xe2e8ce)
    static let charcoal = Color(hex: 0x262626)
    /// The training canvas "read right" green.
    static let green = Color(hex: 0x2f9a4f)
    /// Dark surfaces (camera, refit) regardless of theme.
    static let night = Color(hex: 0x262626)
    static let nightCard = Color(hex: 0x343532)
    static let nightSub = Color(hex: 0xe2e8ce, opacity: 0.6)
    static let cameraBg = Color(hex: 0x141414)

    // Typography
    static func karantina(_ size: CGFloat, bold: Bool = true) -> Font {
        .custom(bold ? "Karantina-Bold" : "Karantina-Regular", fixedSize: size)
    }

    /// Amatic SC 700: the hand-lettered face of the training samples.
    static func amatic(_ size: CGFloat) -> Font {
        .custom("AmaticSC-Bold", fixedSize: size)
    }

    enum Weight { case light, regular, medium, semibold, bold }

    static func rubik(_ size: CGFloat, _ weight: Weight = .regular) -> Font {
        .custom(rubikName(weight), fixedSize: size)
    }

    /// Rubik that follows Dynamic Type (used for the reading body).
    static func rubik(_ size: CGFloat, _ weight: Weight = .regular, relativeTo style: Font.TextStyle) -> Font {
        .custom(rubikName(weight), size: size, relativeTo: style)
    }

    static func rubikName(_ weight: Weight) -> String {
        switch weight {
        case .light: return "Rubik-Light"
        case .regular: return "Rubik-Regular"
        case .medium: return "Rubik-Medium"
        case .semibold: return "Rubik-SemiBold"
        case .bold: return "Rubik-Bold"
        }
    }

    /// cubic-bezier(.2,.8,.2,1)
    static func ease(_ duration: Double = 0.55) -> Animation {
        .timingCurve(0.2, 0.8, 0.2, 1, duration: duration)
    }
}

// MARK: - The binoculars mark

/// The logo: two lenses (r 7 at (9, cy) and (27, cy)) and a bridge, in a 36x24 viewBox, aspect-fit.
/// The header draws it at cy 14 with the bridge at y 11; the home orb at cy 13 with the bridge at y 10.
struct BinocularsShape: Shape {
    var lensY: CGFloat = 14

    func path(in rect: CGRect) -> Path {
        let s = min(rect.width / 36, rect.height / 24)
        let dx = rect.minX + (rect.width - 36 * s) / 2
        let dy = rect.minY + (rect.height - 24 * s) / 2
        var p = Path()
        p.addEllipse(in: CGRect(x: 2, y: lensY - 7, width: 14, height: 14))
        p.addEllipse(in: CGRect(x: 20, y: lensY - 7, width: 14, height: 14))
        p.move(to: CGPoint(x: 16, y: lensY - 3))
        p.addLine(to: CGPoint(x: 20, y: lensY - 3))
        return p.applying(CGAffineTransform(translationX: dx, y: dy).scaledBy(x: s, y: s))
    }
}

/// The logo at a fixed size; `strokeUnits` is the stroke width in viewBox units (3.6 in the header).
struct BinocularsLogo: View {
    var width: CGFloat = 36
    var height: CGFloat = 24
    var strokeUnits: CGFloat = 3.6
    var lensY: CGFloat = 14
    var color: Color = Mk.orange

    var body: some View {
        let s = min(width / 36, height / 24)
        BinocularsShape(lensY: lensY)
            .stroke(color, style: StrokeStyle(lineWidth: strokeUnits * s, lineCap: .round, lineJoin: .round))
            .frame(width: width, height: height)
            .environment(\.layoutDirection, .leftToRight)
            .accessibilityHidden(true)
    }
}

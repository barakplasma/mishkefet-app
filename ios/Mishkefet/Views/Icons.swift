import SwiftUI

/// The design's 24x24 stroke icons, parsed from their SVG path data.
enum Glyph {
    case menu, upload, chevronBack, chevron, pencil, check, copy, share, trash, close, gallery

    fileprivate var data: String {
        switch self {
        case .menu: return "M4 7h16M4 12h16M4 17h10"
        case .upload: return "M12 15V4M7.5 8.5L12 4l4.5 4.5M5 15v4h14v-4"
        case .chevronBack: return "M9 5l7 7-7 7"
        case .chevron: return "M15 5l-7 7 7 7"
        case .pencil: return "M4.5 19.5h4l10-10-4-4-10 10z"
        case .check: return "M5 12.5l4.5 4.5L19 7.5"
        case .copy: return ""
        case .share: return "M12 14V4M7.5 8.5L12 4l4.5 4.5M5 13v6h14v-6"
        case .trash: return "M5 7h14M10 7V5h4v2M7 7l1 12h8l1-12"
        case .close: return "M6 6l12 12M18 6L6 18"
        case .gallery: return "M5 17l5-4.5 3.5 3 2.5-2 3.5 3"
        }
    }

    var isFilled: Bool { false }

    var path: Path {
        switch self {
        case .gallery:
            var p = Path(roundedRect: CGRect(x: 3.5, y: 5, width: 17, height: 14), cornerRadius: 3)
            p.addEllipse(in: CGRect(x: 9 - 1.6, y: 10 - 1.6, width: 3.2, height: 3.2))
            p.addPath(SVGPath.parse(data))
            return p
        case .copy:
            var p = Path(roundedRect: CGRect(x: 8.5, y: 8.5, width: 11, height: 11), cornerRadius: 2.5)
            // M15.5 8.5V6a1.5 1.5 0 0 0-1.5-1.5H6A1.5 1.5 0 0 0 4.5 6v8A1.5 1.5 0 0 0 6 15.5h2.5
            p.move(to: CGPoint(x: 15.5, y: 8.5))
            p.addLine(to: CGPoint(x: 15.5, y: 6))
            p.addQuadCurve(to: CGPoint(x: 14, y: 4.5), control: CGPoint(x: 15.5, y: 4.5))
            p.addLine(to: CGPoint(x: 6, y: 4.5))
            p.addQuadCurve(to: CGPoint(x: 4.5, y: 6), control: CGPoint(x: 4.5, y: 4.5))
            p.addLine(to: CGPoint(x: 4.5, y: 14))
            p.addQuadCurve(to: CGPoint(x: 6, y: 15.5), control: CGPoint(x: 4.5, y: 15.5))
            p.addLine(to: CGPoint(x: 8.5, y: 15.5))
            return p
        default:
            return SVGPath.parse(data)
        }
    }
}

/// Draws a `Glyph` at `size` points; `lineWidth` is in the icon's 24-unit space (as in the SVG).
struct Icon: View {
    var glyph: Glyph
    var size: CGFloat = 24
    var lineWidth: CGFloat = 1.8

    var body: some View {
        let s = size / 24
        let p = glyph.path.applying(CGAffineTransform(scaleX: s, y: s))
        Group {
            if glyph.isFilled {
                p.fill(style: FillStyle())
            } else {
                p.stroke(style: StrokeStyle(lineWidth: lineWidth * s, lineCap: .round, lineJoin: .round))
            }
        }
        .frame(width: size, height: size)
        // SVG coordinates are physical; never mirror them for RTL.
        .environment(\.layoutDirection, .leftToRight)
        .accessibilityHidden(true)
    }
}

/// A minimal SVG path parser (M, L, H, V, C, Z; absolute and relative).
enum SVGPath {
    static func parse(_ d: String) -> Path {
        var p = Path()
        var tokens: [String] = []
        var cur = ""
        for ch in d {
            if ch.isLetter {
                if !cur.isEmpty { tokens.append(cur); cur = "" }
                tokens.append(String(ch))
            } else if ch == " " || ch == "," {
                if !cur.isEmpty { tokens.append(cur); cur = "" }
            } else if ch == "-" {
                if !cur.isEmpty { tokens.append(cur) }
                cur = "-"
            } else {
                cur.append(ch)
            }
        }
        if !cur.isEmpty { tokens.append(cur) }

        var i = 0
        var cmd: Character = "M"
        var pt = CGPoint.zero
        var start = CGPoint.zero
        func num() -> CGFloat {
            defer { i += 1 }
            return CGFloat(Double(tokens[i]) ?? 0)
        }
        while i < tokens.count {
            if let c = tokens[i].first, c.isLetter {
                cmd = c
                i += 1
                if cmd == "z" || cmd == "Z" {
                    p.closeSubpath()
                    pt = start
                    continue
                }
            }
            let rel = cmd.isLowercase
            switch cmd.uppercased() {
            case "M":
                let x = num(), y = num()
                pt = rel ? CGPoint(x: pt.x + x, y: pt.y + y) : CGPoint(x: x, y: y)
                p.move(to: pt)
                start = pt
                cmd = rel ? "l" : "L"
            case "L":
                let x = num(), y = num()
                pt = rel ? CGPoint(x: pt.x + x, y: pt.y + y) : CGPoint(x: x, y: y)
                p.addLine(to: pt)
            case "H":
                let x = num()
                pt = CGPoint(x: rel ? pt.x + x : x, y: pt.y)
                p.addLine(to: pt)
            case "V":
                let y = num()
                pt = CGPoint(x: pt.x, y: rel ? pt.y + y : y)
                p.addLine(to: pt)
            case "C":
                var v = [CGFloat]()
                for _ in 0..<6 { v.append(num()) }
                let o = rel ? pt : .zero
                let c1 = CGPoint(x: o.x + v[0], y: o.y + v[1])
                let c2 = CGPoint(x: o.x + v[2], y: o.y + v[3])
                pt = CGPoint(x: o.x + v[4], y: o.y + v[5])
                p.addCurve(to: pt, control1: c1, control2: c2)
            default:
                i += 1
            }
        }
        return p
    }
}

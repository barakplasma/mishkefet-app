import Foundation

/// Hebrew dates, as the web app's dateLabel(): "היום, 09:12", "אתמול, 18:03", "6 באוקטובר".
enum HebrewDate {
    static let months = ["בינואר", "בפברואר", "במרץ", "באפריל", "במאי", "ביוני",
                         "ביולי", "באוגוסט", "בספטמבר", "באוקטובר", "בנובמבר", "בדצמבר"]

    static var calendar: Calendar = {
        var c = Calendar(identifier: .gregorian)
        c.locale = Locale(identifier: "he_IL")
        c.timeZone = .current
        return c
    }()

    static func time(_ date: Date, calendar: Calendar = calendar) -> String {
        let c = calendar.dateComponents([.hour, .minute], from: date)
        return String(format: "%02d:%02d", c.hour ?? 0, c.minute ?? 0)
    }

    /// Today and yesterday with the time; older days by date, with the year when it is not this one.
    static func label(_ date: Date, now: Date = Date(), calendar: Calendar = calendar) -> String {
        let days = calendar.dateComponents([.day], from: calendar.startOfDay(for: date),
                                           to: calendar.startOfDay(for: now)).day ?? 0
        if days == 0 { return "היום, \(time(date, calendar: calendar))" }
        if days == 1 { return "אתמול, \(time(date, calendar: calendar))" }
        let c = calendar.dateComponents([.day, .month, .year], from: date)
        let year = calendar.component(.year, from: now)
        var s = "\(c.day ?? 1) \(months[(c.month ?? 1) - 1])"
        if let y = c.year, y != year { s += " \(y)" }
        return s
    }
}

enum Titles {
    /// A default title: the first four words of the first non-empty line, else "סריקה <date>".
    static func scan(lines: [String], at date: Date, now: Date = Date(), calendar: Calendar = HebrewDate.calendar) -> String {
        if let first = lines.first(where: { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }) {
            let words = first.split(whereSeparator: { $0.isWhitespace }).prefix(4)
            if !words.isEmpty { return words.joined(separator: " ") }
        }
        return "סריקה \(HebrewDate.label(date, now: now, calendar: calendar))"
    }

    static func lines(_ n: Int) -> String {
        n == 1 ? "שורה אחת" : "\(n) שורות"
    }

    /// The short form on list rows: "שורה" / "5 שורות".
    static func linesShort(_ n: Int) -> String {
        n == 1 ? "שורה" : "\(n) שורות"
    }
}

/// Alignment and word diff (web/src/lib/text.ts).
enum TextDiff {
    /// Levenshtein alignment of `target` against a reading: the edit distance and which target
    /// characters were not read right.
    static func editOps(target: String, read: String) -> (distance: Int, wrongTargetIndices: [Int]) {
        let a = Array(target), b = Array(read)
        var d = [[Int]](repeating: [Int](repeating: 0, count: b.count + 1), count: a.count + 1)
        for i in 0...a.count { d[i][0] = i }
        if !b.isEmpty { for j in 1...b.count { d[0][j] = j } }
        if !a.isEmpty && !b.isEmpty {
            for i in 1...a.count {
                for j in 1...b.count {
                    d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1))
                }
            }
        }
        var wrong: [Int] = []
        var i = a.count, j = b.count
        while i > 0 || j > 0 {
            if i > 0, j > 0, d[i][j] == d[i - 1][j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1) {
                if a[i - 1] != b[j - 1] { wrong.append(i - 1) }
                i -= 1
                j -= 1
            } else if i > 0, d[i][j] == d[i - 1][j] + 1 {
                wrong.append(i - 1)
                i -= 1
            } else {
                j -= 1
            }
        }
        return (d[a.count][b.count], wrong.reversed())
    }

    struct Token: Equatable {
        var text: String
        var changed: Bool
    }

    enum Side { case before, after }

    /// Word-level diff: the tokens of `side`, marked when not in the longest common subsequence.
    static func words(before: String, after: String, side: Side) -> [Token] {
        let a = before.split(whereSeparator: { $0.isWhitespace }).map(String.init)
        let b = after.split(whereSeparator: { $0.isWhitespace }).map(String.init)
        var L = [[Int]](repeating: [Int](repeating: 0, count: b.count + 1), count: a.count + 1)
        if !a.isEmpty && !b.isEmpty {
            for i in stride(from: a.count - 1, through: 0, by: -1) {
                for j in stride(from: b.count - 1, through: 0, by: -1) {
                    L[i][j] = a[i] == b[j] ? L[i + 1][j + 1] + 1 : max(L[i + 1][j], L[i][j + 1])
                }
            }
        }
        var keepA = Set<Int>(), keepB = Set<Int>()
        var i = 0, j = 0
        while i < a.count && j < b.count {
            if a[i] == b[j] {
                keepA.insert(i); keepB.insert(j)
                i += 1; j += 1
            } else if L[i + 1][j] >= L[i][j + 1] {
                i += 1
            } else {
                j += 1
            }
        }
        switch side {
        case .before: return a.enumerated().map { Token(text: $1, changed: !keepA.contains($0)) }
        case .after: return b.enumerated().map { Token(text: $1, changed: !keepB.contains($0)) }
        }
    }
}

/// The copy / share text: the reading's lines, one per line.
enum ReadingText {
    static func plain(_ lines: [String]) -> String {
        lines.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.joined(separator: "\n")
    }
}

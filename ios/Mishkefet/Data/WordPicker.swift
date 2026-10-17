import Foundation

/// Which word to ask for next (web/src/lib/words.ts).
///
/// Round 1 makes sure every letter, final forms included, is written at least once. After that,
/// words are weighted by how often this hand's letters get misread, so the rounds spend their time
/// where the model is weakest. Words from the user's own scans join the pool when the model was
/// unsure of the line they came from; the user never sees that judgment, it only steers practice.
struct WordPicker {
    static let letters: [Character] = Array("אבגדהוזחטיכךלמםנןסעפףצץקרשת")

    struct Attempt {
        var word: String
        var readAs: String
    }

    private let common: [String]
    private let extra: [String]
    private let rates: [Character: Double]
    private var recent: Set<String>
    private var covered: Set<Character>
    private var random: () -> Double

    init(pool: [String], attempts: [Attempt], scanWords: [String], random: @escaping () -> Double = { Double.random(in: 0..<1) }) {
        common = Array(pool.filter { $0.count >= 2 && $0.count <= 7 }.prefix(2000))
        recent = Set(attempts.suffix(120).map(\.word))
        rates = Self.letterErrorRates(attempts)
        let firstRound = attempts.count < 40
        covered = firstRound ? Set(attempts.flatMap { Array($0.word) }) : Set(Self.letters)
        var seen = Set<String>()
        extra = scanWords.filter { w in
            guard (2...7).contains(w.count), w.unicodeScalars.allSatisfy({ (0x5D0...0x5EA).contains($0.value) }) else { return false }
            return seen.insert(w).inserted
        }
        self.random = random
    }

    /// Per-letter misread rate with a mild prior: (errors + 1) / (seen + 4).
    static func letterErrorRates(_ attempts: [Attempt]) -> [Character: Double] {
        var seen: [Character: Int] = [:], err: [Character: Int] = [:]
        for a in attempts {
            let wrong = Set(TextDiff.editOps(target: a.word, read: a.readAs).wrongTargetIndices)
            for (i, c) in a.word.enumerated() {
                seen[c, default: 0] += 1
                if wrong.contains(i) { err[c, default: 0] += 1 }
            }
        }
        var out: [Character: Double] = [:]
        for c in letters { out[c] = Double((err[c] ?? 0) + 1) / Double((seen[c] ?? 0) + 4) }
        return out
    }

    private func weight(_ w: String, rank: Int) -> Double {
        let letters = Array(w)
        let errors = letters.reduce(0.0) { $0 + (rates[$1] ?? 0.25) } / Double(max(1, letters.count))
        let freq = 1 / (1 + Double(rank) / 50).squareRoot()
        let len = (3...6).contains(w.count) ? 1.0 : 0.5
        return errors * errors * freq * len
    }

    mutating func next() -> String {
        guard !common.isEmpty else { return "שלום" }
        let missing = Set(Self.letters.filter { !covered.contains($0) })
        var candidates: [(String, Double)]
        if !missing.isEmpty {
            // Cover the letters this round has not had yet, common words first.
            candidates = common.enumerated().compactMap { i, w in
                let s = Double(Set(w).intersection(missing).count) * (1 / (1 + Double(i) / 30).squareRoot())
                return s > 0 && !recent.contains(w) ? (w, s) : nil
            }
        } else {
            candidates = common.enumerated().map { ($1, weight($1, rank: $0)) }
                + extra.map { ($0, weight($0, rank: 100) * 2) }
            candidates = candidates.filter { !recent.contains($0.0) }
        }
        if candidates.isEmpty { candidates = common.map { ($0, 1) } }
        let total = candidates.reduce(0) { $0 + $1.1 }
        var r = random() * total
        var pick = candidates[candidates.count - 1].0
        for (w, x) in candidates {
            r -= x
            if r <= 0 { pick = w; break }
        }
        recent.insert(pick)
        for c in pick { covered.insert(c) }
        return pick
    }

    /// The bundled pool (spec/train-words.json).
    static let pool: [String] = {
        guard let url = Bundle.main.url(forResource: "train-words", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let words = try? JSONDecoder().decode([String].self, from: data) else { return [] }
        return words
    }()
}

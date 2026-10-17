import Foundation

public enum Metrics {
    /// `1 - sum(edit distance) / sum(len(ref))` over the pairs (1 when there is nothing to compare).
    public static func characterAccuracy(pairs: [(hyp: String, ref: String)]) -> Double {
        var edits = 0, total = 0
        for p in pairs {
            let ref = Array(p.ref)
            edits += editDistance(Array(p.hyp), ref)
            total += ref.count
        }
        return total > 0 ? 1 - Double(edits) / Double(total) : 1
    }

    /// Levenshtein distance.
    public static func editDistance(_ a: [Character], _ b: [Character]) -> Int {
        if a.isEmpty { return b.count }
        if b.isEmpty { return a.count }
        var prev = Array(0...b.count)
        var cur = [Int](repeating: 0, count: b.count + 1)
        for i in 1...a.count {
            cur[0] = i
            for j in 1...b.count {
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1))
            }
            swap(&prev, &cur)
        }
        return prev[b.count]
    }

    /// Character error rate of one line: edit distance / reference length.
    public static func cer(hyp: String, ref: String) -> Double {
        let r = Array(ref)
        if r.isEmpty { return hyp.isEmpty ? 0 : 1 }
        return Double(editDistance(Array(hyp), r)) / Double(r.count)
    }
}

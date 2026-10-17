import Foundation

/// CTC decoding over log-probabilities (spec/decoder.md). All scores are double.
struct Decoder: Sendable {
    var charLM: CharLM?
    var words: WordLM?
    var beamWidth = 12
    var topK = 8
    var lmWeight = 0.4
    var lengthBonus = 0.6
    var wordWeight = 0.2

    static let blank = UInt8(Charset.blank)
    static let space = UInt8(Charset.space)

    @inline(__always) static func lse(_ a: Double, _ b: Double) -> Double {
        if a == -.infinity { return b }
        if b == -.infinity { return a }
        let (hi, lo) = a > b ? (a, b) : (b, a)
        return hi + log1p(exp(lo - hi))
    }

    /// Argmax per frame (ties to the lower index).
    static func argmax(_ lp: UnsafeBufferPointer<Float>, frames: Int, classes: Int) -> [Int] {
        (0..<frames).map { t in
            let row = lp.baseAddress! + t * classes
            var best = 0
            for k in 1..<classes where row[k] > row[best] { best = k }
            return best
        }
    }

    static func greedy(_ lp: [Float], frames: Int, classes: Int) -> [UInt8] {
        let best = lp.withUnsafeBufferPointer { argmax($0, frames: frames, classes: classes) }
        var out: [UInt8] = []
        var prev = -1
        for k in best {
            if k != prev && k != Int(blank) { out.append(UInt8(k)) }
            prev = k
        }
        return out
    }

    /// Mean best-class probability over all frames, or over emitting frames only.
    static func confidence(_ lp: [Float], frames: Int, classes: Int, emittingOnly: Bool) -> Double {
        guard frames > 0 else { return 0 }
        return lp.withUnsafeBufferPointer { p in
            let best = argmax(p, frames: frames, classes: classes)
            var sum = 0.0, n = 0
            for t in 0..<frames {
                if emittingOnly && (best[t] == Int(blank) || (t > 0 && best[t] == best[t - 1])) { continue }
                sum += exp(Double(p[t * classes + best[t]]))
                n += 1
            }
            return n > 0 ? sum / Double(n) : 0
        }
    }

    /// Indices of the k largest values, largest first, ties to the lower index.
    static func topK(_ row: UnsafePointer<Float>, classes: Int, k: Int) -> [Int] {
        var out: [Int] = []
        out.reserveCapacity(k + 1)
        for i in 0..<classes {
            let v = row[i]
            if out.count == k && v <= row[out[k - 1]] { continue }
            var j = out.count
            while j > 0 && row[out[j - 1]] < v { j -= 1 }
            out.insert(i, at: j)
            if out.count > k { out.removeLast() }
        }
        return out
    }

    /// The characters after the last space (the whole prefix if there is none).
    static func lastWord(_ prefix: [UInt8]) -> ArraySlice<UInt8> {
        if let i = prefix.lastIndex(of: space) { return prefix[(i + 1)...] }
        return prefix[...]
    }

    private struct Entry {
        var prefix: [UInt8]
        var pb: Double
        var pnb: Double
    }

    /// Insertion-ordered prefix map.
    private struct OrderedBeams {
        var entries: [Entry] = []
        var index: [[UInt8]: Int] = [:]

        mutating func slot(_ prefix: [UInt8]) -> Int {
            if let i = index[prefix] { return i }
            index[prefix] = entries.count
            entries.append(Entry(prefix: prefix, pb: -.infinity, pnb: -.infinity))
            return entries.count - 1
        }
    }

    /// CTC prefix beam search with the character and word LMs.
    func beamSearch(_ lp: [Float], frames: Int, classes: Int) -> [UInt8] {
        var beams = [Entry(prefix: [], pb: 0, pnb: -.infinity)]
        lp.withUnsafeBufferPointer { lpp in
            for t in 0..<frames {
                let frame = lpp.baseAddress! + t * classes
                let cands = Self.topK(frame, classes: classes, k: min(topK, classes))
                var next = OrderedBeams()
                for beam in beams {
                    let prefix = beam.prefix
                    let ptot = Self.lse(beam.pb, beam.pnb)
                    for c in cands {
                        let p = Double(frame[c])
                        let cu = UInt8(c)
                        if cu == Self.blank {
                            let i = next.slot(prefix)
                            next.entries[i].pb = Self.lse(next.entries[i].pb, ptot + p)
                            continue
                        }
                        let src: Double
                        if let last = prefix.last, last == cu {
                            let i = next.slot(prefix)
                            next.entries[i].pnb = Self.lse(next.entries[i].pnb, beam.pnb + p)
                            src = beam.pb
                        } else {
                            src = ptot
                        }
                        if src == -.infinity { continue }
                        var bonus = 0.0
                        if let lm = charLM { bonus = lmWeight * lm.logprob(prefix[...], cu) + lengthBonus }
                        if let words, wordWeight != 0, cu == Self.space {
                            let closed = Self.lastWord(prefix)
                            if !closed.isEmpty { bonus += wordWeight * words.bonus(closed) }
                        }
                        let i = next.slot(prefix + [cu])
                        next.entries[i].pnb = Self.lse(next.entries[i].pnb, src + p + bonus)
                    }
                }
                // Stable: equal scores keep insertion order.
                let scores = next.entries.map { Self.lse($0.pb, $0.pnb) }
                let order = scores.indices.sorted { scores[$0] > scores[$1] || (scores[$0] == scores[$1] && $0 < $1) }
                beams = order.prefix(beamWidth).map { next.entries[$0] }
                if beams.isEmpty { beams = [Entry(prefix: [], pb: 0, pnb: -.infinity)] }
            }
        }
        var best = beams[0].prefix, bestScore = -Double.infinity
        var first = true
        for b in beams {
            var s = Self.lse(b.pb, b.pnb)
            if let words, wordWeight != 0 {
                let tail = Self.lastWord(b.prefix)
                if !tail.isEmpty { s += wordWeight * words.bonus(tail) }
            }
            if first || s > bestScore {
                best = b.prefix
                bestScore = s
                first = false
            }
        }
        return best
    }
}

import Foundation

/// Stupid-backoff character n-gram (`charlm.bin`, "MKL1"; spec/lm-format.md).
final class CharLM: Sendable {
    let order: Int
    let logBackoff: Double
    let logFloor: Double
    let scale: Double
    /// child[level][i] .. child[level][i + 1] - 1 are node i's children at level + 1 (levels 0..order-2).
    private let child: [[UInt32]]
    private let q: [[UInt16]]
    private let chars: [[UInt8]]

    init(contentsOf url: URL) throws {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard data.count >= 24, data.prefix(4) == Data("MKL1".utf8) else {
            throw MishkefetError.badFile(url.lastPathComponent)
        }
        order = Int(data.u32(at: 4))
        logBackoff = Double(data.f32(at: 8))
        logFloor = Double(data.f32(at: 12))
        scale = Double(data.f32(at: 16))
        let counts = (0..<order).map { Int(data.u32(at: 24 + 4 * $0)) }
        var off = 24 + 4 * order
        var need = off
        for k in 0..<order { need += (k < order - 1 ? 4 * (counts[k] + 1) : 0) + 3 * counts[k] }
        guard order >= 1, data.count >= need else { throw MishkefetError.badFile(url.lastPathComponent) }
        var child: [[UInt32]] = [], q: [[UInt16]] = [], chars: [[UInt8]] = []
        for k in 0..<(order - 1) {
            child.append(data.u32Array(at: off, count: counts[k] + 1))
            off += 4 * (counts[k] + 1)
        }
        for k in 0..<order {
            q.append(data.u16Array(at: off, count: counts[k]))
            off += 2 * counts[k]
        }
        for k in 0..<order {
            chars.append(data.u8Array(at: off, count: counts[k]))
            off += counts[k]
        }
        self.child = child
        self.q = q
        self.chars = chars
    }

    /// Node index of `gram` at level gram.count, or -1.
    private func find(_ gram: UnsafeBufferPointer<UInt8>) -> Int {
        var lo = 0, hi = chars[0].count, node = -1
        for (depth, c) in gram.enumerated() {
            if depth > 0 {
                lo = Int(child[depth - 1][node])
                hi = Int(child[depth - 1][node + 1])
            }
            let arr = chars[depth]
            // lower bound of c in arr[lo..<hi]
            var a = lo, b = hi
            while a < b {
                let mid = (a + b) >> 1
                if arr[mid] < c { a = mid + 1 } else { b = mid }
            }
            if a >= hi || arr[a] != c { return -1 }
            node = a
        }
        return node
    }

    /// log P(c | context), with the backoff penalty for each shortened context.
    func logprob(_ context: ArraySlice<UInt8>, _ c: UInt8) -> Double {
        let ctx = context.suffix(order - 1)
        var gram = [UInt8](ctx)
        gram.append(c)
        var penalty = 0.0
        return gram.withUnsafeBufferPointer { g in
            for length in stride(from: ctx.count, through: 0, by: -1) {
                let sub = UnsafeBufferPointer(rebasing: g[(ctx.count - length)...])
                let node = find(sub)
                if node >= 0 {
                    let v = q[sub.count - 1][node]
                    if v != 65535 { return penalty - Double(v) / scale }
                }
                penalty += logBackoff
            }
            return penalty + logFloor
        }
    }
}

/// Word unigram prior (`words.bin`, "MKW1").
final class WordLM: Sendable {
    let oov: Double
    let discount: Double
    let scale: Double
    private let offsets: [UInt32]
    private let q: [UInt16]
    private let blob: [UInt8]
    private let prefixes: Set<UInt8>

    static let prefixLetters = ["ב", "ה", "ו", "כ", "ל", "מ", "ש"]

    init(contentsOf url: URL, charset: Charset) throws {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard data.count >= 24, data.prefix(4) == Data("MKW1".utf8) else {
            throw MishkefetError.badFile(url.lastPathComponent)
        }
        let n = Int(data.u32(at: 4))
        oov = Double(data.f32(at: 8))
        discount = Double(data.f32(at: 12))
        scale = Double(data.f32(at: 16))
        var off = 24
        guard data.count >= off + 4 * (n + 1) + 2 * n else { throw MishkefetError.badFile(url.lastPathComponent) }
        offsets = data.u32Array(at: off, count: n + 1)
        off += 4 * (n + 1)
        q = data.u16Array(at: off, count: n)
        off += 2 * n
        guard data.count >= off + Int(offsets[n]) else { throw MishkefetError.badFile(url.lastPathComponent) }
        blob = data.u8Array(at: off, count: data.count - off)
        prefixes = Set(Self.prefixLetters.compactMap { charset.index(of: $0).map(UInt8.init) })
    }

    private var count: Int { q.count }

    /// Bytewise comparison of entry `i` with `word` (unsigned, shorter first).
    private func compare(_ i: Int, _ word: ArraySlice<UInt8>) -> Int {
        let a = Int(offsets[i]), b = Int(offsets[i + 1])
        var k = 0
        var j = word.startIndex
        while a + k < b && j < word.endIndex {
            let x = blob[a + k], y = word[j]
            if x != y { return x < y ? -1 : 1 }
            k += 1
            j += 1
        }
        let la = b - a, lb = word.count
        return la == lb ? 0 : (la < lb ? -1 : 1)
    }

    private func get(_ word: ArraySlice<UInt8>) -> Double? {
        var lo = 0, hi = count
        while lo < hi {
            let mid = (lo + hi) >> 1
            if compare(mid, word) < 0 { lo = mid + 1 } else { hi = mid }
        }
        if lo < count && compare(lo, word) == 0 { return -Double(q[lo]) / scale }
        return nil
    }

    func logprob(_ word: ArraySlice<UInt8>) -> Double {
        if word.isEmpty { return 0 }
        if let hit = get(word) { return hit }
        if word.count > 2, prefixes.contains(word[word.startIndex]), let rest = get(word.dropFirst()) {
            return rest + discount
        }
        return oov
    }

    func bonus(_ word: ArraySlice<UInt8>) -> Double {
        max(0, logprob(word) - oov)
    }
}

import Foundation

/// The model's output classes (index 0 is the CTC blank, 38 is space).
public struct Charset: Sendable {
    public static let blank = 0
    public static let space = 38

    public let symbols: [String]
    private let index: [String: Int]

    public init(symbols: [String]) {
        self.symbols = symbols
        var index: [String: Int] = [:]
        for (i, s) in symbols.enumerated() where i != Self.blank && index[s] == nil {
            index[s] = i
        }
        self.index = index
    }

    public init(contentsOf url: URL) throws {
        try self.init(symbols: Self.load(url))
    }

    /// The JSON array of class strings.
    public static func load(_ url: URL) throws -> [String] {
        let data = try Data(contentsOf: url)
        guard let symbols = try? JSONDecoder().decode([String].self, from: data) else {
            throw MishkefetError.badFile(url.lastPathComponent)
        }
        return symbols
    }

    public var count: Int { symbols.count }

    /// Class indices of `text`, one per unicode scalar; scalars outside the charset are dropped.
    public func encode(_ text: String) -> [Int] {
        text.unicodeScalars.compactMap { index[String($0)] }
    }

    public func decode<S: Sequence>(_ labels: S) -> String where S.Element: BinaryInteger {
        var out = ""
        for l in labels {
            let i = Int(l)
            if i > 0, i < symbols.count { out += symbols[i] }
        }
        return out
    }

    public func index(of symbol: String) -> Int? { index[symbol] }
}

import XCTest
@testable import MishkefetCore

final class PreprocessTests: XCTestCase {
    /// Every value within 3/255 of golden/preprocess/*.u8.
    func testMatchesGolden() throws {
        var worst = 0, exact = 0, total = 0
        for line in try Fixtures.lines() {
            let x = Preprocess.run(try Fixtures.gray(line.file))
            XCTAssertEqual(x.width, line.width, line.file)
            let golden = try Data(contentsOf: Fixtures.goldenFile("preprocess/\(Fixtures.stem(line.file)).u8"))
            XCTAssertEqual(golden.count, line.height * line.width, line.file)
            guard x.values.count == golden.count else { continue }
            for (v, g) in zip(x.values, golden) {
                let q = Int((v * 255 + 0.5).rounded(.down))
                let d = abs(q - Int(g))
                worst = max(worst, d)
                if d == 0 { exact += 1 }
                total += 1
            }
        }
        print("preprocess: max diff \(worst)/255, \(String(format: "%.4f", Double(exact) / Double(total))) exact")
        XCTAssertLessThanOrEqual(worst, 3)
    }

    func testOutputWidth() {
        XCTAssertEqual(Preprocess.outputWidth(width: 632, height: 120), 337)
        XCTAssertEqual(Preprocess.outputWidth(width: 10, height: 64), 16)
        XCTAssertEqual(Preprocess.outputWidth(width: 100_000, height: 64), 2560)
        // round half to even: 2.5 -> 2 (then clamped to 16), 33 * 64 / 128 = 16.5 -> 16
        XCTAssertEqual(Preprocess.outputWidth(width: 33, height: 128), 16)
        XCTAssertEqual(Preprocess.outputWidth(width: 35, height: 128), 18)
    }

    func testFlatCropIsBlank() {
        let x = Preprocess.run(GrayImage(width: 40, height: 30, fill: 200))
        XCTAssertTrue(x.values.allSatisfy { $0 == 0 })
    }

    func testTTAViewWidths() {
        let g = GrayImage(width: 100, height: 20, fill: 128)
        XCTAssertEqual(Preprocess.ttaView(g, scale: 0.82).width, 82)
        XCTAssertEqual(Preprocess.ttaView(g, scale: 1.22).width, 122)
        XCTAssertEqual(Preprocess.ttaView(GrayImage(width: 5, height: 2, fill: 1), scale: 0.82).width, 8)
        // A flat image stays flat through the normalized bicubic weights.
        XCTAssertTrue(Preprocess.ttaView(g, scale: 0.82).pixels.allSatisfy { $0 == 128 })
    }

    func testGrayConversionMatchesPIL() throws {
        // PNG lines are stored gray; the conversion must keep the stored values.
        let img = try Fixtures.cgImage(Fixtures.fixtures.appendingPathComponent("lines/p1-01.png"))
        let g = GrayImage(cgImage: img)
        XCTAssertEqual(g.width, 632)
        XCTAssertEqual(g.height, 120)
        // RGB path: a synthetic RGBA image, composited on white.
        let w = 4, h = 1
        var bytes: [UInt8] = [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 0, 0, 0, 0]
        let ctx = CGContext(data: &bytes, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 4 * w,
                            space: CGColorSpace(name: CGColorSpace.sRGB)!,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        let rgba = ctx.makeImage()!
        let gg = GrayImage(cgImage: rgba)
        // PIL: red 76, green 150, blue 29, transparent -> white 255
        XCTAssertEqual(gg.pixels, [76, 150, 29, 255])
    }
}

import XCTest
@testable import MishkefetCore

final class SegmenterTests: XCTestCase {
    /// binarize() output vs golden/pages/*.binary.png.
    func testBinarizeMatchesGolden() throws {
        for page in try Fixtures.pages() {
            let gray = try Fixtures.gray(page.file)
            let (binary, scale) = Segmenter.binarize(gray)
            XCTAssertEqual(scale, page.scale)
            let golden = GrayImage(cgImage: try Fixtures.cgImage(
                Fixtures.goldenFile("pages/\(Fixtures.stem(page.file)).binary.png")))
            XCTAssertEqual(binary.width, golden.width)
            XCTAssertEqual(binary.height, golden.height)
            let equal = zip(binary.pixels, golden.pixels).filter { $0 == $1 }.count
            let frac = Double(equal) / Double(golden.pixels.count)
            print(String(format: "binarize %@: %.5f%% pixels equal (%d differ)", page.file, frac * 100,
                         golden.pixels.count - equal))
            XCTAssertGreaterThanOrEqual(frac, 0.999, page.file)
        }
    }

    /// Same angle; at least 90% of golden boxes matched at IoU >= 0.9.
    func testBoxesMatchGolden() throws {
        for page in try Fixtures.pages() {
            let t0 = DispatchTime.now()
            let r = Segmenter.segment(try Fixtures.gray(page.file))
            let ms = Fixtures.ms(t0)
            XCTAssertEqual(r.angle, page.angle, page.file)
            let matched = page.boxes.filter { g in r.boxes.contains { Fixtures.iou($0, g) >= 0.9 } }.count
            let frac = Double(matched) / Double(page.boxes.count)
            let exact = zip(r.boxes, page.boxes).filter { $0 == $1 }.count
            print(String(format: "segment %@: angle %.1f pitch %.0f (golden %.0f), %d boxes (golden %d), "
                + "%.1f%% matched at IoU 0.9, %d identical in order, %.0f ms",
                         page.file, r.angle, r.pitch, page.pitch, r.boxes.count, page.boxes.count, frac * 100, exact, ms))
            XCTAssertGreaterThanOrEqual(frac, 0.9, page.file)
            XCTAssertEqual(r.crops.count, r.boxes.count)
        }
    }

    /// From PIL's decode of page2.jpg the boxes are identical to the golden ones.
    func testBoxesFromReferenceGrayAreIdentical() throws {
        let url = Fixtures.root.appendingPathComponent("ios/MishkefetCore/Tests/Fixtures/page2-pil-gray.png")
        let g = GrayImage(cgImage: try Fixtures.cgImage(url))
        let page = try XCTUnwrap(try Fixtures.pages().first { $0.file.hasSuffix("page2.jpg") })
        let r = Segmenter.segment(g)
        XCTAssertEqual(r.pitch, page.pitch)
        XCTAssertEqual(r.boxes, page.boxes)
        let golden = GrayImage(cgImage: try Fixtures.cgImage(Fixtures.goldenFile("pages/page2.binary.png")))
        XCTAssertEqual(Segmenter.binarize(g).0, golden)
    }

    func testRotateNearestIdentityAndQuarterTurn() {
        let g = GrayImage(width: 4, height: 4, pixels: (0..<16).map(UInt8.init))
        XCTAssertEqual(Segmenter.rotateNearest(g, angle: 0), g)
        let r = Segmenter.rotateNearest(g, angle: 90)
        // Counter-clockwise about (2, 2): dst (x, y) samples src (4 - y, x).
        XCTAssertEqual(r[0, 1], g[3, 0])
        XCTAssertEqual(r[2, 1], g[3, 2])
    }

    func testOpenRemovesSpecks() {
        var g = GrayImage(width: 10, height: 10)
        g[5, 5] = 255
        for y in 1...3 { for x in 1...3 { g[x, y] = 255 } }
        let o = Morphology.open(g, .rect(width: 2, height: 2))
        XCTAssertEqual(o[5, 5], 0)
        XCTAssertEqual(o[2, 2], 255)
    }

    func testLineKernel() {
        // tan(2 deg) * 53 = 1.85 -> drop 2: three rows covering every column once.
        let k = Morphology.Kernel.line(length: 54, angle: 2)
        XCTAssertEqual(k.rows, 3)
        XCTAssertEqual(k.runs.map { $0.c1 - $0.c0 + 1 }.reduce(0, +), 54)
        XCTAssertEqual(Morphology.Kernel.line(length: 40, angle: 0).rows, 1)
    }

    func testBlankPage() {
        let r = Segmenter.segment(GrayImage(width: 300, height: 400, fill: 250))
        XCTAssertTrue(r.boxes.isEmpty)
        // Every skew score ties at 0 and the first angle wins, as in the reference.
        XCTAssertEqual(r.angle, -6)
    }
}

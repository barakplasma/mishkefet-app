import XCTest
@testable import MishkefetCore

final class OrientationTests: XCTestCase {
    /// The paper crop on the screenshot is within 2 px of the reference; no crop elsewhere.
    func testFindPaper() throws {
        for page in try Fixtures.pages() {
            let found = Orientation.findPaper(try Fixtures.gray(page.file))
            guard let g = page.read.paper else {
                XCTAssertNil(found, page.file)
                continue
            }
            let p = try XCTUnwrap(found, page.file)
            print("findPaper \(page.file): \(p) (golden \(g))")
            for (a, b) in zip([p.x, p.y, p.w, p.h], g) { XCTAssertLessThanOrEqual(abs(a - b), 2, page.file) }
        }
    }

    /// The wide skew of each upright (cropped and turned) page equals the reference's angle.
    func testWideSkewOfUprightPage() throws {
        for page in try Fixtures.pages() {
            var g = try Fixtures.gray(page.file)
            if let p = page.read.paper { g = g.cropped(x: p[0], y: p[1], w: p[2], h: p[3]) }
            g = g.rot90(page.read.turns)
            let (angle, score) = Segmenter.estimateSkewWide(Segmenter.binarize(g).0)
            print(String(format: "skew wide %@: %.1f (golden %.1f), score %.3g", page.file, angle, page.read.angle, score))
            XCTAssertEqual(angle, page.read.angle, page.file)
        }
    }

    func testOtsu() {
        XCTAssertEqual(Orientation.otsu([10, 10, 10, 200, 200, 200]), 10)
        XCTAssertEqual(Orientation.otsu([UInt8](repeating: 7, count: 9)), 0)
    }

    func testRot90() {
        // numpy: rot90([[1,2,3],[4,5,6]]) == [[3,6],[2,5],[1,4]]
        let g = GrayImage(width: 3, height: 2, pixels: [1, 2, 3, 4, 5, 6])
        XCTAssertEqual(g.rot90(1), GrayImage(width: 2, height: 3, pixels: [3, 6, 2, 5, 1, 4]))
        XCTAssertEqual(g.rot90(2), GrayImage(width: 3, height: 2, pixels: [6, 5, 4, 3, 2, 1]))
        XCTAssertEqual(g.rot90(3), GrayImage(width: 2, height: 3, pixels: [4, 1, 5, 2, 6, 3]))
        XCTAssertEqual(g.rot90(4), g)
    }

    func testExpandedRotationKeepsCorners() {
        let (w, h) = Segmenter.expandedSize(width: 1099, height: 1600, angle: 20)
        let r = Segmenter.rotateBicubic(GrayImage(width: 1099, height: 1600, fill: 0), angle: 20, expand: true)
        XCTAssertEqual(r.width, w)
        XCTAssertEqual(r.height, h)
        XCTAssertEqual(r[0, 0], 255)  // a corner of the canvas is outside the page
        XCTAssertEqual(r[w / 2, h / 2], 0)
        let n = Segmenter.rotateNearestExpanded(GrayImage(width: 1099, height: 1600, fill: 255), angle: 20)
        XCTAssertEqual(n.width, w)
        XCTAssertEqual(n[0, 0], 0)
    }
}

final class StraightenTests: XCTestCase {
    /// Same output shape and changed flag as the reference; pixel sum within 1%.
    func testMatchesGolden() throws {
        let cases = try Fixtures.decode([Fixtures.StraightenCase].self, "straighten.json")
        XCTAssertFalse(cases.isEmpty)
        for c in cases {
            let g = try Fixtures.gray(c.file)
            XCTAssertEqual([g.width, g.height], c.in)
            let (out, changed) = Straighten.line(g)
            XCTAssertEqual(changed, c.changed, c.file)
            XCTAssertEqual([out.width, out.height], c.out, c.file)
            let sum = out.pixels.reduce(0) { $0 + Int($1) }
            XCTAssertEqual(Double(sum), Double(c.sum), accuracy: 0.01 * Double(c.sum), c.file)
            if changed, out.width == c.out[0], out.height == c.out[1] {
                let golden = GrayImage(cgImage: try Fixtures.cgImage(
                    Fixtures.goldenFile("straighten/\(Fixtures.stem(c.file)).png")))
                let diff = zip(out.pixels, golden.pixels).map { abs(Int($0) - Int($1)) }
                print(String(format: "straighten %@: %dx%d -> %dx%d, max diff %d, mean diff %.3f", c.file, c.in[0], c.in[1],
                             c.out[0], c.out[1], diff.max() ?? 0, Double(diff.reduce(0, +)) / Double(diff.count)))
            }
        }
    }

    func testLevelLineUnchangedAndCanvasRenderUnchanged() throws {
        let flat = GrayImage(width: 200, height: 40, fill: 255)
        XCTAssertFalse(Straighten.line(flat).changed)
        let strokes = [[CGPoint(x: 0, y: 10), CGPoint(x: 100, y: 10)], [CGPoint(x: 0, y: 30), CGPoint(x: 100, y: 30)]]
        let img = try XCTUnwrap(StrokeRenderer.render(strokes, nChars: 4))
        XCTAssertFalse(Straighten.line(img).changed)
    }

    func testQuadraticFit() throws {
        let xs = (0..<50).map(Double.init)
        let ys = xs.map { 3 - 0.5 * $0 + 0.02 * $0 * $0 }
        let q = try XCTUnwrap(Quadratic(xs: xs, ys: ys, weights: xs.map { _ in 1 }))
        for x in [0.0, 17, 49] { XCTAssertEqual(q(x), 3 - 0.5 * x + 0.02 * x * x, accuracy: 1e-9) }
    }
}

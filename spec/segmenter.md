# Page segmentation

Port of hebocr's `segment_page`, as re-implemented in `reference.py`. Every
threshold scales with the measured line pitch `p`. Images are uint8, ink = 255
in binary masks. Morphology never counts pixels outside the image (erode treats
them as 255, dilate as 0).

## Morphology convention (OpenCV's)
A kernel is a 0/1 matrix of `rows x cols` with anchor `(ax, ay) = (cols // 2, rows // 2)`.
`erode(a)[y][x] = min over kernel cells (i, j) of a[y + i - ay][x + j - ax]`,
`dilate` the same with max (**no** kernel reflection). `open = dilate(erode(a))`,
`close = erode(dilate(a))`. All kernels used here have a single run of ones per row
(or per column, for transposed ones), so each op is a min/max of shifted 1-D
running windows (van Herk / Gil-Werman makes those O(1) per pixel).

## Steps
1. **binarize(gray)**
   - `scale = min(1, 2000 / max(h, w))`. If `scale < 1`: area-resize to
     `round(w*scale) x round(h*scale)` using source-per-destination `1/scale` on both axes.
   - Flatten: `bg = blur(gray, sigma = max(h, w) / 30)` (exact Gaussian as in
     [preprocess.md](preprocess.md), reflect101, `round_u8`);
     `flat = trunc(clamp(float32(gray) / max(float32(bg), 1) * 128, 0, 255))`.
   - `block = max(15, (min(h, w) // 40) | 1)`; `mean = round_u8(gauss(flat, block x block,
     sigma = 0.3*((block-1)*0.5 - 1) + 0.8))` with border **replicate**.
   - `binary = 255 where flat - mean <= -12 else 0`.
   - `binary = open(binary, ones 2x2)`.
2. **estimate_skew(binary)**: `small` = area resize to `round(w/2) x round(h/2)`
   with source-per-destination 2. For `angle = -6 + 0.5*i`, `i = 0..24`: rotate
   `small` (nearest, see below), `profile = row sums` (double), score =
   population variance of `diff(profile)`. First strictly best angle wins.
3. **rotation**: `M = [[c, s, (1-c)*cx - s*cy], [-s, c, s*cx + (1-c)*cy]]`, `c = cos(angle)`,
   `s = sin(angle)`, `cx = w/2`, `cy = h/2`, angle in degrees (positive is
   counter-clockwise on screen). Destination pixel `(x, y)` samples source
   `floor(Minv * (x, y) + 0.5)`; out of bounds is 0. If `|angle| > 1e-3`, rotate
   the binary mask this way and the original RGB the same way with bicubic
   sampling and white fill (platform rotation is fine for RGB: it only feeds crops).
4. **estimate_line_pitch(binary)**: `prof = row sums - mean` (double). If all
   zero: 30. `lo = max(8, trunc(0.004 n))`, `hi = min(n - 1, n // 2)`; if
   `hi <= lo + 2`: 30. `win[k] = sum_t prof[t] * prof[t + lag]` for `lag = lo..hi-1`.
   First `k` in `1..len-2` with `win[k] > win[k-1]`, `win[k] >= win[k+1]`, `win[k] > 0`;
   else argmax. Pitch = `lo + k`.
   This first-peak rule is fragile on very dense pages (a tiny change flips it to
   a sub-harmonic); it is kept because it is what the shipped numbers used.
   **Deviation from hebocr:** hebocr searched lags up to `0.15 n`, so a photo of a
   few large lines (spacing above 15% of the height) fell back to a tiny lag and
   was cut into fragments. Searching up to `n // 2` fixes that and gives identical
   results on every page the old window handled.
5. **remove_rules(binary, p)**: `L = max(40, trunc(2p))`, `t = max(3, trunc(0.28 p))`.
   - Line kernel for angle `a` in `-2, -1, -0.4, 0, 0.4, 1, 2`: `drop = round(tan(a) * (L-1))`
     (half to even), `rows = |drop| + 1`, `y0 = 0 if drop >= 0 else rows - 1`, cells of an
     8-connected line from `(0, y0)` to `(L-1, y0 + drop)` drawn as OpenCV's
     LineIterator does (see `_cv_line` in the reference).
   - `horiz = OR over a of open(binary, K_a)`; `vert = OR over a of open(binary, K_a transposed)`.
   - For `(runs, thin) in ((horiz, 1 wide x t tall), (vert, t wide x 1 tall))`:
     `rules = runs - open(runs, thin)` (saturating), `out = out - dilate(rules, ones 3x3)`.
6. Re-estimate the pitch on the cleaned mask.
7. **find_lines**: `sx = max(8, trunc(0.55p))`, `sy = max(1, trunc(0.05p))`.
   `dilate(sx wide x sy tall)`, then `close(sx wide x 1)`. 8-connected components;
   keep a component's bounding box unless `w < 0.8p`, `h < 0.25p`, `h > 12p`, or
   pixel count `< 0.12 p^2`.
8. **split_tall**: for a box with `n = round(h / p) > 1` (half to even): `prof` = row
   sums of the box in the cleaned mask, smoothed vertically by `[1,4,6,4,1]/16`
   (reflect101). Cuts: start `[0]`; for `k = 1..n-1`: `centre = trunc(k h / n)`,
   `lo = max(last + 2, centre - trunc(0.35p))`, `hi = min(h - 2, centre + trunc(0.35p))`,
   if `hi > lo` append `lo + argmin(smooth[lo:hi])`. End with `h`. Each band is
   tightened to its ink; empty bands are dropped.
9. **merge_overlapping**: sort by `(y, x)`; for each box, merge into the first
   already-kept box with vertical overlap `> 0.5 * min(h)` and horizontal gap
   `< 1.2p` (union box), else keep it.
10. **reading_order**: sort by centre y (stable); a box joins the current row when
    `|cy - mean cy of the row| < 0.45p`; rows top to bottom, inside a row by right
    edge descending.
11. Map back: `trunc(v / scale)` for x, y, w, h (deskewed page coordinates). Crop
    from the rotated RGB with `py = trunc(0.12 h)` and `px = trunc(0.05 h)` of padding,
    clamped to the image.

`golden/pages/*.binary.png` is step 1's output for each fixture page.

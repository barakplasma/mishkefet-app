# Training canvas strokes to a line image

The renderer makes a grayscale crop that then goes through ordinary
[preprocess](preprocess.md), so canvas samples and scans share one path.

Input: strokes as lists of points `(x, y)` in canvas units (y down), the
number of characters in the target word `n`, and render params:

| param | plain (seed 0) | augmented (seed > 0) |
|---|---|---|
| slant | 0 | `(r() * 2 - 1) * 0.25` |
| rotate (deg) | 0 | `(r() * 2 - 1) * 2` |
| radius | 1.2 | `0.9 + r() * 1.1` |
| density | 1 | `0.85 + r() * 0.33` |
| ink | 0.62 | `0.5 + r() * 0.3` |

`r` is mulberry32 seeded with the seed, drawn in that order:

```
state = (state + 0x6D2B79F5) >>> 0
t = imul(state ^ (state >>> 15), state | 1)
t ^= t + imul(t ^ (t >>> 7), t | 61)
return ((t ^ (t >>> 14)) >>> 0) / 4294967296
```

## Algorithm (H = 128)
1. `cx, cy` = centre of the raw points' bounding box. Transform every point:
   `x += slant * (cy - y)`, then rotate by `rotate` degrees about `(cx, cy)`
   (`x' = cx + dx cos - dy sin`, `y' = cy + dx sin + dy cos`).
2. Bounding box `x0, x1, y0, y1` of the transformed points.
   `s = H * ink / max(y1 - y0, 1e-6)`; if `y1 - y0 < 1e-3`, `s = 1`.
   `top = (H - (y1 - y0) * s) / 2`.
3. Density: `width = (x1 - x0) * s + 24`; if `n > 0`,
   `fx = clamp(44.2 * n / width, 0.5, 2)` else 1; `fx *= density`.
4. `W = max(16, ceil((x1 - x0) * s * fx + 24))`. Pixel coords:
   `X = (x - x0) * s * fx + 12`, `Y = (y - y0) * s + top`.
5. Coverage: for each stroke, each segment between consecutive points (a
   one-point stroke is a zero-length segment), for every pixel `(i, j)` within
   `radius + 1` of the segment's box: `d` = distance from `(i + 0.5, j + 0.5)` to the
   segment, `cov = max(cov, clamp(radius + 0.5 - d, 0, 1))`.
6. `gray = floor(255 - cov * 225 + 0.5)`.

Training views: seed 0 plus seeds `(fnv1a32(sample id) & 0xFFFFFF) * 16 + k` for `k = 1..AUG` (see adapter.md).

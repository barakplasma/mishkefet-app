# Line preprocessing

Input: a line crop as RGB(A). Output: float32 `(64, W)`, ink near 1, paper near 0.

## Loading
- Apply the EXIF orientation. Composite any alpha onto white.
- Gray = `(R*19595 + G*38470 + B*7471 + 0x8000) >> 16` (PIL "L", BT.601), uint8.

## preprocess(gray)
1. **Contrast normalize at the crop's own resolution** (before resizing):
   - `sigma = max(h / 8, 3)`; `ksize = floor(sigma*6 + 1 + 0.5) | 1`.
   - Gaussian kernel `k[i] = exp(-(i - (ksize-1)/2)^2 / (2 sigma^2))`, normalized.
   - Separable correlation, rows then columns, in double, border **reflect101**
     (`... 2 1 | 0 1 2 ... n-1 | n-2 ...`; a length-1 axis repeats its only sample).
   - `blur = round_u8(result)` where `round_u8(x) = clamp(floor(x + 0.5), 0, 255)`.
   - `flat = float32(gray) / max(float32(blur), 1e-3)` (float32 division).
   - `lo, hi` = the 2nd and 98th percentiles of `flat` (numpy linear: sort, position
     `q/100 * (n-1)`, interpolate between neighbours), computed in double.
   - If `hi - lo < 1e-6` the result is all 255. Otherwise
     `trunc(clamp((flat - lo) / (hi - lo) * 255, 0, 255))` as uint8 (truncation, not rounding).
2. **Resize to height 64**: `scale = 64 / h`, `new_w = clamp(round(w * scale), 16, 2560)`
   (round half to even, as Python's `round`).
   - `scale < 1`: **area** resize on both axes. Output pixel `i` covers source
     `[i*s, (i+1)*s)` with `s = src/dst`; weights are the overlap lengths, normalized.
   - otherwise **bilinear** with half-pixel centres: `f = (i + 0.5)*s - 0.5`,
     `j = floor(f)`, `a = f - j`; if `j < 0` use `j = 0, a = 0`; if `j >= n-1` use `j = n-1, a = 0`.
   - Both are separable (columns of `wy` then `wx`), computed in double, then `round_u8`.
3. **Invert**: `x = 1 - float32(g) / 255`.

## TTA views
`tta_view(gray, s)`: for `s != 1`, resize horizontally only to `max(8, trunc(w * s))`
columns with PIL's BICUBIC (`a = -0.5`):
- `scale = w / new_w`, `fscale = max(scale, 1)`, `support = 2 * fscale`
- for output `i`: `centre = (i + 0.5) * scale`, `lo = max(trunc(centre - support + 0.5), 0)`,
  `hi = min(trunc(centre + support + 0.5), w)`, weights `cubic((j - centre + 0.5) / fscale)`
  for `j` in `[lo, hi)`, normalized; `round_u8`.

The view then goes through `preprocess` like any crop. Scales: `1.0, 0.82, 1.22`.

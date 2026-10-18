# Straightening a line

Every line crop is straightened before it is read (`read_line`, before the TTA
views), including single-line camera shots and training canvas renders (which
are already straight and come back unchanged). Reference: `straighten_line`.

Handwriting on one page does not share one angle: lines tilt and bend on their
own. The page is deskewed as a whole; this fixes each line's remaining tilt and
curve.

1. `flat = contrast_normalize(gray)` ([preprocess.md](preprocess.md)), uint8.
   `t = otsu(flat)` ([orientation.md](orientation.md) step 2). Ink weight
   `wt = clamp((t - flat) / max(t, 1), 0, 1)` in double. (A fixed threshold let textured
   or grey paper weigh as much as ink and pulled the centreline to the crop's middle.)
2. Per column `x`: `col[x] = sum_y wt`. Columns with `col > 0.5` are kept; if fewer than
   `max(8, 0.15 w)`, return the crop unchanged. Their centre `m[x] = sum_y y*wt / col[x]`.
3. Weighted quadratic fit (numpy `polyfit(xs, m, 2, w=sqrt(col))`, least squares on
   `sqrt(col)`-scaled rows). Up to twice: residuals `r = m - fit(xs)`,
   `rms = sqrt(weighted mean of r^2 with weights col)`, keep columns with
   `|r| < max(0.18 h, 2.5 rms + 1)`; if all kept, stop; refit on the kept ones; if fewer than 8
   remain, return the crop unchanged.
4. `f(x) = fit(x)` for every column; `sigma = sqrt(sum wt*(y - f(x))^2 / sum wt)` over all pixels.
5. `dev = f - mean(f)`. If `max |dev| < 0.15 h`, return the crop **unchanged** (on the
   benchmark, 65 of 225 already-cropped lines are straightened at all).
6. `pad = ceil(max |dev|)`, output height `H = h + 2 pad`, white (255). For each column `x` and
   output row `y`: source row `y - pad + dev[x]`; if within `[0, h-1]`, linear interpolation
   between its two neighbours.
7. Keep only the band around the line: `c = mean(f) + pad`, `band = max(2.5 sigma, 0.22 h)`,
   rows `trunc(max(0, c - band))` to `trunc(min(H, c + band + 1))` (exclusive). This cuts away
   neighbouring lines' ink that the axis-aligned crop caught. `round_u8` the result.

Measured on benchmark lines given a random tilt (up to 15 degrees) and bend (up to 0.35 h):
greedy CER 0.537 untouched, 0.305 straightened. On the real benchmark lines it does no
harm: 0.3002 untouched, 0.2999 straightened.

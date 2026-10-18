# Getting a page upright

Runs before [segmentation](segmenter.md) on every page read (not on single-line
camera shots). Reference: `find_paper`, `orient_page` in `reference.py`.

## 1. Crop to the paper (screenshots)

`find_paper(rgb)` returns a crop rectangle or nothing:

1. Gray (BT.601), area-resize so the long side is 256 (`w = round(W*s)`, `h = round(H*s)`,
   source-per-destination `W/w`, `H/h`).
2. `t = otsu(small)`: the first argmax over `t` of `(mu_T*omega(t) - mu(t))^2 / (omega(t)*(1-omega(t)))`
   on the 256-bin histogram (0/0 counts as 0). `bright = small > t`.
3. Binary close with a `k x k` square, `k = max(3, round(0.03*min(w,h))) | 1`, then binary open
   with 3x3. Pixels outside the image count as dark in every dilation and erosion (scipy's
   `border_value=0`). The result only needs to be right to a pixel at 256 px.
4. Largest 4-connected bright component (scipy `label` default structure). Stop (no crop) if
   it is under 20% of `w*h`, or its bounding box is over 92% of `w*h`.
5. Outside the box: stop unless its mean gray is at most 25 **and** it covers at least
   25% of the image (near-black app bars, not a table or shading).
6. Crop 1.5% inside the box: `X0 = trunc((x0/w + 0.015)*W)`, `X1 = trunc((x1/w - 0.015)*W)`,
   same for y, clamped; stop if under 16 px either way.

The crop is what gets stored and shown as the scan.

## 2. Quarter turns

`orient_page(rgb)` returns `turns` (quarter turns counter-clockwise, as displayed: `rot90`
with `out[y][x] = in[x][W-1-y]`, output `W x H`) and the segmentation of the turned page.

1. `binary = binarize(gray)` ([segmenter.md](segmenter.md) step 1).
2. `(_, s0) = estimate_skew_wide(binary)`, `(_, s1) = estimate_skew_wide(rot90(binary, 1))`.
3. If `max(s0, s1) >= 3 * min(s0, s1)` the candidates are `[0]` if `s0 >= s1` else `[1]`;
   otherwise `[0, 1]` (graph paper and sparse pages fool the stripe test).
4. For each candidate `base`: `seg = segment_page(rot90(rgb, base))`; then for `flip` in
   `false, true`: score = mean, over the **6** widest crops (stable sort by width descending),
   of the **emitting confidence** of the greedy single-view reading of the crop's **middle
   window** (when the crop is wider than `win = trunc(10 * crop height)`, keep columns
   `x0 = (w - win) // 2` to `x0 + win`), rotated 180 degrees when `flip`, read with the
   **general** head. No crops: 0. The first strictly highest `(base, flip)` wins, in the order
   (0, no), (0, yes), (1, no), (1, yes). (Middle windows instead of whole lines: the vote
   needs letters, not lines, and phones without a GPU spent over a minute on whole lines.)
5. `turns = (base + 2*flip) % 4`. If `flip`, segment `rot90(rgb, turns)` again (the 180-degree
   page gets its own segmentation).

The upright (turned) image is what gets stored and shown as the scan.

Measured: all 40 turned benchmark pages (4 turns each) come back upright; turned and
additionally tilted copies (plus or minus 20 to 30 degrees) too.

"""Render handwriting strokes from the training canvas into a line image.

The output is an ordinary grayscale crop (dark ink on white paper) that goes
through the same `preprocess` as a photographed line, so scans and canvas
samples share one path into the model. What the renderer controls is how
much it looks like the paper the model was trained on:

  - ink height fills 62% of the crop (training used 45 to 100%),
  - strokes are 1.5 px wide at the model's 64 px line height (ballpoint is
    1 to 2 px; the canvas itself draws a much fatter finger stroke),
  - the word is stretched horizontally to about 22.1 px per character at
    64 px, the density the model was tuned to (hebocr's _match_density),
    because a finger on glass writes far wider than a pen on paper.

Rasterization is analytic (distance to each segment), so the web and iOS
renderers produce the same pixels without depending on a canvas library.
spec/strokes.md is the prose version of this file.
"""

import math

import numpy as np

H = 128             # rendered crop height; preprocess brings it to 64
INK = 0.62          # ink bbox height as a fraction of H
PAD_X = 12          # paper left and right of the ink, px
RADIUS = 1.2        # stroke half-width at H = 128 (about 1.2 px wide at 64)
PX_PER_CHAR = 44.2  # 22.1 at 64 px, doubled for H = 128
DENSITY_CLIP = (0.5, 2.0)
MIN_INK_H = 1e-6
PAPER, INK_LEVEL = 255, 30


def mulberry32(seed: int):
    """The deterministic PRNG both apps use for augmentation."""
    state = seed & 0xFFFFFFFF

    def nxt() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = state
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    return nxt


def augment_params(seed: int) -> dict:
    """Random rendering variation for one training view. Seed 0 is the plain render."""
    if seed == 0:
        return dict(slant=0.0, rotate=0.0, radius=RADIUS, density=1.0, ink=INK)
    r = mulberry32(seed)
    return dict(
        slant=(r() * 2 - 1) * 0.25,
        rotate=(r() * 2 - 1) * 2.0,
        radius=0.9 + r() * 1.1,
        density=0.85 + r() * 0.33,
        ink=0.5 + r() * 0.3,
    )


def render(strokes: list[list[tuple[float, float]]], n_chars: int, params: dict | None = None) -> np.ndarray | None:
    """Strokes (canvas coordinates, y down) -> uint8 (128, W) crop, or None if empty."""
    p = params or augment_params(0)
    pts = [(float(x), float(y)) for s in strokes for x, y in s]
    if not pts:
        return None
    # slant and rotation about the ink centre, before measuring the box
    xs, ys = [x for x, _ in pts], [y for _, y in pts]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    a = math.radians(p["rotate"])
    ca, sa = math.cos(a), math.sin(a)

    def tf(x, y):
        x = x + p["slant"] * (cy - y)
        dx, dy = x - cx, y - cy
        return cx + dx * ca - dy * sa, cy + dx * sa + dy * ca

    strokes_t = [[tf(x, y) for x, y in s] for s in strokes if s]
    flat = [q for s in strokes_t for q in s]
    x0, x1 = min(q[0] for q in flat), max(q[0] for q in flat)
    y0, y1 = min(q[1] for q in flat), max(q[1] for q in flat)
    ink_h = max(y1 - y0, MIN_INK_H)
    s = H * p["ink"] / ink_h
    # a single dot or a flat dash has no height to scale by; draw it small
    if y1 - y0 < 1e-3:
        s = 1.0
    top = (H - (y1 - y0) * s) / 2
    width = (x1 - x0) * s + 2 * PAD_X
    fx = 1.0
    if n_chars > 0:
        fx = min(max(PX_PER_CHAR * n_chars / width, DENSITY_CLIP[0]), DENSITY_CLIP[1])
    fx *= p["density"]
    W = max(16, int(math.ceil((x1 - x0) * s * fx + 2 * PAD_X)))

    def to_px(q):
        return ((q[0] - x0) * s * fx + PAD_X, (q[1] - y0) * s + top)

    cov = np.zeros((H, W), np.float32)
    r = p["radius"]
    for stroke in strokes_t:
        px = [to_px(q) for q in stroke]
        segs = list(zip(px, px[1:])) if len(px) > 1 else [(px[0], px[0])]
        for (ax, ay), (bx, by) in segs:
            _capsule(cov, ax, ay, bx, by, r)
    gray = PAPER - cov * (PAPER - INK_LEVEL)
    return np.floor(gray + 0.5).astype(np.uint8)


def _capsule(cov, ax, ay, bx, by, r):
    """cov = max(cov, clamp(r + 0.5 - distance to segment, 0, 1)) at pixel centres."""
    h, w = cov.shape
    reach = r + 1
    xa, xb = int(math.floor(min(ax, bx) - reach)), int(math.ceil(max(ax, bx) + reach))
    ya, yb = int(math.floor(min(ay, by) - reach)), int(math.ceil(max(ay, by) + reach))
    xa, ya, xb, yb = max(xa, 0), max(ya, 0), min(xb, w - 1), min(yb, h - 1)
    if xa > xb or ya > yb:
        return
    gy, gx = np.mgrid[ya: yb + 1, xa: xb + 1]
    px, py = gx + 0.5, gy + 0.5
    dx, dy = bx - ax, by - ay
    ll = dx * dx + dy * dy
    if ll > 0:
        t = np.clip(((px - ax) * dx + (py - ay) * dy) / ll, 0.0, 1.0)
    else:
        t = np.zeros_like(px, dtype=np.float64)
    d = np.hypot(px - (ax + t * dx), py - (ay + t * dy))
    c = np.clip(r + 0.5 - d, 0.0, 1.0).astype(np.float32)
    region = cov[ya: yb + 1, xa: xb + 1]
    np.maximum(region, c, out=region)

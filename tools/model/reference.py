"""Reference implementation of the Mishkefet pipeline: the yardstick both apps
are tested against.

Plain numpy (+ scipy.ndimage for connected components) and onnxruntime. It
re-implements hebocr's preprocessing, page segmentation, CTC decoding and
language models from their algorithms, without OpenCV or PIL resampling, so
that every step is something the web and iOS apps can reproduce. It is checked
against hebocr itself by test_reference.py (same text on the fixtures, line
boxes at IoU >= 0.9).

    python reference.py read <image>          # one cropped line
    python reference.py read <image> --page   # a whole page
    python reference.py read <image> --head <head.bin>   # a fine-tuned head

The algorithms are written out in spec/*.md; when this file and a spec
disagree, this file is right and the spec is a bug.
"""

import argparse
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common import BUILD

LINE_HEIGHT = 64
MIN_WIDTH = 16
MAX_WIDTH = 2560
BLANK = 0
SPACE = 38
TTA_SCALES = (1.0, 0.82, 1.22)

DECODE = dict(beam_width=12, top_k=8, lm_weight=0.4, length_bonus=0.6, word_weight=0.2)


# --------------------------------------------------------------------------
# Image primitives. All images are 2-D arrays, uint8 unless stated.
# --------------------------------------------------------------------------

def to_gray(rgb: np.ndarray) -> np.ndarray:
    """ITU-R 601 luma with PIL's fixed-point rounding."""
    if rgb.ndim == 2:
        return rgb.astype(np.uint8)
    r, g, b = (rgb[..., i].astype(np.uint32) for i in range(3))
    return ((r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16).astype(np.uint8)


def round_u8(x: np.ndarray) -> np.ndarray:
    """Round half away from zero, saturate to uint8 (OpenCV saturate_cast)."""
    return np.clip(np.floor(x + 0.5), 0, 255).astype(np.uint8)


def gaussian_kernel(ksize: int, sigma: float) -> np.ndarray:
    """OpenCV getGaussianKernel, including its fixed small-kernel tables."""
    if sigma <= 0:
        fixed = {1: [1.0], 3: [0.25, 0.5, 0.25], 5: [0.0625, 0.25, 0.375, 0.25, 0.0625],
                 7: [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125]}
        if ksize in fixed:
            return np.array(fixed[ksize])
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    x = np.arange(ksize) - (ksize - 1) / 2
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    return k / k.sum()


def _pad_index(n: int, r: int, mode: str) -> np.ndarray:
    """Source indices for a signal of length n padded by r on each side."""
    i = np.arange(-r, n + r)
    if n == 1:
        return np.zeros_like(i)
    if mode == "replicate":
        return np.clip(i, 0, n - 1)
    # reflect101: ... 2 1 | 0 1 2 ... n-1 | n-2 n-3 ...
    period = 2 * (n - 1)
    i = np.mod(i, period)
    return np.where(i >= n, period - i, i)


def convolve_sep(img: np.ndarray, kx: np.ndarray, ky: np.ndarray, border="reflect101") -> np.ndarray:
    """Separable correlation in float64, rows then columns."""
    a = img.astype(np.float64)
    rx, ry = len(kx) // 2, len(ky) // 2
    if rx:
        p = a[:, _pad_index(a.shape[1], rx, border)]
        out = np.zeros_like(a)
        for j, w in enumerate(kx):
            out += w * p[:, j : j + a.shape[1]]
        a = out
    if ry:
        p = a[_pad_index(a.shape[0], ry, border), :]
        out = np.zeros_like(a)
        for j, w in enumerate(ky):
            out += w * p[j : j + a.shape[0], :]
        a = out
    return a


def gaussian_ksize_u8(sigma: float) -> int:
    """OpenCV's kernel size for GaussianBlur(ksize=(0,0)) on 8-bit input."""
    return int(math.floor(sigma * 6 + 1 + 0.5)) | 1


def area_weights(n_src: int, n_dst: int, scale: float) -> np.ndarray:
    """(n_dst, n_src) box-filter weights: dst i covers src [i*scale, (i+1)*scale)."""
    w = np.zeros((n_dst, n_src))
    for i in range(n_dst):
        a, b = i * scale, min((i + 1) * scale, n_src)
        j0, j1 = int(math.floor(a)), int(math.ceil(b))
        for j in range(j0, min(j1, n_src)):
            w[i, j] = min(b, j + 1) - max(a, j)
    s = w.sum(axis=1, keepdims=True)
    return w / np.where(s > 0, s, 1)


def linear_weights(n_src: int, n_dst: int, scale: float) -> np.ndarray:
    """(n_dst, n_src) bilinear weights with half-pixel centres, edges clamped."""
    w = np.zeros((n_dst, n_src))
    for i in range(n_dst):
        f = (i + 0.5) * scale - 0.5
        j = int(math.floor(f))
        f -= j
        if j < 0:
            j, f = 0, 0.0
        if j >= n_src - 1:
            j, f = n_src - 1, 0.0
        w[i, j] += 1 - f
        if f:
            w[i, j + 1] += f
    return w


def resize(img: np.ndarray, new_w: int, new_h: int, mode: str, sx=None, sy=None) -> np.ndarray:
    """Area (shrinking) or linear (enlarging) resize to uint8.

    `sx`, `sy` override the source-per-destination scale, the way OpenCV uses
    1/fx rather than w/new_w when a resize is given as a factor."""
    h, w = img.shape
    sx = w / new_w if sx is None else sx
    sy = h / new_h if sy is None else sy
    make = area_weights if mode == "area" else linear_weights
    wx, wy = make(w, new_w, sx), make(h, new_h, sy)
    return round_u8(wy @ img.astype(np.float64) @ wx.T)


def bicubic_resize_x(gray: np.ndarray, new_w: int) -> np.ndarray:
    """Horizontal-only resize, PIL BICUBIC (a = -0.5, support widened when shrinking)."""
    h, w = gray.shape
    scale = w / new_w
    fscale = max(scale, 1.0)
    support = 2.0 * fscale

    def cubic(x):
        x = abs(x)
        if x < 1:
            return ((1.5 * x - 2.5) * x) * x + 1
        if x < 2:
            return ((-0.5 * x + 2.5) * x - 4) * x + 2
        return 0.0

    wt = np.zeros((new_w, w))
    for i in range(new_w):
        centre = (i + 0.5) * scale
        lo = max(int(centre - support + 0.5), 0)
        hi = min(int(centre + support + 0.5), w)
        ks = np.array([cubic((j - centre + 0.5) / fscale) for j in range(lo, hi)])
        if ks.sum():
            ks /= ks.sum()
        wt[i, lo:hi] = ks
    return round_u8(gray.astype(np.float64) @ wt.T)


def blur_large(gray: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of an 8-bit image with OpenCV's kernel size, as uint8.

    Exact rather than approximated at reduced resolution: page binarization
    turned out to be sensitive to a level or two of background error, which
    moved line boxes."""
    k = gaussian_kernel(gaussian_ksize_u8(sigma), sigma)
    return round_u8(convolve_sep(gray, k, k))


def percentile(a: np.ndarray, q: float) -> float:
    """numpy's default (linear) percentile."""
    s = np.sort(a.ravel().astype(np.float64))
    pos = q / 100.0 * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


# --------------------------------------------------------------------------
# Line preprocessing (hebocr.data.transforms.preprocess)
# --------------------------------------------------------------------------

def contrast_normalize(gray: np.ndarray) -> np.ndarray:
    blur = blur_large(gray, max(gray.shape[0] / 8, 3.0)).astype(np.float32)
    flat = gray.astype(np.float32) / np.maximum(blur, np.float32(1e-3))
    lo, hi = percentile(flat, 2), percentile(flat, 98)
    if hi - lo < 1e-6:
        return np.full_like(gray, 255)
    return np.clip((flat.astype(np.float64) - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def preprocess(gray: np.ndarray) -> np.ndarray:
    """uint8 line crop -> float32 (64, W), ink near 1."""
    g = contrast_normalize(gray)
    h, w = g.shape
    scale = LINE_HEIGHT / max(h, 1)
    new_w = max(MIN_WIDTH, min(int(round(w * scale)), MAX_WIDTH))
    g = resize(g, new_w, LINE_HEIGHT, "area" if scale < 1 else "linear")
    return (1.0 - g.astype(np.float32) / np.float32(255.0)).astype(np.float32)


# --------------------------------------------------------------------------
# Line straightening (spec/straighten.md)
# --------------------------------------------------------------------------

STRAIGHTEN_MIN_DEV = 0.15   # fraction of the crop height the centreline must wander to act
STRAIGHTEN_BAND = 2.5       # keep rows within this many ink standard deviations of the line


def line_centerline(gray: np.ndarray):
    """(f, sigma): a weighted quadratic through the ink's vertical centre per column,
    refitted twice without columns far from it (pieces of neighbouring lines);
    sigma is the ink's spread around it. None when there is too little ink."""
    h, w = gray.shape
    flat = contrast_normalize(gray)
    # ink weight: how far below the crop's own Otsu split a pixel is (textured or
    # grey paper must weigh nothing, or the centreline drifts to the crop's middle)
    t = float(otsu_threshold(flat))
    wt = np.clip((t - flat.astype(np.float64)) / max(t, 1.0), 0.0, 1.0)
    col = wt.sum(axis=0)
    ys = np.arange(h, dtype=np.float64)
    keep = col > 0.5
    if keep.sum() < max(8, w * 0.15):
        return None
    xs = np.arange(w, dtype=np.float64)[keep]
    m = (wt * ys[:, None]).sum(axis=0)[keep] / col[keep]
    wk = col[keep]
    coef = None
    for _ in range(2):
        coef = np.polyfit(xs, m, 2, w=np.sqrt(wk))
        res = m - np.polyval(coef, xs)
        rms = math.sqrt(float(np.average(res * res, weights=wk)))
        good = np.abs(res) < max(0.18 * h, 2.5 * rms + 1.0)
        if good.all():
            break
        xs, m, wk = xs[good], m[good], wk[good]
        if len(xs) < 8:
            return None
    f = np.polyval(coef, np.arange(w, dtype=np.float64))
    dy = ys[:, None] - f[None, :]
    sigma = math.sqrt(float((wt * dy * dy).sum() / max(float(wt.sum()), 1e-6)))
    return f, sigma


def straighten_line(gray: np.ndarray) -> np.ndarray:
    """Flatten a tilted or curved line: shift every column so the ink's centreline
    is level, then keep only a band around it (neighbouring lines' ink is cut away).
    A line whose centreline wanders less than STRAIGHTEN_MIN_DEV of the height is
    returned unchanged."""
    h, w = gray.shape
    fit = line_centerline(gray)
    if fit is None:
        return gray
    f, sigma = fit
    dev = f - f.mean()
    if np.abs(dev).max() < STRAIGHTEN_MIN_DEV * h:
        return gray
    pad = int(math.ceil(float(np.abs(dev).max())))
    H = h + 2 * pad
    out = np.full((H, w), 255.0)
    ys = np.arange(H, dtype=np.float64)
    for x in range(w):
        src = ys - pad + dev[x]
        ok = (src >= 0) & (src <= h - 1)
        s_ = src[ok]
        i0 = np.floor(s_).astype(np.int64)
        i1 = np.minimum(i0 + 1, h - 1)
        a = s_ - i0
        out[ok, x] = gray[i0, x] * (1 - a) + gray[i1, x] * a
    c = float(f.mean()) + pad
    band = max(STRAIGHTEN_BAND * sigma, 0.22 * h)
    top, bot = int(max(0.0, c - band)), int(min(float(H), c + band + 1))
    return round_u8(out[top:bot])


def tta_view(gray: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return gray
    return bicubic_resize_x(gray, max(8, int(gray.shape[1] * scale)))


# --------------------------------------------------------------------------
# Model: ONNX backbone + head applied here
# --------------------------------------------------------------------------

def read_head(path) -> tuple[np.ndarray, np.ndarray]:
    raw = Path(path).read_bytes()
    assert raw[:4] == b"MKH1"
    classes, dim = struct.unpack("<II", raw[4:12])
    w = np.frombuffer(raw, "<f4", classes * dim, 12).reshape(classes, dim)
    b = np.frombuffer(raw, "<f4", classes, 12 + classes * dim * 4)
    return w.astype(np.float32), b.astype(np.float32)


def log_softmax(z: np.ndarray) -> np.ndarray:
    m = z.max(axis=-1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=-1, keepdims=True))


class Model:
    def __init__(self, backbone=None, head=None):
        import onnxruntime as ort

        backbone = backbone or BUILD / "backbone.fp32.onnx"
        self.session = ort.InferenceSession(str(backbone), providers=["CPUExecutionProvider"])
        self.w, self.b = read_head(head or BUILD / "head.bin")

    def features(self, image: np.ndarray) -> np.ndarray:
        """(64, W) float32 -> (W // 4, 512)."""
        x = image[None, None, :, :].astype(np.float32)
        return self.session.run(None, {"image": x})[0][0]

    def logprobs(self, image: np.ndarray, head=None) -> np.ndarray:
        w, b = head if head is not None else (self.w, self.b)
        f = self.features(image).astype(np.float32)
        return log_softmax(f @ w.T + b).astype(np.float32)


# --------------------------------------------------------------------------
# Language models (binary formats, spec/lm-format.md)
# --------------------------------------------------------------------------

class CharLM:
    def __init__(self, path=None):
        raw = Path(path or BUILD / "charlm.bin").read_bytes()
        assert raw[:4] == b"MKL1"
        self.order, = struct.unpack_from("<I", raw, 4)
        self.log_backoff, self.log_floor, self.scale = struct.unpack_from("<fff", raw, 8)
        n = struct.unpack_from(f"<{self.order}I", raw, 24)
        off = 24 + 4 * self.order
        self.child = []
        for k in range(self.order - 1):
            self.child.append(np.frombuffer(raw, "<u4", n[k] + 1, off))
            off += 4 * (n[k] + 1)
        self.lp = []
        for k in range(self.order):
            self.lp.append(np.frombuffer(raw, "<u2", n[k], off))
            off += 2 * n[k]
        self.ch = []
        for k in range(self.order):
            self.ch.append(np.frombuffer(raw, "u1", n[k], off))
            off += n[k]
        self.log_backoff = float(np.float32(self.log_backoff))
        self.log_floor = float(np.float32(self.log_floor))
        self.scale = float(np.float32(self.scale))

    def _find(self, gram) -> int:
        """Node index of `gram` (charset indices) at level len(gram), or -1."""
        lo, hi = 0, len(self.ch[0])
        node = -1
        for depth, c in enumerate(gram):
            if depth:
                lo, hi = int(self.child[depth - 1][node]), int(self.child[depth - 1][node + 1])
            arr = self.ch[depth]
            i = lo + int(np.searchsorted(arr[lo:hi], c))
            if i >= hi or arr[i] != c:
                return -1
            node = i
        return node

    def logprob(self, context, c: int) -> float:
        context = list(context)[-(self.order - 1):]
        penalty = 0.0
        for length in range(len(context), -1, -1):
            gram = context[len(context) - length:] + [c]
            node = self._find(gram)
            if node >= 0:
                q = int(self.lp[len(gram) - 1][node])
                if q != 65535:
                    return penalty - q / self.scale
            penalty += self.log_backoff
        return penalty + self.log_floor


PREFIX_LETTERS = "בהוכלמש"


class WordLM:
    def __init__(self, path=None, chars=None):
        raw = Path(path or BUILD / "words.bin").read_bytes()
        assert raw[:4] == b"MKW1"
        n, = struct.unpack_from("<I", raw, 4)
        self.oov, self.discount, self.scale = (float(np.float32(v)) for v in struct.unpack_from("<fff", raw, 8))
        off = 24
        self.offsets = np.frombuffer(raw, "<u4", n + 1, off)
        off += 4 * (n + 1)
        self.lp = np.frombuffer(raw, "<u2", n, off)
        off += 2 * n
        self.blob = raw[off:]
        self.n = n
        chars = chars or charset()
        self.prefixes = {chars.index(c) for c in PREFIX_LETTERS}

    def _get(self, word: bytes) -> float | None:
        lo, hi = 0, self.n
        while lo < hi:
            mid = (lo + hi) // 2
            w = self.blob[self.offsets[mid]: self.offsets[mid + 1]]
            if w < word:
                lo = mid + 1
            else:
                hi = mid
        if lo < self.n and self.blob[self.offsets[lo]: self.offsets[lo + 1]] == word:
            return -int(self.lp[lo]) / self.scale
        return None

    def logprob(self, word) -> float:
        if not word:
            return 0.0
        key = bytes(word)
        hit = self._get(key)
        if hit is not None:
            return hit
        if len(key) > 2 and key[0] in self.prefixes:
            rest = self._get(key[1:])
            if rest is not None:
                return rest + self.discount
        return self.oov

    def bonus(self, word) -> float:
        return max(0.0, self.logprob(word) - self.oov)


def charset() -> list[str]:
    return json.loads((BUILD / "charset.json").read_text())


# --------------------------------------------------------------------------
# CTC decoding (hebocr.decode)
# --------------------------------------------------------------------------

NEG_INF = float("-inf")


def lse(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    hi, lo = (a, b) if a > b else (b, a)
    return hi + math.log1p(math.exp(lo - hi))


def greedy(logprobs: np.ndarray) -> list[int]:
    out, prev = [], None
    for idx in logprobs.argmax(axis=-1).tolist():
        if idx != prev and idx != BLANK:
            out.append(idx)
        prev = idx
    return out


def confidence(logprobs: np.ndarray, emitting_only: bool) -> float:
    if len(logprobs) == 0:
        return 0.0
    best = logprobs.argmax(axis=-1)
    probs = np.exp(logprobs.max(axis=-1).astype(np.float64))
    if not emitting_only:
        return float(probs.mean())
    keep = [j for j in range(len(best)) if best[j] != BLANK and (j == 0 or best[j] != best[j - 1])]
    return float(probs[keep].mean()) if keep else 0.0


def topk_desc(row: np.ndarray, k: int) -> list[int]:
    """Indices of the k largest values, largest first, ties to the lower index."""
    order = np.lexsort((np.arange(len(row)), -row.astype(np.float64)))
    return order[:k].tolist()


def beam_search(logprobs: np.ndarray, lm: CharLM | None, words: WordLM | None,
                beam_width=12, top_k=8, lm_weight=0.4, length_bonus=0.6, word_weight=0.2) -> list[int]:
    """CTC prefix beam search. Prefixes are tuples of class indices.

    Mirrors hebocr.decode.beam_decode exactly, including dict insertion order
    as the tie-break of the stable sort."""
    beams: dict[tuple, list[float]] = {(): [0.0, NEG_INF]}
    for t in range(logprobs.shape[0]):
        frame = logprobs[t]
        cands = topk_desc(frame, min(top_k, frame.shape[0]))
        nxt: dict[tuple, list[float]] = {}
        for prefix, (pb, pnb) in beams.items():
            ptot = lse(pb, pnb)
            for c in cands:
                p = float(frame[c])
                if c == BLANK:
                    e = nxt.setdefault(prefix, [NEG_INF, NEG_INF])
                    e[0] = lse(e[0], ptot + p)
                    continue
                if prefix and prefix[-1] == c:
                    e = nxt.setdefault(prefix, [NEG_INF, NEG_INF])
                    e[1] = lse(e[1], pnb + p)
                    src = pb
                else:
                    src = ptot
                if src == NEG_INF:
                    continue
                ext = prefix + (c,)
                bonus = (lm_weight * lm.logprob(prefix, c) + length_bonus) if lm is not None else 0.0
                if words is not None and word_weight and c == SPACE:
                    closed = last_word(prefix)
                    if closed:
                        bonus += word_weight * words.bonus(closed)
                e = nxt.setdefault(ext, [NEG_INF, NEG_INF])
                e[1] = lse(e[1], src + p + bonus)
        ranked = sorted(nxt.items(), key=lambda kv: lse(kv[1][0], kv[1][1]), reverse=True)
        beams = dict(ranked[:beam_width]) or {(): [0.0, NEG_INF]}

    def final(kv):
        s = lse(kv[1][0], kv[1][1])
        if words is not None and word_weight:
            tail = last_word(kv[0])
            if tail:
                s += word_weight * words.bonus(tail)
        return s

    return list(max(beams.items(), key=final)[0])


def last_word(prefix) -> tuple:
    """The characters after the last space (Python's rsplit(" ", 1)[-1])."""
    for i in range(len(prefix) - 1, -1, -1):
        if prefix[i] == SPACE:
            return tuple(prefix[i + 1:])
    return tuple(prefix)


# --------------------------------------------------------------------------
# Reading lines
# --------------------------------------------------------------------------

class Reader:
    def __init__(self, model: Model | None = None, lm=True, tta=3):
        self.model = model or Model()
        self.chars = charset()
        self.lm = CharLM() if lm else None
        self.words = WordLM(chars=self.chars) if lm else None
        self.tta = tta

    def text(self, labels) -> str:
        return "".join(self.chars[i] for i in labels)

    def read_view(self, gray: np.ndarray, head=None):
        lp = self.model.logprobs(preprocess(gray), head)
        labels = beam_search(lp, self.lm, self.words, **DECODE) if self.lm else greedy(lp)
        return self.text(labels), lp

    def read_line(self, gray: np.ndarray, head=None) -> dict:
        """Best of the TTA views by emitting-frame confidence (hebocr read_tta),
        after straightening a tilted or curved line."""
        gray = straighten_line(gray)
        best_text, best_score, base_lp = "", NEG_INF, None
        for scale in TTA_SCALES[: max(1, self.tta)]:
            text, lp = self.read_view(tta_view(gray, scale), head)
            if scale == 1.0:
                base_lp = lp
            if self.tta <= 1:
                return {"text": text, "confidence": confidence(lp, False)}
            score = confidence(lp, True) if text.strip() else NEG_INF
            if score > best_score:
                best_score, best_text = score, text
        return {"text": best_text, "confidence": confidence(base_lp, False)}

    def read_page(self, rgb: np.ndarray, head=None) -> dict:
        """Crop to the paper (screenshots), turn it upright, segment, read every
        line. Boxes are in the upright page's deskewed frame (`seg["deskewed"]`)."""
        paper = find_paper(rgb)
        if paper:
            x, y, w, h = paper
            rgb = np.ascontiguousarray(rgb[y: y + h, x: x + w])
        turns, seg = orient_page(rgb, self.model)
        lines = []
        for crop, box in zip(seg["crops"], seg["boxes"]):
            r = self.read_line(to_gray(crop), head)
            lines.append({**r, "box": box})
        return {"paper": paper, "turns": turns, "angle": seg["angle"], "lines": lines, "deskewed": seg["deskewed"]}


# --------------------------------------------------------------------------
# Page segmentation (hebocr.page.segment)
# --------------------------------------------------------------------------

@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int


def _shift(a: np.ndarray, dx: int, dy: int, fill: int) -> np.ndarray:
    """out[y, x] = a[y + dy, x + dx], `fill` outside."""
    h, w = a.shape
    out = np.full_like(a, fill)
    ys, yd = (slice(dy, h), slice(0, h - dy)) if dy >= 0 else (slice(0, h + dy), slice(-dy, h))
    xs, xd = (slice(dx, w), slice(0, w - dx)) if dx >= 0 else (slice(0, w + dx), slice(-dx, w))
    if ys.stop > ys.start and xs.stop > xs.start:
        out[yd, xd] = a[ys, xs]
    return out


def _run_extreme(a: np.ndarray, length: int, op) -> np.ndarray:
    """out[y, x] = op(a[y, x : x + length]) over the in-bounds part (van Herk)."""
    h, w = a.shape
    neutral = 255 if op is np.minimum else 0
    if length == 1:
        return a.copy()
    n = -(-(w + length - 1) // length) * length
    p = np.full((h, n), neutral, dtype=a.dtype)
    p[:, :w] = a
    blocks = p.reshape(h, -1, length)
    fwd = op.accumulate(blocks, axis=2).reshape(h, n)
    bwd = op.accumulate(blocks[:, :, ::-1], axis=2)[:, :, ::-1].reshape(h, n)
    x = np.arange(w)
    return op(bwd[:, x], fwd[:, x + length - 1])


def morph(a: np.ndarray, kernel: np.ndarray, op: str) -> np.ndarray:
    """OpenCV erode/dilate with an arbitrary 0/1 kernel whose rows are single runs.

    Offsets are (col - anchor_x, row - anchor_y) for both erode and dilate, the
    anchor being (cols // 2, rows // 2); pixels outside the image never count."""
    f = np.minimum if op == "erode" else np.maximum
    neutral = 255 if op == "erode" else 0
    rows, cols = kernel.shape
    ax, ay = cols // 2, rows // 2
    # Pad with the neutral value so a window that starts outside the image
    # still sees its in-bounds part.
    h, w = a.shape
    a = np.pad(a, ((rows, rows), (cols, cols)), constant_values=neutral)
    out = np.full_like(a, neutral)
    cache = {}
    for r in range(rows):
        nz = np.flatnonzero(kernel[r])
        if not len(nz):
            continue
        c0, c1 = int(nz[0]), int(nz[-1])
        assert c1 - c0 + 1 == len(nz), "kernel rows must be single runs"
        run = c1 - c0 + 1
        if run not in cache:
            cache[run] = _run_extreme(a, run, f)
        out = f(out, _shift(cache[run], c0 - ax, r - ay, neutral))
    return out[rows: rows + h, cols: cols + w]


def morph_t(a, kernel, op):
    """The same with a kernel whose *columns* are single runs (transposed)."""
    return morph(a.T, kernel.T, op).T


def opening(a, kernel, transposed=False):
    m = morph_t if transposed else morph
    return m(m(a, kernel, "erode"), kernel, "dilate")


def _cv_line(x0, y0, x1, y1):
    """OpenCV LineIterator (connectivity 8) pixel sequence."""
    dx, dy = x1 - x0, y1 - y0
    s = -1 if dx < 0 else 0
    dx = (dx ^ s) - s
    # bt_pix steps along x, istep along y (image stepping simulated on coords)
    sy = -1 if dy < 0 else 0
    dy = (dy ^ sy) - sy
    stepx = -1 if s else 1
    stepy = -1 if sy else 1
    swap = dy > dx
    if swap:
        dx, dy = dy, dx
    err = dx - (dy + dy)
    plus_delta = dx + dx
    minus_delta = -(dy + dy)
    count = dx + 1
    x, y = x0, y0
    pts = []
    for _ in range(count):
        pts.append((x, y))
        mask = -1 if err < 0 else 0
        err += minus_delta + (plus_delta & mask)
        if swap:
            y += stepy
            if mask:
                x += stepx
        else:
            x += stepx
            if mask:
                y += stepy
    return pts


def oriented_line_kernel(length: int, angle: float) -> np.ndarray:
    drop = int(round(math.tan(math.radians(angle)) * (length - 1)))
    height = abs(drop) + 1
    k = np.zeros((height, length), np.uint8)
    y0 = 0 if drop >= 0 else height - 1
    for x, y in _cv_line(0, y0, length - 1, y0 + drop):
        k[y, x] = 1
    return k


def rect(w: int, h: int) -> np.ndarray:
    return np.ones((h, w), np.uint8)


def flatten_page(gray: np.ndarray) -> np.ndarray:
    bg = blur_large(gray, max(gray.shape) / 30.0).astype(np.float32)
    flat = gray.astype(np.float32) / np.maximum(bg, np.float32(1.0))
    return np.clip(flat * np.float32(128.0), 0, 255).astype(np.uint8)


def binarize(gray: np.ndarray, max_side=2000):
    scale = min(1.0, max_side / max(gray.shape))
    if scale < 1.0:
        h, w = gray.shape
        gray = resize(gray, int(round(w * scale)), int(round(h * scale)), "area", sx=1 / scale, sy=1 / scale)
    flat = flatten_page(gray)
    block = max(15, (min(flat.shape) // 40) | 1)
    k = gaussian_kernel(block, 0)
    mean = round_u8(convolve_sep(flat, k, k, border="replicate")).astype(np.int16)
    binary = np.where(flat.astype(np.int16) - mean <= -12, 255, 0).astype(np.uint8)
    binary = opening(binary, rect(2, 2))
    return binary, scale


def rotation_matrix(cx, cy, angle):
    a = math.radians(angle)
    al, be = math.cos(a), math.sin(a)
    return np.array([[al, be, (1 - al) * cx - be * cy], [-be, al, be * cx + (1 - al) * cy]])


def rotate_nearest(img: np.ndarray, angle: float, fill=0) -> np.ndarray:
    """warpAffine(INTER_NEAREST) about the centre; dst samples src at M^-1 (x, y)."""
    h, w = img.shape
    m = rotation_matrix(w / 2, h / 2, angle)
    inv = np.linalg.inv(np.vstack([m, [0, 0, 1]]))[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    sx = np.floor(inv[0, 0] * xs + inv[0, 1] * ys + inv[0, 2] + 0.5).astype(np.int64)
    sy = np.floor(inv[1, 0] * xs + inv[1, 1] * ys + inv[1, 2] + 0.5).astype(np.int64)
    ok = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
    out = np.full_like(img, fill)
    out[ok] = img[sy[ok], sx[ok]]
    return out


def expanded_size(w: int, h: int, angle: float) -> tuple[int, int]:
    """Canvas that holds the whole page rotated by `angle` (PIL's expand=True rule)."""
    a = math.radians(angle)
    c, s_ = math.cos(a), math.sin(a)
    xs, ys = [], []
    for x, y in ((0, 0), (w, 0), (w, h), (0, h)):
        dx, dy = x - w / 2, y - h / 2
        xs.append(c * dx + s_ * dy)
        ys.append(-s_ * dx + c * dy)
    return math.ceil(max(xs)) - math.floor(min(xs)), math.ceil(max(ys)) - math.floor(min(ys))


def rotate_nearest_expand(img: np.ndarray, angle: float, fill=0) -> np.ndarray:
    """Nearest-neighbour rotation onto the expanded canvas, centre to centre."""
    h, w = img.shape
    W, H = expanded_size(w, h, angle)
    a = math.radians(angle)
    c, s_ = math.cos(a), math.sin(a)
    ys, xs = np.mgrid[0:H, 0:W]
    dx, dy = xs - W / 2, ys - H / 2
    # inverse of [[c, s], [-s, c]] is [[c, -s], [s, c]]
    sx = np.floor(c * dx - s_ * dy + w / 2 + 0.5).astype(np.int64)
    sy = np.floor(s_ * dx + c * dy + h / 2 + 0.5).astype(np.int64)
    ok = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
    out = np.full((H, W), fill, dtype=img.dtype)
    out[ok] = img[sy[ok], sx[ok]]
    return out


def stripe_score(small: np.ndarray, angle: float) -> float:
    """Row-profile sharpness at `angle`, per unit width (comparable across orientations)."""
    prof = rotate_nearest(small, angle).sum(axis=1, dtype=np.float64) / (255.0 * small.shape[1])
    return float(np.var(np.diff(prof)))


WIDE_LIMIT = 44


def estimate_skew_wide(binary: np.ndarray) -> tuple[float, float]:
    """(angle, stripe score). A coarse search over +-44 degrees at quarter resolution;
    within +-6 the original fine search decides (so ordinary pages are unchanged),
    beyond it a 0.5-degree search around the coarse answer."""
    h, w = binary.shape
    q = resize(binary, max(1, int(round(w / 4))), max(1, int(round(h / 4))), "area", sx=4.0, sy=4.0)
    best_c, best, worst = 0, -1.0, float("inf")
    for i in range(WIDE_LIMIT + 1):
        a = -WIDE_LIMIT + 2 * i
        sc = stripe_score(q, a)
        worst = min(worst, sc)
        if sc > best:
            best_c, best = a, sc
    half = resize(binary, int(round(w * 0.5)), int(round(h * 0.5)), "area", sx=2.0, sy=2.0)
    if best <= worst:
        # every angle scores the same (a page without ink): leave it level
        return 0.0, stripe_score(half, 0.0)
    if abs(best_c) <= 6:
        angle = estimate_skew(binary)
    else:
        angle, best = best_c - 2.0, -1.0
        for i in range(9):
            a = best_c - 2 + 0.5 * i
            sc = stripe_score(half, a)
            if sc > best:
                angle, best = a, sc
    return angle, stripe_score(half, angle)


def estimate_skew(binary: np.ndarray, limit=6.0, step=0.5) -> float:
    h, w = binary.shape
    sw, sh = int(round(w * 0.5)), int(round(h * 0.5))
    small = resize(binary, sw, sh, "area", sx=2.0, sy=2.0)
    best_angle, best = 0.0, -1.0
    for i in range(int(round(2 * limit / step)) + 1):
        angle = -limit + i * step
        prof = rotate_nearest(small, angle).sum(axis=1, dtype=np.float64)
        score = float(np.var(np.diff(prof)))
        if score > best:
            best_angle, best = angle, score
    return best_angle


def estimate_line_pitch(binary: np.ndarray) -> float:
    prof = binary.sum(axis=1, dtype=np.float64)
    prof = prof - prof.mean()
    if not np.any(prof):
        return 30.0
    n = len(prof)
    lo = max(8, int(n * 0.004))
    # hebocr searched up to 0.15 n, which assumes a page of many lines; a photo
    # of a few big lines (pitch 18% of the height and more) then fell back to a
    # stroke-texture lag and shattered every line. Half the height still needs
    # two lines to see a rhythm, and the first-peak rule leaves every page the
    # old window handled unchanged (checked on the 10 benchmark pages).
    hi = min(n - 1, n // 2)
    if hi <= lo + 2:
        return 30.0
    win = np.array([float(np.dot(prof[: n - lag], prof[lag:])) for lag in range(lo, hi)])
    peaks = [i for i in range(1, len(win) - 1) if win[i] > win[i - 1] and win[i] >= win[i + 1] and win[i] > 0]
    return float(lo + (peaks[0] if peaks else int(np.argmax(win))))


ANGLES = (-2.0, -1.0, -0.4, 0.0, 0.4, 1.0, 2.0)


def remove_rules(binary: np.ndarray, pitch: float) -> np.ndarray:
    length = max(40, int(pitch * 2.0))
    thickness = max(3, int(pitch * 0.28))
    horiz = np.zeros_like(binary)
    vert = np.zeros_like(binary)
    for angle in ANGLES:
        k = oriented_line_kernel(length, angle)
        horiz = np.maximum(horiz, opening(binary, k))
        vert = np.maximum(vert, opening(binary, np.ascontiguousarray(k.T), transposed=True))
    out = binary
    for runs, thin in ((horiz, rect(1, thickness)), (vert, rect(thickness, 1))):
        thick = opening(runs, thin, transposed=thin.shape[0] > 1)
        rules = np.clip(runs.astype(np.int16) - thick, 0, 255).astype(np.uint8)
        out = np.clip(out.astype(np.int16) - morph(rules, rect(3, 3), "dilate"), 0, 255).astype(np.uint8)
    return out


def find_lines(binary: np.ndarray, pitch: float) -> list[Box]:
    from scipy import ndimage

    sx = max(8, int(pitch * 0.55))
    sy = max(1, int(pitch * 0.05))
    k = rect(sx, sy)
    sm = morph_t(morph(binary, rect(sx, 1), "dilate"), rect(1, sy), "dilate") if sy > 1 else morph(binary, k, "dilate")
    sm = morph(morph(sm, rect(sx, 1), "dilate"), rect(sx, 1), "erode")
    labels, n = ndimage.label(sm > 0, structure=np.ones((3, 3)))
    if n == 0:
        return []
    areas = np.bincount(labels.ravel())
    boxes = []
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        y, x = sl[0].start, sl[1].start
        h, w = sl[0].stop - y, sl[1].stop - x
        if w < pitch * 0.8 or h < pitch * 0.25 or areas[i] < pitch * pitch * 0.12:
            continue
        # hebocr dropped every component taller than 12 pitches as a box
        # outline. On dense notes (graph paper, close lines) whole paragraphs
        # smear into one tall component and vanished with it, so only drop
        # tall components that are mostly empty, which outlines are; dense
        # ones go on to split_tall. Same result on all 10 benchmark pages.
        if h > pitch * 12.0 and areas[i] / (w * h) < 0.35:
            continue
        boxes.append(Box(int(x), int(y), int(w), int(h)))
    return boxes


def split_tall(binary: np.ndarray, boxes: list[Box], pitch: float) -> list[Box]:
    out = []
    k5 = gaussian_kernel(5, 0)
    for b in boxes:
        n = int(round(b.h / pitch))
        if n <= 1:
            out.append(b)
            continue
        roi = binary[b.y: b.y + b.h, b.x: b.x + b.w]
        prof = roi.sum(axis=1, dtype=np.float64)
        smooth = convolve_sep(prof.reshape(-1, 1), np.array([1.0]), k5)[:, 0]
        cuts = [0]
        for k in range(1, n):
            centre = int(k * b.h / n)
            lo = max(cuts[-1] + 2, centre - int(pitch * 0.35))
            hi = min(b.h - 2, centre + int(pitch * 0.35))
            if hi > lo:
                cuts.append(lo + int(np.argmin(smooth[lo:hi])))
        cuts.append(b.h)
        for top, bottom in zip(cuts, cuts[1:]):
            band = roi[top:bottom]
            rows = np.flatnonzero(band.sum(axis=1) > 0)
            cols = np.flatnonzero(band.sum(axis=0) > 0)
            if not len(rows) or not len(cols):
                continue
            out.append(Box(b.x + int(cols[0]), b.y + top + int(rows[0]),
                           int(cols[-1] - cols[0] + 1), int(rows[-1] - rows[0] + 1)))
    return out


def merge_overlapping(boxes: list[Box], pitch: float) -> list[Box]:
    merged: list[Box] = []
    for b in sorted(boxes, key=lambda b: (b.y, b.x)):
        for i, o in enumerate(merged):
            v = min(b.y + b.h, o.y + o.h) - max(b.y, o.y)
            gap = max(b.x, o.x) - min(b.x + b.w, o.x + o.w)
            if v > min(b.h, o.h) * 0.5 and gap < pitch * 1.2:
                x0, y0 = min(b.x, o.x), min(b.y, o.y)
                x1, y1 = max(b.x + b.w, o.x + o.w), max(b.y + b.h, o.y + o.h)
                merged[i] = Box(x0, y0, x1 - x0, y1 - y0)
                break
        else:
            merged.append(b)
    return merged


def reading_order(boxes: list[Box], pitch: float) -> list[Box]:
    if not boxes:
        return []
    ordered = sorted(boxes, key=lambda b: b.y + b.h / 2)
    rows = [[ordered[0]]]
    for b in ordered[1:]:
        c = b.y + b.h / 2
        rc = sum(o.y + o.h / 2 for o in rows[-1]) / len(rows[-1])
        if abs(c - rc) < pitch * 0.45:
            rows[-1].append(b)
        else:
            rows.append([b])
    return [b for row in rows for b in sorted(row, key=lambda b: -(b.x + b.w))]


def rotate_rgb(rgb: np.ndarray, angle: float, expand: bool = False) -> np.ndarray:
    """Rotate about the centre with bicubic sampling, white outside.

    The apps use their platform's bicubic/bilinear rotation; crops only feed
    the recognizer, whose own preprocessing absorbs the difference."""
    from PIL import Image

    im = Image.fromarray(rgb)
    if expand:
        # same canvas as rotate_nearest_expand, scaled to this image
        W, H = expanded_size(im.width, im.height, angle)
        out = im.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
        if out.size != (W, H):
            out = out.crop(((out.width - W) // 2, (out.height - H) // 2, (out.width - W) // 2 + W, (out.height - H) // 2 + H))
        return np.asarray(out)
    return np.asarray(im.rotate(angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255)))


def crop_box(img: np.ndarray, b: Box, pad=0.12) -> np.ndarray:
    py, px = int(b.h * pad), int(b.h * 0.05)
    h, w = img.shape[:2]
    return img[max(b.y - py, 0): min(b.y + b.h + py, h), max(b.x - px, 0): min(b.x + b.w + px, w)]


def segment_page(rgb: np.ndarray, max_side=2000) -> dict:
    """Page -> {"angle", "boxes" (deskewed page coords), "crops" (RGB)}."""
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    binary, scale = binarize(to_gray(rgb), max_side)
    angle, _ = estimate_skew_wide(binary)
    if abs(angle) > 6:
        # a strongly tilted photo: rotate onto a canvas big enough to keep the corners
        binary = rotate_nearest_expand(binary, angle)
        work = rotate_rgb(rgb, angle, expand=True)
    elif abs(angle) > 1e-3:
        binary = rotate_nearest(binary, angle)
        work = rotate_rgb(rgb, angle)
    else:
        work = rgb
    pitch = estimate_line_pitch(binary)
    binary = remove_rules(binary, pitch)
    pitch = estimate_line_pitch(binary)
    boxes = find_lines(binary, pitch)
    boxes = split_tall(binary, boxes, pitch)
    boxes = merge_overlapping(boxes, pitch)
    boxes = reading_order(boxes, pitch)
    inv = 1.0 / scale
    full = [Box(int(b.x * inv), int(b.y * inv), int(b.w * inv), int(b.h * inv)) for b in boxes]
    return {
        "angle": angle,
        "pitch": pitch,
        "boxes": [{"x": b.x, "y": b.y, "w": b.w, "h": b.h} for b in full],
        "crops": [crop_box(work, b) for b in full],
        "deskewed": work,
    }


# --------------------------------------------------------------------------
# Orientation: sideways or upside-down photos (spec/orientation.md)
# --------------------------------------------------------------------------

def otsu_threshold(values: np.ndarray) -> int:
    """Otsu's threshold over uint8 values (first maximum of the between-class variance)."""
    hist = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    p = hist / max(hist.sum(), 1.0)
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mu[-1] * omega - mu) ** 2 / (omega * (1 - omega))
    return int(np.argmax(np.nan_to_num(sb)))


def find_paper(rgb: np.ndarray, work: int = 256):
    """(x, y, w, h) of the paper when it sits inside near-black surroundings (a
    screenshot's app bars, a photo on a black screen), else None.

    At 256 px: Otsu-split brightness, close gaps (ink, ruling) with a square of
    3% of the short side, open with 3x3, keep the largest bright component. Crop
    only when that component is at least 20% of the image, its box is under 92%
    of it, and everything outside the box averages below 25 and covers at least a
    quarter of the image. The crop sits 1.5% inside the box, so no strip of the
    dark surround survives to look like a ruled edge."""
    from scipy import ndimage

    g = to_gray(rgb)
    H, W = g.shape
    s = work / max(H, W)
    w, h = max(1, int(round(W * s))), max(1, int(round(H * s)))
    small = resize(g, w, h, "area", sx=W / w, sy=H / h)
    bright = small > otsu_threshold(small)
    k = max(3, int(round(min(w, h) * 0.03))) | 1
    bright = ndimage.binary_closing(bright, structure=np.ones((k, k)), border_value=0)
    bright = ndimage.binary_opening(bright, structure=np.ones((3, 3)))
    lab, n = ndimage.label(bright)
    if n == 0:
        return None
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    i = int(np.argmax(sizes))
    if sizes[i] < 0.2 * w * h:
        return None
    sl = ndimage.find_objects(lab)[i - 1]
    y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
    if (y1 - y0) * (x1 - x0) > 0.92 * w * h:
        return None
    outside = np.ones(small.shape, bool)
    outside[y0:y1, x0:x1] = False
    if float(small[outside].mean()) > 25 or float(outside.mean()) < 0.25:
        return None
    m = 0.015
    X0, Y0 = max(0, int((x0 / w + m) * W)), max(0, int((y0 / h + m) * H))
    X1, Y1 = min(W, int((x1 / w - m) * W)), min(H, int((y1 / h - m) * H))
    if X1 - X0 < 16 or Y1 - Y0 < 16:
        return None
    return X0, Y0, X1 - X0, Y1 - Y0


def rot90(img: np.ndarray, k: int) -> np.ndarray:
    """k quarter turns counter-clockwise (as displayed)."""
    return np.ascontiguousarray(np.rot90(img, k % 4))


VOTE_LINES = 6       # widest crops read for the orientation vote
VOTE_WINDOW = 10.0   # read only the middle of each, this many crop-heights wide


def orientation_votes(model: "Model", seg: dict, flip: bool) -> float:
    """Mean emitting confidence of the 6 widest crops, read greedily at one scale,
    each cut to its middle window (10 crop heights wide, about 640 px at the
    model's 64 px): the vote needs letters, not whole lines, and phones without
    a GPU paid minutes for full lines."""
    crops = sorted(seg["crops"], key=lambda c: -c.shape[1])[:VOTE_LINES]
    if not crops:
        return 0.0
    vals = []
    for c in crops:
        h, w = c.shape[:2]
        win = int(VOTE_WINDOW * h)
        if w > win:
            x0 = (w - win) // 2
            c = c[:, x0: x0 + win]
        g = to_gray(rot90(c, 2) if flip else c)
        vals.append(confidence(model.logprobs(preprocess(g)), True))
    return float(np.mean(vals))


def orient_page(rgb: np.ndarray, model: "Model", max_side=2000) -> tuple[int, dict]:
    """(quarter turns ccw that make the page upright, its segmentation).

    Sideways: text rows make a sharply striped row profile, so compare the best
    stripe score upright and turned 90 degrees; when they are within 3x of each
    other (graph paper, sparse pages) let the model decide between both.
    Upside down: read the middles of the widest lines both ways, keep the more confident."""
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    binary, _ = binarize(to_gray(rgb), max_side)
    _, s0 = estimate_skew_wide(binary)
    _, s1 = estimate_skew_wide(rot90(binary, 1))
    if max(s0, s1) >= 3 * min(s0, s1):
        bases = [0 if s0 >= s1 else 1]
    else:
        bases = [0, 1]
    best = None
    for base in bases:
        seg = segment_page(rot90(rgb, base), max_side)
        for flip in (False, True):
            c = orientation_votes(model, seg, flip)
            if best is None or c > best[0]:
                best = (c, base, flip, seg)
    _, base, flip, seg = best
    turns = (base + (2 if flip else 0)) % 4
    if flip:
        seg = segment_page(rot90(rgb, turns), max_side)
    return turns, seg


# --------------------------------------------------------------------------

def load_rgb(path) -> np.ndarray:
    from PIL import Image, ImageOps

    im = ImageOps.exif_transpose(Image.open(path))
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        white = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(white, im)
    return np.asarray(im.convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read")
    r.add_argument("image")
    r.add_argument("--page", action="store_true")
    r.add_argument("--head")
    r.add_argument("--tta", type=int, default=3)
    r.add_argument("--greedy", action="store_true")
    args = ap.parse_args()

    reader = Reader(Model(head=args.head), lm=not args.greedy, tta=args.tta)
    rgb = load_rgb(args.image)
    if args.page:
        res = reader.read_page(rgb)
        print(f"skew {res['angle']}")
        for line in res["lines"]:
            print(line["box"], f"{line['confidence']:.3f}", line["text"])
    else:
        print(reader.read_line(to_gray(rgb))["text"])


if __name__ == "__main__":
    main()

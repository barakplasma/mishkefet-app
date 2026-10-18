"""reference.py against hebocr / OpenCV, primitive by primitive and end to end.

    .venv/bin/python -m pytest -q test_reference.py

The reference never imports OpenCV; these tests are where OpenCV appears, as
the thing being matched.
"""

import json
import math
import pickle
import random

import cv2
import numpy as np
import pytest
from PIL import Image

import reference as R
from common import HF_DIR, SPEC, hebocr

hebocr()
from hebocr.data import transforms as T  # noqa: E402
from hebocr.page import segment as S  # noqa: E402

FIX = json.loads((SPEC / "fixtures/fixtures.json").read_text())
rng = np.random.default_rng(0)


def fixture_gray(f):
    return np.asarray(Image.open(SPEC / "fixtures" / f).convert("L"))


# -- primitives -------------------------------------------------------------

@pytest.mark.parametrize("length", list(range(40, 260, 7)))
def test_line_kernel_matches_opencv(length):
    for angle in R.ANGLES:
        assert np.array_equal(R.oriented_line_kernel(length, angle), S._oriented_line_kernel(length, angle))


def random_binary(h=120, w=300, p=0.3):
    return np.where(rng.random((h, w)) < p, 255, 0).astype(np.uint8)


@pytest.mark.parametrize("kernel", [
    np.ones((2, 2), np.uint8), np.ones((3, 3), np.uint8), np.ones((1, 9), np.uint8),
    S._oriented_line_kernel(40, 2.0), S._oriented_line_kernel(41, -1.0), S._oriented_line_kernel(57, 0.4),
])
def test_morphology_matches_opencv(kernel):
    a = random_binary(p=0.7)
    assert np.array_equal(R.morph(a, kernel, "erode"), cv2.erode(a, kernel))
    assert np.array_equal(R.morph(a, kernel, "dilate"), cv2.dilate(a, kernel))
    kt = np.ascontiguousarray(kernel.T)
    assert np.array_equal(R.morph_t(a, kt, "erode"), cv2.erode(a, kt))
    assert np.array_equal(R.morph_t(a, kt, "dilate"), cv2.dilate(a, kt))
    assert np.array_equal(R.opening(a, kernel), cv2.morphologyEx(a, cv2.MORPH_OPEN, kernel))


@pytest.mark.parametrize("shape,new", [((97, 803), (64, 530)), ((40, 300), (64, 480)), ((1500, 1100), (750, 550))])
def test_resize_close_to_opencv(shape, new):
    img = rng.integers(0, 256, shape).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 2)
    mode = "area" if new[0] < shape[0] else "linear"
    ours = R.resize(img, new[1], new[0], mode)
    theirs = cv2.resize(img, (new[1], new[0]), interpolation=cv2.INTER_AREA if mode == "area" else cv2.INTER_LINEAR)
    assert np.abs(ours.astype(int) - theirs).max() <= 1


@pytest.mark.parametrize("sigma", [3.0, 7.5, 12.4])
def test_gaussian_close_to_opencv(sigma):
    img = rng.integers(0, 256, (90, 400)).astype(np.uint8)
    ours = R.blur_large(img, sigma)
    theirs = cv2.GaussianBlur(img, (0, 0), sigma)
    # OpenCV's 8-bit fixed-point kernel differs by up to 2 levels at corners
    assert np.abs(ours.astype(int) - theirs).max() <= 2


def test_preprocess_close_to_hebocr():
    for line in FIX["lines"]:
        g = fixture_gray(line["file"])
        ours, theirs = R.preprocess(g), T.preprocess(Image.fromarray(g))[0]
        assert ours.shape == theirs.shape
        # blur differs by a level or two, which the contrast stretch amplifies
        assert np.abs(ours - theirs).max() <= 5 / 255 + 1e-6, line["file"]


# -- language models ----------------------------------------------------------

def test_char_lm_matches_pickle():
    from hebocr.lm import CharNGramLM

    py = CharNGramLM.load(HF_DIR / "hebrew_char6.pkl")
    ours = R.CharLM()
    chars = R.charset()
    random.seed(1)
    texts = [l["truth"] for l in FIX["lines"]]
    for _ in range(3000):
        t = random.choice(texts)
        i = random.randrange(len(t))
        ctx = t[max(0, i - random.randint(0, 7)): i]
        if any(c not in chars for c in ctx + t[i]):
            continue
        a = py.logprob(ctx, t[i])
        b = ours.logprob([chars.index(c) for c in ctx], chars.index(t[i]))
        assert abs(a - b) < 2e-4, (ctx, t[i], a, b)
    for c in range(1, len(chars)):
        assert abs(py.logprob("של", chars[c]) - ours.logprob([chars.index("ש"), chars.index("ל")], c)) < 2e-4


def test_word_lm_matches_pickle():
    from hebocr.wordlm import WordUnigramLM

    py = WordUnigramLM.load(HF_DIR / "hebrew_words.pkl")
    ours = R.WordLM()
    chars = R.charset()
    words = " ".join(l["truth"] for l in FIX["lines"]).split() + ["ושלום", "בבית", "הספר", "xyz", "שלום,"]
    for w in words:
        if any(c not in chars for c in w):
            continue
        assert abs(py.bonus(w) - ours.bonus([chars.index(c) for c in w])) < 2e-4, w


# -- decoding -----------------------------------------------------------------

def test_beam_matches_hebocr():
    import torch
    from hebocr.charset import Charset
    from hebocr.decode import beam_decode
    from hebocr.lm import CharNGramLM
    from hebocr.wordlm import WordUnigramLM

    chars = R.charset()
    cs = Charset(chars=chars)
    lm, wl = CharNGramLM.load(HF_DIR / "hebrew_char6.pkl"), WordUnigramLM.load(HF_DIR / "hebrew_words.pkl")
    model = R.Model()
    ours_lm, ours_wl = R.CharLM(), R.WordLM()
    for line in FIX["lines"][:6]:
        lp = model.logprobs(R.preprocess(fixture_gray(line["file"])))
        theirs = beam_decode(torch.from_numpy(lp)[None], torch.tensor([lp.shape[0]]), cs, lm=lm, word_lm=wl,
                             beam_width=12, lm_weight=0.4, length_bonus=0.6, word_weight=0.2)[0]
        ours = "".join(chars[i] for i in R.beam_search(lp, ours_lm, ours_wl, **R.DECODE))
        assert ours == theirs


# -- end to end ---------------------------------------------------------------

def cer(a, b):
    from rapidfuzz.distance import Levenshtein

    return Levenshtein.distance(a, b) / max(1, len(b))


def test_lines_match_hebocr():
    from hebocr.lm import CharNGramLM
    from hebocr.recognize import Recognizer
    from hebocr.wordlm import WordUnigramLM

    rec = Recognizer(HF_DIR / "mishkefet-v1.pt", device="cpu", lm=CharNGramLM.load(HF_DIR / "hebrew_char6.pkl"),
                     lm_weight=0.4, beam_width=12, tta=3, word_lm=WordUnigramLM.load(HF_DIR / "hebrew_words.pkl"),
                     word_weight=0.2)
    reader = R.Reader()
    # hebocr does not straighten lines; compare the shared pipeline without it
    # (straightening alone takes these lines' CER against the truth from 0.268 to 0.241)
    R.STRAIGHTEN_MIN_DEV, saved = 1e9, R.STRAIGHTEN_MIN_DEV
    total = 0.0
    for line in FIX["lines"]:
        g = fixture_gray(line["file"])
        theirs = rec.read_tta([Image.fromarray(g).convert("RGB")])[0]
        ours = reader.read_line(g)["text"]
        total += cer(ours, theirs)
    R.STRAIGHTEN_MIN_DEV = saved
    # Not exact: our Gaussian differs from OpenCV's 8-bit one by a level here and there,
    # which flips readings on these uncertain early-modern lines (about 0.027 CER on
    # average). Equivalence is measured where it counts: on the 225 benchmark lines both
    # give median CER 0.1875 and mean 0.263 (eval_bench.py).
    assert total / len(FIX["lines"]) < 0.03


def iou(a, b):
    x0, y0 = max(a["x"], b["x"]), max(a["y"], b["y"])
    x1, y1 = min(a["x"] + a["w"], b["x"] + b["w"]), min(a["y"] + a["h"], b["y"] + b["h"])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / (a["w"] * a["h"] + b["w"] * b["h"] - inter)


def test_segmentation_matches_hebocr():
    # The rendered pages exercise where we deliberately differ from hebocr (spec/segmenter.md,
    # orientation.md), and so does page2: hebocr dropped a dense block of it as a "box outline";
    # keeping it doubles line recall against the Pinkas annotations (0.303 to 0.636).
    for page in [p for p in FIX["pages"] if p["file"].endswith("page1.jpg")]:
        im = Image.open(SPEC / "fixtures" / page["file"]).convert("RGB")
        _, boxes, angle = S.segment_page(im)
        theirs = [{"x": b.x, "y": b.y, "w": b.w, "h": b.h} for b in boxes]
        ours = R.segment_page(np.asarray(im))
        assert ours["angle"] == angle
        matched = sum(max((iou(o, t) for t in theirs), default=0) >= 0.9 for o in ours["boxes"])
        assert matched >= 0.9 * len(theirs), (matched, len(theirs), len(ours["boxes"]))

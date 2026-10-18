"""Regenerate spec/golden/: the vectors the web and iOS engines are tested against.

    python golden.py

Everything is produced by reference.py / adapter.py / strokes.py with the
shipped artifacts (fp16 backbone), so an app that reproduces these numbers
reproduces the reference pipeline.

Layout (all little endian):
  preprocess/<line>.u8     preprocess output, round(x * 255), shape in lines.json
  logprobs/<line>.f32      log-probabilities (T, 198) from the fp16 backbone
  pages/<page>.binary.png  binarize() output, for debugging a segmenter port
  lines.json               per fixture line: shape, greedy, beam (tta 1), read (tta 3), confidence
  pages.json               per fixture page: angle, pitch, boxes, line texts
  lm.json                  char LM log-probs and word bonuses at sample points
  ctc.json                 CTC loss + gradient on small fixed tensors
  adapter.json             a short deterministic training run (losses, weight checksums)
  strokes.json             stroke sets, their mulberry32 augmentation params, render checksums
  strokes/<name>.png       the plain renders
"""

import json
import math
import shutil

import numpy as np
from PIL import Image

import adapter as AD
import reference as R
import strokes as ST
import synth_strokes as SS
from common import BUILD, SPEC

OUT = SPEC / "golden"
FIX = json.loads((SPEC / "fixtures/fixtures.json").read_text())
BACKBONE = BUILD / "backbone.fp16.onnx"


def stem(f):
    return f.split("/")[-1].rsplit(".", 1)[0]


def gray(f):
    return R.to_gray(np.asarray(Image.open(SPEC / "fixtures" / f).convert("RGB")))


def lines(model, reader1, reader3, chars):
    out = []
    for line in FIX["lines"] + FIX.get("curved", []):
        g = gray(line["file"])
        x = R.preprocess(g)
        (OUT / "preprocess" / f"{stem(line['file'])}.u8").write_bytes(
            np.floor(x * 255 + 0.5).astype(np.uint8).tobytes())
        lp = model.logprobs(x)
        (OUT / "logprobs" / f"{stem(line['file'])}.f32").write_bytes(lp.astype("<f4").tobytes())
        greedy = "".join(chars[i] for i in R.greedy(lp))
        beam = "".join(chars[i] for i in R.beam_search(lp, reader1.lm, reader1.words, **R.DECODE))
        r3 = reader3.read_line(g)
        out.append({
            "file": line["file"], "height": int(x.shape[0]), "width": int(x.shape[1]), "frames": int(lp.shape[0]),
            "greedy": greedy, "beam": beam, "read": r3["text"], "confidence": r3["confidence"],
            "emitting_confidence": R.confidence(lp, True),
        })
    return out


def pages(reader3):
    """Per page: binarize output, the plain segmentation (no crop or turning), and the
    full read (paper crop, quarter turns, wide deskew, straightening)."""
    out = []
    for page in FIX["pages"]:
        rgb = np.asarray(Image.open(SPEC / "fixtures" / page["file"]).convert("RGB"))
        binary, scale = R.binarize(R.to_gray(rgb))
        Image.fromarray(binary).save(OUT / "pages" / f"{stem(page['file'])}.binary.png", optimize=True)
        seg = R.segment_page(rgb)
        res = reader3.read_page(rgb)
        out.append({
            "file": page["file"], "scale": scale, "angle": seg["angle"], "pitch": seg["pitch"], "boxes": seg["boxes"],
            "read": {
                "paper": list(res["paper"]) if res["paper"] else None, "turns": res["turns"], "angle": res["angle"],
                "deskewed": [int(res["deskewed"].shape[1]), int(res["deskewed"].shape[0])],
                "boxes": [l["box"] for l in res["lines"]], "lines": [l["text"] for l in res["lines"]],
            },
        })
    return out


def straighten_vectors():
    """straighten_line on the fixture lines (mostly unchanged) and the curved ones."""
    out = []
    for line in FIX["lines"] + FIX.get("curved", []):
        g = gray(line["file"])
        s = R.straighten_line(g)
        out.append({"file": line["file"], "in": [int(g.shape[1]), int(g.shape[0])], "out": [int(s.shape[1]), int(s.shape[0])],
                    "changed": s is not g, "sum": int(s.astype(np.int64).sum())})
        if s is not g:
            Image.fromarray(s).save(OUT / "straighten" / f"{stem(line['file'])}.png", optimize=True)
    return out


def lm_vectors(reader, chars):
    rng = np.random.default_rng(3)
    texts = [l["truth"] for l in FIX["lines"]]
    char_cases = []
    for _ in range(400):
        t = texts[rng.integers(len(texts))]
        i = int(rng.integers(len(t)))
        ctx = t[max(0, i - int(rng.integers(0, 8))): i]
        if any(c not in chars for c in ctx + t[i]):
            continue
        ci = [chars.index(c) for c in ctx]
        char_cases.append({"context": ci, "char": chars.index(t[i]), "logprob": reader.lm.logprob(ci, chars.index(t[i]))})
    for c in range(1, len(chars)):
        ci = [chars.index("ש"), chars.index("ל")]
        char_cases.append({"context": ci, "char": c, "logprob": reader.lm.logprob(ci, c)})
    words = sorted(set(" ".join(texts).split()) | {"ושלום", "בבית", "הספר", "שלום,", "ב", "של"})
    word_cases = []
    for w in words:
        if any(c not in chars for c in w):
            continue
        wi = [chars.index(c) for c in w]
        word_cases.append({"word": wi, "text": w, "bonus": reader.words.bonus(wi)})
    return {"char": char_cases, "word": word_cases}


def ctc_vectors():
    rng = np.random.default_rng(5)
    cases = []
    for T, C, label in [(12, 6, [1, 2, 2, 3]), (30, 10, [4, 4, 4]), (40, 198, [12, 5, 38, 7, 7, 1]), (3, 5, [1, 1, 1]), (1, 4, [2])]:
        z = (rng.standard_normal((T, C)) * 2).astype(np.float32)
        loss, g = AD.ctc_loss_grad(z, label)
        cases.append({"T": T, "C": C, "label": label, "logits": z.ravel().tolist(), "loss": loss, "grad": g.ravel().tolist()})
    return cases


def adapter_vectors():
    rng = np.random.default_rng(7)
    D, C = 16, 7
    W0 = (rng.standard_normal((C, D)) * 0.3).astype(np.float32)
    b0 = (rng.standard_normal(C) * 0.1).astype(np.float32)
    samples = []
    for T, label in [(10, [1, 2]), (14, [3, 3, 4]), (8, [5])]:
        samples.append(((rng.standard_normal((T, D))).astype(np.float32), label))
    runs = []
    for rank in (0, 3):
        cfg = AD.Config(rank=rank, steps=6, lr=0.01, l2=0.05, l2_adapter=0.01, seed=11)
        losses = []
        head = AD.init_head(W0, b0, cfg)
        A0 = None if head.A is None else head.A.tolist()
        head = AD.train(W0, b0, samples, cfg, on_step=lambda s, l: losses.append(l))
        runs.append({
            "rank": rank, "steps": cfg.steps, "lr": cfg.lr, "l2": cfg.l2, "l2_adapter": cfg.l2_adapter,
            "A0": A0, "losses": losses,
            "W": head.W.ravel().tolist(), "b": head.b.ravel().tolist(),
            "A": None if head.A is None else head.A.ravel().tolist(),
            "B": None if head.B is None else head.B.ravel().tolist(),
        })
    return {"D": D, "C": C, "W0": W0.ravel().tolist(), "b0": b0.ravel().tolist(),
            "samples": [{"features": f.ravel().tolist(), "T": len(f), "label": l} for f, l in samples],
            "runs": runs}


def stroke_vectors(reader1, chars):
    out = []
    words = ["שלום", "מכתב", "ירושלים", "צבע", "ים"]
    for i, w in enumerate(words):
        st = SS.word_strokes(w, i % 2, size=110, scale=0.9)
        st = [[(round(x, 2), round(y, 2)) for x, y in s] for s in st]
        entry = {"word": w, "strokes": st, "renders": []}
        for seed in (0, 1, 2):
            p = ST.augment_params(seed)
            img = ST.render(st, len(w), p)
            if seed == 0:
                Image.fromarray(img).save(OUT / "strokes" / f"{i}.png", optimize=True)
            entry["renders"].append({
                "seed": seed, "params": p, "height": int(img.shape[0]), "width": int(img.shape[1]),
                "sum": int(img.astype(np.int64).sum()), "ink_pixels": int((img < 128).sum()),
            })
        entry["read_greedy"] = "".join(chars[k] for k in R.greedy(reader1.model.logprobs(R.preprocess(ST.render(st, len(w))))))
        out.append(entry)
    rng_vals = [ST.mulberry32(42)() for _ in range(1)]
    g = ST.mulberry32(42)
    return {"mulberry32_seed42": [g() for _ in range(8)], "sets": out, "_": rng_vals}


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for d in ("preprocess", "logprobs", "pages", "strokes", "straighten"):
        (OUT / d).mkdir(parents=True)
    model = R.Model(backbone=BACKBONE)
    reader1 = R.Reader(model, tta=1)
    reader3 = R.Reader(model, tta=3)
    chars = R.charset()
    write = lambda name, obj: (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    write("lines.json", lines(model, reader1, reader3, chars))
    write("pages.json", pages(reader3))
    write("straighten.json", straighten_vectors())
    write("lm.json", lm_vectors(reader1, chars))
    write("ctc.json", ctc_vectors())
    write("adapter.json", adapter_vectors())
    write("strokes.json", stroke_vectors(reader1, chars))
    shutil.copy(BUILD / "charset.json", SPEC / "charset.json")
    print("wrote", OUT)


if __name__ == "__main__":
    main()

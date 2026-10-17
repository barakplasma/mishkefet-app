"""Measure the reference pipeline on the ivrit.ai benchmark (local only).

The benchmark is test-only and licensed by ivrit.ai, so nothing from it is
committed; results land in ~/.cache/mishkefet/eval/.

    python eval_bench.py seg                      # reference vs hebocr line boxes on the 10 pages
    python eval_bench.py lines --impl ref         # median line CER, reference pipeline
    python eval_bench.py lines --impl hebocr      # the same with hebocr itself
    python eval_bench.py lines --impl ref --backbone ~/.cache/mishkefet/build/backbone.fp16.onnx
"""

import argparse
import io
import json
import statistics
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageOps

import reference as R
from common import HF_DIR, hebocr

BENCH = Path.home() / ".cache/mishkefet/data/bench/data"
OUT = Path.home() / ".cache/mishkefet/eval"


def rows(kind):
    t = pq.read_table(BENCH / kind / "test-00000-of-00001.parquet").to_pylist()
    for r in t:
        im = Image.open(io.BytesIO(r["image"]["bytes"]))
        r["pil"] = ImageOps.exif_transpose(im).convert("RGB")
        yield r


def cer(hyp, ref):
    from rapidfuzz.distance import Levenshtein

    return Levenshtein.distance(hyp, ref) / max(1, len(ref))


def iou(a, b):
    x0, y0 = max(a["x"], b["x"]), max(a["y"], b["y"])
    x1, y1 = min(a["x"] + a["w"], b["x"] + b["w"]), min(a["y"] + a["h"], b["y"] + b["h"])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / (a["w"] * a["h"] + b["w"] * b["h"] - inter)


def seg():
    hebocr()
    from hebocr.page import segment as S

    for r in rows("pages"):
        _, boxes, angle = S.segment_page(r["pil"])
        theirs = [dict(x=b.x, y=b.y, w=b.w, h=b.h) for b in boxes]
        t0 = time.time()
        ours = R.segment_page(np.asarray(r["pil"]))
        dt = time.time() - t0
        best = [max((iou(o, t) for t in theirs), default=0) for o in ours["boxes"]]
        print(f"{r['page_id'] if 'page_id' in r else ''} {r['pil'].size} angle {angle}/{ours['angle']} "
              f"boxes {len(theirs)}/{len(ours['boxes'])} matched {sum(b >= 0.9 for b in best)} ({dt:.1f}s)")


def lines(impl, backbone, tta, limit):
    OUT.mkdir(parents=True, exist_ok=True)
    if impl == "hebocr":
        hebocr()
        from hebocr.lm import CharNGramLM
        from hebocr.recognize import Recognizer
        from hebocr.wordlm import WordUnigramLM

        rec = Recognizer(HF_DIR / "mishkefet-v1.pt", device="cpu", lm=CharNGramLM.load(HF_DIR / "hebrew_char6.pkl"),
                         lm_weight=0.4, beam_width=12, tta=tta,
                         word_lm=WordUnigramLM.load(HF_DIR / "hebrew_words.pkl"), word_weight=0.2)
        read = lambda im: rec.read_tta([im])[0] if tta > 1 else rec.read([im])[0]
    else:
        reader = R.Reader(R.Model(backbone=backbone), tta=tta)
        read = lambda im: reader.read_line(R.to_gray(np.asarray(im)))["text"]

    results = []
    for i, r in enumerate(rows("lines")):
        if limit and i >= limit:
            break
        hyp = read(r["pil"])
        results.append({"id": r.get("line_id", i), "hyp": hyp, "ref": r["text"], "cer": cer(hyp, r["text"])})
    tag = f"lines-{impl}-{Path(backbone).stem if backbone else 'fp32'}-tta{tta}"
    (OUT / f"{tag}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    kept = [x["cer"] for x in results if x["hyp"].strip()]
    print(tag, "n", len(results), "median CER (blanks dropped)", round(statistics.median(kept), 4),
          "no-drop", round(statistics.median([x["cer"] if x["hyp"].strip() else 1.0 for x in results]), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["seg", "lines"])
    ap.add_argument("--impl", default="ref", choices=["ref", "hebocr"])
    ap.add_argument("--backbone")
    ap.add_argument("--tta", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    seg() if a.what == "seg" else lines(a.impl, a.backbone, a.tta, a.limit)


if __name__ == "__main__":
    main()

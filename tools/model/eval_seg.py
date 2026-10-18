"""Score page segmentation against the ivrit.ai benchmark's gold line boxes (local only).

    python eval_seg.py                  # the reference segmenter
    python eval_seg.py --extra DIR      # also report box counts for images in DIR (e.g. pulled from the phone)

Recall: gold lines matched by a predicted box at IoU >= 0.5 (boxes compared in
the deskewed frame when the page was rotated, which only matters for a few
degrees). Precision: predicted boxes matching some gold line.
"""

import argparse
from pathlib import Path

import numpy as np

import reference as R
from eval_bench import iou, rows


def score(fn=R.segment_page):
    tot_r = tot_p = 0.0
    out = []
    for r in rows("pages"):
        rgb = np.asarray(r["pil"])
        seg = fn(rgb)
        gold = [l["bbox"] for l in r["lines"]]
        pred = seg["boxes"]
        rec = sum(any(iou(g, p) >= 0.5 for p in pred) for g in gold) / max(1, len(gold))
        prec = sum(any(iou(g, p) >= 0.5 for g in gold) for p in pred) / max(1, len(pred))
        out.append((r["page_id"][:8], len(gold), len(pred), round(rec, 3), round(prec, 3), seg["pitch"]))
        tot_r += rec
        tot_p += prec
    n = len(out)
    return tot_r / n, tot_p / n, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra")
    a = ap.parse_args()
    rec, prec, rows_ = score()
    for row in rows_:
        print(*row)
    print(f"MEAN recall {rec:.3f} precision {prec:.3f}")
    if a.extra:
        for f in sorted(Path(a.extra).glob("*.jpg")):
            if f.stem.endswith(("-thumb", "-small")):
                continue
            seg = R.segment_page(R.load_rgb(f))
            print(f.stem[:8], "pitch", seg["pitch"], "boxes", len(seg["boxes"]))


if __name__ == "__main__":
    main()

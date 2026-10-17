"""Simulate on-device writer adaptation on the ivrit.ai benchmark's 10 writers.

Each benchmark page is one writer. Per writer: adapt the head on `--train`
lines (about 5 words each, so 8 lines is roughly one 40-word round), then
measure greedy CER on that writer's remaining lines (did it learn the hand?)
and on other writers' lines (did it forget everyone else?).

    python adapt_sim.py features      # cache backbone features (3 TTA views)
    python adapt_sim.py run --rank 0 --lr 1e-3 --steps 150 --l2 1e-2

Local only; the benchmark is never committed or trained into the shipped
weights. It only picks the adaptation hyperparameters in spec/adapt.json.
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

import adapter as AD
import reference as R
from eval_bench import cer, rows

CACHE = Path.home() / ".cache/mishkefet/eval/sim-features.npz"


def features():
    model = R.Model()
    chars = R.charset()
    out = {}
    meta = []
    for i, r in enumerate(rows("lines")):
        g = R.to_gray(np.asarray(r["pil"]))
        label = [chars.index(c) for c in r["text"] if c in chars]
        for v, s in enumerate(R.TTA_SCALES):
            out[f"{i}_{v}"] = model.features(R.preprocess(R.tta_view(g, s))).astype(np.float16)
        meta.append({"i": i, "page": r["page_id"], "text": r["text"], "label": label})
    np.savez_compressed(CACHE, **out)
    CACHE.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False))
    print(len(meta), "lines cached")


def decode(head, f, chars):
    return "".join(chars[i] for i in R.greedy(head.logprobs(f.astype(np.float32))))


def run(a):
    feats = np.load(CACHE)
    meta = json.loads(CACHE.with_suffix(".json").read_text())
    chars = R.charset()
    W0, b0 = R.read_head(R.BUILD / "head.bin")
    base = AD.Head(W0, b0)
    by_page = defaultdict(list)
    for m in meta:
        by_page[m["page"]].append(m)
    cfg = AD.Config(rank=a.rank, steps=a.steps, lr=a.lr, l2=a.l2)
    rng = np.random.default_rng(0)
    res = []
    all_lines = [m for m in meta]
    for page, lines in sorted(by_page.items()):
        idx = rng.permutation(len(lines))
        train = [lines[j] for j in idx[: a.train]]
        test = [lines[j] for j in idx[a.train:]]
        views = range(3) if a.aug else range(1)
        samples = [(feats[f"{m['i']}_{v}"].astype(np.float32), m["label"]) for m in train for v in views]
        head = AD.train(W0, b0, samples, cfg)
        f0 = lambda m: feats[f"{m['i']}_0"]
        own_b = statistics.mean(cer(decode(base, f0(m), chars), m["text"]) for m in test)
        own_a = statistics.mean(cer(decode(head, f0(m), chars), m["text"]) for m in test)
        others = [m for m in all_lines if m["page"] != page][:: 7]
        oth_b = statistics.mean(cer(decode(base, f0(m), chars), m["text"]) for m in others)
        oth_a = statistics.mean(cer(decode(head, f0(m), chars), m["text"]) for m in others)
        res.append((own_b, own_a, oth_b, oth_a))
        print(f"{page[:8]} n_train {len(train)} own {own_b:.3f} -> {own_a:.3f}   others {oth_b:.3f} -> {oth_a:.3f}", flush=True)
    m = np.array(res).mean(axis=0)
    print(f"MEAN own {m[0]:.4f} -> {m[1]:.4f} ({(m[1]-m[0])/m[0]*100:+.1f}%)  others {m[2]:.4f} -> {m[3]:.4f} ({(m[3]-m[2])/m[2]*100:+.1f}%)  cfg {vars(a)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["features", "run"])
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l2", type=float, default=1e-2)
    ap.add_argument("--train", type=int, default=8)
    ap.add_argument("--aug", type=int, default=1)
    a = ap.parse_args()
    features() if a.what == "features" else run(a)


if __name__ == "__main__":
    main()

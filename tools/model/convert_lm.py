"""Convert the two pickled language models to the compact binaries the apps read.

    python -I convert_lm.py       # writes $BUILD/charlm.bin and $BUILD/words.bin

Both formats are specified in spec/lm-format.md. Characters are stored as
charset indices (one byte each, every index < 256), so the apps never handle
Hebrew strings inside the decoder.
"""

import json
import math
import pickle
import struct

import numpy as np

from common import BUILD, HF_DIR

LP_SCALE = 3000.0  # stored log-prob = -q / LP_SCALE, q in [0, 65534]
ABSENT = 65535


def quantize(lp: float) -> int:
    q = int(round(-lp * LP_SCALE))
    if not 0 <= q < ABSENT:
        raise ValueError(f"log-prob {lp} out of range")
    return q


def charset():
    chars = json.loads((HF_DIR / "charset.json").read_text())["chars"]
    assert len(chars) < 256
    return chars, {c: i for i, c in enumerate(chars)}


def convert_char_lm(path):
    chars, idx = charset()
    data = pickle.load(open(HF_DIR / "hebrew_char6.pkl", "rb"))
    order, counts, total = data["order"], data["counts"], data["total"]

    # gram (tuple of indices) -> quantized log P(last | rest), or ABSENT
    levels: list[dict[tuple, int]] = [dict() for _ in range(order + 1)]
    for n in range(1, order + 1):
        for gram, num in counts[n].items():
            if "\x02" in gram or any(c not in idx for c in gram) or num <= 0:
                continue
            den = counts[n - 1].get(gram[:-1], 0) if n > 1 else total
            key = tuple(idx[c] for c in gram)
            levels[n][key] = quantize(math.log(num / den)) if den else ABSENT
    # Every node's parent must exist for the trie walk; a missing parent is a
    # context that was never counted on its own, so it carries no estimate.
    for n in range(order, 1, -1):
        for key in list(levels[n]):
            parent = key[:-1]
            if parent not in levels[n - 1]:
                levels[n - 1][parent] = ABSENT

    keys = [sorted(levels[n]) for n in range(order + 1)]
    child_start = []
    for n in range(1, order):
        # children of node i at level n are a contiguous run at level n + 1,
        # because both levels are sorted lexicographically
        starts = np.zeros(len(keys[n]) + 1, dtype="<u4")
        pos = {k: i for i, k in enumerate(keys[n])}
        counts_per = np.zeros(len(keys[n]), dtype=np.int64)
        for k in keys[n + 1]:
            counts_per[pos[k[:-1]]] += 1
        starts[1:] = np.cumsum(counts_per)
        child_start.append(starts)

    with open(path, "wb") as fh:
        fh.write(b"MKL1")
        fh.write(struct.pack("<I", order))
        fh.write(struct.pack("<f", math.log(data["backoff"])))
        fh.write(struct.pack("<f", math.log(1.0 / max(total, 1))))
        fh.write(struct.pack("<f", LP_SCALE))
        fh.write(struct.pack("<I", 0))
        fh.write(struct.pack(f"<{order}I", *[len(keys[n]) for n in range(1, order + 1)]))
        for starts in child_start:
            fh.write(starts.tobytes())
        for n in range(1, order + 1):
            fh.write(np.array([levels[n][k] for k in keys[n]], dtype="<u2").tobytes())
        if fh.tell() % 2:
            raise AssertionError
        for n in range(1, order + 1):
            fh.write(np.array([k[-1] for k in keys[n]], dtype="u1").tobytes())
    return {n: len(keys[n]) for n in range(1, order + 1)}


def convert_words(path):
    chars, idx = charset()
    data = pickle.load(open(HF_DIR / "hebrew_words.pkl", "rb"))
    total = data["total"]
    entries = []
    for word, count in data["counts"].items():
        if count <= 0 or any(c not in idx for c in word):
            continue
        entries.append((bytes(idx[c] for c in word), quantize(math.log(count / total))))
    entries.sort()
    offsets = np.zeros(len(entries) + 1, dtype="<u4")
    offsets[1:] = np.cumsum([len(w) for w, _ in entries])
    with open(path, "wb") as fh:
        fh.write(b"MKW1")
        fh.write(struct.pack("<I", len(entries)))
        fh.write(struct.pack("<f", data["oov_logprob"]))
        fh.write(struct.pack("<f", data["prefix_discount"]))
        fh.write(struct.pack("<f", LP_SCALE))
        fh.write(struct.pack("<I", 0))
        fh.write(offsets.tobytes())
        fh.write(np.array([q for _, q in entries], dtype="<u2").tobytes())
        fh.write(b"".join(w for w, _ in entries))
    return len(entries)


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    print("char LM nodes per level:", convert_char_lm(BUILD / "charlm.bin"))
    print("words:", convert_words(BUILD / "words.bin"))
    for name in ("charlm.bin", "words.bin"):
        print(name, (BUILD / name).stat().st_size)


if __name__ == "__main__":
    main()

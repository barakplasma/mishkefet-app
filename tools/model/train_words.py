"""Write spec/train-words.json: the pool the training screen draws words from.

Frequent Hebrew words of 2 to 7 letters from the model's own word prior
(Hebrew Wikipedia counts), most frequent first. The apps pick from it
adaptively (letters the user's hand gets misread on weigh more), so the list
only has to be common, real and cover every letter including final forms.
"""

import json
import pickle
import re

from common import HF_DIR, SPEC

HEB = re.compile(r"^[א-ת]{2,7}$")
FINALS = set("ךםןףץ")
MEDIAL_OF = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}


def ok(w: str) -> bool:
    if not HEB.match(w):
        return False
    # final forms only at the end, and their medial forms never at the end
    if any(c in FINALS for c in w[:-1]):
        return False
    if w[-1] in MEDIAL_OF.values() and w[-1] in "כמנפצ":
        return False
    return True


def main(n=3000):
    data = pickle.load(open(HF_DIR / "hebrew_words.pkl", "rb"))
    ranked = sorted(((c, w) for w, c in data["counts"].items() if ok(w)), reverse=True)
    words = [w for _, w in ranked[:n]]
    letters = set("".join(words))
    missing = set("אבגדהוזחטיכךלמםנןסעפףצץקרשת") - letters
    assert not missing, missing
    (SPEC / "train-words.json").write_text(json.dumps(words, ensure_ascii=False))
    print(len(words), words[:20])


if __name__ == "__main__":
    main()

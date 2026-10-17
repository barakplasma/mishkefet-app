"""Cut the committed test fixtures out of the Pinkas corpus (CC-BY-4.0).

    python fixtures.py        # writes spec/fixtures/{pages,lines}/ and fixtures.json

Pinkas (https://zenodo.org/records/3569694) is early modern community records,
not modern cursive, so these fixtures test *fidelity* (the apps reproduce the
reference), not accuracy. Accuracy is measured locally on the ivrit.ai
benchmark by eval_bench.py, which is never committed.
"""

import io
import json
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from common import SPEC, hebocr

ARCHIVE = Path.home() / ".cache/mishkefet/data/pinkas.zip"
PAGES = ["Page 113_1", "Page 107_1"]
LINES_PER_PAGE = 6
PAGE_MAX_SIDE = 1600


def main():
    hebocr()
    from hebocr.data.hebrew_ink import _crop_polygon, _polygon, _usable

    out = SPEC / "fixtures"
    (out / "pages").mkdir(parents=True, exist_ok=True)
    (out / "lines").mkdir(parents=True, exist_ok=True)
    meta = {"source": "Pinkas dataset, https://zenodo.org/records/3569694, CC-BY-4.0", "pages": [], "lines": []}

    with zipfile.ZipFile(ARCHIVE) as z:
        root_dir = "pinkas_dataset/pinkas_dataset_images_and_xmls/"
        for p, name in enumerate(PAGES):
            image = Image.open(io.BytesIO(z.read(root_dir + name + ".jpg")))
            image = ImageOps.exif_transpose(image).convert("RGB")
            scale = min(1.0, PAGE_MAX_SIDE / max(image.size))
            page_file = f"page{p + 1}.jpg"
            page_img = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
            page_img.save(out / "pages" / page_file, quality=88)
            meta["pages"].append({"file": f"pages/{page_file}", "source": name})

            xml = ET.fromstring(z.read(root_dir + name + ".xml"))
            ns = {"p": xml.tag.split("}")[0].strip("{")}
            gray = image.convert("L")
            taken = 0
            for line in xml.findall(".//p:TextLine", ns):
                uni = line.find("p:TextEquiv/p:Unicode", ns)
                coords = line.find("p:Coords", ns)
                if uni is None or not uni.text or coords is None:
                    continue
                text = _usable(uni.text)
                poly = _polygon(coords.get("points", ""))
                if text is None or poly is None or len(text) < 12:
                    continue
                crop = _crop_polygon(gray, poly)
                if crop is None:
                    continue
                # Lines are the sizes a phone would hand the recognizer: keep
                # the crop's own resolution but cap it so the fixtures stay small.
                if crop.height > 120:
                    s = 120 / crop.height
                    crop = crop.resize((round(crop.width * s), 120), Image.LANCZOS)
                f = f"lines/p{p + 1}-{taken + 1:02d}.png"
                crop.save(out / f)
                meta["lines"].append({"file": f, "truth": text})
                taken += 1
                if taken >= LINES_PER_PAGE:
                    break

    (out / "fixtures.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(len(meta["pages"]), "pages,", len(meta["lines"]), "lines")


if __name__ == "__main__":
    main()

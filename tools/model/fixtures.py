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


BIG_LINES = ["שלום לכולם וברוכים הבאים", "זהו דף עם מעט שורות גדולות", "המרווח בין השורות רחב מאוד", "כדי לבדוק את חיתוך השורות", "תודה רבה"]


def big_lines_page(out):
    """A photo-like page of five big lines, line spacing about 19% of its height.

    hebocr's segmenter only searched spacings up to 15% of the height and cut
    pages like this into fragments; it is the regression fixture for that.
    Rendered here (macOS Corsiva Hebrew), so it carries no third-party licence."""
    from PIL import ImageDraw, ImageFilter, ImageFont

    w, h, pitch = 986, 451, 84
    im = Image.new("RGB", (w, h), (226, 212, 186))
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Corsiva.ttc", 62)
    for i, line in enumerate(BIG_LINES):
        text = line[::-1]  # PIL without libraqm lays out left to right
        tw = d.textlength(text, font=font)
        d.text(((w - tw) / 2, 16 + i * pitch), text, fill=(30, 28, 26), font=font, stroke_width=3, stroke_fill=(30, 28, 26))
    im = im.filter(ImageFilter.GaussianBlur(0.6))
    im.save(out / "pages" / "big-lines.png", optimize=True)
    return {"file": "pages/big-lines.png", "source": "rendered for this repo", "lines": len(BIG_LINES)}


def turned_tilted_page(out):
    """page2 turned a quarter (clockwise as shot sideways) and tilted 20 degrees on a white canvas."""
    im = Image.open(out / "pages" / "page2.jpg").convert("RGB").rotate(-90, expand=True)
    im = im.rotate(20, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
    im.thumbnail((1600, 1600), Image.LANCZOS)
    im.save(out / "pages" / "turned-tilted.jpg", quality=88)
    return {"file": "pages/turned-tilted.jpg", "source": "page2.jpg turned 270 and tilted 20 degrees"}


def screenshot_page(out):
    """The big-lines page inside near-black phone chrome with white UI text, like a screenshot."""
    from PIL import ImageDraw, ImageFont

    page = Image.open(out / "pages" / "big-lines.png").convert("RGB")
    W, H = 1179, 2556
    shot = Image.new("RGB", (W, H), (8, 8, 8))
    page = page.resize((W, round(page.height * W / page.width)), Image.LANCZOS)
    top = 700
    shot.paste(page, (0, top))
    d = ImageDraw.Draw(shot)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 44)
    d.text((60, 60), "23:59", fill=(240, 240, 240), font=font)
    d.text((420, 230), "r/hebrew", fill=(240, 240, 240), font=font)
    d.text((60, top + page.height + 120), "Today's class", fill=(240, 240, 240), font=font)
    d.rounded_rectangle((40, H - 260, 300, H - 160), 50, outline=(120, 120, 120), width=4)
    shot.save(out / "pages" / "screenshot.png", optimize=True)
    return {"file": "pages/screenshot.png", "source": "big-lines.png inside rendered app chrome"}


def curved_lines(out, lines):
    """Three Pinkas lines each given its own tilt and bend (CC-BY derivatives)."""
    rng = np.random.default_rng(11)
    res = []
    for i, line in enumerate(lines[:3]):
        g = np.asarray(Image.open(out / line["file"]).convert("L"))
        h, w = g.shape
        bend = (0.3, -0.25, 0.2)[i]
        angle = (8.0, -11.0, 5.0)[i]
        pad = int(abs(bend) * h) + 4
        canvas = np.full((h + 2 * pad, w), 255, np.uint8)
        for x in range(w):
            dy = int(round(bend * h * np.sin(np.pi * x / max(1, w - 1))))
            canvas[pad - dy: pad - dy + h, x] = g[:, x]
        im = Image.fromarray(canvas).rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=255)
        f = f"lines/curved-{i + 1}.png"
        im.save(out / f)
        res.append({"file": f, "truth": line["truth"], "source": line["file"], "angle": angle, "bend": bend})
    return res


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

    meta["pages"].append(big_lines_page(out))
    meta["pages"].append(turned_tilted_page(out))
    meta["pages"].append(screenshot_page(out))
    meta["curved"] = curved_lines(out, meta["lines"])
    (out / "fixtures.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(len(meta["pages"]), "pages,", len(meta["lines"]), "lines")


if __name__ == "__main__":
    main()

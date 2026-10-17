"""Stand-in canvas strokes for tests: a word in a Hebrew script font, skeletonized
into one-pixel paths and walked into polylines, as if traced with a finger.

Not handwriting, but it exercises the whole canvas path (strokes -> render ->
preprocess -> model -> decode) with known labels, and gives the spec vectors
their inputs. Rendering goes right to left like a hand would.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONTS = ["/System/Library/Fonts/Supplemental/Corsiva.ttc", "/System/Library/Fonts/Supplemental/Raanana.ttc"]


def word_strokes(word: str, font_index=0, size=150, scale=1.0) -> list[list[tuple[float, float]]]:
    from skimage.morphology import skeletonize

    font = ImageFont.truetype(FONTS[font_index % len(FONTS)], size)
    # PIL without libraqm lays text out left to right; draw the reversed
    # string so the picture reads right to left like real Hebrew.
    word = word[::-1]
    box = font.getbbox(word)
    w, h = box[2] - box[0] + 40, box[3] - box[1] + 40
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).text((20 - box[0], 20 - box[1]), word, fill=255, font=font)
    sk = skeletonize(np.asarray(im) > 127)
    ys, xs = np.nonzero(sk)
    pts = set(zip(xs.tolist(), ys.tolist()))
    strokes = []
    # walk 8-connected paths, starting from the rightmost unvisited pixel
    while pts:
        start = max(pts, key=lambda p: (p[0], -p[1]))
        path = [start]
        pts.discard(start)
        cur = start
        while True:
            nb = [(cur[0] + dx, cur[1] + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx or dy)]
            nb = [p for p in nb if p in pts]
            if not nb:
                break
            cur = min(nb, key=lambda p: abs(p[0] - cur[0]) + abs(p[1] - cur[1]))
            pts.discard(cur)
            path.append(cur)
        if len(path) >= 3:
            strokes.append([(x * scale, y * scale) for x, y in path[::2]])
    return strokes

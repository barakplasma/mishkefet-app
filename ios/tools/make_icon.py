#!/usr/bin/env python3
"""Renders the app icons from the in-app binoculars mark. Run from ios/: python3 tools/make_icon.py

The mark is the design's logo (Shared/MishkefetStyle.swift, BinocularsShape): two lenses of
radius 7 at (9, 14) and (27, 14) and a bridge from (16, 11) to (20, 11), in a 36x24 viewBox,
stroked with round caps:
  * icon-1024.png         orange #ff7f11 binoculars on the light background #e2e8ce (the default icon)
  * icon-1024-dark.png    orange binoculars on the dark background #262626
  * icon-1024-tinted.png  light-grey binoculars on black, for the system's tinted appearance
Rendered at 4x and box-downscaled (anti-aliased), sRGB, no alpha.
"""
import math
from pathlib import Path

from PIL import Image, ImageDraw

S = 1024
SS = 4
N = S * SS

LENSES = [(9.0, 14.0), (27.0, 14.0)]
RADIUS = 7.0
BRIDGE = ((16.0, 11.0), (20.0, 11.0))
STROKE = 3.2          # viewBox units (the header logo uses 3.6; a touch lighter reads better large)
SCALE = 21.0          # viewBox units -> icon px: the mark spans about 72% of the canvas width
CENTER = (18.0, 14.0)  # the lenses' midpoint sits at the canvas centre


def render(bg, fg, path):
    img = Image.new("RGB", (N, N), bg)
    d = ImageDraw.Draw(img)
    k = SCALE * SS
    r = STROKE * k / 2
    ox = N / 2 - CENTER[0] * k
    oy = N / 2 - CENTER[1] * k

    def dab(x, y):
        X, Y = x * k + ox, y * k + oy
        d.ellipse((X - r, Y - r, X + r, Y + r), fill=fg)

    # Dense round dabs = a stroke with round caps and joins.
    for cx, cy in LENSES:
        steps = 2400
        for i in range(steps):
            a = 2 * math.pi * i / steps
            dab(cx + RADIUS * math.cos(a), cy + RADIUS * math.sin(a))
    (x0, y0), (x1, y1) = BRIDGE
    for i in range(401):
        t = i / 400
        dab(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
    small = img.resize((S, S), Image.BOX)
    small.save(path, optimize=True)
    print(path)


if __name__ == "__main__":
    out = Path("Mishkefet/Resources/Assets.xcassets/AppIcon.appiconset")
    render((0xE2, 0xE8, 0xCE), (0xFF, 0x7F, 0x11), out / "icon-1024.png")
    render((0x26, 0x26, 0x26), (0xFF, 0x7F, 0x11), out / "icon-1024-dark.png")
    render((0x00, 0x00, 0x00), (0xD8, 0xD8, 0xD8), out / "icon-1024-tinted.png")

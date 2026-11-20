"""Generates assets/class_0.jpg, class_1.jpg, class_2.jpg (wine glass illustrations)."""
from PIL import Image, ImageDraw, ImageFilter

import math
S = 3  # supersample

def wfrac(t):
    return max(0.0, math.sin(math.pi * (0.12 + 0.88 * t))) ** 0.55

W, H = 400 * S, 480 * S
COLORS = {0: (122, 24, 40), 1: (176, 70, 88), 2: (70, 18, 38)}
LEVEL = {0: 0.55, 1: 0.45, 2: 0.62}

def glass(cls):
    bg = Image.new("RGB", (W, H), (240, 230, 216))
    d = ImageDraw.Draw(bg)
    for y in range(H):  # soft vertical gradient
        t = y / H
        d.line([(0, y), (W, y)], fill=(int(246 - 14 * t), int(238 - 18 * t), int(226 - 22 * t)))
    # table shadow
    sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(sh).ellipse((W * .25, H * .86, W * .75, H * .93), fill=(60, 30, 25, 110))
    sh = sh.filter(ImageFilter.GaussianBlur(14 * S))
    bg.paste(sh, (0, 0), sh)
    d = ImageDraw.Draw(bg)

    cx = W / 2
    top, bowl_bot = H * .12, H * .56
    half = W * .24
    # bowl outline polygon (tulip shape)
    def bowl_pts(y_top, y_bot, scale=1.0):
        pts_l, pts_r = [], []
        n = 40
        for i in range(n + 1):
            t = i / n
            y = y_top + (y_bot - y_top) * t
            w = half * 1.08 * scale * wfrac(t)
            pts_l.append((cx - w, y)); pts_r.append((cx + w, y))
        return pts_l + pts_r[::-1]

    # wine fill
    y_fill = bowl_bot - (bowl_bot - top) * LEVEL[cls]
    fill = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    fd = ImageDraw.Draw(fill)
    pts = bowl_pts(top, bowl_bot, 0.97)
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).polygon(pts, fill=255)
    wine = Image.new("RGBA", (W, H), COLORS[cls] + (255,))
    wd = ImageDraw.Draw(wine)
    for y in range(int(y_fill), int(bowl_bot) + 2):  # darker toward the base
        t = (y - y_fill) / max(1, bowl_bot - y_fill)
        c = tuple(int(COLORS[cls][i] * (1 - .35 * t)) for i in range(3))
        wd.line([(0, y), (W, y)], fill=c + (255,))
    wd.rectangle((0, 0, W, y_fill), fill=(0, 0, 0, 0))
    wm = Image.new("L", (W, H), 0)
    wm.paste(mask.point(lambda v: v), (0, 0))
    below = Image.new("L", (W, H), 0)
    ImageDraw.Draw(below).rectangle((0, y_fill, W, H), fill=255)
    from PIL import ImageChops
    wm = ImageChops.multiply(mask, below)
    bg.paste(wine.convert("RGB"), (0, 0), wm)
    d = ImageDraw.Draw(bg)
    # wine surface ellipse
    wy = y_fill
    ww = half * 1.08 * 0.97 * wfrac((wy - top) / (bowl_bot - top))
    d.ellipse((cx - ww, wy - 9 * S, cx + ww, wy + 9 * S), fill=tuple(min(255, int(c * 1.25)) for c in COLORS[cls]))
    # glass outline, stem, foot
    d.line(bowl_pts(top, bowl_bot)[:41], fill=(255, 255, 255), width=3 * S)
    d.line(bowl_pts(top, bowl_bot)[41:][::-1], fill=(255, 255, 255), width=3 * S)
    d.line(bowl_pts(top, bowl_bot)[:41], fill=(200, 190, 180), width=1 * S)
    d.line(bowl_pts(top, bowl_bot)[41:][::-1], fill=(200, 190, 180), width=1 * S)
    d.rectangle((cx - 3 * S, bowl_bot, cx + 3 * S, H * .85), fill=(252, 250, 246), outline=(200, 190, 180))
    d.ellipse((cx - 70 * S, H * .85 - 7 * S, cx + 70 * S, H * .85 + 9 * S), fill=(252, 250, 246), outline=(200, 190, 180))
    # highlight
    d.arc((cx - half * 1.0, top + 30 * S, cx + half * 1.0, bowl_bot - 10 * S), 110, 150, fill=(255, 255, 255), width=4 * S)
    return bg.resize((W // S, H // S), Image.LANCZOS)

for c in (0, 1, 2):
    glass(c).save(f"assets/class_{c}.jpg", quality=90)

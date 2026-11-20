"""Generates assets/hero.jpg (vineyard rows at dusk). Replace with any photo you like."""
import math, random
from PIL import Image, ImageDraw, ImageFilter

W, H = 1600, 520
random.seed(7)
img = Image.new("RGB", (W, H))
d = ImageDraw.Draw(img)
top, bot = (58, 18, 30), (196, 120, 92)
for y in range(H):
    t = y / H
    d.line([(0, y), (W, y)], fill=tuple(int(top[i] + (bot[i] - top[i]) * t) for i in range(3)))
img = img.filter(ImageFilter.GaussianBlur(2))
d = ImageDraw.Draw(img)

# sun glow
glow = Image.new("RGB", (W, H), (0, 0, 0))
gd = ImageDraw.Draw(glow)
gd.ellipse((1050, 170, 1350, 470), fill=(255, 170, 110))
glow = glow.filter(ImageFilter.GaussianBlur(90))
img = Image.blend(img, Image.composite(glow, img, glow.convert("L")), 0.55)
d = ImageDraw.Draw(img)

def ridge(base, amp, phase, color):
    pts = [(0, H)]
    for x in range(0, W + 8, 8):
        y = base + amp * math.sin(x / 260 + phase) + amp * 0.4 * math.sin(x / 90 + phase * 2)
        pts.append((x, y))
    pts.append((W, H))
    d.polygon(pts, fill=color)

ridge(300, 22, 0.5, (96, 44, 52))
ridge(340, 26, 2.0, (72, 30, 40))

# vineyard rows converging to a vanishing point
vx, vy = 800, 372
d.polygon([(0, 372), (W, 372), (W, H), (0, H)], fill=(44, 20, 28))
for i in range(-14, 15):
    x0 = vx + i * 150
    shade = 52 + int(18 * (1 - abs(i) / 14))
    d.line([(vx + i * 6, vy), (x0, H)], fill=(shade, 38, 34), width=3)
    for k in range(7):
        t = 0.15 + k * 0.13
        px = vx + i * 6 + (x0 - vx - i * 6) * t
        py = vy + (H - vy) * t
        r = 2 + 6 * t
        d.ellipse((px - r, py - r * 0.7, px + r, py + r * 0.7), fill=(78, 70, 42))

img = img.filter(ImageFilter.GaussianBlur(0.8))
# film grain
px = img.load()
for _ in range(60000):
    x, y = random.randrange(W), random.randrange(H)
    r, g, b = px[x, y]
    n = random.randint(-9, 9)
    px[x, y] = (max(0, min(255, r + n)), max(0, min(255, g + n)), max(0, min(255, b + n)))
img.save("assets/hero.jpg", quality=88)

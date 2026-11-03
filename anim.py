"""anim.py - tiny code-drawn animation engine (Pillow + numpy only, no downloads).
A scene = template + extras + camera + big_text. Everything moves: parallax, characters, weather, camera."""
import math, random, re, subprocess
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 720, 1280
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

SCENES = ["mountain_day", "mountain_sunset", "mountain_night", "mountain_storm", "climb", "summit", "ocean_day",
          "ocean_storm", "underwater", "space", "earth_space", "rocket_launch", "city_day", "city_night", "forest",
          "desert", "arctic", "volcano", "abstract"]
EXTRAS = ["snow", "rain", "birds", "sun", "moon", "stars", "clouds", "fish", "rocket", "person", "planet", "lightning",
          "embers", "shake"]
CAMERAS = ["zoom_in", "zoom_out", "pan_left", "pan_right", "tilt_up", "tilt_down", "none"]


def lerp(a, b, t): return a + (b - a) * t
def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def ease(t): t = clamp(t); return t * t * (3 - 2 * t)
def back(t): t = clamp(t); c = 1.70158; return 1 + (c + 1) * (t - 1) ** 3 + c * (t - 1) ** 2
def mix(c1, c2, t): return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))


SKY = {"day": ((80, 150, 230), (205, 232, 250)), "dawn": ((70, 90, 165), (255, 195, 145)),
       "sunset": ((55, 45, 115), (255, 140, 85)), "night": ((6, 8, 28), (32, 42, 95)),
       "storm": ((38, 42, 55), (105, 112, 125)), "space": ((2, 2, 12), (22, 12, 52)),
       "water": ((20, 110, 160), (4, 22, 58)), "red": ((55, 6, 8), (210, 75, 30))}
MT = {"day": ((150, 178, 208), (52, 68, 92)), "dawn": ((195, 145, 165), (58, 48, 84)),
      "sunset": ((205, 112, 122), (58, 33, 64)), "night": ((42, 58, 105), (11, 15, 33)),
      "storm": ((98, 103, 118), (30, 33, 42)), "red": ((125, 42, 40), (28, 9, 11)),
      "space": ((40, 40, 70), (15, 15, 30)), "water": ((30, 90, 120), (8, 30, 55))}


@lru_cache(None)
def sky_img(name):
    top, bot = SKY[name]
    g = np.linspace(0, 1, H)[:, None, None]
    arr = np.array(top)[None, None, :] * (1 - g) + np.array(bot)[None, None, :] * g
    return Image.fromarray(np.repeat(arr, W, axis=1).astype("uint8"), "RGB")


@lru_cache(None)
def font(size): return ImageFont.truetype(FONT, int(size))


class Ctx:
    def __init__(s, img, t, dur, cam, sky):
        s.img, s.d, s.t, s.dur, s.p = img, ImageDraw.Draw(img, "RGBA"), t, dur, t / dur
        s.Z, s.cx, s.cy, s.sky, s.surface = cam[0], cam[1], cam[2], sky, None

    def k(s, par): return 1 + (s.Z - 1) * (0.25 + 0.75 * par)

    def m(s, x, y, par):
        k = s.k(par)
        return W / 2 + (x - W / 2 - s.cx * par) * k, H * 0.55 + (y - H * 0.55 - s.cy * par) * k

    def poly(s, pts, par, fill):
        s.d.polygon([s.m(x, y, par) for x, y in pts], fill=fill)

    def circle(s, x, y, r, par, fill):
        sx, sy = s.m(x, y, par); r *= s.k(par)
        s.d.ellipse([sx - r, sy - r, sx + r, sy + r], fill=fill)

    def ellipse(s, x, y, rx, ry, par, fill):
        sx, sy = s.m(x, y, par); k = s.k(par)
        s.d.ellipse([sx - rx * k, sy - ry * k, sx + rx * k, sy + ry * k], fill=fill)

    def line(s, pts, par, fill, width):
        s.d.line([s.m(x, y, par) for x, y in pts], fill=fill, width=max(1, int(width * s.k(par))), joint="curve")


# ---------------------------------------------------------------- terrain
@lru_cache(None)
def ridge(seed, i, n, base_y, ph, hero):
    rng = random.Random(seed * 131 + i)
    peaks = [(rng.uniform(-80, W + 80), ph * rng.uniform(0.5, 1.0), rng.uniform(170, 320)) for _ in range(4)]
    if hero:
        peaks.append(hero)
    pts = []
    for x in range(-320, W + 321, 8):
        h = max(max(0.0, pk[1] * (1 - abs(x - pk[0]) / pk[2])) for pk in peaks)
        h += 7 * math.sin(x * 0.09 + seed) + 4 * math.sin(x * 0.23 + i)
        pts.append((x, base_y - max(h, 0)))
    return tuple(pts), tuple(peaks)


def surface_y(pts, x):
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return lerp(y0, y1, (x - x0) / (x1 - x0))
    return pts[-1][1]


def mountains(c, seed=1, layers=3, snow=True, pal=None, hero=None, shade=True):
    far, near = MT[pal or c.sky]
    for i in range(layers):
        f = i / max(layers - 1, 1)
        par = 0.12 + 0.3 * f
        base = H * (0.66 + 0.05 * i)
        ph = H * lerp(0.27, 0.36, f)
        hv = hero if (hero and i == layers - 1) else None
        pts, peaks = ridge(seed, i, layers, base, ph, hv)
        col = mix(far, near, f ** 0.8)
        poly = list(pts) + [(W + 320, H + 500), (-320, H + 500)]
        c.poly(poly, par, col + (255,))
        if shade:
            for px, pheight, pw in peaks:
                ay = surface_y(pts, px)
                sp = [(px, ay)] + [p for p in pts if px < p[0] <= px + pw * 0.5] + [(px, base + 60)]
                if len(sp) > 3:
                    c.poly(sp, par, (0, 0, 20, 38))
        if snow:
            sy = base - ph * 0.62 * (1.4 if hv else 1)
            run = []
            for x, y in list(pts) + [(W + 400, base)]:
                if y < sy:
                    run.append((x, y))
                elif run:
                    bot = [(px_, py_ + (sy - py_) * clamp(0.42 + 0.4 * abs(math.sin(px_ * 0.11 + seed + i)))) for px_, py_ in reversed(run)]
                    if len(run) > 1:
                        c.poly(run + bot, par, (246, 249, 255, 255) if c.sky not in ("night",) else (170, 185, 225, 255))
                    run = []
        if i == layers - 1:
            c.surface = (pts, par)


def hills(c, color=(40, 70, 45), base=0.8, par=0.9, seed=2, amp=26):
    pts = [(x, H * base + amp * math.sin(x * 0.012 + seed) + amp * 0.5 * math.sin(x * 0.031)) for x in range(-320, W + 321, 12)]
    c.poly(pts + [(W + 320, H + 500), (-320, H + 500)], par, color + (255,))
    c.surface = (tuple(pts), par)


def pines(c, n=14, base=0.78, par=0.7, color=(20, 60, 40), seed=3, sway=True):
    rng = random.Random(seed)
    for k in range(n):
        x = -150 + k * (W + 300) / n + rng.uniform(-20, 20)
        h = rng.uniform(110, 190); w = h * 0.42
        y = H * base + rng.uniform(-10, 20)
        sw = (math.sin(c.t * 1.3 + k) * 3) if sway else 0
        c.poly([(x - 6, y), (x + 6, y), (x + 6, y - h * 0.2), (x - 6, y - h * 0.2)], par, (60, 40, 30, 255))
        for j in range(3):
            yy = y - h * 0.15 - j * h * 0.27
            ww = w * (1 - j * 0.22)
            c.poly([(x - ww, yy), (x + ww, yy), (x + sw * (j + 1), yy - h * 0.42)], par, color + (255,))


# ---------------------------------------------------------------- sky things
def sun(c, x=0.7, y=0.2, r=62, rise=False, color=(255, 232, 150)):
    yy = H * y + (lerp(180, 0, ease(c.p)) if rise else 0)
    for k in range(6, 0, -1):
        c.circle(W * x, yy, r * (1 + k * 0.55), 0.02, color + (14,))
    c.circle(W * x, yy, r, 0.02, color + (255,))


def moon(c, x=0.25, y=0.18, r=50):
    for k in range(5, 0, -1):
        c.circle(W * x, H * y, r * (1 + k * 0.5), 0.02, (200, 215, 255, 12))
    c.circle(W * x, H * y, r, 0.02, (238, 240, 250, 255))
    for dx, dy, rr in ((-14, -8, 10), (12, 10, 14), (6, -20, 6)):
        c.circle(W * x + dx, H * y + dy, rr, 0.02, (205, 210, 225, 255))


def stars(c, n=90, seed=5, ymax=0.62):
    rng = random.Random(seed)
    for k in range(n):
        x, y, ph = rng.uniform(0, W), rng.uniform(0, H * ymax), rng.uniform(0, 6)
        a = int(120 + 120 * math.sin(c.t * 2 + ph))
        sx, sy = c.m(x, y, 0.03); r = rng.choice((1, 1, 2))
        c.d.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(255, 255, 255, clamp(a, 40, 255) if False else max(40, min(255, a))))


def clouds(c, n=5, y=0.14, seed=7, color=(255, 255, 255, 235), speed=14):
    rng = random.Random(seed)
    for k in range(n):
        x0, yy, s = rng.uniform(0, W + 400), H * y + rng.uniform(-60, 180), rng.uniform(0.8, 1.6)
        x = (x0 + c.t * speed * s) % (W + 500) - 250
        for dx, dy, rx, ry in ((0, 0, 70, 28), (-45, 8, 45, 22), (48, 8, 52, 24), (10, -18, 40, 26)):
            c.ellipse(x + dx * s, yy + dy * s, rx * s, ry * s, 0.1 + 0.05 * k % 0.3, color)


def birds(c, n=5, y=0.3, seed=9):
    rng = random.Random(seed)
    for k in range(n):
        x = (rng.uniform(0, W) + c.t * rng.uniform(50, 90)) % (W + 200) - 100
        yy = H * y + rng.uniform(-70, 90) + 12 * math.sin(c.t * 2 + k)
        fl = math.sin(c.t * 9 + k * 2) * 9; s = rng.uniform(0.8, 1.3)
        c.line([(x - 16 * s, yy - fl), (x, yy), (x + 16 * s, yy - fl)], 0.3, (25, 25, 35, 255), 3)


def planet(c, x=0.7, y=0.3, r=70, color=(214, 140, 90), ring=False):
    cx_, cy_ = W * x, H * y
    if ring:
        c.ellipse(cx_, cy_, r * 1.8, r * 0.38, 0.06, (230, 210, 170, 150))
    c.circle(cx_, cy_, r, 0.06, color + (255,))
    c.circle(cx_ + r * 0.25, cy_ + r * 0.2, r * 0.82, 0.06, (0, 0, 30, 70))
    c.circle(cx_ - r * 0.3, cy_ - r * 0.3, r * 0.35, 0.06, (255, 255, 255, 40))


def earth(c, y=1.0, r=520):
    for k in range(5, 0, -1):
        c.circle(W / 2, H * y, r + k * 22, 0.1, (90, 170, 255, 16))
    c.circle(W / 2, H * y, r, 0.1, (28, 84, 190, 255))
    for dx, dy, rx, ry in ((-120, -380, 130, 60), (110, -300, 90, 50), (-30, -440, 70, 35)):
        c.ellipse(W / 2 + dx + c.t * 6, H * y + dy, rx, ry, 0.1, (60, 150, 80, 255))


# ---------------------------------------------------------------- water
def ocean(c, base=0.62, colors=((30, 110, 170), (20, 85, 140), (12, 60, 110)), amp=14, speed=1.6):
    for i, col in enumerate(colors):
        par = 0.35 + 0.2 * i
        pts = [(x, H * (base + 0.05 * i) + amp * (1 + i * 0.4) * math.sin(x * 0.02 + c.t * speed * (1 + 0.3 * i) + i * 2))
               for x in range(-320, W + 321, 14)]
        c.poly(pts + [(W + 320, H + 500), (-320, H + 500)], par, col + (255,))
        if i == 0:
            c.d.line([c.m(x, y, par) for x, y in pts], fill=(255, 255, 255, 120), width=3)


def underwater(c):
    for k in range(5):
        x = 80 + k * 150 + 25 * math.sin(c.t * 0.5 + k)
        c.poly([(x, -100), (x + 60, -100), (x + 230, H), (x + 120, H)], 0.05, (255, 255, 255, 14))
    pts = [(x, H * 0.9 + 18 * math.sin(x * 0.02)) for x in range(-320, W + 321, 14)]
    c.poly(pts + [(W + 320, H + 500), (-320, H + 500)], 0.8, (10, 40, 55, 255))
    rng = random.Random(4)
    for k in range(26):
        x, sp, r = rng.uniform(0, W), rng.uniform(40, 120), rng.uniform(3, 9)
        y = H - ((rng.uniform(0, H) + c.t * sp) % (H + 60))
        c.circle(x + 10 * math.sin(c.t * 2 + k), y, r, 0.4, (200, 235, 255, 90))


def fish(c, n=7, seed=11):
    rng = random.Random(seed)
    for k in range(n):
        d = 1 if k % 2 == 0 else -1
        x = (rng.uniform(0, W) + d * c.t * rng.uniform(40, 100)) % (W + 300) - 150
        y = H * rng.uniform(0.25, 0.8) + 14 * math.sin(c.t * 2 + k); s = rng.uniform(0.8, 1.5)
        col = rng.choice(((255, 150, 60, 255), (255, 210, 80, 255), (120, 200, 255, 255)))
        c.ellipse(x, y, 26 * s, 12 * s, 0.5, col)
        c.poly([(x - d * 22 * s, y), (x - d * 42 * s, y - 13 * s + 4 * math.sin(c.t * 10 + k)), (x - d * 42 * s, y + 13 * s)], 0.5, col)
        c.circle(x + d * 12 * s, y - 3 * s, 2.5 * s, 0.5, (10, 10, 10, 255))


# ---------------------------------------------------------------- people & machines
def person(c, x0=0.15, x1=0.55, action="climb", scale=1.0, speed_ease=True):
    if not c.surface:
        pts, par = tuple((x, H * 0.8) for x in range(-320, W + 321, 40)), 0.9
    else:
        pts, par = c.surface
    prog = ease(c.p) if speed_ease else c.p
    wx = lerp(W * x0, W * x1, prog) if action != "stand" else W * x1
    wy = surface_y(pts, wx)
    if action in ("climb", "walk"):
        ahead = surface_y(pts, wx + 6)
    else:
        ahead = wy
    ang = math.atan2(wy - ahead, 6) * 0.0  # keep figure upright
    ph = c.t * (7 if action == "climb" else 6)
    s = 34 * scale
    jacket, pants, skin = (232, 92, 40, 255), (40, 44, 60, 255), (240, 200, 160, 255)
    base = (wx, wy)
    hip = (wx, wy - s * 1.0)
    sh = (wx, wy - s * 1.9)
    swing = math.sin(ph) * (0.6 if action != "stand" else 0.05)
    for sg in (-1, 1):
        fx = wx + sg * swing * s * 0.8
        fy = wy - max(0, -sg * math.sin(ph)) * s * 0.35
        c.line([hip, (fx, fy)], par, pants, 7 * scale)
    c.line([hip, sh], par, jacket, 13 * scale)
    c.circle(wx, sh[1] - s * 0.4, s * 0.34, par, skin)
    c.circle(wx, sh[1] - s * 0.5, s * 0.36, par, (200, 30, 30, 255) if action != "stand" else (200, 30, 30, 255))
    c.poly([(wx - s * 0.22, sh[1] + 4), (wx - s * 0.62, sh[1] + 4), (wx - s * 0.62, hip[1] - 2), (wx - s * 0.22, hip[1] - 2)], par, (60, 90, 60, 255))
    for sg in (-1, 1):
        hx = wx + sg * (math.sin(ph + (1.6 if action == "climb" else 0)) * s * 0.7 + (s * 0.4 if action == "climb" else 0))
        hy = sh[1] + s * 0.5 - (math.cos(ph) * s * 0.35 if action == "climb" else 0) - (s * 0.9 if action == "stand" and sg == 1 else 0)
        c.line([(wx, sh[1] + 6), (hx, hy)], par, jacket, 6 * scale)
    if action == "stand":
        fx_, fy_ = wx + s * 0.4, sh[1] - s * 0.95
        c.line([(fx_, fy_ + s * 1.3), (fx_, fy_ - s * 0.9)], par, (230, 230, 230, 255), 3)
        c.poly([(fx_, fy_ - s * 0.9), (fx_ + s * 0.9 + 4 * math.sin(c.t * 6), fy_ - s * 0.62), (fx_, fy_ - s * 0.3)], par, (220, 40, 40, 255))


def rocket(c, x=0.5, rise=True):
    prog = ease(c.p) ** 1.6 if rise else 0
    y = lerp(H * 0.72, -H * 0.5, prog)
    cx_ = W * x + (2.5 * math.sin(c.t * 40) if prog < 0.2 else 0)
    for k in range(14):
        c.circle(cx_ + 10 * math.sin(k + c.t * 5), y + 90 + k * 26, 18 + k * 5, 0.8, (230, 230, 235, int(150 - k * 10)))
    fl = 40 + 22 * math.sin(c.t * 30)
    c.poly([(cx_ - 18, y + 80), (cx_ + 18, y + 80), (cx_, y + 80 + fl * 2.4)], 0.8, (255, 140, 30, 255))
    c.poly([(cx_ - 9, y + 80), (cx_ + 9, y + 80), (cx_, y + 80 + fl * 1.5)], 0.8, (255, 235, 120, 255))
    c.poly([(cx_ - 30, y + 70), (cx_ + 30, y + 70), (cx_ + 30, y - 70), (cx_ - 30, y - 70)], 0.8, (236, 238, 245, 255))
    c.poly([(cx_ - 30, y - 70), (cx_ + 30, y - 70), (cx_, y - 140)], 0.8, (220, 50, 50, 255))
    c.poly([(cx_ - 30, y + 20), (cx_ - 62, y + 78), (cx_ - 30, y + 70)], 0.8, (220, 50, 50, 255))
    c.poly([(cx_ + 30, y + 20), (cx_ + 62, y + 78), (cx_ + 30, y + 70)], 0.8, (220, 50, 50, 255))
    c.circle(cx_, y - 20, 14, 0.8, (90, 170, 230, 255))


def city(c, night=False, seed=6):
    for layer, (par, col) in enumerate(((0.3, (70, 90, 120) if not night else (18, 24, 52)), (0.6, (38, 50, 74) if not night else (9, 12, 28)))):
        rng = random.Random(seed + layer); x = -250
        while x < W + 250:
            w, h = rng.uniform(50, 110), rng.uniform(180, 520) * (1.0 if layer else 0.75)
            by = H * (0.72 + 0.06 * layer)
            c.poly([(x, by), (x + w, by), (x + w, by - h), (x, by - h)], par, col + (255,))
            for wy in range(int(by - h + 16), int(by - 12), 26):
                for wx in range(int(x + 10), int(x + w - 12), 20):
                    on = rng.random() < (0.55 if night else 0.2)
                    if on and (night or rng.random() < 0.5):
                        fl = 255 if rng.random() > 0.04 or int(c.t * 3) % 2 else 120
                        c.poly([(wx, wy), (wx + 9, wy), (wx + 9, wy + 13), (wx, wy + 13)], par, (255, 220, 110, fl))
            x += w + rng.uniform(4, 14)
    c.poly([(-320, H * 0.84), (W + 320, H * 0.84), (W + 320, H + 500), (-320, H + 500)], 0.9, (20, 22, 30, 255))


def dunes(c, seed=3):
    for i, col in enumerate(((236, 180, 100), (214, 150, 78), (180, 118, 58))):
        par = 0.25 + 0.3 * i
        pts = [(x, H * (0.66 + 0.07 * i) + 40 * math.sin(x * 0.006 + i * 2 + seed) + 15 * math.sin(x * 0.02 + i)) for x in range(-320, W + 321, 14)]
        c.poly(pts + [(W + 320, H + 500), (-320, H + 500)], par, col + (255,))
        if i == 2:
            c.surface = (tuple(pts), par)
    for x, h in ((140, 130), (560, 100)):
        y = surface_y(c.surface[0], x) + 6
        c.line([(x, y), (x, y - h)], 0.8, (60, 130, 70, 255), 16)
        c.line([(x, y - h * 0.5), (x - 34, y - h * 0.5), (x - 34, y - h * 0.8)], 0.8, (60, 130, 70, 255), 11)
        c.line([(x, y - h * 0.4), (x + 32, y - h * 0.4), (x + 32, y - h * 0.7)], 0.8, (60, 130, 70, 255), 11)


def aurora(c):
    for k, col in enumerate(((60, 255, 160), (120, 120, 255), (60, 220, 230))):
        for j in range(26):
            x = j * 30 - 20
            top = H * (0.12 + 0.08 * k) + 40 * math.sin(c.t * 0.8 + j * 0.35 + k)
            c.poly([(x, top), (x + 32, top + 8 * math.sin(j)), (x + 32, top + 260), (x, top + 260)], 0.05, col + (22,))


def volcano_glow(c):
    px, ph = W * 0.5, H * 0.55
    for k in range(6, 0, -1):
        c.circle(px, H * 0.81 - ph, 40 + k * 28, 0.35, (255, 110, 30, 16))
    lava = [(px - 18, H * 0.81 - ph + 4), (px + 18, H * 0.81 - ph + 4), (px + 55 + 10 * math.sin(c.t * 2), H * 0.81 - ph + 190), (px - 40, H * 0.81 - ph + 220)]
    c.poly(lava, 0.35, (255, 120, 30, 230))


def embers(c, n=60, seed=8):
    rng = random.Random(seed)
    for k in range(n):
        x, sp, r = rng.uniform(150, W - 150), rng.uniform(60, 180), rng.uniform(2, 5)
        y = H * 0.55 - ((rng.uniform(0, H) + c.t * sp) % (H * 0.75))
        c.circle(x + 30 * math.sin(c.t * 1.5 + k), y, r, 0.4, (255, rng.choice((120, 170, 210)), 40, 220))


def snow(c, n=130, seed=12):
    rng = random.Random(seed)
    for k in range(n):
        x = (rng.uniform(0, W) + c.t * rng.uniform(-20, 20) + 12 * math.sin(c.t + k)) % W
        y = (rng.uniform(0, H) + c.t * rng.uniform(50, 130)) % H
        r = rng.uniform(1.5, 4.5)
        c.d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, 215))


def rain(c, n=160, seed=13):
    rng = random.Random(seed)
    for k in range(n):
        x = (rng.uniform(0, W) + c.t * 90) % W
        y = (rng.uniform(0, H) + c.t * rng.uniform(900, 1300)) % H
        c.d.line([x, y, x - 8, y + 26], fill=(190, 205, 235, 150), width=2)


def lightning(c, seed=14):
    cyc = (c.t * 0.9) % 3.1
    if cyc < 0.3:
        a = int(150 * (1 - cyc / 0.3))
        c.d.rectangle([0, 0, W, H], fill=(255, 255, 255, a))
        rng = random.Random(int(c.t * 0.9 / 3.1) + seed)
        x, y, pts = rng.uniform(150, W - 150), 0, []
        while y < H * 0.62:
            pts.append((x, y)); x += rng.uniform(-45, 45); y += rng.uniform(40, 80)
        c.d.line(pts, fill=(255, 255, 230, 255), width=6)


ELEMENTS = {k: v for k, v in dict(mountains=mountains, hills=hills, pines=pines, sun=sun, moon=moon, stars=stars,
            clouds=clouds, birds=birds, planet=planet, earth=earth, ocean=ocean, underwater=underwater, fish=fish,
            person=person, rocket=rocket, city=city, dunes=dunes, aurora=aurora, volcano_glow=volcano_glow,
            embers=embers, snow=snow, rain=rain, lightning=lightning).items()}

HERO = (W * 0.6, H * 0.50, W * 0.75)


def build(scene):
    """-> (sky, sky_to, [(element, kwargs), ...])"""
    S = scene
    E = []
    sky, sky_to = "day", None
    if S == "mountain_day":
        E = [("sun", {}), ("clouds", {}), ("mountains", {"seed": 2}), ("birds", {})]
    elif S == "mountain_sunset":
        sky = "sunset"; E = [("sun", {"x": 0.5, "y": 0.4, "r": 80, "color": (255, 170, 90)}), ("clouds", {"color": (255, 190, 170, 200)}), ("mountains", {"seed": 4})]
    elif S == "mountain_night":
        sky = "night"; E = [("stars", {}), ("moon", {}), ("mountains", {"seed": 5})]
    elif S == "mountain_storm":
        sky = "storm"; E = [("clouds", {"n": 7, "color": (70, 75, 88, 240), "speed": 30}), ("mountains", {"seed": 6}), ("rain", {}), ("lightning", {})]
    elif S == "climb":
        E = [("sun", {"x": 0.25, "y": 0.2}), ("clouds", {"n": 4}), ("mountains", {"seed": 3, "hero": HERO}), ("person", {"x0": 0.08, "x1": 0.56, "action": "climb", "scale": 1.7}), ("snow", {"n": 60})]
    elif S == "summit":
        sky = "dawn"; E = [("sun", {"x": 0.5, "y": 0.33, "rise": True, "r": 70, "color": (255, 200, 130)}), ("clouds", {"n": 4, "color": (255, 210, 190, 210)}),
                           ("mountains", {"seed": 3, "hero": HERO}), ("person", {"x0": 0.6, "x1": 0.6, "action": "stand", "scale": 1.7})]
    elif S == "ocean_day":
        E = [("sun", {"x": 0.3}), ("clouds", {}), ("ocean", {}), ("birds", {})]
    elif S == "ocean_storm":
        sky = "storm"; E = [("clouds", {"n": 7, "color": (65, 70, 82, 240), "speed": 35}), ("ocean", {"colors": ((42, 70, 95), (30, 52, 75), (20, 36, 56)), "amp": 30, "speed": 2.6}), ("rain", {}), ("lightning", {})]
    elif S == "underwater":
        sky = "water"; E = [("underwater", {}), ("fish", {})]
    elif S == "space":
        sky = "space"; E = [("stars", {"n": 160, "ymax": 1.0}), ("planet", {"ring": True}), ("planet", {"x": 0.2, "y": 0.72, "r": 44, "color": (90, 140, 210)})]
    elif S == "earth_space":
        sky = "space"; E = [("stars", {"n": 160, "ymax": 1.0}), ("earth", {"y": 1.08})]
    elif S == "rocket_launch":
        sky, sky_to = "day", "space"; E = [("clouds", {"n": 3}), ("hills", {"color": (60, 110, 60), "base": 0.84}), ("rocket", {})]
    elif S == "city_day":
        E = [("sun", {}), ("clouds", {}), ("city", {}), ("birds", {})]
    elif S == "city_night":
        sky = "night"; E = [("stars", {"n": 50}), ("moon", {}), ("city", {"night": True})]
    elif S == "forest":
        sky = "dawn"; E = [("sun", {"x": 0.7, "y": 0.28}), ("hills", {"color": (36, 74, 52), "base": 0.7, "par": 0.3}),
                           ("pines", {"base": 0.72, "par": 0.4, "color": (26, 78, 56), "n": 12, "seed": 3}),
                           ("pines", {"base": 0.86, "par": 0.8, "color": (14, 52, 38), "n": 9, "seed": 8}), ("birds", {})]
    elif S == "desert":
        E = [("sun", {"x": 0.65, "y": 0.22, "r": 70, "color": (255, 240, 170)}), ("dunes", {}), ("person", {"x0": 0.1, "x1": 0.9, "action": "walk"})]
    elif S == "arctic":
        sky = "night"; E = [("stars", {}), ("aurora", {}), ("mountains", {"seed": 9, "pal": "night"}), ("snow", {})]
    elif S == "volcano":
        sky = "red"; E = [("clouds", {"n": 5, "color": (45, 30, 30, 235), "y": 0.1}), ("mountains", {"seed": 2, "layers": 2, "pal": "red", "snow": False,
                          "hero": (W * 0.5, H * 0.55, W * 0.6)}), ("volcano_glow", {}), ("embers", {})]
    else:  # abstract
        sky = "night"; E = [("stars", {"n": 120, "ymax": 1.0}), ("planet", {"x": 0.5, "y": 0.45, "r": 110, "color": (120, 90, 200)})]
    return sky, sky_to, E


KEYWORDS = [(r"everest|climb|summit|peak|mountaineer", "climb"), (r"mountain|alps|himalaya", "mountain_day"), (r"storm|thunder|lightning", "mountain_storm"),
            (r"ocean|sea |waves|ship", "ocean_day"), (r"deep|underwater|fish|reef", "underwater"), (r"rocket|launch|astronaut", "rocket_launch"),
            (r"planet|galaxy|space|star", "space"), (r"earth|world", "earth_space"), (r"city|building|skyscraper", "city_day"),
            (r"forest|tree|jungle", "forest"), (r"desert|sand|dune", "desert"), (r"ice|arctic|polar|frozen|snow", "arctic"),
            (r"volcano|lava|eruption", "volcano"), (r"night|dark|moon", "mountain_night")]


def make_spec(seg, i):
    scene = str(seg.get("scene", "")).strip().lower()
    if scene not in SCENES:
        scene = next((v for k, v in KEYWORDS if re.search(k, seg.get("say", "").lower())), "abstract")
    extras = [e for e in (seg.get("extras") or []) if e in EXTRAS][:3]
    cam = seg.get("camera") if seg.get("camera") in CAMERAS else CAMERAS[i % 6]
    txt = re.sub(r"[^\w\s,.%$°/+!?-]", "", str(seg.get("big_text", ""))).upper().strip()[:16]
    sky, sky_to, els = build(scene)
    for e in extras:
        if e == "sun": els.append(("sun", {}))
        elif e == "person": els.append(("person", {"x0": 0.15, "x1": 0.8, "action": "walk"}))
        elif e in ("snow", "rain", "birds", "moon", "stars", "clouds", "fish", "planet", "lightning", "embers"):
            els.append((e, {}))
    return {"scene": scene, "sky": sky, "sky_to": sky_to, "els": els, "cam": cam, "text": txt, "shake": "shake" in extras, "idx": i}


def camera(name, p, t, shake):
    e = ease(p)
    Z, cx, cy = 1.0 + 0.05 * p, 0.0, 0.0
    if name == "zoom_in": Z = lerp(1.0, 1.32, e)
    elif name == "zoom_out": Z = lerp(1.34, 1.0, e)
    elif name == "pan_left": cx, Z = lerp(80, -80, e), 1.12
    elif name == "pan_right": cx, Z = lerp(-80, 80, e), 1.12
    elif name == "tilt_up": cy, Z = lerp(20, -150, e), 1.1
    elif name == "tilt_down": cy, Z = lerp(-120, 40, e), 1.1
    if shake:
        cx += 9 * math.sin(t * 47); cy += 7 * math.cos(t * 53)
    return Z, cx, cy


def frame(spec, t, dur):
    p = t / dur
    sky = sky_img(spec["sky"]).copy()
    if spec["sky_to"]:
        sky = Image.blend(sky, sky_img(spec["sky_to"]), clamp(ease(p) * 1.4 - 0.1))
    c = Ctx(sky, t, dur, camera(spec["cam"], p, t, spec["shake"]), spec["sky"])
    for name, kw in spec["els"]:
        ELEMENTS[name](c, **kw)
    if spec["text"]:
        s = back(clamp((t - 0.25) / 0.45)); 
        if s > 0.02:
            size = 118 * s
            f = font(max(12, round(size / 4) * 4))
            txt = spec["text"]
            wd = c.d.textlength(txt, font=f)
            if wd > W - 60:
                f = font(max(12, round(size * (W - 60) / wd / 4) * 4))
            c.d.text((W / 2, H * 0.12), txt, font=f, fill=(255, 224, 70, 255), anchor="mm", stroke_width=7, stroke_fill=(20, 20, 20, 255))
    if t < 0.18:
        c.d.rectangle([0, 0, W, H], fill=(0, 0, 0, int(255 * (1 - t / 0.18))))
    return sky


def render_segment(spec, dur, out_path, vf_extra="", fps=30):
    n = max(1, int(round(dur * fps)))
    vf = "scale=1080:1920:flags=lanczos,setsar=1" + ("," + vf_extra if vf_extra else "")
    pr = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
                           "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(fps), str(out_path)],
                          stdin=subprocess.PIPE)
    for f in range(n):
        pr.stdin.write(frame(spec, f / fps, n / fps).tobytes())
    pr.stdin.close()
    if pr.wait() != 0:
        raise SystemExit("ffmpeg failed while rendering a segment")

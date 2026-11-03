"""anim.py - tiny code-drawn animation engine (Pillow + numpy only, no downloads).
A scene = template + extras + camera + big_text. Everything moves: parallax, characters, weather, camera."""
import math, random, re, subprocess
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFont
try:
    import icons
except Exception:  # cairosvg/emoji missing -> only the hand-animated objects are available
    icons = None

W, H = 720, 1280
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

SCENES = ["mountain_day", "mountain_sunset", "mountain_night", "mountain_storm", "climb", "summit", "ocean_day",
          "ocean_storm", "underwater", "space", "earth_space", "rocket_launch", "city_day", "city_night", "forest",
          "desert", "arctic", "volcano", "abstract", "stage_nature", "stage_lab", "stage_dark", "stage_warm", "stage_bright"]
EXTRAS = ["snow", "rain", "birds", "sun", "moon", "stars", "clouds", "person", "lightning", "embers", "shake"]
CAMERAS = ["zoom_in", "zoom_out", "pan_left", "pan_right", "tilt_up", "tilt_down", "none"]


def lerp(a, b, t): return a + (b - a) * t
def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def ease(t): t = clamp(t); return t * t * (3 - 2 * t)
def back(t): t = clamp(t); c = 1.70158; return 1 + (c + 1) * (t - 1) ** 3 + c * (t - 1) ** 2
def mix(c1, c2, t): return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))


SKY = {"day": ((80, 150, 230), (205, 232, 250)), "dawn": ((70, 90, 165), (255, 195, 145)),
       "sunset": ((55, 45, 115), (255, 140, 85)), "night": ((6, 8, 28), (32, 42, 95)),
       "storm": ((38, 42, 55), (105, 112, 125)), "space": ((2, 2, 12), (22, 12, 52)),
       "water": ((20, 110, 160), (4, 22, 58)), "red": ((55, 6, 8), (210, 75, 30)),
       "lab": ((8, 36, 52), (24, 100, 110)), "warm": ((250, 140, 70), (255, 214, 130)), "bright": ((255, 226, 130), (255, 250, 215)),
       "dark": ((14, 10, 34), (52, 32, 92)), "meadow": ((120, 195, 250), (222, 240, 190))}
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



# ---------------------------------------------------------------- animated objects (any topic)
OBJECTS = ["beetle", "frog", "spider", "cat", "bird", "fish", "thermometer", "flask", "explosion", "stopwatch", "heart", "brain",
           "lightbulb", "coin", "fire", "drop", "atom", "gear", "magnifier", "warning", "question", "bolt"]


class O:
    """relative drawing helper: coordinates in units around (x, y), u pixels per unit"""
    def __init__(s, c, x, y, u, par=0.6): s.c, s.x, s.y, s.u, s.par = c, x, y, u, par
    def P(s, dx, dy): return (s.x + dx * s.u, s.y + dy * s.u)
    def poly(s, pts, col): s.c.poly([s.P(a, b) for a, b in pts], s.par, col + ((255,) if len(col) == 3 else ()))
    def ell(s, dx, dy, rx, ry, col): s.c.ellipse(s.x + dx * s.u, s.y + dy * s.u, rx * s.u, ry * s.u, s.par, col + ((255,) if len(col) == 3 else ()))
    def circ(s, dx, dy, r, col): s.ell(dx, dy, r, r, col)
    def line(s, pts, col, w): s.c.line([s.P(a, b) for a, b in pts], s.par, col + ((255,) if len(col) == 3 else ()), w * 130 * s.u)


def rot(pts, a, cx=0, cy=0):
    ca, sa = math.cos(a), math.sin(a)
    return [(cx + (x - cx) * ca - (y - cy) * sa, cy + (x - cx) * sa + (y - cy) * ca) for x, y in pts]


def o_beetle(o, t):
    dark = (22, 20, 24)
    for side in (-1, 1):
        for k, yy in enumerate((-30, 5, 40)):
            sw = math.sin(t * 9 + k * 2.1 + (0 if side > 0 else math.pi)) * 10
            o.line([(side * 50, yy), (side * 92 + sw * 0.4 * side, yy + (k - 1) * 22 + sw), (side * 118, yy + (k - 1) * 36 + sw * 1.3)], dark, 0.055)
        a = math.sin(t * 4 + side) * 8
        o.line([(side * 12, -95), (side * 40 + a, -135), (side * 60 + a * 1.5, -152)], dark, 0.04)
    o.ell(0, 20, 62, 82, (34, 46, 92))
    o.ell(-22, -5, 14, 46, (70, 90, 150, 120))
    o.line([(0, -55), (0, 100)], (12, 18, 40), 0.03)
    o.ell(0, 94, 36, 16, (232, 112, 40))
    o.ell(0, -86, 31, 28, (208, 98, 44))
    for sx in (-1, 1):
        o.circ(sx * 13, -92, 7, (255, 255, 255)); o.circ(sx * 13, -92, 3.5, (10, 10, 10))
    pulse = max(0, math.sin(t * 6))
    for k in range(3):
        o.circ(-6 + k * 6, 112 + 22 * k * pulse, 5 + 3 * pulse, (255, 150 + 40 * k, 60, int(190 * pulse)))


def o_frog(o, t):
    g, dg = (84, 176, 84), (52, 124, 60)
    o.ell(-92, 78, 44, 22, dg); o.ell(92, 78, 44, 22, dg)
    o.ell(0, 32, 96, 72, g); o.ell(0, 48, 60, 40, (196, 232, 150))
    o.ell(0, -32, 82, 50, g)
    th = 1 + 0.18 * math.sin(t * 5)
    o.ell(0, 6, 34 * th, 14 * th, (150, 210, 110))
    bl = 1 if (t % 2.4) > 0.15 else 0.15
    for sx in (-1, 1):
        o.circ(sx * 48, -78, 27, g); o.circ(sx * 48, -78, 19, (255, 255, 255)); o.ell(sx * 48, -78, 9, 12 * bl, (15, 15, 15))
    o.line([(-58, -20), (-30, -8), (0, -4), (30, -8), (58, -20)], (40, 90, 40), 0.03)


def o_spider(o, t):
    bob = math.sin(t * 2.5) * 14
    o.line([(0, -520), (0, -40 + bob)], (230, 230, 240, 200), 0.012)
    o.y += bob * o.u * 0.0
    for side in (-1, 1):
        for k in range(4):
            sw = math.sin(t * 5 + k * 1.2 + (side > 0) * 1.5) * 8
            yy = -22 + k * 16
            o.line([(side * 20, yy + 20 + bob), (side * (75 + k * 6), yy - 28 + sw + bob), (side * (115 + k * 9), yy + 30 + k * 12 + sw + bob)], (20, 18, 24), 0.05)
    o.circ(0, 28 + bob, 46, (36, 30, 44)); o.circ(0, -18 + bob, 27, (50, 42, 60))
    o.ell(0, 30 + bob, 14, 22, (200, 40, 40, 200))
    for sx in (-1, 1):
        o.circ(sx * 10, -24 + bob, 6, (255, 60, 60)); o.circ(sx * 10, -24 + bob, 2.5, (10, 0, 0))


def o_cat(o, t):
    fur, dk = (242, 160, 70), (205, 120, 40)
    tail = math.sin(t * 3) * 25
    o.line([(70, 90), (130, 60 + tail), (150, 0 + tail * 1.4)], fur, 0.1)
    o.ell(0, 80, 82, 66, fur); o.ell(0, 92, 46, 40, (252, 226, 190))
    o.circ(0, -20, 84, fur)
    o.poly([(-75, -62), (-35, -92), (-84, -135)], fur); o.poly([(75, -62), (35, -92), (84, -135)], fur)
    o.poly([(-68, -78), (-45, -90), (-76, -118)], (240, 150, 150)); o.poly([(68, -78), (45, -90), (76, -118)], (240, 150, 150))
    bl = 1 if (t % 3) > 0.15 else 0.1
    for sx in (-1, 1):
        o.ell(sx * 32, -28, 14, 18 * bl, (255, 255, 255)); o.ell(sx * 32, -28, 6, 16 * bl, (20, 20, 20))
        o.line([(sx * 50, -2), (sx * 125, -12)], (250, 250, 250, 220), 0.012); o.line([(sx * 50, 6), (sx * 125, 12)], (250, 250, 250, 220), 0.012)
    o.poly([(-9, -2), (9, -2), (0, 9)], (230, 100, 120))
    o.line([(0, 9), (0, 18), (-14, 24)], (120, 60, 50), 0.02); o.line([(0, 18), (14, 24)], (120, 60, 50), 0.02)


def o_bird(o, t):
    fl = math.sin(t * 9)
    o.poly([(-60, 20), (-130, 40 - fl * 10), (-60, 50)], (40, 90, 170))
    o.ell(0, 15, 74, 52, (70, 150, 230)); o.ell(10, 30, 46, 30, (210, 235, 255))
    o.circ(62, -28, 36, (70, 150, 230))
    o.poly([(92, -30), (140, -20), (92, -10)], (255, 190, 50))
    o.circ(72, -36, 8, (255, 255, 255)); o.circ(74, -36, 4, (10, 10, 10))
    o.poly(rot([(-10, 0), (-60, -90 - fl * 50), (40, -20)], 0, 0, 0), (40, 100, 190))
    o.poly([(-10, 0), (-70, 40 + fl * 40), (35, 20)], (30, 80, 170, 200))


def o_fish(o, t):
    wag = math.sin(t * 7) * 18
    o.poly([(-85, 0), (-150, -42 + wag), (-150, 42 + wag)], (255, 120, 40))
    o.ell(0, 0, 112, 62, (255, 150, 50)); o.poly([(-10, -55), (30, -95), (50, -50)], (255, 110, 40))
    for k in (-30, 0, 30):
        o.line([(k, -50), (k + 8, 50)], (255, 255, 255, 180), 0.05)
    o.circ(66, -10, 11, (255, 255, 255)); o.circ(69, -10, 5, (10, 10, 10))
    for k in range(3):
        o.circ(130 + k * 14, -40 - ((t * 60 + k * 30) % 90), 5 + k, (255, 255, 255, 120))


def o_thermometer(o, t):
    lvl = ease(clamp(t / 2.2)); col = mix((70, 140, 255), (255, 50, 40), lvl)
    o.poly([(-26, -170), (26, -170), (26, 80), (-26, 80)], (240, 244, 250)); o.circ(0, -170, 26, (240, 244, 250)); o.circ(0, 110, 56, (240, 244, 250))
    o.poly([(-13, 80 - lvl * 240 - 10), (13, 80 - lvl * 240 - 10), (13, 90), (-13, 90)], col); o.circ(0, -170 + 0, 0, col); o.circ(0, 110, 44, col)
    for k in range(8):
        o.line([(30, -150 + k * 28), (48 if k % 2 == 0 else 40, -150 + k * 28)], (240, 244, 250, 230), 0.02)
    if lvl > 0.85:
        for k in range(5):
            a = t * 5 + k * 1.3; o.circ(0 + 85 * math.cos(a), 110 + 60 * math.sin(a * 1.4) - ((t * 50 + k * 20) % 60), 7, (255, 160, 60, 160))


def o_flask(o, t):
    lc = mix((70, 220, 130), (170, 90, 240), (math.sin(t * 1.3) + 1) / 2)
    o.poly([(-28, -150), (28, -150), (28, -60), (110, 90), (110, 110), (-110, 110), (-110, 90), (-28, -60)], (225, 240, 250, 90))
    surf = 8 * math.sin(t * 4)
    o.poly([(-70, 8 + surf), (70, 8 - surf), (108, 92), (108, 108), (-108, 108), (-108, 92)], lc)
    for k in range(8):
        bx = -70 + (k * 41) % 140; by = 100 - ((t * 70 + k * 37) % 120)
        o.circ(bx + 6 * math.sin(t * 3 + k), by, 5 + k % 3 * 2, (255, 255, 255, 170))
    o.poly([(-34, -158), (34, -158), (34, -148), (-34, -148)], (210, 225, 240))
    o.line([(-110, 110), (110, 110)], (210, 225, 240), 0.03)


def o_explosion(o, t):
    ph = (t % 1.5) / 1.5; sc = 0.35 + 0.9 * ease(clamp(ph * 1.4)); fade = int(255 * (1 - clamp((ph - 0.7) / 0.3)))
    for rad, col in ((150, (255, 90, 30)), (115, (255, 170, 40)), (70, (255, 240, 150))):
        pts = []
        for k in range(32):
            r = rad * sc * (1.0 if k % 2 == 0 else 0.62) * (1 + 0.1 * math.sin(k * 7 + t * 20))
            a = k / 32 * math.tau; pts.append((r * math.cos(a), r * math.sin(a)))
        o.poly(pts, col[:3] + (fade,))
    for k in range(14):
        a = k / 14 * math.tau + 0.2; r1, r2 = 150 * sc, 150 * sc + 90 * ph
        o.line([(r1 * math.cos(a), r1 * math.sin(a)), (r2 * math.cos(a), r2 * math.sin(a))], (255, 230, 120, fade), 0.03)


def o_stopwatch(o, t):
    o.poly([(-14, -150), (14, -150), (14, -125), (-14, -125)], (180, 185, 195)); o.circ(0, 0, 118, (60, 70, 90)); o.circ(0, 0, 104, (245, 247, 250))
    for k in range(12):
        a = k / 12 * math.tau; o.line([(88 * math.sin(a), -88 * math.cos(a)), (100 * math.sin(a), -100 * math.cos(a))], (50, 55, 70), 0.025)
    a = (t / 1.5) * math.tau
    o.poly(rot([(-5, 10), (5, 10), (3, -88), (-3, -88)], a), (230, 50, 50)); o.circ(0, 0, 9, (50, 55, 70))
    sw = ease(clamp(((t % 1.5) / 1.5)))
    o.poly([(0, 0)] + [(70 * math.sin(k / 20 * a), -70 * math.cos(k / 20 * a)) for k in range(21)], (230, 50, 50, 60))


def o_heart(o, t):
    b = (t * 1.6) % 1; sc = 1 + 0.16 * math.exp(-((b - 0.1) / 0.08) ** 2) + 0.1 * math.exp(-((b - 0.3) / 0.08) ** 2)
    pts = [(16 * math.sin(a) ** 3 * 8.5 * sc, -(13 * math.cos(a) - 5 * math.cos(2 * a) - 2 * math.cos(3 * a) - math.cos(4 * a)) * 8.5 * sc) for a in [k / 60 * math.tau for k in range(60)]]
    o.poly(pts, (232, 40, 70)); o.ell(-45 * sc, -50 * sc, 18, 26, (255, 150, 170, 150))


def o_brain(o, t):
    pk = (244, 160, 180); pulse = 1 + 0.04 * math.sin(t * 4)
    for dx, dy, r in ((-62, -10, 58), (62, -10, 58), (-30, -62, 56), (30, -62, 56), (0, -20, 64), (-45, 38, 50), (45, 38, 50), (0, 48, 46)):
        o.circ(dx * pulse, dy * pulse, r, pk)
    for pts in ([(-70, -20), (-40, -40), (-20, -10)], [(70, -20), (40, -40), (20, -10)], [(-30, 30), (0, 10), (30, 30)], [(0, -90), (0, -30)]):
        o.line(pts, (210, 110, 140), 0.03)
    for k in range(6):
        a = t * 3 + k * 1.05; r = 110 + 20 * math.sin(t * 5 + k)
        if math.sin(t * 6 + k * 2) > 0.2:
            o.circ(r * math.cos(a), -20 + r * math.sin(a) * 0.8, 6, (255, 240, 120))


def o_lightbulb(o, t):
    on = clamp((t - 0.3) / 0.25) * (0.85 + 0.15 * math.sin(t * 25))
    for k in range(6, 0, -1):
        o.circ(0, -30, 85 + k * 22, (255, 230, 100, int(18 * on)))
    o.circ(0, -30, 85, mix((200, 205, 215), (255, 236, 120), on)); o.poly([(-40, 40), (40, 40), (28, 78), (-28, 78)], mix((200, 205, 215), (255, 236, 120), on))
    o.line([(-18, 10), (-8, -20), (0, 0), (8, -20), (18, 10)], (255, 160, 40) if on > 0.5 else (140, 140, 150), 0.02)
    for k in range(3):
        o.poly([(-34, 82 + k * 16), (34, 82 + k * 16), (34, 92 + k * 16), (-34, 92 + k * 16)], (120, 125, 140))
    if on > 0.5:
        for k in range(10):
            a = k / 10 * math.tau; o.line([(125 * math.cos(a), -30 + 125 * math.sin(a)), (160 * math.cos(a), -30 + 160 * math.sin(a))], (255, 230, 100, int(200 * on)), 0.03)


def o_coin(o, t):
    w = abs(math.cos(t * 3)); w = max(w, 0.05)
    o.ell(0, 0, 110 * w, 110, (190, 140, 20)); o.ell(0, 0, 98 * w, 98, (250, 200, 50)); o.ell(0, 0, 78 * w, 78, (255, 222, 100))
    if w > 0.5:
        f = font(max(12, int(130 * o.u * w))); sx, sy = o.c.m(o.x, o.y, o.par)
        o.c.d.text((sx, sy), "$", font=f, fill=(190, 140, 20, 255), anchor="mm")


def o_fire(o, t):
    o.poly([(-90, 100), (90, 100), (90, 120), (-90, 120)], (90, 55, 35))
    for scale, col in ((1.0, (255, 100, 30)), (0.72, (255, 170, 40)), (0.42, (255, 240, 150))):
        pts = [(-80 * scale, 100)]
        for k in range(9):
            x = -80 * scale + k * 20 * scale; h = 100 + (170 * scale) * (1 - abs(k - 4) / 4.5) + 22 * math.sin(t * 9 + k * 1.7) * scale
            pts.append((x, 100 - h))
        pts += [(80 * scale, 100)]
        o.poly(pts, col)
    for k in range(7):
        o.circ(-60 + k * 20 + 10 * math.sin(t * 3 + k), 80 - ((t * 90 + k * 40) % 220), 4, (255, 210, 90, 200))


def o_drop(o, t):
    sq = 1 + 0.05 * math.sin(t * 4)
    pts = [(0, -140)] + [(80 * math.sin(a) * sq, 20 - 80 * math.cos(a) * 0.9 + 0) for a in [k / 24 * math.pi + math.pi / 2 for k in range(-12, 13)]]
    o.poly([(0, -150), (-72, 20), (-60, 70), (0, 100), (60, 70), (72, 20)], (60, 150, 240)); o.circ(0, 40, 66, (60, 150, 240))
    o.ell(-22, 30, 12, 26, (255, 255, 255, 150))
    for k in range(3):
        r = ((t * 0.8 + k / 3) % 1); o.ell(0, 150, 60 + r * 100, 14 + r * 22, (150, 200, 255, int(160 * (1 - r))))


def o_atom(o, t):
    for k in range(3):
        a = k * math.pi / 3
        o.poly(rot([(130 * math.cos(q / 40 * math.tau), 46 * math.sin(q / 40 * math.tau)) for q in range(41)], a), (80, 200, 255, 0))
        pts = rot([(130 * math.cos(q / 60 * math.tau), 46 * math.sin(q / 60 * math.tau)) for q in range(61)], a)
        o.line(pts, (150, 210, 255, 210), 0.02)
        e = t * (2.2 + k * 0.5) + k
        ex, ey = rot([(130 * math.cos(e), 46 * math.sin(e))], a)[0]
        o.circ(ex, ey, 11, (255, 220, 90))
    o.circ(0, 0, 26, (255, 90, 90)); o.circ(10, -8, 10, (90, 140, 255))


def o_gear(o, t):
    a = t * 0.9; pts = []
    for k in range(48):
        r = 120 if (k // 3) % 2 == 0 else 96; ang = k / 48 * math.tau + a; pts.append((r * math.cos(ang), r * math.sin(ang)))
    o.poly(pts, (150, 160, 180)); o.circ(0, 0, 70, (120, 130, 150)); o.circ(0, 0, 32, (40, 48, 70))
    pts = []
    for k in range(32):
        r = 62 if (k // 2) % 2 == 0 else 48; ang = -k / 32 * math.tau - a * 1.4; pts.append((r * math.cos(ang) + 0, r * math.sin(ang)))


def o_magnifier(o, t):
    mx, my = 26 * math.cos(t * 1.6), 16 * math.sin(t * 2.2)
    o.line([(mx + 62, my + 62), (mx + 135, my + 135)], (110, 70, 40), 0.1)
    o.circ(mx, my, 92, (90, 100, 120)); o.circ(mx, my, 78, (200, 235, 255, 140)); o.ell(mx - 28, my - 30, 16, 28, (255, 255, 255, 120))


def o_warning(o, t):
    sh = 4 * math.sin(t * 25) * (1 if (t % 1.2) < 0.4 else 0)
    o.poly([(sh, -140), (125 + sh, 90), (-125 + sh, 90)], (30, 30, 30)); o.poly([(sh, -112), (96 + sh, 72), (-96 + sh, 72)], (255, 205, 40))
    o.poly([(-9 + sh, -50), (9 + sh, -50), (6 + sh, 22), (-6 + sh, 22)], (30, 30, 30)); o.circ(sh, 46, 10, (30, 30, 30))


def o_question(o, t):
    b = 8 * math.sin(t * 3); f = font(max(12, int(300 * o.u)))
    o.circ(0, b, 125, (110, 90, 230)); o.circ(0, b, 108, (140, 120, 250))
    sx, sy = o.c.m(o.x, o.y + b * o.u, o.par); o.c.d.text((sx, sy), "?", font=f, fill=(255, 255, 255, 255), anchor="mm")


def o_bolt(o, t):
    fl = 0.8 + 0.2 * math.sin(t * 22)
    for k in range(5, 0, -1):
        o.circ(0, 0, 90 + k * 25, (255, 230, 80, int(14 * fl)))
    o.poly([(25, -150), (-70, 20), (-8, 20), (-38, 150), (80, -30), (14, -30), (60, -150)], (255, 225, 60)); o.poly([(22, -128), (-48, 12), (6, 12)], (255, 250, 190, 160))



# ---------------------------------------------------------------- more objects: food, nature, things, people
def arc_pts(cx, cy, r, a0, a1, n=24):
    return [(cx + r * math.cos(a0 + (a1 - a0) * k / n), cy + r * math.sin(a0 + (a1 - a0) * k / n)) for k in range(n + 1)]


def o_tomato(o, t):
    w = 1 + 0.025 * math.sin(t * 3)
    o.ell(0, 22, 108 * w, 98 / w, (222, 48, 40)); o.ell(-38, -8, 22, 34, (255, 140, 130, 150))
    pts = []
    for k in range(10):
        r = 52 if k % 2 == 0 else 20; a = k / 10 * math.tau - math.pi / 2; pts.append((r * math.cos(a), -72 + r * 0.45 * math.sin(a)))
    o.poly(pts, (60, 150, 60)); o.line([(0, -72), (6, -112)], (60, 120, 50), 0.07)


def o_apple(o, t):
    o.circ(-42, 20, 78, (220, 40, 50)); o.circ(42, 20, 78, (220, 40, 50)); o.ell(0, 40, 100, 80, (220, 40, 50))
    o.ell(-45, 0, 14, 30, (255, 160, 160, 150)); o.line([(0, -50), (6, -110)], (100, 70, 40), 0.07)
    sw = math.sin(t * 2) * 6
    o.poly([(8, -92), (60 + sw, -128), (82 + sw, -96), (40, -80)], (80, 180, 70))


def o_banana(o, t):
    outer = arc_pts(0, -40, 150, 0.35, math.pi - 0.35); inner = arc_pts(0, -70, 120, math.pi - 0.45, 0.45)
    o.poly(outer + inner, (255, 214, 50)); o.line(arc_pts(0, -55, 135, 0.5, math.pi - 0.5), (230, 180, 30), 0.03)
    o.circ(outer[0][0], outer[0][1], 12, (110, 80, 40)); o.circ(outer[-1][0], outer[-1][1], 12, (110, 80, 40))


def o_pizza(o, t):
    o.poly([(-115, -80), (115, -80), (0, 135)], (255, 200, 70)); o.poly([(-120, -100), (120, -100), (120, -66), (-120, -66)], (214, 150, 70))
    for dx, dy, r in ((-40, -30, 22), (36, -10, 22), (-4, 40, 20), (20, -52, 14)):
        o.circ(dx, dy, r, (200, 50, 40))
    for k in range(3):
        o.line([(-20 + k * 22, 75 - k * 24), (-14 + k * 22, 108 - k * 30 - 4 * math.sin(t * 3 + k))], (255, 240, 190), 0.02)


def o_burger(o, t):
    o.ell(0, -62, 112, 62, (222, 150, 70)); o.ell(0, 62, 106, 34, (214, 140, 62))
    o.poly([(-112, 8), (112, 8), (112, 36), (-112, 36)], (90, 52, 36)); o.poly([(-118, -8), (118, -8), (100, 12), (-100, 12)], (255, 205, 50))
    o.line([(-114 + k * 19, 0 + (6 if k % 2 else -4)) for k in range(13)], (90, 190, 70), 0.06)
    for dx, dy in ((-40, -80), (10, -92), (50, -70), (-8, -60)):
        o.ell(dx, dy, 8, 4, (255, 240, 200))


def o_coffee(o, t):
    o.ell(0, 100, 130, 22, (235, 235, 240)); o.poly([(-80, -30), (80, -30), (64, 96), (-64, 96)], (245, 245, 250))
    o.ell(0, -30, 80, 16, (110, 70, 40)); o.line(arc_pts(92, 26, 36, -1.6, 1.6), (245, 245, 250), 0.09)
    for k in range(3):
        pts = [(-30 + k * 30 + 12 * math.sin(t * 3 + q * 0.9 + k), -48 - q * 22) for q in range(6)]
        o.line(pts, (255, 255, 255, 190), 0.035)


def o_egg(o, t):
    w = 1 + 0.02 * math.sin(t * 3)
    o.ell(0, 20, 82 * w, 108, (248, 238, 214)); o.ell(0, -50, 62 * w, 64, (248, 238, 214)); o.ell(-26, 0, 14, 38, (255, 255, 255, 150))


def o_ketchup(o, t):
    o.poly([(-48, -40), (48, -40), (58, 120), (-58, 120)], (210, 36, 36)); o.poly([(-48, -40), (48, -40), (22, -92), (-22, -92)], (210, 36, 36))
    o.poly([(-16, -92), (16, -92), (14, -130), (-14, -130)], (250, 250, 250)); o.poly([(-40, 10), (40, 10), (40, 78), (-40, 78)], (250, 250, 250))
    o.circ(0, 44, 18, (210, 36, 36)); o.ell(-30, 20, 8, 44, (255, 140, 140, 120))
    r = (t * 1.2) % 1; o.ell(0, -150 - r * 90 * 0, 11, 16, (200, 30, 30, int(255 * (1 - r)))) if False else None
    o.circ(0, -138 - r * 110 + 0, 10, (200, 30, 30, int(255 * (1 - r * 0.6))))


def o_tree(o, t):
    sw = math.sin(t * 1.6) * 6
    o.poly([(-18, 130), (18, 130), (14, 10), (-14, 10)], (110, 76, 46))
    for dx, dy, r in ((-52, -10, 62), (52, -10, 62), (0, -62, 74), (-10, -20, 70)):
        o.circ(dx + sw * 0.5, dy, r, (64, 158, 80))
    for dx, dy in ((-30, -40), (30, -10), (0, -80)):
        o.circ(dx + sw * 0.5, dy, 9, (220, 50, 50))


def o_flower(o, t):
    sw = math.sin(t * 2) * 5
    o.line([(0, 130), (sw, 20)], (70, 150, 70), 0.06); o.poly([(0, 80), (60, 40), (66, 76)], (80, 170, 80))
    for k in range(8):
        a = k / 8 * math.tau + t * 0.5; o.circ(sw + 54 * math.cos(a), -20 + 54 * math.sin(a), 32, (255, 120, 170))
    o.circ(sw, -20, 36, (255, 210, 60))


def o_sun(o, t):
    for k in range(12):
        a = k / 12 * math.tau + t * 0.6; o.poly(rot([(-14, -110), (14, -110), (0, -155)], a), (255, 190, 40))
    o.circ(0, 0, 96, (255, 214, 60)); o.circ(0, 0, 80, (255, 232, 110))


def o_moon(o, t):
    for k in range(5, 0, -1):
        o.circ(0, 0, 100 + k * 18, (210, 220, 255, 14))
    pts = arc_pts(0, 0, 105, -math.pi / 2, math.pi / 2) + arc_pts(-48, 0, 100, math.pi / 2, -math.pi / 2)
    o.poly(pts, (245, 242, 220)); o.circ(36, -30, 9, (220, 215, 190)); o.circ(52, 28, 13, (220, 215, 190))


def o_star(o, t):
    sc = 1 + 0.07 * math.sin(t * 4); pts = []
    for k in range(10):
        r = (118 if k % 2 == 0 else 50) * sc; a = k / 10 * math.tau - math.pi / 2 + math.sin(t * 2) * 0.1; pts.append((r * math.cos(a), r * math.sin(a) + 8))
    o.poly(pts, (255, 206, 40)); o.circ(-14, -10, 14, (255, 245, 170))


def o_snowflake(o, t):
    for k in range(6):
        a = k / 6 * math.tau + t * 0.4; ca, sa = math.cos(a), math.sin(a)
        o.line([(0, 0), (125 * ca, 125 * sa)], (200, 235, 255), 0.04)
        for d in (50, 85):
            o.line([(d * ca - 26 * sa, d * sa + 26 * ca), (d * ca, d * sa), (d * ca + 26 * sa, d * sa - 26 * ca)], (200, 235, 255), 0.03)


def o_car(o, t):
    b = math.sin(t * 9) * 2
    o.poly([(-140, 40 + b), (-130, -10 + b), (-70, -20 + b), (-40, -70 + b), (50, -70 + b), (90, -20 + b), (140, -8 + b), (146, 40 + b)], (222, 52, 52))
    o.poly([(-34, -58 + b), (-58, -18 + b), (-8, -18 + b), (-8, -58 + b)], (190, 225, 250)); o.poly([(6, -58 + b), (6, -18 + b), (76, -18 + b), (44, -58 + b)], (190, 225, 250))
    for wx in (-82, 84):
        o.circ(wx, 52 + b, 36, (30, 30, 36)); o.circ(wx, 52 + b, 18, (190, 195, 205))
        a = t * 14; o.line([(wx + 18 * math.cos(a), 52 + b + 18 * math.sin(a)), (wx - 18 * math.cos(a), 52 + b - 18 * math.sin(a))], (90, 95, 105), 0.03)
    for k in range(3):
        o.circ(-160 - ((t * 120 + k * 40) % 100), 54 + 6 * k, 6 + k * 2, (230, 230, 230, int(150 - ((t * 120 + k * 40) % 100))))


def o_plane(o, t):
    tilt = math.sin(t * 1.5) * 0.08
    pts = lambda p: rot(p, tilt)
    o.poly(pts([(-150, 0), (-120, -20), (120, -20), (160, 0), (120, 20), (-120, 20)]), (240, 244, 250))
    o.poly(pts([(-20, 0), (60, 0), (-40, 110), (-80, 110)]), (200, 206, 220)); o.poly(pts([(-20, -4), (60, -4), (-40, -100), (-80, -100)]), (215, 220, 232))
    o.poly(pts([(-150, 0), (-130, -60), (-100, -60), (-100, 0)]), (222, 52, 52))
    for k in range(5):
        o.circ(pts([(-55 + k * 28, -6)])[0][0], pts([(-55 + k * 28, -6)])[0][1], 6, (110, 170, 230))
    for k in range(6):
        o.circ(-170 - ((t * 140 + k * 30) % 150), 4, 7 + k, (255, 255, 255, int(170 - ((t * 140 + k * 30) % 150))))


def o_house(o, t):
    o.poly([(-100, -10), (100, -10), (100, 110), (-100, 110)], (240, 205, 150)); o.poly([(-122, -6), (0, -100), (122, -6)], (200, 70, 60))
    o.poly([(-22, 30), (22, 30), (22, 110), (-22, 110)], (120, 80, 50)); o.poly([(48, 22), (84, 22), (84, 58), (48, 58)], (170, 215, 250)); o.poly([(-84, 22), (-48, 22), (-48, 58), (-84, 58)], (170, 215, 250))
    o.poly([(58, -88), (84, -88), (84, -40), (58, -52)], (150, 60, 50))
    for k in range(4):
        o.circ(72 + 10 * math.sin(t * 2 + k), -100 - ((t * 40 + k * 25) % 100), 8 + k * 3, (230, 230, 235, int(190 - k * 40)))


def o_robot(o, t):
    w = math.sin(t * 5) * 0.5
    o.line([(0, -112), (0, -140)], (150, 160, 180), 0.04); o.circ(0, -146, 9, (255, 80, 80) if int(t * 3) % 2 else (255, 190, 190))
    o.poly([(-70, -110), (70, -110), (70, -20), (-70, -20)], (170, 182, 205)); o.poly([(-58, -10), (58, -10), (58, 100), (-58, 100)], (140, 152, 178))
    for sx in (-1, 1):
        o.circ(sx * 30, -66, 17, (90, 240, 255)); o.circ(sx * 30, -66, 7, (20, 40, 70))
        o.line([(sx * 58, 0), (sx * (96 + 10 * w), 40 + 20 * w * sx), (sx * (118 + 10 * w), 70 + 20 * w * sx)], (120, 132, 158), 0.06)
    o.poly([(-34, -38), (34, -38), (34, -28), (-34, -28)], (60, 70, 100)); o.circ(0, 30, 16, (255, 190, 60))


def o_phone(o, t):
    o.poly([(-62, -140), (62, -140), (62, 140), (-62, 140)], (30, 34, 46)); o.poly([(-52, -118), (52, -118), (52, 118), (-52, 118)], (120, 180, 255))
    for k in range(3):
        p = clamp((t * 1.5 - k * 0.5) % 2.0, 0, 1)
        o.poly([(-40, -90 + k * 60), (40 * p, -90 + k * 60), (40 * p, -62 + k * 60), (-40, -62 + k * 60)], (255, 255, 255, 230))
    o.circ(0, 128, 5, (90, 94, 110))


def o_laptop(o, t):
    o.poly([(-120, -100), (120, -100), (120, 50), (-120, 50)], (60, 66, 84)); o.poly([(-108, -88), (108, -88), (108, 38), (-108, 38)], (20, 28, 44))
    o.poly([(-150, 56), (150, 56), (170, 90), (-170, 90)], (190, 196, 210))
    for k in range(5):
        L = 70 + (k * 37) % 70; p = clamp((t * 0.8 - k * 0.2) % 1.6, 0, 1)
        o.poly([(-92, -72 + k * 20), (-92 + L * p, -72 + k * 20), (-92 + L * p, -62 + k * 20), (-92, -62 + k * 20)], (110, 255, 160))


def o_book(o, t):
    f = math.sin(t * 2) * 6
    o.poly([(-130, -80), (0, -60 + f), (0, 100), (-130, 80)], (250, 246, 232)); o.poly([(130, -80), (0, -60 + f), (0, 100), (130, 80)], (240, 236, 220))
    o.poly([(-138, -86), (-130, -80), (-130, 90), (-138, 96)], (60, 100, 190)); o.poly([(138, -86), (130, -80), (130, 90), (138, 96)], (60, 100, 190))
    for k in range(5):
        o.line([(-110, -40 + k * 22), (-20, -26 + k * 22)], (170, 170, 180), 0.02); o.line([(110, -40 + k * 22), (20, -26 + k * 22)], (170, 170, 180), 0.02)


def o_trophy(o, t):
    for sx in (-1, 1):
        o.line(arc_pts(sx * 92, -48, 36, -1.6 if sx < 0 else 1.6 + 3.14, 1.6 if sx < 0 else -1.6 + 3.14 + 0), (230, 170, 30), 0.06)
    o.poly([(-82, -110), (82, -110), (60, 20), (20, 62), (-20, 62), (-60, 20)], (255, 205, 50)); o.poly([(-14, 62), (14, 62), (14, 108), (-14, 108)], (230, 170, 30))
    o.poly([(-60, 108), (60, 108), (70, 134), (-70, 134)], (110, 70, 40)); o.ell(-34, -50, 12, 40, (255, 245, 170, 170))
    sp = (t * 1.2) % 1; o.circ(60, -90 - sp * 30, 8 * (1 - sp), (255, 255, 255))


def o_key(o, t):
    a = math.sin(t * 2) * 0.15
    P = lambda p: rot(p, a)
    o.poly(P([(-20, -10), (150, -10), (150, 18), (-20, 18)]), (240, 190, 50)); o.poly(P([(90, 18), (110, 18), (110, 56), (90, 56)]), (240, 190, 50)); o.poly(P([(124, 18), (144, 18), (144, 44), (124, 44)]), (240, 190, 50))
    pts = rot(arc_pts(-60, 4, 62, 0, math.tau, 30), a); o.poly(pts, (240, 190, 50)); o.poly(rot(arc_pts(-60, 4, 30, 0, math.tau, 24), a), (30, 40, 70) if False else (200, 150, 30))


def o_crown(o, t):
    o.poly([(-120, 70), (-120, -50), (-60, 10), (0, -80), (60, 10), (120, -50), (120, 70)], (255, 206, 50)); o.poly([(-120, 70), (120, 70), (120, 96), (-120, 96)], (230, 170, 30))
    for dx, c in ((-120, (230, 50, 60)), (0, (60, 150, 240)), (120, (60, 200, 110))):
        o.circ(dx, -58 if dx != 0 else -88, 13, c)
    sp = (t * 1.3) % 1; o.circ(-70, -10 - sp * 30, 8 * (1 - sp), (255, 255, 255))


def o_globe(o, t):
    o.circ(0, 0, 112, (40, 110, 210))
    for bx, by, r in ((-60, -30, 40), (30, 20, 46), (80, -40, 30), (-20, 70, 30)):
        x = ((bx + 112 + t * 40) % 224) - 112; fade = clamp((112 - abs(x) - r * 0.4) / 40)
        if fade > 0.05:
            o.circ(x, by, r * (0.5 + 0.5 * fade), (70, 170, 90, int(255 * fade)))
    o.circ(0, 0, 112, (255, 255, 255, 0)); o.circ(-36, -40, 30, (255, 255, 255, 45))
    o.line(arc_pts(0, 0, 118, 0, math.tau, 40), (60, 70, 100), 0.03)


def o_rocket(o, t):
    sh = math.sin(t * 40) * 2; fl = 40 + 24 * math.sin(t * 30)
    o.poly([(-22 + sh, 100), (22 + sh, 100), (sh, 100 + fl * 2.2)], (255, 140, 30)); o.poly([(-10 + sh, 100), (10 + sh, 100), (sh, 100 + fl * 1.4)], (255, 235, 120))
    o.poly([(-34 + sh, 90), (34 + sh, 90), (34 + sh, -70), (-34 + sh, -70)], (238, 240, 248)); o.poly([(-34 + sh, -70), (34 + sh, -70), (sh, -150)], (224, 56, 56))
    o.poly([(-34 + sh, 30), (-72 + sh, 100), (-34 + sh, 90)], (224, 56, 56)); o.poly([(34 + sh, 30), (72 + sh, 100), (34 + sh, 90)], (224, 56, 56)); o.circ(sh, -20, 17, (90, 170, 230))


def o_tooth(o, t):
    pts = [(-70, -80), (-30, -100), (0, -86), (30, -100), (70, -80), (82, -20), (60, 50), (56, 120), (30, 120), (14, 50), (-14, 50), (-30, 120), (-56, 120), (-60, 50), (-82, -20)]
    o.poly(pts, (250, 250, 252)); o.ell(-34, -56, 12, 24, (255, 255, 255)); o.line([(-50, -70), (-20, -40)], (225, 232, 245), 0.03)
    sp = (t * 1.2) % 1; o.circ(60, -70 - sp * 20, 9 * (1 - sp), (255, 255, 255)); o.circ(60, -70 - sp * 20, 3, (140, 200, 255))


def o_eye(o, t):
    bl = 1 if (t % 2.6) > 0.18 else 0.1; lx, ly = 22 * math.sin(t * 1.3), 8 * math.sin(t * 1.9)
    o.poly(arc_pts(0, 0, 150, -2.3, -0.84, 20) if False else [(-140, 0), (-70, -70 * bl), (0, -86 * bl), (70, -70 * bl), (140, 0), (70, 70 * bl), (0, 86 * bl), (-70, 70 * bl)], (250, 250, 252))
    if bl > 0.5:
        o.circ(lx, ly, 52, (60, 140, 220)); o.circ(lx, ly, 26, (10, 12, 20)); o.circ(lx - 14, ly - 16, 10, (255, 255, 255))
    o.line([(-140, 0), (-70, -70 * bl), (0, -86 * bl), (70, -70 * bl), (140, 0)], (60, 50, 60), 0.04)


def o_dog(o, t):
    pant = math.sin(t * 8) * 4
    o.ell(-86, -20, 34, 66, (120, 80, 50)); o.ell(86, -20, 34, 66, (120, 80, 50))
    o.circ(0, -10, 84, (220, 170, 110)); o.ell(0, 36, 50, 38, (245, 225, 190))
    for sx in (-1, 1):
        o.circ(sx * 34, -34, 13, (255, 255, 255)); o.circ(sx * 34, -34, 7, (20, 20, 20))
    o.ell(0, 20, 20, 13, (30, 30, 36)); o.ell(0, 66 + pant, 16, 22 + pant, (240, 110, 130))
    o.line([(0, 30), (0, 52), (-22, 62)], (90, 60, 40), 0.025); o.line([(0, 52), (22, 62)], (90, 60, 40), 0.025)


def o_human(o, t):
    w = math.sin(t * 5)
    o.line([(-22, 40), (-26, 130)], (50, 60, 90), 0.1); o.line([(22, 40), (26, 130)], (50, 60, 90), 0.1)
    o.poly([(-48, -50), (48, -50), (40, 60), (-40, 60)], (60, 130, 220))
    o.line([(-46, -40), (-90, 10)], (60, 130, 220), 0.1); o.line([(46, -40), (90 + 8 * w, -90 + 14 * w), (104 + 12 * w, -128 + 10 * w)], (60, 130, 220), 0.1)
    o.circ(104 + 12 * w, -132 + 10 * w, 12, (240, 200, 160)); o.circ(-90, 14, 12, (240, 200, 160))
    o.circ(0, -92, 38, (240, 200, 160)); o.poly([(-38, -96), (38, -96), (30, -128), (-30, -128)], (70, 50, 40))
    for sx in (-1, 1):
        o.circ(sx * 14, -92, 4.5, (20, 20, 30))
    o.line([(-12, -76), (0, -70), (12, -76)], (160, 70, 70), 0.025)


def o_dinosaur(o, t):
    st = math.sin(t * 6) * 6
    o.poly([(-120, 30), (-170, 70), (-60, 56)], (80, 170, 90)); o.ell(0, 20, 100, 62, (90, 180, 100)); o.poly([(50, 0), (90, -90), (126, -96), (130, -60), (96, -40), (86, 14)], (90, 180, 100))
    o.ell(116, -86, 36, 26, (90, 180, 100)); o.circ(122, -94, 7, (255, 255, 255)); o.circ(124, -94, 3.5, (10, 10, 10))
    for k in range(5):
        o.poly([(-70 + k * 32, -34 + abs(k - 2) * 8), (-54 + k * 32, -64 + abs(k - 2) * 10), (-38 + k * 32, -36 + abs(k - 2) * 8)], (230, 150, 60))
    for lx, ph in ((-40, 0), (40, 3)):
        o.poly([(lx - 18, 60), (lx + 18, 60), (lx + 14 + st * (1 if ph else -1), 128), (lx - 18 + st * (1 if ph else -1), 128)], (70, 150, 80))
    o.poly([(-60, 40), (-30, 50), (-50, 110)], (70, 150, 80)) if False else None


OBJ_FUNCS = dict(beetle=o_beetle, frog=o_frog, spider=o_spider, cat=o_cat, bird=o_bird, fish=o_fish, thermometer=o_thermometer,
                 flask=o_flask, explosion=o_explosion, stopwatch=o_stopwatch, heart=o_heart, brain=o_brain, lightbulb=o_lightbulb,
                 coin=o_coin, fire=o_fire, drop=o_drop, atom=o_atom, gear=o_gear, magnifier=o_magnifier, warning=o_warning,
                 question=o_question, bolt=o_bolt, tomato=o_tomato, apple=o_apple, banana=o_banana, pizza=o_pizza, burger=o_burger,
                 coffee=o_coffee, egg=o_egg, ketchup=o_ketchup, tree=o_tree, flower=o_flower, sun=o_sun, moon=o_moon, star=o_star,
                 snowflake=o_snowflake, car=o_car, plane=o_plane, house=o_house, robot=o_robot, phone=o_phone, laptop=o_laptop, book=o_book,
                 trophy=o_trophy, key=o_key, crown=o_crown, globe=o_globe, rocket=o_rocket, tooth=o_tooth, eye=o_eye, dog=o_dog,
                 human=o_human, dinosaur=o_dinosaur)
OBJECTS = list(OBJ_FUNCS.keys())
LAYOUT = {1: [(0.5, 0.40, 1.0)], 2: [(0.30, 0.42, 0.55), (0.70, 0.42, 0.55)], 3: [(0.2, 0.46, 0.42), (0.5, 0.37, 0.46), (0.8, 0.46, 0.42)]}


ICON_STYLES = ["bob", "wiggle", "pulse", "swing"]


def icon_draw(c, img, x, y, size, t, style, par=0.6):
    k = c.k(par)
    sx, sy = c.m(x, y, par)
    sz = max(8, int(size * k))
    sc, ang, dy = 1.0, 0.0, 0.0
    if style == "bob": dy = -12 * math.sin(t * 3)
    elif style == "wiggle": ang = 8 * math.sin(t * 4)
    elif style == "pulse": sc = 1 + 0.07 * math.sin(t * 4)
    elif style == "swing": ang = 14 * math.sin(t * 2.2)
    sz = max(8, int(sz * sc))
    im = img.resize((sz, sz), Image.LANCZOS)
    if ang:
        im = im.rotate(ang, resample=Image.BICUBIC, expand=True)
    c.d.ellipse([sx - sz * 0.30, sy + sz * 0.46, sx + sz * 0.30, sy + sz * 0.54], fill=(0, 0, 0, 46))
    c.img.paste(im, (int(sx - im.width / 2), int(sy - im.height / 2 + dy)), im)


def objects(c, names=()):
    items = [n for n in names if isinstance(n, tuple) or (isinstance(n, str) and n in OBJ_FUNCS)][:3]
    for i, (n, (fx, fy, sc)) in enumerate(zip(items, LAYOUT.get(len(items), []))):
        lt = c.t - 0.12 * i
        if lt < 0:
            continue
        pop = back(clamp(lt / 0.45))
        if pop < 0.03:
            continue
        if isinstance(n, tuple):
            icon_draw(c, n[1], W * fx, H * fy, 430 * sc * pop, lt, n[2])
            continue
        u = 2.0 * sc * pop
        o = O(c, W * fx, H * fy + 9 * math.sin(c.t * 2.2 + i), u)
        OBJ_FUNCS[n](o, lt)


def rays(c, color=(255, 255, 255), n=14, alpha=40):
    cx_, cy_ = W / 2, H * 0.4
    for k in range(n):
        if k % 2:
            continue
        a0 = k / n * math.tau + c.t * 0.25; a1 = a0 + math.tau / n
        c.poly([(cx_, cy_), (cx_ + 1600 * math.cos(a0), cy_ + 1600 * math.sin(a0)), (cx_ + 1600 * math.cos(a1), cy_ + 1600 * math.sin(a1))], 0.02, color + (alpha,))


def grid(c, color=(120, 230, 230)):
    for k in range(-2, 16):
        y = (k * 100 + c.t * 20) % (H + 200) - 100
        c.d.line([0, y, W, y], fill=color + (28,), width=2)
    for k in range(0, 9):
        c.d.line([k * 100, 0, k * 100, H], fill=color + (28,), width=2)


def motes(c, n=36, color=(255, 255, 255), seed=21, size=5):
    rng = random.Random(seed)
    for k in range(n):
        x = (rng.uniform(0, W) + 18 * math.sin(c.t * 0.8 + k)) % W; y = H - ((rng.uniform(0, H) + c.t * rng.uniform(15, 50)) % H)
        r = rng.uniform(2, size); c.d.ellipse([x - r, y - r, x + r, y + r], fill=color + (130,))


ELEMENTS = {k: v for k, v in dict(mountains=mountains, hills=hills, pines=pines, sun=sun, moon=moon, stars=stars,
            clouds=clouds, birds=birds, planet=planet, earth=earth, ocean=ocean, underwater=underwater, fish=fish,
            person=person, rocket=rocket, city=city, dunes=dunes, aurora=aurora, volcano_glow=volcano_glow,
            embers=embers, snow=snow, rain=rain, lightning=lightning, objects=objects, rays=rays, grid=grid, motes=motes).items()}

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
    elif S == "stage_nature":
        sky = "meadow"; E = [("clouds", {"n": 3, "y": 0.1}), ("motes", {"color": (255, 255, 200)}), ("hills", {"color": (80, 160, 80), "base": 0.82, "par": 0.4}), ("hills", {"color": (52, 120, 62), "base": 0.9, "par": 0.8, "seed": 5})]
    elif S == "stage_lab":
        sky = "lab"; E = [("grid", {}), ("motes", {"color": (150, 255, 240)})]
    elif S == "stage_dark":
        sky = "dark"; E = [("stars", {"n": 70, "ymax": 1.0}), ("rays", {"color": (150, 120, 255), "alpha": 22})]
    elif S == "stage_warm":
        sky = "warm"; E = [("rays", {"color": (255, 255, 255), "alpha": 44}), ("motes", {"color": (255, 240, 200)})]
    elif S == "stage_bright":
        sky = "bright"; E = [("rays", {"color": (255, 255, 255), "alpha": 90}), ("motes", {"color": (255, 150, 60), "size": 7})]
    else:  # abstract
        sky = "night"; E = [("stars", {"n": 120, "ymax": 1.0}), ("planet", {"x": 0.5, "y": 0.45, "r": 110, "color": (120, 90, 200)})]
    return sky, sky_to, E


KEYWORDS = [(r"everest|climb|summit|peak|mountaineer", "climb"), (r"mountain|alps|himalaya", "mountain_day"), (r"storm|thunder", "mountain_storm"),
            (r"ocean|sea |waves|ship", "ocean_day"), (r"deep sea|underwater|reef", "underwater"), (r"rocket|launch|astronaut", "rocket_launch"),
            (r"planet|galaxy|space|stars", "space"), (r"earth|world", "earth_space"), (r"city|building|skyscraper", "city_day"),
            (r"forest|jungle", "forest"), (r"desert|sand|dune", "desert"), (r"arctic|polar|iceberg", "arctic"),
            (r"volcano|lava|eruption", "volcano")]
OBJ_KEYWORDS = [(r"ketchup|sauce|condiment", "ketchup"), (r"tomato", "tomato"), (r"apple", "apple"), (r"banana", "banana"), (r"pizza", "pizza"),
                (r"burger|sandwich|fast food", "burger"), (r"coffee|tea\b|caffeine|cafe", "coffee"), (r"\begg", "egg"), (r"fruit|berry|berries|vegetable|plant|seed", "tomato"),
                (r"tree|wood|paper", "tree"), (r"flower|bloom|pollen", "flower"), (r"\bsun\b|sunshine|daylight", "sun"), (r"moon|lunar", "moon"), (r"\bstar\b|famous|best|record", "star"),
                (r"snow|ice\b|cold|freez|winter", "snowflake"), (r"\bcars?\b|drive|road|vehicle|speeding", "car"), (r"plane|flight|airline|aircraft", "plane"), (r"house|home|room|live in", "house"),
                (r"robot|\bai\b|artificial|computer", "robot"), (r"phone|smartphone|app\b|text message", "phone"), (r"laptop|internet|online|code|website", "laptop"),
                (r"book|read|story|wrote|writer|history", "book"), (r"win|won|champion|prize|trophy|medal", "trophy"), (r"key\b|unlock|password", "key"), (r"king|queen|royal|crown|empire", "crown"),
                (r"globe|country|countries|planet earth|worldwide|across the world", "globe"), (r"tooth|teeth|dentist", "tooth"), (r"\beyes?\b|vision|see\b|sight|watch", "eye"),
                (r"\bdogs?\b|puppy|wolf", "dog"), (r"dinosaur|prehistoric|fossil|jurassic", "dinosaur"), (r"people|person|human|man\b|woman|children|kids|everyone", "human"),
(r"beetle|insect|\bbug|ant\b|ants\b|bee\b|bees\b", "beetle"), (r"frog|toad", "frog"), (r"spider|web", "spider"), (r"\bcat|kitten|\bdog|animal|predator", "cat"),
                (r"bird|fly|flies|wing", "bird"), (r"fish|shark|whale", "fish"), (r"hot|boil|degree|heat|temperature|warm|°", "thermometer"),
                (r"chemic|liquid|mix|enzyme|acid|react|experiment|lab", "flask"), (r"explo|blast|bang|shoot|fire[sd]?\b|spray|burst|cannon", "explosion"),
                (r"second|fast|speed|quick|time|minute|hour|instant", "stopwatch"), (r"heart|love|blood", "heart"), (r"brain|smart|think|mind|memory|intelligen", "brain"),
                (r"idea|light|invent|discover|clever|solution", "lightbulb"), (r"money|coin|cost|price|rich|dollar|\$", "coin"), (r"burn|flame|ignite|fire", "fire"),
                (r"water|rain|drop|liquid|sweat", "drop"), (r"atom|energy|particle|molecule|physics", "atom"), (r"machine|engine|gear|mechanic|design|system", "gear"),
                (r"search|find|found|study|scientist|research|look", "magnifier"), (r"danger|warning|deadly|toxic|poison|risk|safe", "warning"),
                (r"why|mystery|secret|unknown|question|how", "question"), (r"lightning|electric|shock|power|volt", "bolt")]


def infer_objects(text):
    t = text.lower(); out = []
    for pat, name in OBJ_KEYWORDS:
        if re.search(pat, t) and name not in out:
            out.append(name)
    return out[:2]


def resolve_objects(tokens, say):
    """hand-animated name -> as is; any other noun/emoji -> ('icon', image, style) from the ~11,000-icon libraries"""
    out = []
    for n, tok in enumerate(tokens):
        tok = str(tok).strip()
        if tok.lower() in OBJ_FUNCS:
            out.append(tok.lower()); continue
        if icons is None:
            continue
        try:
            r = icons.resolve(tok)
            img = icons.get_image(*r) if r else None
        except Exception as e:
            print("  icon lookup failed for", tok, "->", type(e).__name__, e, flush=True); continue
        if img is not None:
            out.append(("icon", img, ICON_STYLES[(len(tok) + n) % 4], tok))
    if not out and icons is not None:           # AI gave nothing usable: try the sentence's own nouns
        words = sorted(set(re.findall(r"[A-Za-z]{4,}", say)), key=len, reverse=True)[:7]
        for w in words:
            try:
                r = icons.resolve(w); img = icons.get_image(*r) if r else None
            except Exception:
                continue
            if img is not None:
                out.append(("icon", img, ICON_STYLES[len(w) % 4], w))
            if len(out) >= 2:
                break
    return out[:3]


def make_spec(seg, i):
    say = seg.get("say", "")
    raw_objs = seg.get("objects") or []
    if isinstance(raw_objs, str):
        raw_objs = [raw_objs]
    objs = resolve_objects(raw_objs[:3], say) if raw_objs else []
    scene = str(seg.get("scene", "")).strip().lower()
    if scene not in SCENES:
        scene = next((v for k, v in KEYWORDS if re.search(k, say.lower())), "")
    if not scene:
        scene = ["stage_nature", "stage_lab", "stage_warm", "stage_dark", "stage_bright"][i % 5]
    if not objs and scene.startswith("stage_"):
        objs = infer_objects(say) or resolve_objects([], say) or ["question"]
    extras = [e for e in (seg.get("extras") or []) if e in EXTRAS][:2]
    cam = seg.get("camera") if seg.get("camera") in CAMERAS else CAMERAS[i % 6]
    txt = re.sub(r"[^\w\s,.%$°/+!?-]", "", str(seg.get("big_text", ""))).upper().strip()[:16]
    sky, sky_to, els = build(scene)
    have = {n for n, _ in els}
    if objs:
        els.append(("objects", {"names": objs}))
    for e in extras:
        if e in have or e in ("shake",):
            continue
        if e == "person":
            els.append(("person", {"x0": 0.15, "x1": 0.8, "action": "walk"}))
        elif e in ("sun", "snow", "rain", "birds", "moon", "stars", "clouds", "fish", "planet", "lightning", "embers"):
            els.append((e, {}))
        have.add(e)
    return {"scene": scene, "sky": sky, "sky_to": sky_to, "els": els, "cam": cam, "text": txt, "shake": "shake" in extras, "idx": i, "objs": [o if isinstance(o, str) else o[3] for o in objs]}


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

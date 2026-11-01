"""S8 Paper Cut-out Documentary ("paper-doc") — generic renderer.

The look of the YT03 Bombardier Beetle reel, for any topic: kraft-paper page (or a dark X-ray page),
"FIG. 03 · <HEADER>" strip, a two-line title top-left with one line in red, real photos turned into
paper cut-outs (cream cut edge, grain, drop shadow), paper props (stamps, sticky notes, tickets, calendar,
ruler, flasks, thermometer, bursts, name tags), layered paper scenery, effects (spray, smoke, rings,
crosshair, glows, sparkles), a slow push-in per scene, and word-synced sound (whoosh per scene, thud per
stamp, pops, sizzles, riser + boom).

Everything that changes per video lives in a scene plan (JSON) — see ../REFERENCE.md.

  python3 scripts/paper_doc.py render PLAN.json              # full render -> plan "out"
  python3 scripts/paper_doc.py stills PLAN.json OUTDIR 1.0 3.5 …   # check frames first
  python3 scripts/paper_doc.py contact PLAN.json OUTDIR      # one still per scene, as a sheet
  python3 scripts/paper_doc.py words WORDS.json [FROM TO]    # word times to write the plan
  python3 scripts/paper_doc.py cutout PHOTO OUT.png [--rotate 90]   # Apple Vision cut-out
  python3 scripts/paper_doc.py selftest OUTDIR [PHOTO]       # 6 s test render (checks the install)

Sounds: the AST BTS library (02_Audio/Edit sound) when the engine runs inside that project, otherwise the
CC0 sounds in ../sounds, otherwise generated here (sizzle, boil, boom, riser, music drone).
"""
import json
import math
import random
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
PACK = HERE.parent                      # the skill folder (sounds/, prompts/, examples/)
PACK_SOUNDS = PACK / "sounds"


def find_ast_root():
    """The AST BTS project root (has 02_Audio/Edit sound), searched upwards from here and from the cwd."""
    for start in (HERE, Path.cwd().resolve()):
        for p in (start, *start.parents):
            if (p / "02_Audio/Edit sound").is_dir():
                return p
    return None


ROOT = find_ast_root()
SFX_DIR = ROOT / "02_Audio/Edit sound" if ROOT else None
SR = 48000

W, H, FPS = 1080, 1920, 30          # set from the plan

FUT = "/System/Library/Fonts/Supplemental/Futura.ttc"
MONO = "/System/Library/Fonts/Menlo.ttc"
HEAD = "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MONOF = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

INK = (28, 22, 16)
RED = (196, 36, 28)
CREAM = (244, 234, 212)
KRAFT = (204, 178, 134)
NAVY = (24, 36, 62)
XRAY = (196, 226, 255)
YELLOW = (250, 222, 88)
GREEN = (46, 140, 70)
COLORS = {"ink": INK, "red": RED, "cream": CREAM, "green": GREEN, "yellow": YELLOW, "navy": NAVY,
          "xray": XRAY, "white": (255, 255, 255), "orange": (224, 104, 38), "blue": (90, 170, 255)}

# named sounds: AST library file (start, duration) -> else ../sounds/<name>.mp3 (CC0) -> else synth(name)
SOUNDS = {
    "whoosh": ("15 Epidemic Top 50 (ES)/12 Whoosh Soft (ES).mp3", 0, 0.8),
    "thud": ("16 Free Shareable SFX (CC0)/34 Hit - Body Thud (CC0).mp3", 0, None),
    "pop": ("15 Epidemic Top 50 (ES)/11 Pop Small (ES).mp3", 0, None),
    "mpop": ("15 Epidemic Top 50 (ES)/28 Mouth Pop (ES).mp3", 0, None),
    "question": ("16 Free Shareable SFX (CC0)/24 UI - Question (CC0).mp3", 0, None),
    "tick": ("16 Free Shareable SFX (CC0)/15 UI - Click (CC0).mp3", 0, None),
    "ding": ("16 Free Shareable SFX (CC0)/30 Hit - Bell Ding (CC0).mp3", 0, None),
    "sizzle": ("12 Home Kitchen Daily Life/Frying Intense (ES).mp3", 1.0, 2.2),
    "boil": ("12 Home Kitchen Daily Life/Boiling Water (ES).mp3", 5.0, 3.0),
    "boom": ("15 Epidemic Top 50 (ES)/26 Boom Cinematic Impact (ES).mp3", 0, None),
    "riser": ("15 Epidemic Top 50 (ES)/02 Riser Suspense Short (ES).mp3", 0, None),
    "glitch": ("16 Free Shareable SFX (CC0)/19 UI - Glitch (CC0).mp3", 0, None),
}
# default sound per element type: (sound, dB, offset s)
DEFAULT_SFX = {"stamp": ("thud", -6, 0.1), "sticky": ("question", -8, 0), "name_tag": ("pop", -8, 0),
               "flask": ("pop", -10, 0), "burst": ("mpop", -6, 0), "warning": ("question", -10, 0),
               "spray": ("sizzle", -14, 0), "flash": ("mpop", -8, 0), "bubbles": ("boil", -14, 0)}


@lru_cache(maxsize=None)
def font(kind, size):
    if kind == "head":
        return ImageFont.truetype(HEAD, size)
    if kind == "bold":
        return ImageFont.truetype(BOLD, size)
    if kind == "mono":
        return ImageFont.truetype(MONOF, size)
    raise KeyError(kind)


def col(c, default=RED):
    if c is None:
        return default
    if isinstance(c, str):
        return COLORS[c]
    return tuple(c)


# ---------------------------------------------------------------- easing
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_out(x):
    return 1 - (1 - clamp(x)) ** 3


def ease_in_out(x):
    x = clamp(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_back(x):
    x = clamp(x)
    c1 = 1.9
    return 1 + (c1 + 1) * (x - 1) ** 3 + c1 * (x - 1) ** 2


def prog(t, t0, d):
    return clamp((t - t0) / d) if d > 0 else float(t >= t0)


def keyframes(v, lt):
    """Number, or [[t, v], …] (linear between keys, held outside)."""
    if not isinstance(v, list):
        return v
    if lt <= v[0][0]:
        return v[0][1]
    for (t0, a), (t1, b) in zip(v, v[1:]):
        if lt <= t1:
            return a + (b - a) * ease_in_out((lt - t0) / max(1e-6, t1 - t0))
    return v[-1][1]


# ---------------------------------------------------------------- pages + title
def make_page(dark=False):
    rng = np.random.default_rng(11 if dark else 3)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    base = np.array((30, 34, 40) if dark else KRAFT, np.float32)
    noise = rng.normal(0, 5 if dark else 7, (H, W)).astype(np.float32)
    fib = np.zeros((H, W), np.float32)
    for _ in range(int(260 * H / 1920)):
        y = rng.integers(0, H)
        x0 = rng.integers(0, W)
        ln = rng.integers(60, 260)
        fib[y, x0:x0 + ln] += rng.uniform(-10, 10)
    fib = np.array(Image.fromarray((fib + 128).clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2)),
                   np.float32) - 128
    d = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2)
    vig = 1 - 0.28 * np.clip(d, 0, 1.3) ** 2
    fold = np.ones((H, W), np.float32)
    fx, fy = W * 0.5, H * 0.5
    fold -= 0.07 * np.exp(-((xx - fx) / 3) ** 2) - 0.035 * np.exp(-((xx - fx - 6) / 5) ** 2)
    fold -= 0.05 * np.exp(-((yy - fy) / 3) ** 2) - 0.03 * np.exp(-((yy - fy - 6) / 5) ** 2)
    img = (base[None, None, :] + noise[..., None] + fib[..., None]) * (vig * fold)[..., None]
    page = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(page, "RGBA")
    grid = (255, 255, 255, 14) if dark else (90, 60, 30, 16)
    for x in range(0, W, 60):
        dr.line((x, 0, x, H), fill=grid, width=1)
    for y in range(0, H, 60):
        dr.line((0, y, W, y), fill=grid, width=1)
    lc = (220, 220, 220, 120) if dark else (60, 40, 20, 110)
    dr.line((60, 196, W - 60, 196), fill=lc, width=2)
    for i, x in enumerate(range(60, W - 59, 24)):
        dr.line((x, 196, x, 186 if i % 5 else 176), fill=lc, width=2)
    return page


def header(canvas, fig, dark, text):
    d = ImageDraw.Draw(canvas)
    c = (210, 210, 210) if dark else (70, 50, 30)
    d.text((62, 212), f"FIG. {fig:02d}", font=font("mono", 22), fill=c)
    tw = d.textlength(text, font=font("mono", 22))
    d.text((W - 62 - tw, 212), text, font=font("mono", 22), fill=c)


def draw_title(canvas, l1, l2, red_line, lt, dark):
    """Two-line title top-left; line `red_line` (1 or 2, 0 = none) in red. Lines slide up and fade in."""
    size = 168
    while size > 90:
        f = font("head", size)
        if max(f.getlength(l1 or ""), f.getlength(l2 or "")) <= W - 150:
            break
        size -= 6
    f = font("head", size)
    base = CREAM if dark else INK
    y = 270
    for i, line in enumerate([l1, l2]):
        if not line:
            continue
        p = ease_out(prog(lt, 0.05 + 0.12 * i, 0.32))
        if p <= 0:
            y += int(size * 0.98)
            continue
        c = RED if (i + 1) == red_line else base
        layer = Image.new("RGBA", (W, size + 40), (0, 0, 0, 0))
        ImageDraw.Draw(layer).text((72, 0), line, font=f, fill=c + (255,))
        if p < 1:
            layer.putalpha(layer.getchannel("A").point(lambda v: int(v * p)))
        canvas.alpha_composite(layer, (0, int(y + 36 * (1 - p))))
        y += int(size * 0.98)


# ---------------------------------------------------------------- cut-out helpers
SS = 2
PAD = 40


def finish(img, outline=7, shadow=True, grain=True):
    """Paper cut-out: cream cut edge + paper grain + soft drop shadow. img is RGBA at final size."""
    w, h = img.size
    big = Image.new("RGBA", (w + PAD * 2, h + PAD * 2), (0, 0, 0, 0))
    big.alpha_composite(img, (PAD, PAD))
    a = big.getchannel("A")
    if outline:
        edge = a.filter(ImageFilter.MaxFilter(outline * 2 + 1)).filter(ImageFilter.GaussianBlur(0.8))
        cut = Image.new("RGBA", big.size, CREAM + (255,))
        cut.putalpha(edge)
        cut.alpha_composite(big)
        big = cut
        a = big.getchannel("A")
    if grain:
        rng = np.random.default_rng(w * 7 + h)
        g = (rng.normal(1.0, 0.035, (big.size[1], big.size[0]))).clip(0.85, 1.1)
        arr = np.array(big).astype(np.float32)
        arr[..., :3] *= g[..., None]
        big = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    if shadow:
        sh = Image.new("RGBA", big.size, (0, 0, 0, 0))
        sh.putalpha(a.filter(ImageFilter.GaussianBlur(10)).point(lambda v: int(v * 0.42)))
        shifted = Image.new("RGBA", big.size, (0, 0, 0, 0))
        shifted.paste(sh, (9, 14))
        shifted.alpha_composite(big)
        big = shifted
    return big


def down(img):
    return img.resize((img.width // SS, img.height // SS), Image.LANCZOS)


def place(canvas, img, cx, cy, scale=1.0, alpha=1.0, rot=0.0):
    if scale <= 0.01 or alpha <= 0.01:
        return
    im = img
    if rot:
        im = im.rotate(rot, resample=Image.BICUBIC, expand=True)
    if abs(scale - 1) > 1e-3:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.BICUBIC)
    if alpha < 1:
        im = im.copy()
        im.putalpha(im.getchannel("A").point(lambda v: int(v * alpha)))
    canvas.alpha_composite(im, (int(cx - im.width / 2), int(cy - im.height / 2)))


def bez(p0, p1, p2, n=24):
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]) for t in np.linspace(0, 1, n)]


# ---------------------------------------------------------------- subjects (photo cut-outs)
class Subjects:
    """plan["subjects"] = {name: {"image": path, "rotate": deg, "anchors": {"tail": [x, y]}}}.
    A photo without transparency is cut out once with Apple Vision (cached next to it as *_cutout.png)."""

    def __init__(self, spec, resolve):
        self.spec = spec
        self.resolve = resolve
        self._src = {}

    def src(self, name):
        if name not in self._src:
            s = self.spec[name]
            path = self.resolve(s["image"])
            img = Image.open(path)
            if img.mode != "RGBA" or img.getchannel("A").getextrema()[0] == 255:
                cut = path.with_name(path.stem + "_cutout.png")
                if not cut.exists():
                    vision_cutout(path, cut)
                img = Image.open(cut)
            img = img.convert("RGBA")
            bbox = img.getchannel("A").point(lambda v: 255 if v > 20 else 0).getbbox()
            if bbox and s.get("crop", True) and not s.get("anchors"):
                img = img.crop(bbox)
            if s.get("rotate"):
                img = img.rotate(s["rotate"], expand=True, resample=Image.BICUBIC)
            self._src[name] = img
        return self._src[name]

    def anchor(self, name, key):
        """Anchor in source pixels: named anchor, 'center', or [rx, ry] fraction of the cut-out."""
        img = self.src(name)
        if isinstance(key, (list, tuple)):
            return key[0] * img.width, key[1] * img.height
        if key == "center":
            return img.width / 2, img.height / 2
        return tuple(self.spec[name]["anchors"][key])

    @lru_cache(maxsize=None)
    def image(self, name, mode, height):
        src = self.src(name)
        k = height / src.height
        img = src.resize((max(1, int(src.width * k)), max(1, int(src.height * k))), Image.LANCZOS)
        if mode == "silhouette":
            sil = Image.new("RGBA", img.size, (20, 16, 12, 255))
            sil.putalpha(img.getchannel("A"))
            return finish(sil, outline=0, shadow=False)
        if mode == "xray":
            return finish(xray_look(img), outline=0, shadow=False, grain=False)
        rgb = img.convert("RGB")
        if self.spec[name].get("enhance", True):
            rgb = ImageEnhance.Contrast(ImageEnhance.Color(rgb).enhance(1.15)).enhance(1.08)
            rgb = rgb.filter(ImageFilter.UnsharpMask(radius=2, percent=80, threshold=2))
        out = rgb.convert("RGBA")
        out.putalpha(img.getchannel("A"))
        return finish(out, outline=7, shadow=True)


def xray_look(img):
    """Glowing blue line drawing of a cut-out for dark 'inside the body' pages."""
    a = img.getchannel("A")
    g = img.convert("L")
    edges = g.filter(ImageFilter.FIND_EDGES).point(lambda v: min(255, v * 3))
    edges = ImageChops.multiply(edges, a.filter(ImageFilter.MinFilter(5)))
    rim = ImageChops.subtract(a.filter(ImageFilter.MaxFilter(5)), a.filter(ImageFilter.MinFilter(5)))
    lines = ImageChops.lighter(edges.point(lambda v: int(v * 0.7)), rim)
    out = Image.new("RGBA", img.size, XRAY + (0,))
    fill = a.point(lambda v: int(v * 0.10))
    out.putalpha(ImageChops.lighter(fill, lines))
    glow = out.filter(ImageFilter.GaussianBlur(6))
    glow.alpha_composite(out)
    return glow


def vision_cutout(photo, out_png):
    swift = HERE / "subject_cutout.swift"
    r = subprocess.run(["swift", str(swift), str(photo), str(out_png)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"cut-out failed for {photo}: {r.stdout}{r.stderr}")


# ---------------------------------------------------------------- props (static images, cached)
@lru_cache(maxsize=None)
def stamp(text, color=RED, size=92, rot=-8):
    f = font("head", size)
    tw = int(f.getlength(text))
    w, h = tw + 70, size + 50
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((4, 4, w - 5, h - 5), 14, fill=CREAM + (225,), outline=color + (255,), width=8)
    d.text((35, 18), text, font=f, fill=color + (255,))
    rng = np.random.default_rng(len(text) * 31)
    holes = (rng.random((h, w)) > 0.16).astype(np.uint8) * 255
    hm = Image.fromarray(holes).filter(ImageFilter.GaussianBlur(0.7))
    img.putalpha(ImageChops.multiply(img.getchannel("A"), hm))
    return img.rotate(rot, resample=Image.BICUBIC, expand=True)


@lru_cache(maxsize=None)
def sticky_note(text="?"):
    w = h = 460
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.polygon([(0, 0), (w, 0), (w, h - 40), (w - 40, h), (0, h)], fill=YELLOW + (255,))
    d.polygon([(w, h - 40), (w - 40, h), (w - 46, h - 46)], fill=(214, 186, 60, 255))
    size = 380
    while font("head", size).getlength(text) > w - 60:
        size -= 10
    f = font("head", size)
    d.text(((w - f.getlength(text)) / 2, (h - size * 1.15) / 2 + 10), text, font=f, fill=RED + (255,))
    return finish(img, outline=0)


@lru_cache(maxsize=None)
def ticket(strike="CARTOON", ok="REAL", number="0013", top="ADMIT ONE"):
    w, h = 820, 330
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, w, h), 18, fill=CREAM + (255,))
    for y in range(20, h, 34):
        d.ellipse((-12, y - 10, 12, y + 10), fill=(0, 0, 0, 0))
        d.ellipse((w - 12, y - 10, w + 12, y + 10), fill=(0, 0, 0, 0))
    d.line((300, 26, 300, h - 26), fill=(150, 120, 80, 255), width=3)
    d.text((40, 36), top, font=font("mono", 30), fill=(110, 80, 50))
    d.text((40, 90), "NO.", font=font("mono", 26), fill=(110, 80, 50))
    d.text((40, 124), number, font=font("head", 120), fill=INK)
    size = 96
    while max(font("head", size).getlength(strike), font("head", size).getlength(ok) + 130) > w - 360:
        size -= 4
    f = font("head", size)
    d.text((340, 40), strike, font=f, fill=(120, 110, 100))
    d.line((334, 40 + size * 0.62, 340 + f.getlength(strike) + 8, 40 + size * 0.62), fill=RED, width=12)
    d.text((340, 170), ok, font=f, fill=INK)
    x0 = 360 + f.getlength(ok)
    d.line((x0, 230, x0 + 34, 266, x0 + 100, 180), fill=GREEN, width=18, joint="curve")
    return finish(img, outline=0)


@lru_cache(maxsize=None)
def calendar(top="2026", big="AAJ"):
    w, h = 500, 560
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 30, w, h), 16, fill=(250, 246, 236, 255))
    d.rounded_rectangle((0, 30, w, 180), 16, fill=RED + (255,))
    d.rectangle((0, 150, w, 180), fill=RED + (255,))
    for x in (120, 380):
        d.rounded_rectangle((x - 14, 0, x + 14, 70), 10, fill=(90, 90, 95, 255))
    bs = 250
    while font("head", bs).getlength(big) > w - 50:
        bs -= 10
    for txt, y, c, fn in [(top, 50, CREAM, font("head", 110)), (big, 210 + (250 - bs) // 2, INK, font("head", bs))]:
        d.text(((w - fn.getlength(txt)) / 2, y), txt, font=fn, fill=c)
    return finish(img, outline=0)


@lru_cache(maxsize=None)
def ruler(units="CM", count=4):
    w, h = 940, 170
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, w, h), 10, fill=(232, 206, 150, 255))
    step = (w - 230) / count            # leaves room for the units label after the last number
    for i in range(0, count * 10 + 1):
        x = 70 + i * step / 10
        ln = 70 if i % 10 == 0 else (44 if i % 5 == 0 else 26)
        d.line((x, 0, x, ln), fill=INK, width=4 if i % 10 == 0 else 2)
        if i % 10 == 0:
            d.text((x - 14, 78), str(i // 10), font=font("head", 60), fill=INK)
    d.text((w - 110, 100), units, font=font("head", 50), fill=INK)
    return finish(img, outline=0)


@lru_cache(maxsize=None)
def flask(label, liquid):
    w, h = 300, 520
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((110, 0, 190, 40), 8, fill=(60, 60, 66, 255))
    d.rectangle((118, 40, 182, 120), fill=(225, 238, 240, 255))
    d.rounded_rectangle((20, 110, 280, 520), 40, fill=(225, 238, 240, 255))
    d.rounded_rectangle((34, 230, 266, 506), 30, fill=liquid + (255,))
    d.rounded_rectangle((44, 270, 256, 430), 10, fill=CREAM + (255,))
    lines = label.split("\n")
    y = 350 - len(lines) * 34
    for ln in lines:
        size = 64
        while font("head", size).getlength(ln) > 196:
            size -= 4
        fl = font("head", size)
        d.text(((w - fl.getlength(ln)) / 2, y), ln, font=fl, fill=INK)
        y += 68
    d.line((50, 130, 50, 220), fill=(255, 255, 255, 200), width=10)
    return finish(img)


@lru_cache(maxsize=None)
def warning(mark="!"):
    w, h = 520, 460
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.polygon([(w / 2, 0), (w, h), (0, h)], fill=INK + (255,))
    d.polygon([(w / 2, 46), (w - 46, h - 26), (46, h - 26)], fill=YELLOW + (255,))
    f = font("head", 300)
    d.text(((w - f.getlength(mark)) / 2, 100), mark, font=f, fill=INK)
    return finish(img)


@lru_cache(maxsize=64)
def thermometer(level, labels=("0", "50", "100°C")):
    w, h = 420, 1000
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((105, 0, 195, 860), 45, fill=(236, 242, 244, 255))
    d.ellipse((60, 800, 240, 980), fill=(236, 242, 244, 255))
    top = 820 - 740 * level
    d.rounded_rectangle((130, top, 170, 860), 20, fill=RED + (255,))
    d.ellipse((84, 824, 216, 956), fill=RED + (255,))
    n = len(labels)
    for i, lab in enumerate(labels):
        y = 820 - 740 * (i / max(1, n - 1))
        d.line((195, y, 230, y), fill=INK, width=6)
        d.text((236, y - 30), lab, font=font("head", 52), fill=INK)
    return finish(img)


@lru_cache(maxsize=None)
def burst(text, size=1.0, color=YELLOW, seed=2, spikes=14):
    rng = random.Random(seed)
    w = h = int(700 * size)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pts = []
    for i in range(spikes * 2):
        a = math.pi * 2 * i / (spikes * 2)
        r = (w / 2 - 6) * (1 if i % 2 == 0 else rng.uniform(0.55, 0.68))
        pts.append((w / 2 + math.cos(a) * r, h / 2 + math.sin(a) * r))
    d.polygon(pts, fill=RED + (255,))
    d.polygon([(w / 2 + (x - w / 2) * 0.86, h / 2 + (y - h / 2) * 0.86) for x, y in pts], fill=color + (255,))
    fs = int(220 * size)
    while font("head", fs).getlength(text) > w * 0.62:
        fs -= 6
    f = font("head", fs)
    d.text(((w - f.getlength(text)) / 2, h / 2 - fs * 0.62), text, font=f, fill=RED)
    return finish(img, outline=0)


@lru_cache(maxsize=None)
def name_tag(line1, line2="", sub=""):
    w, h = 820, 360
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.polygon([(90, 0), (w, 0), (w, h), (90, h), (0, h / 2)], fill=CREAM + (255,))
    d.ellipse((50, h / 2 - 22, 94, h / 2 + 22), fill=(0, 0, 0, 0), outline=(150, 120, 80, 255), width=6)
    size = 104
    while max(font("head", size).getlength(line1), font("head", size).getlength(line2)) > w - 180:
        size -= 4
    d.text((140, 40), line1, font=font("head", size), fill=INK)
    d.text((140, 160), line2, font=font("head", size), fill=RED)
    if sub:
        d.text((150, 292), sub, font=font("mono", 30), fill=(110, 80, 50))
    return finish(img, outline=0)


@lru_cache(maxsize=None)
def firecracker(label="BOMB"):
    w, h = 300, 720
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.line(bez((150, 230), (60, 120), (130, 20)), fill=(70, 60, 50, 255), width=10)
    d.rounded_rectangle((40, 220, 260, 700), 24, fill=RED + (255,))
    for y in (260, 640):
        d.rectangle((40, y, 260, y + 30), fill=(230, 180, 60, 255))
    fb = font("head", 66)
    while fb.getlength(label) > 200:
        fb = font("head", fb.size - 4)
    d.text(((w - fb.getlength(label)) / 2, 420), label, font=fb, fill=CREAM)
    return finish(img)


@lru_cache(maxsize=None)
def ant(charred=False, size=1.0):
    """Small drawn ant (top view, head up) — the stock 'enemy' / 'victim' for insect topics."""
    s = SS * size
    img = Image.new("RGBA", (int(300 * s), int(420 * s)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    body = (40, 28, 22) if charred else (112, 52, 26)
    leg = (30, 22, 18) if charred else (82, 38, 20)
    S = lambda pts: [(x * s, y * s) for x, y in pts]
    for side in (-1, 1):
        d.line(S(bez((150 + 12 * side, 70), (150 + 60 * side, 20), (150 + 110 * side, 30))), fill=leg, width=int(5 * s))
        for hip, knee, foot in [((145, 135), (90, 110), (40, 70)), ((145, 150), (80, 165), (30, 200)),
                                ((148, 165), (95, 230), (60, 300))]:
            pts = [(150 + (p[0] - 150) * (-side), p[1]) for p in (hip, knee, foot)]
            d.line(S(pts), fill=leg, width=int(7 * s), joint="curve")
    d.ellipse(S([(116, 44), (184, 112)]), fill=body)
    d.ellipse(S([(126, 104), (174, 182)]), fill=body)
    d.ellipse(S([(138, 178), (162, 202)]), fill=body)
    d.ellipse(S([(100, 196), (200, 360)]), fill=body)
    if not charred:
        d.ellipse(S([(124, 220), (150, 300)]), fill=(150, 80, 45))
    d.line(S([(140, 60), (130, 40)]), fill=leg, width=int(5 * s))
    d.line(S([(160, 60), (170, 40)]), fill=leg, width=int(5 * s))
    return finish(down(img))


@lru_cache(maxsize=None)
def photo_card(path, width=620, caption=""):
    """Polaroid-style print of a full photo (no cut-out), for places, screenshots, people."""
    src = Image.open(path).convert("RGB")
    k = width / src.width
    ph = src.resize((width, int(src.height * k)), Image.LANCZOS)
    cap_h = 110 if caption else 40
    img = Image.new("RGBA", (width + 60, ph.height + 30 + cap_h), (250, 246, 236, 255))
    img.paste(ph, (30, 30))
    if caption:
        f = font("head", 64)
        while f.getlength(caption) > width:
            f = font("head", f.size - 4)
        ImageDraw.Draw(img).text(((img.width - f.getlength(caption)) / 2, ph.height + 46), caption, font=f, fill=INK)
    return finish(img, outline=0)


# ---------------------------------------------------------------- layered paper scenery
GREENS = [(70, 116, 50), (96, 142, 60), (58, 96, 42), (124, 160, 68), (84, 130, 56)]


def wavy_top(rng, y0, amp, x0=-80, x1=None, step=60):
    x1 = W + 80 if x1 is None else x1
    return [(x, y0 + rng.uniform(-amp, amp)) for x in range(x0, x1 + step, step)]


def blade(d, rng, bx, by, h, wbase, lean, c):
    n = 8
    left, right = [], []
    for i in range(n + 1):
        u = i / n
        x = bx + lean * h * u * u
        y = by - h * u
        half = wbase * (1 - u) * 0.5 + 0.5
        left.append((x - half, y))
        right.append((x + half, y))
    d.polygon(left + right[::-1], fill=c + (255,))


def tuft(d, rng, cx, cy, n, hmin, hmax, wbase=22, spread=60):
    for _ in range(n):
        blade(d, rng, cx + rng.uniform(-spread, spread), cy + rng.uniform(-6, 10), rng.uniform(hmin, hmax),
              rng.uniform(wbase * 0.7, wbase * 1.2), rng.uniform(-0.35, 0.35), rng.choice(GREENS))


def leaf(rng, length, c):
    w = int(length * 0.46)
    img = Image.new("RGBA", (length, w), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pts = [(i / 20 * length, w / 2 - math.sin(i / 20 * math.pi) * w * 0.46) for i in range(21)]
    pts += [((1 - i / 20) * length, w / 2 + math.sin((1 - i / 20) * math.pi) * w * 0.46) for i in range(21)]
    d.polygon(pts, fill=c + (255,))
    dark = tuple(max(0, v - 40) for v in c)
    d.line((6, w / 2, length - 4, w / 2), fill=dark + (255,), width=4)
    for k in range(1, 5):
        x = length * k / 5.5
        d.line((x, w / 2, x + length * 0.1, w / 2 - w * 0.3), fill=dark + (200,), width=3)
        d.line((x, w / 2, x + length * 0.1, w / 2 + w * 0.3), fill=dark + (200,), width=3)
    return img


@lru_cache(maxsize=None)
def scenery_back(seed):
    """Ground seen from above with a grass horizon behind the subject (torn paper layers)."""
    rng = random.Random(seed)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    hy = int(H * 0.41)
    far = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(far)
    d.polygon(wavy_top(rng, hy, 14) + [(W + 80, H + 50), (-80, H + 50)], fill=(108, 76, 46, 255))
    for x in range(-40, W + 60, 34):
        tuft(d, rng, x, hy + 10 + rng.uniform(-8, 8), 3, 50, 120, 20, 18)
    layer.alpha_composite(finish(far, outline=5, grain=True), (-PAD, -PAD))
    near = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(near)
    d.polygon(wavy_top(rng, hy + 80, 20, step=45) + [(W + 80, H + 50), (-80, H + 50)], fill=(150, 110, 68, 255))
    for _ in range(900):
        x, y = rng.uniform(0, W), rng.uniform(hy + 90, H)
        r = rng.uniform(1.5, 4.5)
        d.ellipse((x - r, y - r, x + r, y + r), fill=rng.choice([(120, 86, 52), (176, 134, 88), (100, 72, 44)]) + (255,))
    for _ in range(5):
        x, y = rng.uniform(40, W - 40), rng.uniform(hy + 170, H - 120)
        for _ in range(9):
            r = rng.uniform(24, 60)
            ox, oy = rng.uniform(-60, 60), rng.uniform(-40, 40)
            d.ellipse((x + ox - r, y + oy - r, x + ox + r, y + oy + r), fill=rng.choice(GREENS[1:4]) + (255,))
    layer.alpha_composite(finish(near, outline=6, grain=True), (-PAD, -PAD))
    for _ in range(9):
        r = rng.uniform(16, 40)
        peb = Image.new("RGBA", (int(r * 2.4), int(r * 2)), (0, 0, 0, 0))
        g = rng.randint(140, 185)
        ImageDraw.Draw(peb).ellipse((0, 0, r * 2.4 - 1, r * 2 - 1), fill=(g, g - 6, g - 14, 255))
        place(layer, finish(peb, outline=3), rng.uniform(30, W - 30), rng.uniform(hy + 90, H - 80), rot=rng.uniform(0, 180))
    tw = Image.new("RGBA", (420, 60), (0, 0, 0, 0))
    td = ImageDraw.Draw(tw)
    td.line((10, 34, 200, 28, 410, 38), fill=(92, 62, 38, 255), width=16, joint="curve")
    td.line((200, 28, 270, 6), fill=(92, 62, 38, 255), width=9)
    place(layer, finish(tw, outline=4), rng.uniform(160, 380), rng.uniform(H * 0.78, H * 0.88), rot=rng.uniform(-25, 25))
    for _ in range(6):
        c = rng.choice([(110, 150, 60), (150, 168, 70), (186, 150, 60), (160, 96, 44)])
        lf = finish(leaf(rng, int(rng.uniform(150, 240)), c), outline=4)
        place(layer, lf, rng.choice([rng.uniform(60, 300), rng.uniform(W - 300, W - 60)]), rng.uniform(hy + 110, H - 100),
              rot=rng.uniform(0, 360))
    return layer


@lru_cache(maxsize=None)
def scenery_front(seed):
    rng = random.Random(seed * 7 + 1)
    front = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(front)
    for cx in list(range(-60, 300, 46)) + list(range(W - 280, W + 80, 46)):
        edge = min(abs(cx - (-60)), abs(cx - (W + 60)))
        hmax = 520 - edge * 0.9
        tuft(d, rng, cx, H + 30, 4, max(120, hmax * 0.5), max(180, hmax), 30, 24)
    out = finish(front, outline=6)
    place(out, finish(leaf(rng, 380, (78, 126, 54)), outline=5), 120, H - 130, rot=58)
    return out


def draw_scenery_back(canvas, seed, p):
    lay = scenery_back(seed)
    s = 1.0 + 0.01 * p
    if s > 1.001:
        lay = lay.resize((int(W * s), int(H * s)), Image.BILINEAR)
    canvas.alpha_composite(lay, ((W - lay.width) // 2, (H - lay.height) // 2))


def draw_scenery_front(canvas, seed, p, t):
    lay = scenery_front(seed)
    s = 1.0 + 0.06 * p
    lay = lay.resize((int(W * s), int(H * s)), Image.BILINEAR)
    canvas.alpha_composite(lay, (int((W - lay.width) / 2 + 6 * math.sin(t * 2.1)), H - lay.height))


# ---------------------------------------------------------------- layered paper water (sea / pond topics)
SEA = [(150, 206, 214), (92, 168, 196), (52, 120, 168), (30, 78, 128), (18, 46, 86)]
WEED = [(64, 128, 84), (92, 150, 80), (46, 104, 78), (120, 160, 90)]
WATER_TOP = 660                       # surface line (title stays on paper above it)


@lru_cache(maxsize=None)
def water_layers(seed):
    """Torn-paper sea strata, light at the surface, deep navy below. Each layer is wider than the
    canvas so it can drift sideways (parallax)."""
    rng = random.Random(seed)
    out = []
    span = H - WATER_TOP
    for i, c in enumerate(SEA):
        lay = Image.new("RGBA", (W + 240, H + 60), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        y0 = WATER_TOP + span * (0, 0.16, 0.36, 0.58, 0.8)[i]
        amp = (18, 26, 30, 34, 30)[i]
        pts = []
        for k, x in enumerate(range(-80, W + 320, 45)):
            pts.append((x, y0 + amp * math.sin(k * 0.9 + rng.uniform(0, 6)) * 0.6 + rng.uniform(-amp, amp) * 0.4))
        d.polygon(pts + [(W + 320, H + 60), (-80, H + 60)], fill=c + (255,))
        if i == 0:                                           # sun ripples on the top layer
            for _ in range(26):
                x, y = rng.uniform(0, W + 200), rng.uniform(y0 + 40, y0 + span * 0.15)
                ln = rng.uniform(40, 120)
                d.line((x, y, x + ln, y), fill=(226, 244, 246, 200), width=5)
        out.append(finish(lay, outline=5 if i == 0 else 4, shadow=True, grain=True))
    return out


@lru_cache(maxsize=None)
def water_rays():
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    for x0, w0, a in ((180, 70, 34), (430, 110, 28), (700, 80, 30), (900, 60, 24)):
        d.polygon([(x0, WATER_TOP), (x0 + w0, WATER_TOP), (x0 + w0 * 3.2 - 160, H), (x0 - 260, H)],
                  fill=(240, 250, 250, a))
    return lay.filter(ImageFilter.GaussianBlur(14))


@lru_cache(maxsize=None)
def water_front(seed):
    """Sea floor (sand + pebbles) and paper seaweed fronds at both sides."""
    rng = random.Random(seed * 13 + 5)
    floor = Image.new("RGBA", (W + 160, 360), (0, 0, 0, 0))
    d = ImageDraw.Draw(floor)
    d.polygon([(x, 120 + rng.uniform(-22, 22)) for x in range(-60, W + 220, 50)] + [(W + 220, 400), (-60, 400)],
              fill=(214, 184, 132, 255))
    for _ in range(500):
        x, y = rng.uniform(0, W + 160), rng.uniform(140, 360)
        r = rng.uniform(1.5, 4)
        d.ellipse((x - r, y - r, x + r, y + r), fill=rng.choice([(186, 154, 104), (232, 208, 160)]) + (255,))
    floor = finish(floor, outline=5)
    for _ in range(7):
        r = rng.uniform(16, 34)
        peb = Image.new("RGBA", (int(r * 2.4), int(r * 2)), (0, 0, 0, 0))
        g = rng.randint(120, 170)
        ImageDraw.Draw(peb).ellipse((0, 0, r * 2.4 - 1, r * 2 - 1), fill=(g - 20, g, g + 6, 255))
        place(floor, finish(peb, outline=3), rng.uniform(40, W + 120), rng.uniform(190, 300), rot=rng.uniform(0, 180))
    fronds = []
    for cx in (40, 120, 210, W - 200, W - 110, W - 30):
        h = rng.uniform(380, 620)
        wd = 70
        img = Image.new("RGBA", (wd * 4, int(h) + 20), (0, 0, 0, 0))
        dd = ImageDraw.Draw(img)
        c = rng.choice(WEED)
        n, left, right = 14, [], []
        for k in range(n + 1):
            u = k / n
            x = wd * 2 + 26 * math.sin(u * 5 + rng.uniform(0, 1))
            y = h + 10 - h * u
            half = 26 * (1 - u * 0.8) + 4 * math.sin(u * 17)
            left.append((x - half, y))
            right.append((x + half, y))
        dd.polygon(left + right[::-1], fill=c + (255,))
        vein = tuple(max(0, v - 30) for v in c)
        dd.line([((a[0] + b[0]) / 2, a[1]) for a, b in zip(left, right)], fill=vein + (255,), width=3)
        fronds.append((cx, finish(img, outline=4), rng.uniform(0, 6)))
    return floor, fronds


def draw_water_back(canvas, seed, p, t):
    for i, lay in enumerate(water_layers(seed)):
        dx = -120 + 40 * math.sin(t * (0.35 + 0.12 * i) + i) - 30 * p * (i + 1) / 5
        dy = 5 * math.sin(t * 1.1 + i * 0.8)
        canvas.alpha_composite(lay, (int(dx) - PAD, int(dy) - PAD))
    canvas.alpha_composite(water_rays())
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))            # drifting specks (plankton)
    d = ImageDraw.Draw(layer)
    rng = random.Random(seed * 3)
    for _ in range(70):
        x0, y0, sp = rng.uniform(0, W), rng.uniform(WATER_TOP + 60, H), rng.uniform(10, 34)
        y = WATER_TOP + 60 + (y0 - WATER_TOP - 60 - sp * t) % (H - WATER_TOP - 60)
        x = x0 + 14 * math.sin(t * 0.8 + y0)
        r = rng.uniform(2, 5)
        d.ellipse((x - r, y - r, x + r, y + r), fill=(236, 248, 250, rng.randint(90, 180)))
    canvas.alpha_composite(layer)


def draw_water_front(canvas, seed, p, t):
    floor, fronds = water_front(seed)
    canvas.alpha_composite(floor, (int(-80 + 20 * p), H - floor.height + PAD))
    for cx, img, ph in fronds:
        place(canvas, img, cx + 6 * p, H - img.height / 2 + 60, rot=4 * math.sin(t * 1.3 + ph))


def polyp(n=5, seed=3, color=(236, 196, 186)):
    """Paper polyp colony: a stalk on the sea floor with branches, each tipped by a tiny tentacled head."""
    rng = random.Random(seed)
    w, h = 520, 640
    img = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    stalk = (150, 112, 92)
    base = (w / 2 * SS, (h - 20) * SS)
    tips = []
    for k in range(n):
        ang = math.radians(-90 + (k - (n - 1) / 2) * 17 + rng.uniform(-5, 5))
        ln = rng.uniform(300, 470) * SS
        mid = (base[0] + math.cos(ang) * ln * 0.5 + rng.uniform(-30, 30) * SS, base[1] + math.sin(ang) * ln * 0.5)
        tip = (base[0] + math.cos(ang) * ln, base[1] + math.sin(ang) * ln)
        d.line(bez(base, mid, tip), fill=stalk + (255,), width=13 * SS, joint="curve")
        tips.append((tip, ang))
    for (tx, ty), ang in tips:
        r = 34 * SS
        for j in range(9):                                     # tentacles
            a = ang + math.radians(-80 + j * 20)
            d.line((tx, ty, tx + math.cos(a) * r * 2.1, ty + math.sin(a) * r * 2.1), fill=color + (255,), width=4 * SS)
        d.ellipse((tx - r, ty - r * 1.2, tx + r, ty + r * 0.8), fill=color + (255,))
        d.ellipse((tx - r * 0.45, ty - r * 0.6, tx + r * 0.45, ty + r * 0.2), fill=(214, 90, 96, 255))
    return finish(down(img), outline=5)


# ---------------------------------------------------------------- effects (drawn per frame)
def spray(canvas, origin, direction_deg, t_since, n=120, length=520, hot=True, seed=1):
    if t_since < 0:
        return
    rng = random.Random(seed)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    ox, oy = origin
    a0 = math.radians(direction_deg)
    for _ in range(n):
        birth = rng.uniform(0, 1.6)
        age = t_since - birth
        life = rng.uniform(0.45, 0.8)
        if age < 0 or age > life:
            continue
        u = age / life
        ang = a0 + rng.uniform(-0.28, 0.28)
        dist = length * (u ** 0.7) * rng.uniform(0.7, 1.05)
        x, y = ox + math.cos(ang) * dist, oy + math.sin(ang) * dist
        r = 14 + 52 * u * rng.uniform(0.6, 1.2)
        c = ((255, int(200 - 150 * u), int(60 - 50 * u)) if rng.random() < 0.7 else (255, 245, 170)) if hot else (240, 240, 235)
        d.ellipse((x - r, y - r, x + r, y + r), fill=c + (int(255 * (1 - u ** 1.5)),))
        if u > 0.45 and rng.random() < 0.25:
            r2 = r * 1.6
            sx, sy = x + rng.uniform(-30, 30), y - 40 * u
            d.ellipse((sx - r2, sy - r2, sx + r2, sy + r2), fill=(250, 245, 235, int(45 * (1 - u))))
    glow = layer.filter(ImageFilter.GaussianBlur(16))
    canvas.alpha_composite(glow)
    canvas.alpha_composite(glow)
    canvas.alpha_composite(layer.filter(ImageFilter.GaussianBlur(2)))


def smoke(canvas, cx, cy, t_since, n=14, seed=4, spread=120):
    if t_since < 0:
        return
    rng = random.Random(seed)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for _ in range(n):
        age = t_since - rng.uniform(0, 0.8)
        if age < 0 or age > 1.8:
            continue
        u = age / 1.8
        x = cx + rng.uniform(-spread, spread) * (0.4 + u)
        y = cy - 260 * u + rng.uniform(-30, 30)
        r = 30 + 90 * u
        g = rng.randint(70, 110)
        d.ellipse((x - r, y - r, x + r, y + r), fill=(g, g, g, int(150 * (1 - u))))
    canvas.alpha_composite(layer.filter(ImageFilter.GaussianBlur(14)))


def crosshair(canvas, cx, cy, r, p):
    if p <= 0:
        return
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    rr = r * (1.6 - 0.6 * ease_out(p))
    al = int(255 * clamp(p * 2))
    d.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), outline=RED + (al,), width=10)
    d.ellipse((cx - rr * 0.55, cy - rr * 0.55, cx + rr * 0.55, cy + rr * 0.55), outline=RED + (al,), width=6)
    for a in range(4):
        ang = a * math.pi / 2
        d.line((cx + math.cos(ang) * rr * 0.7, cy + math.sin(ang) * rr * 0.7,
                cx + math.cos(ang) * rr * 1.25, cy + math.sin(ang) * rr * 1.25), fill=RED + (al,), width=10)
    canvas.alpha_composite(layer)


def rings(canvas, cx, cy, t_since, color=RED, n=3):
    if t_since < 0:
        return
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for i in range(n):
        u = (t_since - i * 0.22) / 0.9
        if 0 <= u <= 1:
            r = 80 + 360 * u
            a = int(255 * (1 - u))
            d.arc((cx - r, cy - r, cx + r, cy + r), -60, 60, fill=color + (a,), width=14)
            d.arc((cx - r, cy - r, cx + r, cy + r), 120, 240, fill=color + (a,), width=14)
    canvas.alpha_composite(layer)


# ---------------------------------------------------------------- generated sounds (no files needed)
def _box(x, k):
    """Moving-average low-pass (k samples) along axis 0."""
    k = max(1, int(k))
    c = np.cumsum(np.pad(x, (k, 0)), dtype=np.float64)
    return ((c[k:] - c[:-k]) / k).astype(np.float32)


def _stereo(m, peak=0.7):
    m = m / (np.abs(m).max() + 1e-9) * peak
    return np.stack([m, m], axis=1).astype(np.float32)


def synth(name, dur=None):
    rng = np.random.default_rng(abs(hash(name)) % 2 ** 32)
    if name == "sizzle":
        n = int((dur or 2.2) * SR)
        x = rng.normal(0, 1, n).astype(np.float32)
        hiss = x - _box(x, 6)
        crack = np.zeros(n, np.float32)
        for i in rng.integers(0, n - 400, int(n / SR * 90)):
            crack[i:i + 300] += rng.normal(0, 1, 300) * np.exp(-np.arange(300) / 40)
        env = np.minimum(1, np.arange(n) / (0.08 * SR)) * np.minimum(1, (n - np.arange(n)) / (0.4 * SR))
        return _stereo((hiss * 0.5 + crack) * env, 0.6)
    if name == "boil":
        n = int((dur or 3.0) * SR)
        out = _box(rng.normal(0, 1, n).astype(np.float32), 40) * 0.6
        for i in rng.integers(0, n - 4000, int(n / SR * 18)):
            L = int(rng.integers(1500, 4000))
            t = np.arange(L) / SR
            f = rng.uniform(250, 700)
            out[i:i + L] += np.sin(2 * np.pi * (f * t + 900 * t * t)) * np.exp(-t * 35)
        env = np.minimum(1, np.arange(n) / (0.3 * SR)) * np.minimum(1, (n - np.arange(n)) / (0.6 * SR))
        return _stereo(out * env, 0.55)
    if name == "boom":
        n = int(2.5 * SR)
        t = np.arange(n) / SR
        f = 35 + 60 * np.exp(-t * 6)
        sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.2)
        noise = _box(rng.normal(0, 1, n).astype(np.float32), 25) * np.exp(-t * 9) * 3
        return _stereo(sub + noise, 0.9)
    if name == "riser":
        n = int(2.8 * SR)
        t = np.arange(n) / SR
        u = t / t[-1]
        x = rng.normal(0, 1, n).astype(np.float32)
        bright = x - _box(x, 30)
        tone = np.sin(2 * np.pi * np.cumsum(180 + 900 * u ** 2) / SR)
        return _stereo((bright * u + tone * 0.5 * u) * u ** 1.5, 0.7)
    if name == "drone":                      # dark low music bed: two detuned notes, slow swell, soft pulse
        n = int((dur or 60) * SR)
        t = np.arange(n) / SR
        lfo = 0.6 + 0.4 * np.sin(2 * np.pi * t / 7.0)
        tone = sum(np.sin(2 * np.pi * f * t + p) * a for f, p, a in
                   ((55.0, 0, 1.0), (55.4, 1, 0.8), (82.4, 2, 0.5), (110.3, 3, 0.25)))
        air = _box(rng.normal(0, 1, n).astype(np.float32), 120) * 2.5
        pulse = np.zeros(n, np.float32)
        for k in np.arange(0.5, t[-1], 1.25):
            i = int(k * SR)
            L = min(int(0.35 * SR), n - i)
            tt = np.arange(L) / SR
            pulse[i:i + L] += np.sin(2 * np.pi * 48 * tt) * np.exp(-tt * 9) * 1.2
        return _stereo((tone * lfo + air + pulse), 0.6)
    if name == "whoosh":
        n = int(0.6 * SR)
        u = np.arange(n) / n
        x = rng.normal(0, 1, n).astype(np.float32)
        return _stereo((x - _box(x, 12)) * np.sin(np.pi * u) ** 2, 0.5)
    # short pop for anything else (pop, mpop, tick, …)
    n = int(0.12 * SR)
    t = np.arange(n) / SR
    return _stereo(np.sin(2 * np.pi * (600 * t - 1500 * t * t)) * np.exp(-t * 40), 0.6)


# ---------------------------------------------------------------- the renderer
class Reel:
    def __init__(self, plan_path):
        global W, H, FPS
        self.path = Path(plan_path).resolve()
        self.plan = json.loads(self.path.read_text())
        meta = self.plan["meta"]
        W, H = meta.get("size", [1080, 1920])
        FPS = meta.get("fps", 30)
        # paths are relative to the video folder (the plan's folder, or its parent when the plan is in 06_Edit)
        self.video_dir = self.path.parent.parent if self.path.parent.name == "06_Edit" else self.path.parent
        self.meta = meta
        self.scenes = sorted(self.plan["scenes"], key=lambda s: s["start"])
        self.subjects = Subjects(self.plan.get("subjects", {}), self.resolve)
        self.dur = meta.get("duration") or self._voice_len() + 1.6
        self.out = self.resolve(meta["out"])
        self.page = self.dark_page = None

    def resolve(self, p):
        p = str(p)
        if p.startswith("/"):
            return Path(p)
        if p.startswith("@sfx/"):                     # AST BTS sound library (only inside that project)
            return (SFX_DIR / p[5:]) if SFX_DIR else Path("/nonexistent") / p[5:]
        if p.startswith("@pack/"):                    # this skill folder (sounds/, examples/)
            return PACK / p[6:]
        if p.startswith("@root/"):
            return (ROOT or self.video_dir) / p[6:]
        return self.video_dir / p

    def _voice_len(self):
        if not self.meta.get("voice"):
            raise ValueError("meta.duration is required when there is no meta.voice")
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                            str(self.resolve(self.meta["voice"]))], capture_output=True, text=True)
        return float(r.stdout.strip())

    # ---- timing
    def scene_at(self, t):
        idx = 0
        for i, sc in enumerate(self.scenes):
            if t >= sc["start"]:
                idx = i
        end = self.scenes[idx + 1]["start"] if idx + 1 < len(self.scenes) else self.dur
        return idx, self.scenes[idx]["start"], end

    # ---- element geometry
    def base_pos(self, el, lt, ids):
        if "on" in el:                                  # a point on another element (anchor / rel)
            x, y = self.point_on(ids[el["on"]], el.get("anchor", el.get("rel", "center")), lt, ids)
            x, y = x + el.get("dx", 0), y + el.get("dy", 0)
        else:
            x, y = el.get("x", W / 2), el.get("y", H * 0.6)
        mv = el.get("move")
        if mv:
            p = ease_out(prog(lt, mv.get("at", 0), mv.get("dur", 0.6)))
            x, y = x + (mv["to"][0] - x) * p, y + (mv["to"][1] - y) * p
        sh = el.get("shake")
        if sh and lt >= sh.get("at", 0):
            u = lt - sh.get("at", 0)
            dur = sh.get("dur")
            if dur is None or u <= dur:
                amp = sh.get("amp", 8)
                if sh.get("decay") == "linear" and dur:
                    amp *= 1 - u / dur
                elif sh.get("decay") == "exp":
                    amp *= math.exp(-u * sh.get("k", 9))
                x += amp * math.sin(u * sh.get("freq", 50))
        fl = el.get("float")                            # gentle bob (things in water)
        if fl:
            fl = fl if isinstance(fl, dict) else {}
            ph = fl.get("phase", 0.0)
            y += fl.get("amp", 16) * math.sin(lt * fl.get("freq", 1.6) + ph)
            x += fl.get("amp", 16) * 0.4 * math.sin(lt * fl.get("freq", 1.6) * 0.5 + ph)
        return x, y

    def point_on(self, el, key, lt, ids):
        """Screen position of an anchor (or [rx, ry] fraction) of a cut-out element at scene time lt."""
        cx, cy = self.base_pos(el, lt, ids)
        name, height = el["subject"], self.cut_height(el)
        src = self.subjects.src(name)
        k = height / src.height
        fw, fh = int(src.width * k) + 2 * PAD, int(src.height * k) + 2 * PAD
        ax, ay = self.subjects.anchor(name, key)
        dx, dy = ax * k + PAD - fw / 2, ay * k + PAD - fh / 2
        sc = self.anim(el, lt)[0] if el.get("anim", "pop") != "none" else 1.0
        sc *= el.get("scale", 1.0)
        th = math.radians(el.get("rot", 0))
        rx, ry = dx * math.cos(th) + dy * math.sin(th), -dx * math.sin(th) + dy * math.cos(th)
        return cx + rx * sc, cy + ry * sc

    def cut_height(self, el):
        """Cut-out height in px; "max_width" shrinks wide subjects to fit."""
        h = el.get("height", 700)
        if el.get("max_width"):
            src = self.subjects.src(el["subject"])
            h = min(h, el["max_width"] * src.height / src.width)
        return int(h)

    def anim(self, el, lt):
        """(scale, alpha) for the element's entrance."""
        at, kind = el.get("at", 0.0), el.get("anim", "pop")
        if lt < at or ("until" in el and lt >= el["until"]):
            return 0.0, 0.0
        d = el.get("in", 0.3)
        if kind == "pop":
            p = prog(lt, at, d)
            return 0.6 + 0.4 * ease_back(p), clamp(p * 3)
        if kind == "slam":
            p = prog(lt, at, 0.14)
            return 1.7 - 0.7 * ease_out(p), clamp(p * 1.4)
        if kind == "punch":
            return 0.4 + 0.75 * ease_back(prog(lt, at, d)), 1.0
        if kind == "fade":
            return 1.0, prog(lt, at, d)
        return 1.0, 1.0

    # ---- elements
    def image_for(self, el, lt):
        t = el["type"]
        if t == "cutout":
            return self.subjects.image(el["subject"], el.get("mode", "color"), self.cut_height(el))
        if t == "stamp":
            return stamp(el["text"], col(el.get("color")), el.get("size", 100), el.get("tilt", -8))
        if t == "sticky":
            return sticky_note(el.get("text", "?"))
        if t == "ticket":
            return ticket(el.get("strike", "CARTOON"), el.get("ok", "REAL"), el.get("number", "0013"), el.get("top", "ADMIT ONE"))
        if t == "calendar":
            return calendar(str(el.get("top", "2026")), str(el.get("big", "AAJ")))
        if t == "ruler":
            return ruler(el.get("units", "CM"), el.get("count", 4))
        if t == "flask":
            return flask(el["label"], col(el.get("liquid"), (120, 190, 255)))
        if t == "warning":
            return warning(el.get("mark", "!"))
        if t == "thermometer":
            lvl = keyframes(el.get("level", 0), lt - el.get("at", 0))
            return thermometer(round(lvl, 3), tuple(el.get("labels", ["0", "50", "100°C"])))
        if t == "burst":
            return burst(el["text"], el.get("size", 1.0), col(el.get("color"), YELLOW))
        if t == "name_tag":
            return name_tag(el["line1"], el.get("line2", ""), el.get("sub", ""))
        if t == "firecracker":
            return firecracker(el.get("label", "BOMB"))
        if t == "ant":
            return ant(el.get("charred", False), el.get("size", 1.0))
        if t == "polyp":
            return polyp(el.get("n", 5), el.get("seed", 3), col(el.get("color"), (236, 196, 186)))
        if t == "photo":
            return photo_card(self.resolve(el["image"]), el.get("width", 620), el.get("caption", ""))
        return None

    def draw_element(self, c, el, lt, ids):
        t = el["type"]
        at = el.get("at", 0.0)
        if "until" in el and lt >= el["until"]:
            return
        img = self.image_for(el, lt)
        if img is not None:
            sc, al = self.anim(el, lt)
            x, y = self.base_pos(el, lt, ids)
            place(c, img, x, y, sc * el.get("scale", 1.0), al * el.get("alpha", 1.0), el.get("rot", 0))
            return
        x, y = self.base_pos(el, lt, ids)
        if t == "spray":
            spray(c, (x, y), el.get("dir", 90), lt - at, el.get("n", 120), el.get("length", 520),
                  el.get("hot", True), el.get("seed", 1))
        elif t == "smoke":
            smoke(c, x, y, lt - at, el.get("n", 14), el.get("seed", 4), el.get("spread", 120))
        elif t == "rings":
            rings(c, x, y, lt - at, col(el.get("color")), el.get("n", 3))
        elif t == "crosshair":
            crosshair(c, x, y, el.get("r", 360), prog(lt, at, el.get("in", 0.4)))
        elif t == "text":
            p = prog(lt, at, el.get("in", 0.3))
            if p > 0:
                f = font(el.get("font", "head"), el.get("size", 120))
                layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                dd = ImageDraw.Draw(layer)
                tw = dd.textlength(el["text"], font=f)
                dd.text((x - tw / 2, y - el.get("size", 120) * 0.6), el["text"], font=f,
                        fill=col(el.get("color"), INK) + (int(255 * p),))
                c.alpha_composite(layer)
        elif t == "label":
            p = prog(lt, at, 0.25)
            if p > 0:
                dark = el.get("dark", False)
                layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                ImageDraw.Draw(layer).text((x, y), el["text"], font=font("mono", el.get("size", 40)),
                                           fill=((220, 235, 255) if dark else INK) + (int(255 * p),))
                c.alpha_composite(layer)
        elif t == "bracket":
            p = ease_out(prog(lt, at, el.get("dur", 0.6)))
            if p > 0:
                layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                dd = ImageDraw.Draw(layer)
                x0, x1 = el["x0"], el["x0"] + (el["x1"] - el["x0"]) * p
                yy = el["y"]
                dd.line((x0, yy, x1, yy), fill=RED, width=10)
                dd.line((x0, yy - 25, x0, yy + 25), fill=RED, width=10)
                if p > 0.98:
                    dd.line((x1, yy - 25, x1, yy + 25), fill=RED, width=10)
                c.alpha_composite(layer)
        elif t == "glow":                              # chamber / organ / hotspot, level 0..1 (keyframes)
            if lt < at:
                return
            lvl = keyframes(el.get("level", 1.0), lt - at)
            rx, ry = el.get("rx", 60), el.get("ry", el.get("rx", 60))
            layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            dd = ImageDraw.Draw(layer)
            if el.get("outline", True):
                dd.ellipse((x - rx, y - ry, x + rx, y + ry), outline=col(el.get("outline_color"), XRAY) + (230,),
                           width=el.get("width", 6))
            if lvl > 0:
                dd.ellipse((x - rx + 10, y - ry + 10, x + rx - 10, y + ry - 10),
                           fill=col(el.get("color"), (90, 170, 255)) + (int(220 * lvl),))
            c.alpha_composite(layer.filter(ImageFilter.GaussianBlur(1.5)))
            c.alpha_composite(layer.filter(ImageFilter.GaussianBlur(8)))
        elif t == "flow":                              # droplets travelling from here to "to"
            p = prog(lt, at, el.get("dur", 1.2))
            if 0 < p < 1:
                x1, y1 = self.point_on(ids[el["to_on"]], el.get("to_rel", "center"), lt, ids) if "to_on" in el else el["to"]
                layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                dd = ImageDraw.Draw(layer)
                for k in range(el.get("n", 5)):
                    u = (p * 3 + k / 5) % 1
                    px, py = x + (x1 - x) * u, y + (y1 - y) * u
                    dd.ellipse((px - 9, py - 9, px + 9, py + 9), fill=col(el.get("color"), (120, 190, 255)) + (230,))
                c.alpha_composite(layer.filter(ImageFilter.GaussianBlur(1.5)))
        elif t == "sparkle":
            if lt < at:
                return
            st = lt - at
            layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            dd = ImageDraw.Draw(layer)
            rng = random.Random(el.get("seed", 9))
            for _ in range(el.get("n", 14)):
                u = (st * 2 + rng.uniform(0, 1)) % 1
                a = rng.uniform(0, math.pi * 2)
                rr = 20 + el.get("r", 90) * u
                px, py = x + math.cos(a) * rr, y + math.sin(a) * rr
                s = 10 * (1 - u)
                cc = (255, 240, 140, int(255 * (1 - u)))
                dd.line((px - s, py, px + s, py), fill=cc, width=4)
                dd.line((px, py - s, px, py + s), fill=cc, width=4)
            c.alpha_composite(layer.filter(ImageFilter.GaussianBlur(1)))
        elif t == "spark":                             # fuse spark (follows its own position / move)
            layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            dd = ImageDraw.Draw(layer)
            rng = random.Random(int(lt * 30))
            for _ in range(12):
                a = rng.uniform(0, math.pi * 2)
                r = rng.uniform(10, 50)
                dd.line((x, y, x + math.cos(a) * r, y + math.sin(a) * r), fill=(255, 220, 90, 255), width=4)
            if lt >= at:
                c.alpha_composite(layer.filter(ImageFilter.GaussianBlur(1)))
        elif t == "bubbles":
            st = lt - at
            if st > 0:
                layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                dd = ImageDraw.Draw(layer)
                rng = random.Random(3)
                for _ in range(el.get("n", 26)):
                    u = (st * 0.9 + rng.uniform(0, 1)) % 1
                    bx = rng.uniform(el.get("x0", 600), el.get("x1", 960))
                    by = el.get("y", 1560) - el.get("rise", 700) * u
                    r = rng.uniform(14, 40) * (0.5 + u)
                    dd.ellipse((bx - r, by - r, bx + r, by + r), outline=(250, 250, 250, int(230 * (1 - u))), width=5)
                c.alpha_composite(layer)
        elif t == "sound":                             # no picture, only its sfx at "at"
            pass
        elif t == "flash":
            if lt >= at:
                fl = 1 - prog(lt, at, el.get("dur", 0.25))
                if fl > 0:
                    c.alpha_composite(Image.new("RGBA", (W, H), col(el.get("color"), (255, 240, 200)) + (int(el.get("alpha", 120) * fl),)))
        else:
            raise ValueError(f"unknown element type {t!r}")

    def render_frame(self, t):
        if self.page is None:
            self.page, self.dark_page = make_page(False), make_page(True)
        idx, start, end = self.scene_at(t)
        sc = self.scenes[idx]
        lt = t - start
        dark = sc.get("page") == "dark"
        canvas = (self.dark_page if dark else self.page).copy().convert("RGBA")
        header(canvas, idx + 1, dark, self.meta.get("header", ""))
        p = clamp(lt / max(0.5, end - start))
        seed = sc.get("scenery")
        water = sc.get("water")
        if seed:
            draw_scenery_back(canvas, seed, p)
        if water:
            draw_water_back(canvas, water, p, t)
        ids = {el["id"]: el for el in sc.get("elements", []) if "id" in el}
        for el in sc.get("elements", []):
            self.draw_element(canvas, el, lt, ids)
        if seed:
            draw_scenery_front(canvas, seed, p, t)
        if water and sc.get("seabed", True):
            draw_water_front(canvas, water, p, t)
        title = sc.get("title", ["", ""]) + [""]
        draw_title(canvas, title[0], title[1], sc.get("red", 2), lt, dark)
        zoom = 1.0 + 0.035 * p + 0.02 * math.exp(-lt * 14)
        if zoom > 1.0005:
            zw, zh = int(W * zoom), int(H * zoom)
            big = canvas.resize((zw, zh), Image.BICUBIC)
            ox, oy = (zw - W) // 2, int((zh - H) * 0.52)
            canvas = big.crop((ox, oy, ox + W, oy + H))
        fade_end = self.meta.get("end_fade", 0.5)
        if fade_end and t > self.dur - fade_end:
            canvas = Image.blend(Image.new("RGBA", (W, H), (0, 0, 0, 255)), canvas, clamp((self.dur - t) / fade_end))
        return canvas.convert("RGB")

    # ---- audio
    def load_audio(self, path, start=0.0, dur=None, extra_af=None):
        cmd = ["ffmpeg", "-v", "error", "-ss", str(start), "-i", str(path)]
        if dur:
            cmd += ["-t", str(dur)]
        if extra_af:
            cmd += ["-af", extra_af]
        cmd += ["-ac", "2", "-ar", str(SR), "-f", "f32le", "-"]
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        return np.frombuffer(raw, np.float32).reshape(-1, 2).copy()

    @lru_cache(maxsize=None)
    def sound(self, name):
        if name in SOUNDS:
            f, st, du = SOUNDS[name]
            if SFX_DIR and (SFX_DIR / f).exists():
                return self.load_audio(SFX_DIR / f, st, du)
            if (PACK_SOUNDS / f"{name}.mp3").exists():
                return self.load_audio(PACK_SOUNDS / f"{name}.mp3")
            return synth(name, du)
        path = self.resolve(name)
        if not path.exists():
            raise FileNotFoundError(f"sound not found: {name} -> {path}")
        return self.load_audio(path)

    @staticmethod
    def put(mix, clip, t, db=0.0, trim=None):
        if trim:
            clip = fade(clip[:int(trim * SR)].copy(), 0.02, 0.3)
        i = int(t * SR)
        if i < 0:
            clip, i = clip[-i:], 0
        n = min(len(clip), len(mix) - i)
        if n > 0:
            mix[i:i + n] += clip[:n] * 10 ** (db / 20)

    def build_audio(self):
        mix = np.zeros((int(self.dur * SR), 2), np.float32)
        if self.meta.get("voice"):
            self.put(mix, self.load_audio(self.resolve(self.meta["voice"])), self.meta.get("voice_at", 0.0))
        mu = self.meta.get("music")
        if mu:
            path = self.resolve(mu["file"]) if mu.get("file") else None
            if path is not None and path.exists():
                af = f"atempo={mu['atempo']}" if mu.get("atempo") else None
                m = self.load_audio(path, mu.get("start", 0), None, af)
            else:
                if path is not None:
                    print(f"  music not found ({path.name}) -> generated drone")
                m = synth("drone", self.dur)
            m = fade(m[: int(self.dur * SR)], 0.5, 1.5)
            self.put(mix, m, 0.0, mu.get("db", -21))
        wdb = self.meta.get("whoosh_db", -20)
        if wdb is not None:
            for sc in self.scenes[1:]:
                self.put(mix, fade(self.sound("whoosh").copy(), 0.01, 0.3), sc["start"] - 0.12, wdb)
        for sc in self.scenes:
            for el in sc.get("elements", []):
                spec = el.get("sfx", "default")
                if spec is False or spec is None:
                    continue
                if spec == "default":
                    if el["type"] not in DEFAULT_SFX:
                        continue
                    name, db, off = DEFAULT_SFX[el["type"]]
                else:
                    name, db, off = spec, el.get("sfx_db", -8), 0.0
                db = el.get("sfx_db", db)
                self.put(mix, self.sound(name), sc["start"] + el.get("at", 0.0) + off + el.get("sfx_offset", 0.0), db,
                         el.get("sfx_trim"))
        for s in self.plan.get("sounds", []):
            self.put(mix, self.sound(s["sfx"]), s["t"] - s.get("lead", 0.0), s.get("db", -8), s.get("trim"))
        return mix

    def mux(self, video_path):
        wav = self.out.with_name("_paperdoc_mix.wav")
        data = np.clip(self.build_audio(), -1, 1).astype(np.float32)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-", str(wav)],
                       input=data.tobytes(), check=True)
        err = subprocess.run(["ffmpeg", "-i", str(wav), "-af", "ebur128", "-f", "null", "-"],
                             capture_output=True, text=True).stderr
        gain = self.meta.get("lufs", -14.0) - float(err.rsplit("I:", 1)[1].split("LUFS")[0])
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video_path), "-i", str(wav),
                        "-filter_complex", f"[1:a]volume={gain:.2f}dB,alimiter=limit=0.75:level=false[a]",
                        "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k",
                        "-ar", "48000", "-shortest", "-movflags", "+faststart", str(self.out)], check=True)
        wav.unlink()

    def render(self):
        self.out.parent.mkdir(parents=True, exist_ok=True)
        silent = self.out.with_name("_paperdoc_video.mp4")
        enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                                "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
                                str(silent)], stdin=subprocess.PIPE)
        n = int(self.dur * FPS)
        for i in range(n):
            enc.stdin.write(self.render_frame(i / FPS).tobytes())
            if i % 150 == 0:
                print(f"  {100 * i // n}% ({i}/{n} frames)", flush=True)
        enc.stdin.close()
        enc.wait()
        self.mux(silent)
        silent.unlink()
        print("done ->", self.out)


def fade(clip, fin=0.05, fout=0.3):
    a, b = int(fin * SR), int(fout * SR)
    if a:
        clip[:a] *= np.linspace(0, 1, a)[:, None]
    if b:
        clip[len(clip) - b:] *= np.linspace(1, 0, b)[:, None]
    return clip


# ---------------------------------------------------------------- CLI
def cmd_words(path, t0=0.0, t1=1e9):
    words = json.loads(Path(path).read_text())
    if isinstance(words, dict) and "transcription" in words:          # raw whisper JSON
        words = [{"w": w["text"].strip(), "s": w["offsets"]["from"] / 1000, "e": w["offsets"]["to"] / 1000}
                 for w in words["transcription"] if w["text"].strip()]
    for w in words:
        if t0 <= w["s"] <= t1:
            print(f"{w['s']:7.2f} {w['e']:7.2f}  {w['w']}")


def cmd_cutout(photo, out, rotate=0):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + "_raw.png")
    vision_cutout(photo, tmp)
    img = Image.open(tmp).convert("RGBA")
    bbox = img.getchannel("A").point(lambda v: 255 if v > 20 else 0).getbbox()
    img = img.crop(bbox)
    if rotate:
        img = img.rotate(rotate, expand=True, resample=Image.BICUBIC)
    img.save(out)
    tmp.unlink()
    print("saved", out, img.size)


def cmd_contact(reel, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    thumbs = []
    for i, sc in enumerate(reel.scenes):
        end = reel.scenes[i + 1]["start"] if i + 1 < len(reel.scenes) else reel.dur
        t = sc["start"] + min(end - sc["start"] - 0.05, max(1.2, (end - sc["start"]) * 0.7))
        im = reel.render_frame(t)
        thumbs.append((t, im.resize((W // 4, H // 4))))
    cols = 5
    tw, th = W // 4, H // 4
    sheet = Image.new("RGB", (tw * cols, th * ((len(thumbs) + cols - 1) // cols)), (0, 0, 0))
    for i, (t, im) in enumerate(thumbs):
        ImageDraw.Draw(im).text((6, 6), f"{i + 1} · {t:.1f}s", font=font("mono", 18), fill=(255, 255, 0))
        sheet.paste(im, ((i % cols) * tw, (i // cols) * th))
    path = outdir / f"{reel.out.stem}_contact.jpg"
    sheet.save(path, quality=82)
    print("contact sheet ->", path)


def cmd_selftest(outdir, photo=None):
    """Render a 6 s reel with no voice to check fonts, ffmpeg, sounds and (with a photo) the Vision cut-out."""
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    s2 = [{"type": "sticky", "text": "?", "x": 540, "y": 1150, "rot": -4},
          {"type": "stamp", "text": "100% SACH", "size": 110, "tilt": -10, "x": 640, "y": 1480, "anim": "slam", "at": 0.8}]
    subjects = {}
    if photo:
        subjects = {"thing": {"image": str(Path(photo).resolve())}}
        s2 = [{"id": "t", "type": "cutout", "subject": "thing", "height": 800, "max_width": 900, "x": 540, "y": 1200},
              {"type": "stamp", "text": "100% SACH", "size": 110, "tilt": -10, "x": 680, "y": 1650, "anim": "slam", "at": 0.8}]
    plan = {"meta": {"out": str(outdir / "paper_doc_selftest.mp4"), "duration": 6.0, "header": "SELF TEST",
                     "music": {"db": -21}},
            "subjects": subjects,
            "sounds": [{"t": 4.2, "sfx": "riser", "lead": 2.8, "db": -16}, {"t": 4.2, "sfx": "boom", "db": -8}],
            "scenes": [
                {"start": 0.0, "title": ["PAPER-DOC", "SELF TEST"], "red": 2, "scenery": 1,
                 "elements": [{"type": "name_tag", "line1": "IT", "line2": "WORKS", "sub": "fonts · ffmpeg · sound", "x": 540, "y": 1300, "rot": -3, "at": 0.4}]},
                {"start": 2.0, "title": ["YAKEEN", "NAHI?"], "red": 2, "elements": s2},
                {"start": 4.2, "title": ["BAS", "PHAT!"], "red": 2, "page": "dark",
                 "elements": [{"type": "burst", "text": "PHAT!", "x": 540, "y": 1250, "rot": -5, "anim": "punch"},
                              {"type": "spray", "x": 540, "y": 1500, "dir": 90, "at": 0.1, "sfx": False}]}]}
    path = outdir / "selftest_plan.json"
    path.write_text(json.dumps(plan, indent=1))
    reel = Reel(path)
    print("sounds from:", "AST BTS library" if SFX_DIR else f"{PACK_SOUNDS} (CC0) + generated")
    reel.render()


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return
    cmd = argv[1]
    if cmd == "render":
        Reel(argv[2]).render()
    elif cmd == "stills":
        reel = Reel(argv[2])
        outdir = Path(argv[3])
        outdir.mkdir(parents=True, exist_ok=True)
        for t in map(float, argv[4:]):
            reel.render_frame(t).save(outdir / f"{reel.out.stem}_{t:05.2f}.jpg", quality=82)
        print("stills ->", outdir)
    elif cmd == "contact":
        cmd_contact(Reel(argv[2]), argv[3])
    elif cmd == "words":
        cmd_words(argv[2], *(float(x) for x in argv[3:5]))
    elif cmd == "selftest":
        cmd_selftest(argv[2] if len(argv) > 2 else "paper_doc_selftest", argv[3] if len(argv) > 3 else None)
    elif cmd == "cutout":
        rot = int(argv[argv.index("--rotate") + 1]) if "--rotate" in argv else 0
        cmd_cutout(argv[2], argv[3], rot)
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv)

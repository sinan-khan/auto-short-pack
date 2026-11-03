"""icons.py - ~11,000 ready-made objects, fetched on demand from open libraries (no files to store in your repo).
  * Twemoji (colour emoji, ~3,600 base images)  - CC-BY 4.0  -> credit line is added to upload_info.txt automatically
  * Material Design Icons (~7,200 glyphs)       - Apache-2.0 -> drawn white on a coloured badge
  * Font Awesome Free solid (~1,400 glyphs)     - CC-BY 4.0  -> credit line added automatically
resolve("tomato") / resolve("dna") / resolve("🚀")  ->  (kind, key)   get_image(kind, key) -> 512px RGBA PIL image."""
import hashlib, io, json, re, urllib.request
from functools import lru_cache
from pathlib import Path

CACHE = Path(__file__).resolve().parent / "work" / "icons"
MDI_BASE = "https://raw.githubusercontent.com/Templarian/MaterialDesign-SVG/master/"
TWEMOJI = ["https://raw.githubusercontent.com/jdecked/twemoji/main/assets/svg/{}.svg",
           "https://raw.githubusercontent.com/twitter/twemoji/master/assets/svg/{}.svg"]
USED = set()          # sources used in this run (for the credit line)
STOP = set("a an the of and or to in on at for with from by is are was were be been it its this that these those very more most as into over under".split())
FA_META = "https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/metadata/icons.json"
PALETTE = [(231, 76, 60), (52, 152, 219), (46, 204, 113), (155, 89, 182), (241, 160, 15), (230, 100, 34), (26, 188, 156), (233, 30, 99), (88, 110, 220)]


def _get(url, binary=True, timeout=40):
    req = urllib.request.Request(url, headers={"User-Agent": "short-maker"})
    data = urllib.request.urlopen(req, timeout=timeout).read()
    return data if binary else data.decode("utf8")


def _stem(w):
    w = w.lower()
    for suf, rep in (("ies", "y"), ("sses", "ss"), ("ves", "f"), ("es", ""), ("s", "")):
        if len(w) > 3 and w.endswith(suf) and not w.endswith("ss"):
            return w[: -len(suf)] + rep
    return w


def _words(text):
    return [_stem(w) for w in re.findall(r"[a-zA-Z]+", text) if w.lower() not in STOP]


@lru_cache(None)
def _mdi_index():
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / "mdi_meta.json"
    if not p.exists():
        p.write_text(_get(MDI_BASE + "meta.json", binary=False))
    out = []
    for i in json.loads(p.read_text()):
        if i.get("deprecated"):
            continue
        toks = set(_words(i["name"].replace("-", " ")))
        for a in i.get("aliases", []):
            toks |= set(_words(a.replace("-", " ")))
        out.append(("mdi", i["name"], frozenset(toks), len(_words(i["name"].replace("-", " ")))))
    return out


@lru_cache(None)
def _fa_meta():
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / "fa_meta.json"
    if not p.exists():
        p.write_text(_get(FA_META, binary=False))
    return json.loads(p.read_text())


@lru_cache(None)
def _fa_index():
    out = []
    for name, d in _fa_meta().items():
        if "solid" not in d.get("free", []) or "solid" not in d.get("svg", {}):
            continue
        toks = set(_words(name.replace("-", " ")))
        toks |= set(_words(d.get("label", "")))
        for t in d.get("search", {}).get("terms", []):
            toks |= set(_words(str(t)))
        out.append(("fa", name, frozenset(toks), len(_words(name.replace("-", " ")))))
    return out


@lru_cache(None)
def _emoji_index():
    import emoji
    out = []
    for ch, d in emoji.EMOJI_DATA.items():
        if d.get("status", 2) != emoji.STATUS["fully_qualified"] and d.get("status", 2) != emoji.STATUS["component"]:
            continue
        name = d["en"].strip(":").replace("_", " ")
        if "skin tone" in name or "keycap" in name or "regional" in name or name.startswith("flag"):
            continue
        toks = set(_words(name))
        for a in d.get("alias", []):
            toks |= set(_words(a.strip(":").replace("_", " ")))
        out.append(("emoji", ch, frozenset(toks), len(_words(name))))
    return out


def count():
    return len(_mdi_index()), len(_emoji_index()), len(_fa_index())


def resolve(query):
    """Best icon for a noun/phrase or an emoji character. Returns (kind, key) or None."""
    import emoji
    q = str(query).strip()
    if not q:
        return None
    found = emoji.emoji_list(q)
    if found:
        return ("emoji", found[0]["emoji"])
    Q = _words(q.replace("-", " ").replace("_", " "))
    if not Q:
        return None
    qs = set(Q)
    best, best_s = None, 0.0
    for kind, key, toks, n in (_emoji_index() + _mdi_index() + _fa_index()):
        hit = len(qs & toks)
        if not hit:
            continue
        if n == len(Q) and toks >= qs and len(toks) <= len(qs) + 1:
            s = 100.0
        elif toks >= qs:
            s = 60.0 - 2 * (len(toks) - len(qs))
        else:
            frac = hit / len(qs)
            if frac < 0.5:
                continue
            s = 20.0 * frac - 0.5 * len(toks)
        s += 3.0 if kind == "emoji" else (-0.5 if kind == "fa" else 0.0)
        if s > best_s:
            best, best_s = (kind, key), s
    return best


def _emoji_code(ch):
    return "-".join(f"{ord(c):x}" for c in ch)


def _png_from_svg(svg_bytes, size=512):
    import cairosvg
    return cairosvg.svg2png(bytestring=svg_bytes, output_width=size, output_height=size)


def get_image(kind, key, size=512):
    from PIL import Image, ImageDraw, ImageFilter
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = hashlib.md5(f"{kind}:{key}".encode()).hexdigest()[:12]
    p = CACHE / f"{tag}.png"
    if p.exists():
        return Image.open(p).convert("RGBA")
    if kind == "emoji":
        code = _emoji_code(key)
        cands = [code, code.replace("-fe0f", ""), code.split("-200d")[0]]
        svg = None
        for cd in dict.fromkeys(cands):
            for tpl in TWEMOJI:
                try:
                    svg = _get(tpl.format(cd)); break
                except Exception:
                    continue
            if svg:
                break
        if not svg:
            return None
        USED.add("twemoji")
        img = Image.open(io.BytesIO(_png_from_svg(svg, size))).convert("RGBA")
    else:
        try:
            if kind == "fa":
                e = _fa_meta()[key]["svg"]["solid"]
                svg = e["raw"]
            else:
                svg = _get(MDI_BASE + f"svg/{key}.svg").decode("utf8")
        except Exception:
            return None
        svg = svg.replace("<path ", '<path fill="#ffffff" ')
        m = re.search(r'viewBox="([\d.\s-]+)"', svg)
        vw, vh = (float(v) for v in m.group(1).split()[2:4]) if m else (24.0, 24.0)
        box = int(size * 0.60); k = box / max(vw, vh)
        import cairosvg
        glyph = Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg.encode("utf8"), output_width=max(2, round(vw * k)), output_height=max(2, round(vh * k))))).convert("RGBA")
        col = PALETTE[int(hashlib.md5(key.encode()).hexdigest(), 16) % len(PALETTE)]
        S = size * 2
        badge = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        d = ImageDraw.Draw(badge)
        d.ellipse([20, 20, S - 20, S - 20], fill=col + (255,))
        d.ellipse([20, 20, S - 20, S - 20], outline=tuple(min(255, c + 70) for c in col) + (255,), width=14)
        hl = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        ImageDraw.Draw(hl).ellipse([int(S * 0.16), int(S * 0.1), int(S * 0.62), int(S * 0.42)], fill=(255, 255, 255, 46))
        badge = Image.alpha_composite(badge, hl.filter(ImageFilter.GaussianBlur(14))).resize((size, size), Image.LANCZOS)
        badge.alpha_composite(glyph, ((size - glyph.width) // 2, (size - glyph.height) // 2))
        img = badge
        USED.add("fa" if kind == "fa" else "mdi")
    img.save(p)
    return img


def credit_line():
    parts = []
    if "twemoji" in USED:
        parts.append("Emoji graphics: Twemoji, CC-BY 4.0 (c) Twitter, Inc. and contributors.")
    if "fa" in USED:
        parts.append("Icons: Font Awesome Free, CC-BY 4.0 (fontawesome.com).")
    return " ".join(parts)

#!/usr/bin/env python3
"""One run -> a finished 9:16 Short (video + title/description). Free tools only.
LLM: GitHub Models (uses the workflow's GITHUB_TOKEN). Voice: edge-tts. Render: paper-doc engine + ffmpeg."""
import asyncio, json, os, random, re, subprocess, sys, urllib.error, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK, OUT = ROOT / "work", ROOT / "out"
ENGINE = ROOT / "paper-doc" / "scripts" / "paper_doc.py"
TOPIC = os.environ.get("TOPIC", "").strip()
VOICE = os.environ.get("VOICE", "en-US-GuyNeural")
MODEL = os.environ.get("MODEL", "openai/gpt-4.1-mini")
OFFLINE = os.environ.get("OFFLINE") == "1"
GAP, LEAD, TAIL = 0.25, 0.3, 1.2
COLORS = {"red", "ink", "green", "yellow", "navy", "orange", "blue"}

PROMPT = """You write scripts for faceless 30-45 second YouTube Shorts / Reels of surprising, TRUE facts.
{topic_rule}
Avoid these topics already used: {used}
Return ONLY JSON:
{{"topic": "short topic name",
 "title": "YouTube title under 70 chars",
 "description": "2 short sentences",
 "hashtags": ["#shorts", "..."],
 "facts_to_verify": ["each factual claim with a number or name"],
 "segments": [
  {{"say": "one spoken sentence, max 14 words",
    "title": ["LINE 1", "LINE 2"],
    "red": 2,
    "dark": false,
    "prop": {{"kind": "stamp|sticky|burst|text|name_tag|warning|rings|sparkle", "text": "short", "line1": "", "line2": "", "sub": ""}}}}
 ]}}
Rules: 8 to 11 segments, 85-110 spoken words in total. Segment 1 is a hook with the strangest fact first;
the last segment is a twist or payoff that makes people share. Real, well documented facts only; no made-up numbers.
Each title line is UPPERCASE, max 13 characters, 1-2 words. 'text' props max 9 characters. 'sticky' text is "?" or "!".
'dark' true for 'how it works' segments. Mix prop kinds; use name_tag for the subject's name once."""

SAMPLE = {"topic": "Mantis shrimp", "title": "The shrimp that punches like a bullet", "description": "Test.",
 "hashtags": ["#shorts"], "facts_to_verify": [], "segments": [
 {"say": "Imagine a shrimp that can punch through glass.", "title": ["A SHRIMP THAT", "PUNCHES?"], "red": 2, "prop": {"kind": "sticky", "text": "?"}},
 {"say": "This is not a movie, it is completely real.", "title": ["IT IS", "REAL"], "red": 2, "prop": {"kind": "stamp", "text": "TRUE", "color": "green"}},
 {"say": "Meet the mantis shrimp, about ten centimetres long.", "title": ["MEET THE", "MANTIS SHRIMP"], "red": 2, "prop": {"kind": "name_tag", "line1": "MANTIS", "line2": "SHRIMP", "sub": "10 CM LONG"}},
 {"say": "Its punch moves at about twenty three metres per second.", "title": ["FASTER THAN", "A BULLET?"], "red": 2, "dark": True, "prop": {"kind": "burst", "text": "23 M/S"}},
 {"say": "The punch collapses water into bubbles that flash with heat.", "title": ["BOILING", "BUBBLES"], "red": 2, "dark": True, "prop": {"kind": "rings"}},
 {"say": "Share this with someone who loves weird animals.", "title": ["MANTIS", "SHRIMP"], "red": 1, "prop": {"kind": "burst", "text": "WOW!"}}]}


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def used_topics():
    p = ROOT / "topics.txt"
    return [l.strip() for l in p.read_text().splitlines() if l.strip()][-60:] if p.exists() else []


def _extract_json(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        raise ValueError("no JSON object in reply: " + txt[:200])
    return json.loads(m.group(0))


def _ask_github(prompt, model, json_mode):
    import requests
    payload = {"model": model, "temperature": 0.9, "messages": [{"role": "user", "content": prompt}]}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    r = requests.post("https://models.github.ai/inference/chat/completions", json=payload, timeout=120, headers={
        "Authorization": "Bearer " + os.environ["GITHUB_TOKEN"], "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"})
    info = f"HTTP {r.status_code} ct={r.headers.get('content-type')} redirects={[h.status_code for h in r.history]} body={r.text[:300]!r}"
    if r.status_code != 200:
        raise RuntimeError(info)
    try:
        return _extract_json(r.json()["choices"][0]["message"]["content"])
    except Exception as e:
        raise RuntimeError(f"{type(e).__name__}: {e} | {info}")


def _ask_gemini(prompt):
    import requests
    model = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
    r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                      params={"key": os.environ["GEMINI_API_KEY"]}, timeout=120,
                      json={"contents": [{"parts": [{"text": prompt}]}],
                            "generationConfig": {"temperature": 0.9, "responseMimeType": "application/json"}})
    info = f"HTTP {r.status_code} body={r.text[:300]!r}"
    if r.status_code != 200:
        raise RuntimeError(info)
    try:
        return _extract_json(r.json()["candidates"][0]["content"]["parts"][0]["text"])
    except Exception as e:
        raise RuntimeError(f"{type(e).__name__}: {e} | {info}")


def _ask_compat(name, url, key, model, prompt, json_mode):
    """Any OpenAI-compatible chat endpoint (Groq, OpenRouter, GitHub legacy)."""
    import requests
    payload = {"model": model, "temperature": 0.9, "messages": [{"role": "user", "content": prompt}]}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    r = requests.post(url, json=payload, timeout=120, headers={"Authorization": "Bearer " + key})
    info = f"HTTP {r.status_code} ct={r.headers.get('content-type')} body={r.text[:300]!r}"
    if r.status_code != 200:
        raise RuntimeError(info)
    try:
        return _extract_json(r.json()["choices"][0]["message"]["content"])
    except Exception as e:
        raise RuntimeError(f"{type(e).__name__}: {e} | {info}")


def llm():
    rule = (f"Topic: {TOPIC}." if TOPIC else
            "Pick ONE topic yourself from this random category: " + random.choice(
                ["animals", "the human body", "space", "history", "ocean", "insects", "physics", "food science", "plants", "geology"]) +
            ". Choose something surprising that most people do not know.")
    prompt = PROMPT.format(topic_rule=rule, used=", ".join(used_topics()) or "none")
    env = os.environ.get
    attempts = []
    if env("GROQ_API_KEY"):
        for model in ("llama-3.3-70b-versatile", "llama-3.1-8b-instant"):
            for jm in (True, False):
                attempts.append((f"groq {model} json={jm}", lambda model=model, jm=jm: _ask_compat(
                    "groq", "https://api.groq.com/openai/v1/chat/completions", env("GROQ_API_KEY"), model, prompt, jm)))
    if env("OPENROUTER_API_KEY"):
        for model in ("meta-llama/llama-3.3-70b-instruct:free", "google/gemma-3-27b-it:free", "mistralai/mistral-small-3.2-24b-instruct:free"):
            attempts.append((f"openrouter {model}", lambda model=model: _ask_compat(
                "openrouter", "https://openrouter.ai/api/v1/chat/completions", env("OPENROUTER_API_KEY"), model, prompt, False)))
    if env("GEMINI_API_KEY"):
        attempts.append(("gemini", lambda: _ask_gemini(prompt)))
    if env("GITHUB_TOKEN"):
        attempts.append(("github-legacy gpt-4o-mini", lambda: _ask_compat(
            "github", "https://models.inference.ai.azure.com/chat/completions", env("GITHUB_TOKEN"), "gpt-4o-mini", prompt, False)))
        attempts.append(("github gpt-4.1-mini", lambda: _ask_github(prompt, "openai/gpt-4.1-mini", False)))
    if not attempts:
        raise SystemExit("No LLM key found. Add a repo secret GROQ_API_KEY (free at console.groq.com).")
    errors = []
    for name, fn in attempts:
        try:
            return fn()
        except Exception as e:
            errors.append(f"{name}: {e}")
            print("LLM attempt failed ->", errors[-1][:500], flush=True)
    raise SystemExit("LLM FAILED. See the 'LLM attempt failed' lines above for the raw server reply.")


def clean(s, n):
    s = re.sub(r"[^\w\s'?!.%°/+-]", "", str(s).upper()).strip()
    return s[:n].strip()


def two_lines(title):
    words = " ".join(map(str, title)).upper().split() if isinstance(title, list) else str(title).upper().split()
    words = [clean(w, 13) for w in words if w] or ["WOW"]
    best = (10 ** 9, [" ".join(words), ""])
    for k in range(1, len(words) + 1):
        a, b = " ".join(words[:k]), " ".join(words[k:])
        score = max(len(a), len(b))
        if score < best[0]:
            best = (score, [a, b])
    a, b = best[1]
    return [a[:13].strip(), b[:13].strip()] if b else [a[:13].strip(), ""]


def element(p, dark):
    k = str(p.get("kind", "")).lower()
    base = {"x": 540, "y": 1180, "at": 0.4}
    t = clean(p.get("text", ""), 9)
    if k == "stamp":
        c = p.get("color") if p.get("color") in COLORS else "red"
        return {**base, "type": "stamp", "text": t or "TRUE", "color": c, "tilt": random.choice([-8, -5, 6]), "anim": "slam"}
    if k == "sticky":
        return {**base, "type": "sticky", "text": (t or "?")[:2], "rot": random.choice([-4, 3])}
    if k == "burst":
        return {**base, "type": "burst", "text": t or "WOW!", "size": 0.85, "color": "yellow", "rot": -4, "anim": "punch"}
    if k == "text":
        return {**base, "type": "text", "text": t or "!", "size": 140, "color": "xray" if dark else "ink"}
    if k == "name_tag":
        return {**base, "type": "name_tag", "line1": clean(p.get("line1") or t, 10) or "NAME", "line2": clean(p.get("line2"), 10),
                "sub": clean(p.get("sub"), 24), "rot": -3}
    if k == "warning":
        return {**base, "type": "warning", "mark": "!"}
    if k == "rings":
        return {**base, "type": "rings", "color": "blue", "n": 4}
    return {**base, "type": "sparkle", "n": 14, "r": 380}


async def _edge(text, path):
    import edge_tts
    await edge_tts.Communicate(text, VOICE, rate="+4%").save(str(path))


def tts(text, path):
    try:
        asyncio.run(_edge(text, path))
        if Path(path).stat().st_size > 1000:
            return
        raise RuntimeError("edge-tts produced an empty file")
    except Exception as e:
        print("edge-tts failed, falling back to gTTS:", type(e).__name__, e, flush=True)
    try:
        from gtts import gTTS
        gTTS(text=text, lang="en").save(str(path))
    except Exception as e:
        raise SystemExit(f"TTS FAILED (edge-tts and gTTS): {type(e).__name__} {e}")


def duration(p):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)]).strip())


def main():
    WORK.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
    data = SAMPLE if OFFLINE else llm()
    segs = [s for s in data.get("segments", []) if str(s.get("say", "")).strip()][:12]
    if len(segs) < 3:
        raise SystemExit("LLM returned too few segments")
    print("Topic:", data.get("topic"), "| segments:", len(segs))

    # voice: one clip per segment -> exact scene timing
    wavs, durs = [], []
    sh(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", str(GAP), str(WORK / "gap.wav")])
    sh(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", str(LEAD), str(WORK / "lead.wav")])
    for i, s in enumerate(segs):
        wav = WORK / f"seg{i}.wav"
        if OFFLINE:
            n = len(s["say"].split()) * 0.38
            sh(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", str(n), str(wav)])
        else:
            mp3 = WORK / f"seg{i}.mp3"
            tts(s["say"], mp3)
            sh(["ffmpeg", "-v", "error", "-y", "-i", str(mp3), "-ar", "44100", "-ac", "2", str(wav)])
        wavs.append(wav); durs.append(duration(wav))
    lst = ["file 'lead.wav'"]
    for w in wavs:
        lst += [f"file '{w.name}'", "file 'gap.wav'"]
    (WORK / "list.txt").write_text("\n".join(lst))
    sh(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(WORK / "list.txt"), str(WORK / "voice.wav")])

    # scene plan
    scenes, t = [], LEAD
    for i, (s, d) in enumerate(zip(segs, durs)):
        dark = bool(s.get("dark"))
        sc = {"start": round(t, 2), "title": two_lines(s.get("title", ["WOW", ""])), "red": 2 if s.get("red", 2) == 2 else 1,
              "page": "dark" if dark else "kraft", "elements": [element(s.get("prop", {}), dark)]}
        if not dark and i % 2 == 0:
            sc["scenery"] = i % 5 + 1
        if not sc["title"][1]:
            sc["title"] = [sc["title"][0], ""]
        scenes.append(sc)
        t += d + GAP
    total = round(t + TAIL, 2)
    plan = {"meta": {"voice": str(WORK / "voice.wav"), "out": str(OUT / "short.mp4"), "duration": total,
                     "header": clean(data.get("topic", "FACTS"), 18) + " FILES", "music": {"db": -21}, "whoosh_db": -20},
            "subjects": {}, "sounds": [{"t": round(t - d - GAP + 0.1, 2), "sfx": "boom", "db": -9}], "scenes": scenes}
    (WORK / "plan.json").write_text(json.dumps(plan, indent=1))
    print("Rendering", total, "s ...")
    sh([sys.executable, str(ENGINE), "render", str(WORK / "plan.json")])

    tags = " ".join(data.get("hashtags", ["#shorts"]))
    (OUT / "upload_info.txt").write_text(
        f"TITLE:\n{data.get('title','')}\n\nDESCRIPTION:\n{data.get('description','')}\n\n{tags}\n\n"
        "CHECK THESE FACTS BEFORE POSTING:\n" + "\n".join("- " + str(f) for f in data.get("facts_to_verify", [])) + "\n")
    if not OFFLINE:
        with open(ROOT / "topics.txt", "a") as f:
            f.write(str(data.get("topic", "")).strip() + "\n")
    print("DONE ->", OUT / "short.mp4")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""One run -> a finished 9:16 Short: AI story + CODE-DRAWN ANIMATION that matches each sentence + voice + captions + music.
Free tools only. LLM: Groq/Gemini/etc. Animation: anim.py (Pillow). Voice: edge-tts / gTTS. Render: ffmpeg."""
import asyncio, json, os, random, re, subprocess, sys, urllib.error, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import anim

ROOT = Path(__file__).resolve().parent
WORK, OUT = ROOT / "work", ROOT / "out"
TOPIC = os.environ.get("TOPIC", "").strip()
VOICE = os.environ.get("VOICE", "en-US-GuyNeural")
MODEL = os.environ.get("MODEL", "openai/gpt-4.1-mini")
OFFLINE = os.environ.get("OFFLINE") == "1"
GAP = 0.08            # tiny breath between sentences (no dead air)
SPEED = 1.08          # voice speed-up for a punchier feel
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

PROMPT = """You write scripts for faceless 35-45 second animated YouTube Shorts telling a surprising TRUE story or fact.
{topic_rule}
Avoid these topics already used: {used}
Each sentence is shown as an ANIMATED SCENE built from a fixed library, so choose visuals that LITERALLY show the sentence. Return ONLY JSON:
{{"topic": "short topic name",
 "title": "YouTube title under 70 chars",
 "description": "2 short sentences",
 "hashtags": ["#shorts", "..."],
 "facts_to_verify": ["each factual claim with a number or name"],
 "segments": [
  {{"say": "one spoken sentence, max 14 words, no emojis",
    "scene": "ONE of: {scenes}",
    "objects": ["1-3 of: {objects}"],
    "extras": ["0-2 of: {extras}"],
    "camera": "ONE of: {cameras}",
    "big_text": "optional key number or word shown huge, max 12 chars, e.g. 100°C (empty string if none)"}}
 ]}}
Rules: 9 to 12 segments, 90-115 spoken words in total. Segment 1 is a hook with the strangest fact first;
the last segment is a twist that makes people share. Real, well documented facts only; no invented numbers.
HOW TO PICK VISUALS: most segments should use a stage_* scene (stage_nature, stage_lab, stage_warm, stage_dark, stage_bright)
plus 1-3 'objects' that show what the sentence is about (an animal sentence -> that animal; heat -> thermometer or fire;
chemicals -> flask; speed -> stopwatch; a blast -> explosion; a clever idea -> lightbulb). Use a world scene (mountains, ocean,
space, city, forest, desert, volcano, climb...) ONLY when the story is really set in that place, and then usually leave objects empty.
Never use a scene that does not match the story (no mountains for an insect story). Vary the stage and camera between segments.
Use big_text for the most surprising number in about half of the segments."""

SAMPLE = {"topic": "Bombardier beetle", "title": "The beetle that shoots boiling acid", "description": "Test.", "hashtags": ["#shorts"],
 "facts_to_verify": [], "segments": [
 {"say": "This tiny beetle can shoot boiling liquid from its body.", "scene": "stage_nature", "objects": ["beetle"], "extras": [], "camera": "zoom_in", "big_text": ""},
 {"say": "The spray is almost one hundred degrees Celsius.", "scene": "stage_warm", "objects": ["thermometer"], "extras": [], "camera": "none", "big_text": "100°C"},
 {"say": "Inside its body, two chemicals are kept apart until danger arrives.", "scene": "stage_lab", "objects": ["flask"], "extras": [], "camera": "pan_right", "big_text": ""},
 {"say": "When they mix, enzymes make them explode into hot spray.", "scene": "stage_dark", "objects": ["explosion"], "extras": ["shake"], "camera": "zoom_in", "big_text": ""},
 {"say": "It repeats this about five hundred times every second.", "scene": "stage_bright", "objects": ["stopwatch"], "extras": [], "camera": "zoom_out", "big_text": "500/SEC"},
 {"say": "Frogs and spiders learn quickly not to attack it.", "scene": "stage_nature", "objects": ["frog", "spider"], "extras": [], "camera": "pan_left", "big_text": ""},
 {"say": "Scientists study its design to build safer engines.", "scene": "stage_lab", "objects": ["magnifier", "gear"], "extras": [], "camera": "none", "big_text": ""}]}


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
    payload = {"model": model, "temperature": 0.9, "max_tokens": 6000, "messages": [{"role": "user", "content": prompt}]}
    if "gpt-oss" in model:
        payload["reasoning_effort"] = "low"
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


def groq_models(key):
    """Ask Groq which chat models this key can use, best first."""
    import requests
    try:
        r = requests.get("https://api.groq.com/openai/v1/models", headers={"Authorization": "Bearer " + key}, timeout=30)
        ids = [m["id"] for m in r.json().get("data", []) if m.get("active", True)]
        print("Groq models available:", ids, flush=True)
    except Exception as e:
        print("Could not list Groq models:", e, flush=True)
        ids = []
    skip = ("whisper", "guard", "tts", "playai", "orpheus", "embed", "safeguard", "distil", "compound")
    ids = [i for i in ids if not any(k in i.lower() for k in skip)]
    prefs = ("gpt-oss-120b", "llama-3.3-70b", "70b", "120b", "qwen", "kimi", "gpt-oss-20b", "llama", "gemma", "mixtral")
    def rank(i):
        for n, k in enumerate(prefs):
            if k in i.lower():
                return n
        return len(prefs)
    ids.sort(key=rank)
    return ids[:4] or ["llama-3.3-70b-versatile"]


def llm():
    rule = (f"Topic: {TOPIC}." if TOPIC else
            "Pick ONE topic yourself from this random category: " + random.choice(
                ["animals", "the human body", "space", "history", "ocean", "insects", "physics", "food science", "plants", "geology"]) +
            ". Choose something surprising that most people do not know.")
    prompt = PROMPT.format(topic_rule=rule, used=", ".join(used_topics()) or "none", scenes=", ".join(anim.SCENES), objects=", ".join(anim.OBJECTS),
                           extras=", ".join(anim.EXTRAS), cameras=", ".join(anim.CAMERAS))
    env = os.environ.get
    attempts = []
    if env("GROQ_API_KEY"):
        for model in groq_models(env("GROQ_API_KEY")):
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



def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def used_topics():
    p = ROOT / "topics.txt"
    return [l.strip() for l in p.read_text().splitlines() if l.strip()][-60:] if p.exists() else []


async def _edge(text, path):
    import edge_tts
    await edge_tts.Communicate(text, VOICE, rate="+0%").save(str(path))


def _groq_tts(text, path):
    """Groq Orpheus: expressive free AI voice. Needs GROQ_API_KEY + terms accepted once in the Groq console."""
    import requests
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("no GROQ_API_KEY")
    r = requests.post("https://api.groq.com/openai/v1/audio/speech", headers={"Authorization": "Bearer " + key}, timeout=90,
                      json={"model": "canopylabs/orpheus-v1-english", "input": text[:195],
                            "voice": os.environ.get("GROQ_VOICE", "troy"), "response_format": "wav"})
    if r.status_code != 200 or len(r.content) < 2000:
        raise RuntimeError(f"HTTP {r.status_code} {r.text[:300]}")
    Path(path).write_bytes(r.content)


def _edge_tts(text, path):
    asyncio.run(_edge(text, path))
    if Path(path).stat().st_size < 1000:
        raise RuntimeError("empty file")


def _gtts(text, path):
    from gtts import gTTS
    gTTS(text=text, lang="en").save(str(path))


def tts(text, path):
    errors = []
    for name, fn in (("groq-orpheus", _groq_tts), ("edge-tts", _edge_tts), ("gTTS", _gtts)):
        try:
            fn(text, path)
            print(f"  voice: {name}", flush=True)
            return
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__} {str(e)[:250]}")
            print("TTS failed ->", errors[-1], flush=True)
    raise SystemExit("ALL TTS ENGINES FAILED:\n" + "\n".join(errors))


def duration(p):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)]).strip())


# ---------- captions ----------
def chunk_words(text, limit=15):
    words = re.sub(r"[^\w\s'’%.,?!$-]", "", text).upper().split()
    chunks, cur = [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > limit:
            chunks.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        chunks.append(cur)
    return chunks or [text.upper()[:limit]]


def caption_filters(text, dur, i):
    chunks = chunk_words(text)
    total = sum(len(c) for c in chunks)
    t, parts = 0.0, []
    for n, c in enumerate(chunks):
        d = dur * len(c) / total
        f = WORK / f"cap{i}_{n}.txt"
        f.write_text(c)
        end = dur + 1 if n == len(chunks) - 1 else t + d
        parts.append(f"drawtext=fontfile={FONT}:textfile={f}:fontsize=80:fontcolor=white:borderw=9:bordercolor=black"
                     f":x=(w-text_w)/2:y=h*0.66:enable='between(t,{t:.3f},{end:.3f})'")
        t += d
    return ",".join(parts)


def main():
    WORK.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
    data = SAMPLE if OFFLINE else llm()
    segs = [s for s in data.get("segments", []) if str(s.get("say", "")).strip()][:13]
    if len(segs) < 3:
        raise SystemExit("LLM returned too few segments")
    topic = str(data.get("topic", "facts"))
    print("Topic:", topic, "| segments:", len(segs), flush=True)

    # 1) voice, one clip per sentence, silence trimmed at both ends -> no dead air
    wavs, durs = [], []
    trim = "silenceremove=start_periods=1:start_threshold=-45dB,areverse,silenceremove=start_periods=1:start_threshold=-45dB,areverse"
    for i, s in enumerate(segs):
        raw = WORK / f"seg{i}_raw.wav"
        if OFFLINE:
            n = len(s["say"].split()) * 0.38
            sh(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=frequency={200 + 30 * i}:duration={n}", "-af",
                "volume=0.3,adelay=500|500,apad=pad_dur=0.6", "-ar", "44100", "-ac", "2", str(raw)])
        else:
            mp3 = WORK / f"seg{i}.mp3"
            tts(s["say"], mp3)
            sh(["ffmpeg", "-v", "error", "-y", "-i", str(mp3), "-ar", "44100", "-ac", "2", str(raw)])
        wav = WORK / f"seg{i}.wav"
        sh(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-af", f"{trim},atempo={SPEED},apad=pad_dur={GAP}", str(wav)])
        wavs.append(wav); durs.append(duration(wav))
    (WORK / "list.txt").write_text("\n".join(f"file '{w.name}'" for w in wavs))
    sh(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(WORK / "list.txt"), "-c", "copy", str(WORK / "voice.wav")])
    total = sum(durs)
    print(f"Voice: {total:.1f}s", flush=True)

    # 2) one ANIMATED scene per sentence, captions burned in
    vids = []
    for i, (s, d) in enumerate(zip(segs, durs)):
        spec = anim.make_spec(s, i)
        out = WORK / f"v{i}.mp4"
        anim.render_segment(spec, d, out, vf_extra=caption_filters(s["say"], d - GAP, i))
        vids.append(out)
        print(f"  segment {i + 1}/{len(segs)} ok ({spec['scene']}, {spec['cam']})", flush=True)
    (WORK / "vlist.txt").write_text("\n".join(f"file '{v.name}'" for v in vids))
    sh(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(WORK / "vlist.txt"), "-c", "copy", str(WORK / "video.mp4")])

    # 3) continuous music bed + voice, loudness-normalised
    sh(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=110", "-f", "lavfi", "-i", "sine=frequency=165",
        "-f", "lavfi", "-i", "sine=frequency=220", "-filter_complex",
        "[0][1][2]amix=inputs=3,lowpass=f=700,tremolo=f=0.3:d=0.5", "-t", f"{total:.2f}", str(WORK / "music.wav")])
    fade = max(total - 1.0, 0)
    sh(["ffmpeg", "-v", "error", "-y", "-i", str(WORK / "video.mp4"), "-i", str(WORK / "voice.wav"), "-i", str(WORK / "music.wav"),
        "-filter_complex",
        f"[2:a]volume=0.10,afade=t=in:d=0.8,afade=t=out:st={fade:.2f}:d=1[m];[1:a][m]amix=inputs=2:duration=first:dropout_transition=0,"
        "volume=2,loudnorm=I=-14:TP=-1.5[a]",
        "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-t", f"{total:.2f}", str(OUT / "short.mp4")])

    tags = " ".join(data.get("hashtags", ["#shorts"]))
    (OUT / "upload_info.txt").write_text(
        f"TITLE:\n{data.get('title','')}\n\nDESCRIPTION:\n{data.get('description','')}\n\n{tags}\n\n"
        "CHECK THESE FACTS BEFORE POSTING:\n" + "\n".join("- " + str(f) for f in data.get("facts_to_verify", [])) +
        "\n\nANIMATION: generated by code (no third-party footage).\n")
    if not OFFLINE:
        with open(ROOT / "topics.txt", "a") as f:
            f.write(topic.strip() + "\n")
    print("DONE ->", OUT / "short.mp4", flush=True)


if __name__ == "__main__":
    main()

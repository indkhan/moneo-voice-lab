"""Spike 1: Woof TTS speed and Whistle STT accuracy/latency on this laptop CPU (round trip)."""
import json, os, re, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "models", "woof"))
OUT = os.path.join(ROOT, "audio"); os.makedirs(OUT, exist_ok=True)

SENTS = [
    "Yo, Netto's forty two euros this month.",
    "Let me look through your transactions.",
    "You spent 312.40 euros on groceries in October, about 18 percent more than September.",
    "Your net worth is 48,250 euros, up 1,300 since last month.",
    "There's a card on screen for the 12.40 at Rewe, tap it to save.",
    "Rent of 950 euros goes out on the third, and your Scalable portfolio is up 2.4 percent.",
]
USER = [
    "How much did I spend on groceries last month?",
    "By the way I spent twelve forty at Netto and I have eighty euros in cash.",
    "No, I meant September, not October.",
    "What is my net worth right now and how did it change?",
]

def resample(x, a, b):
    n = int(len(x) * b / a)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)

def words(s): return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()
def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))

res = {"tts": [], "stt": [], "stream": []}
from underdog_woof import Woof
for voice in ("juno", "mira"):
    t = time.perf_counter(); tts = Woof(voice=voice); load = time.perf_counter() - t
    tts("Warm up.")
    for i, s in enumerate(SENTS):
        t = time.perf_counter(); wav = tts(s, seed=1); dt = time.perf_counter() - t
        tts.save(os.path.join(OUT, f"{voice}_{i}.wav"), wav)
        res["tts"].append({"voice": voice, "chars": len(s), "audio_s": round(len(wav) / 24000, 2), "synth_s": round(dt, 2), "x_realtime": round(len(wav) / 24000 / dt, 2), "load_s": round(load, 2)})
        print(res["tts"][-1], flush=True)

juno = Woof(voice="juno")
import cactus
clips = [(s, resample(juno(s, seed=2), 24000, 16000)) for s in USER]
cactus.transcribe(clips[0][1])  # warm up / load
for s, a in clips + [(s, resample(juno(s, seed=3), 24000, 16000)) for s in SENTS[2:4]]:
    t = time.perf_counter(); r = cactus.transcribe(a, keywords=["Netto", "Rewe", "Scalable"]); dt = time.perf_counter() - t
    res["stt"].append({"ref": s, "hyp": r["text"], "wer": round(wer(s, r["text"]), 3), "audio_s": round(len(a) / 16000, 2), "wall_ms": round(dt * 1000), "ttft_ms": round(r["ttft_ms"], 1), "lang": r["language"]})
    print(res["stt"][-1], flush=True)

# Streaming: feed 0.5 s chunks, record when each word gets committed relative to audio time.
for s, a in clips[:2]:
    chunks = [a[i:i + 8000] for i in range(0, len(a), 8000)]
    for ev in cactus.stream(iter(chunks), language="en"):
        res["stream"].append({"received_s": ev.get("received"), "pass_ms": round(ev.get("pass_ms", 0), 1), "text": ev.get("text"), "pending": ev.get("pending")})
        print(res["stream"][-1], flush=True)

json.dump(res, open(os.path.join(ROOT, "spike", "spike1.json"), "w"), indent=1)

"""Transcribe each clip with Whistle (local, CPU), with the workspace's names as keywords like the app does."""
import glob, json, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
import soundfile as sf, numpy as np
import cactus

folder = sys.argv[1]
names = ["Rewe", "Lidl", "Amazon", "Netflix", "Spotify", "Uber", "Shell", "Vapiano", "Starbucks", "N26", "Cash wallet",
         "Groceries", "Shopping", "Subscriptions", "Transport", "Restaurants", "Coffee", "Rent", "Salary"]
out = []
first = True
for path in sorted(glob.glob(f"{folder}/u*.wav")):
    audio, rate = sf.read(path, dtype="float32")
    assert rate == 16000
    if first:
        cactus.transcribe(audio[:16000]); first = False
    row = {"clip": os.path.basename(path), "audio_s": round(len(audio) / 16000, 1)}
    for label, kw in (("plain", None), ("names", names)):
        t = time.perf_counter()
        r = cactus.transcribe(audio, keywords=kw) if kw else cactus.transcribe(audio)
        row[label] = r["text"].strip(); row[label + "_ms"] = round((time.perf_counter() - t) * 1000)
    out.append(row)
    print(f"{row['clip']} {row['audio_s']:5.1f}s {row['names_ms']:5d}ms | {row['names']}\n{'':24}plain | {row['plain']}", flush=True)
json.dump(out, open(f"{folder}/transcripts.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

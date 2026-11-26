"""Transcribe WAVs under audio/<dir>/ with Whistle (cactus). Results cached in transcripts.json.
python transcribe.py <dir> [<dir> ...] [--seeds 0,1,2]"""
import json, os, sys, time, wave
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "transcripts.json")
import cactus

args = [a for a in sys.argv[1:] if not a.startswith("--")]
seeds = None
for a in sys.argv[1:]:
    if a.startswith("--seeds="):
        seeds = {f"seed{s}" for s in a.split("=")[1].split(",")}
done = {}
if os.path.exists(OUT):
    with open(OUT, encoding="utf-8") as f:
        done = json.load(f)


def load16k(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    if sr != 16000:
        x = np.interp(np.arange(0, len(x), sr / 16000.0), np.arange(len(x)), x).astype(np.float32)
    return x


for d in args:
    folder = os.path.join(HERE, "audio", d)
    if not os.path.isdir(folder):
        print("missing", d); continue
    for name in sorted(os.listdir(folder)):
        key = f"{d}/{name}"
        if not name.endswith(".wav") or key in done:
            continue
        if seeds is not None and "seed" in name and name[:-4].split("_")[-1] not in seeds:
            continue
        x = load16k(os.path.join(folder, name))
        t0 = time.perf_counter()
        try:
            text = cactus.transcribe(x, language="en")["text"].strip() if len(x) > 800 else ""
        except Exception as ex:  # noqa: BLE001
            text = ""; print("ERR", key, repr(ex))
        done[key] = text
        print(key, "%.1fs" % (time.perf_counter() - t0), "|", text, flush=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(done, f, indent=1, ensure_ascii=False)

"""Score amount fidelity: every number in the sentence must come back exactly after TTS -> Whistle."""
import json, os, sys, wave
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("CACTUS_TELEMETRY", "0")
import cactus
from textnorm import numbers_recovered
from fidelity_gen import FID

def load16k(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate(); x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    n = int(len(x) * 16000 / sr)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)

out = {}
for folder in sorted(os.listdir(os.path.join(HERE, "audio"))):
    if not folder.startswith("fid_"): continue
    ok = nums_ok = nums_all = 0; fails = []
    for i, (text, nums) in enumerate(FID):
        p = os.path.join(HERE, "audio", folder, f"{i:02d}.wav")
        if not os.path.exists(p): continue
        hyp = cactus.transcribe(load16k(p), language="en")["text"]
        good, found = numbers_recovered(nums, hyp)
        g = sum(bool(x) for x in good) if isinstance(good, (list, tuple)) else int(good)
        allok = (g == len(nums)) if isinstance(good, (list, tuple)) else bool(good)
        ok += allok; nums_ok += g if isinstance(good, (list, tuple)) else (len(nums) if good else 0); nums_all += len(nums)
        if not allok: fails.append({"text": text, "heard": hyp})
    out[folder] = {"sentences_exact": ok, "of": len(FID), "numbers_ok": nums_ok, "numbers": nums_all, "fails": fails}
    print(folder, f"sentences exact {ok}/{len(FID)}", flush=True)
    for f in fails[:6]: print("   ", f["text"], "=>", f["heard"])
json.dump(out, open(os.path.join(HERE, "res_fidelity.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)

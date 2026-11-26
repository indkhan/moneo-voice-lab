"""Transcribe every clean clip with Whistle (word timestamps); store results in corpus/clean_asr.json."""
import os, json, time
os.environ["CACTUS_TELEMETRY"] = "0"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import numpy as np, soundfile as sf, cactus
HERE = os.path.dirname(os.path.abspath(__file__))
man = json.load(open(os.path.join(HERE, "corpus", "manifest.json"), encoding="utf-8"))
out = {}
x, _ = sf.read(man["assistant"][0]["wav"], dtype="float32"); cactus.transcribe(x, language="en")  # warm up
for kind in ["assistant", "interrupt", "backchannel"]:
    for e in man[kind]:
        x, sr = sf.read(e["wav"], dtype="float32")
        t0 = time.perf_counter(); r = cactus.transcribe(x, language="en", word_timestamps=True); ms = (time.perf_counter() - t0) * 1e3
        rms = float(np.sqrt(np.mean(x ** 2))); pk = float(np.abs(x).max())
        # active region by energy
        env = np.sqrt(np.convolve(x ** 2, np.ones(320) / 320, mode="same")); act = np.where(env > 0.1 * env.max())[0]
        a0, a1 = (act[0] / sr, act[-1] / sr) if len(act) else (0, 0)
        out[os.path.basename(e["wav"])] = {"text": r["text"], "words": r.get("words"), "ms": ms, "dur": len(x) / sr, "act": [a0, a1]}
        print(f'{os.path.basename(e["wav"]):22s} dur={len(x)/sr:5.2f} act={a0:4.2f}-{a1:4.2f} rms={rms:.3f} pk={pk:.2f} {ms:5.0f}ms | {e["text"][:40]!r} -> {r["text"]!r}', flush=True)
        if kind == "assistant" and e["id"] == 0:
            print(r)
json.dump(out, open(os.path.join(HERE, "corpus", "clean_asr.json"), "w", encoding="utf-8"), indent=1)

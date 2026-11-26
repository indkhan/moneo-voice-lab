"""A/B of the fast ELU (interleaved so machine noise hits both), exactness of fast vs original, guard on runaway seeds."""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "woof_stream"))
import stream_runtime as srt
from stream_runtime import WoofStream, rt, SR
from common import SENTENCES, timed_stream
out = {"threads": os.environ.get("OPENBLAS_NUM_THREADS", "default")}
tts = WoofStream(voice="juno", schedule=(1, 2, 4)); tts.warmup()
ab = {"orig": [], "fast": []}
diffs = []
for text in (SENTENCES[4], SENTENCES[7]):
    for seed in range(4):
        wavs = {}
        for name in ("orig", "fast"):
            srt.set_fast_elu(name == "fast")
            wav, m = timed_stream(lambda: tts.stream(text, seed=seed, guard=False), SR)
            fr = tts.stats["frames"]
            ab[name].append({"dec_ms_per_frame": tts.stats["t_dec"] / fr * 1e3, "speech_ms_per_frame": tts.stats["t_speech"] / fr * 1e3,
                             "seanet_ms_per_frame": None, "rtf_rest": m["rtf_rest"], "ttfa_ms": m["ttfa_ms"]})
            wavs[name] = wav
        n = min(len(wavs["orig"]), len(wavs["fast"]))
        d = np.abs(wavs["orig"][:n] - wavs["fast"][:n])
        diffs.append({"same_length": len(wavs["orig"]) == len(wavs["fast"]), "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
                      "snr_db": float(10 * np.log10((wavs["orig"][:n] ** 2).sum() / max(1e-30, (d.astype(np.float64) ** 2).sum())))})
med = lambda k, name: float(np.median([r[k] for r in ab[name]]))
out["fast_elu"] = {name: {k: med(k, name) for k in ("dec_ms_per_frame", "speech_ms_per_frame", "rtf_rest", "ttfa_ms")} for name in ab}
out["fast_elu"]["waveform_diff_vs_original_elu"] = diffs
print(json.dumps(out["fast_elu"], indent=1))
# guard behaviour on short backchannels, both voices: frames with and without the guard
srt.set_fast_elu(True)
g = []
for voice in ("juno", "mira"):
    t = tts if voice == "juno" else WoofStream(voice="mira", schedule=(1, 2, 4))
    for text in ("Mm-hm.", "Ok.", "Sure, one moment.", SENTENCES[7]):
        for seed in range(8 if len(text) < 10 else 4):
            a = np.concatenate(list(t.stream(text, seed=seed, guard=False)))
            b = np.concatenate(list(t.stream(text, seed=seed, guard=True)))
            g.append({"voice": voice, "text": text, "seed": seed, "unguarded_s": len(a) / SR, "guarded_s": len(b) / SR, "guard": t.stats.get("aborted")})
            if t.stats.get("aborted") or len(a) != len(b):
                print(g[-1])
out["guard"] = g
print("guard fired on", sum(1 for r in g if r["guard"]), "of", len(g), "takes")
with open(os.path.join(HERE, "res_extra.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)

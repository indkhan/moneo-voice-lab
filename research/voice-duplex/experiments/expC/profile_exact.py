"""Step 1 + 2: profile the original path per frame, and check streamed audio against the whole-sentence decode."""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "woof_stream"))
from stream_runtime import WoofStream, rt, FRAME, SR

TEXTS = ["Mm-hm.", "Let me look through your transactions.",
         "You spent 312.40 euros on groceries in October, about 18 percent more than September."]
out = {"profile": [], "exact": []}

tts = WoofStream(voice="juno")
tts.warmup(); tts.full("Hello there, this is a warm up.", seed=0)
m, sp, dec = tts.m, tts.m.speech, tts.m.dec

# ---------------- profile the ORIGINAL path, split by stage ----------------
for text in TEXTS:
    ntext = tts.norm(text)
    best = None
    for rep in range(3):
        rt.RNG.seed(0)
        ids = rt.encode_text(ntext)
        t0 = time.perf_counter()
        x = np.concatenate([m.voice, sp.text_emb[np.asarray(ids)], sp.bos[None]], 0).astype(np.float32)
        p0 = x.shape[0]
        for i in range(sp.L):
            x = sp._prefix_block(i, x, p0)
        t_prefix = time.perf_counter() - t0
        rt.RNG.seed(0)
        t0 = time.perf_counter()
        lat = m.latents(ntext)
        t_lat = time.perf_counter() - t0
        T = len(lat)
        den = lat * m.std + m.mean
        t0 = time.perf_counter()
        h = dec.transformer(den @ dec.inp_w + dec.inp_b)
        t_tf = time.perf_counter() - t0
        for kind, op in dec.ops:
            if kind == "res":
                op[0].reset(); op[1].reset()
            elif op is not None:
                op.reset()
        t0 = time.perf_counter()
        xlow = dec._run(h, dec.ops[:dec.split])
        t_low = time.perf_counter() - t0
        rows = dec.chunk * xlow.shape[0] // h.shape[0]
        t0 = time.perf_counter()
        wav = np.concatenate([dec._run(np.ascontiguousarray(xlow[c:c + rows]), dec.ops[dec.split:])
                              for c in range(0, xlow.shape[0], rows)])[:, 0]
        t_high = time.perf_counter() - t0
        r = {"text": text, "normalized": ntext, "prefix_tokens": p0, "frames": T, "audio_s": T * FRAME / SR,
             "prefix_ms": t_prefix * 1e3, "latents_total_ms": t_lat * 1e3,
             "ar_ms_per_frame": (t_lat - t_prefix) / T * 1e3,
             "dec_transformer_ms_per_frame": t_tf / T * 1e3, "dec_seanet_low_ms_per_frame": t_low / T * 1e3,
             "dec_seanet_high_ms_per_frame": t_high / T * 1e3,
             "dec_total_ms_per_frame": (t_tf + t_low + t_high) / T * 1e3,
             "rtf_x_realtime": T * FRAME / SR / (t_lat + t_tf + t_low + t_high)}
        if best is None or r["rtf_x_realtime"] > best["rtf_x_realtime"]:
            best = r
    out["profile"].append(best)
    print(json.dumps(best, indent=1))

# ---------------- exactness: streamed vs whole-sentence, same seed ----------------
for voice in ("juno", "mira"):
    t = tts if voice == "juno" else WoofStream(voice="mira")
    for text in TEXTS:
        for fc, ch in ((1, 1), (1, 4), (2, 8), (3, 5)):
            t.first_chunk, t.chunk = fc, ch
            ref = t.full(text, seed=7)
            chunks = list(t.stream(text, seed=7, guard=False))
            got = np.concatenate(chunks)
            same_len = len(ref) == len(got)
            n = min(len(ref), len(got))
            d = np.abs(ref[:n] - got[:n])
            r = {"voice": voice, "text": text, "first_chunk": fc, "chunk": ch, "same_length": same_len,
                 "samples": int(len(ref)), "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
                 "ref_rms": float(np.sqrt((ref ** 2).mean())), "ref_peak": float(np.abs(ref).max()),
                 "snr_db": float(10 * np.log10((ref[:n] ** 2).sum() / max(1e-30, (d.astype(np.float64) ** 2).sum()))),
                 "first_chunk_samples": int(len(chunks[0])), "n_chunks": len(chunks)}
            out["exact"].append(r)
            print(json.dumps(r))

with open(os.path.join(HERE, "results_profile_exact.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)

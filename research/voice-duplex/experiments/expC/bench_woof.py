"""Woof streaming benchmark. Thread counts must be set in the environment before numpy loads, so run one process
per setting:  OPENBLAS_NUM_THREADS=2 python bench_woof.py sweep|main  <tag>"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "woof_stream"))
from stream_runtime import WoofStream, rt, SR, FRAME
from common import SENTENCES, timed_stream, write_wav, med

mode, tag = sys.argv[1], sys.argv[2]
threads = os.environ.get("OPENBLAS_NUM_THREADS", "default")
out = {"threads": threads, "mode": mode}


def run(tts, text, seed, guard=True):
    wav, m = timed_stream(lambda: tts.stream(text, seed=seed, guard=guard), SR)
    m.update(seed=seed, frames=tts.stats.get("frames"), aborted=tts.stats.get("aborted"),
             speech_ms_per_frame=tts.stats.get("t_speech", 0) / max(1, tts.stats.get("frames", 1)) * 1e3,
             dec_ms_per_frame=tts.stats.get("t_dec", 0) / max(1, tts.stats.get("frames", 1)) * 1e3)
    return wav, m


if mode == "sweep":
    t0 = time.perf_counter()
    tts = WoofStream(voice="juno")
    out["load_s"] = time.perf_counter() - t0
    t0 = time.perf_counter(); tts.warmup(); out["first_call_s"] = time.perf_counter() - t0
    texts = [SENTENCES[4], SENTENCES[7]]
    out["chunking"] = []
    for sched in (((1, 2, 4),) if os.environ.get("SWEEP_QUICK") else ((1, 1), (1, 4), (2, 4), (1, 2, 4), (1, 2, 4, 8), (4, 4), (8, 8))):
        tts.schedule = sched
        rows = [run(tts, t, s, guard=False)[1] for t in texts for s in range(3)]
        r = {"schedule": list(sched), "ttfa_ms_median": med([x["ttfa_ms"] for x in rows]),
             "ttfa_ms_max": max(x["ttfa_ms"] for x in rows), "rtf_rest_median": med([x["rtf_rest"] for x in rows]),
             "underruns_total": sum(x["underruns"] for x in rows),
             "needed_prebuffer_ms_max": max(x["needed_prebuffer_ms"] for x in rows),
             "speech_ms_per_frame": med([x["speech_ms_per_frame"] for x in rows]),
             "dec_ms_per_frame": med([x["dec_ms_per_frame"] for x in rows])}
        out["chunking"].append(r)
        print(threads, json.dumps(r), flush=True)
    # original whole-sentence path, fp32 vs int8 decoder
    m = tts.m
    m8 = rt.WoofNP(os.path.join(os.path.dirname(rt.__file__), "..", "woof_int8.npz"), voice="juno", dec_mode="int8")
    out["whole_sentence"] = []
    for text in texts:
        nt = tts.norm(text)
        tl, td, td8, dur = [], [], [], 0
        for s in range(3):
            rt.RNG.seed(s)
            t0 = time.perf_counter(); lat = m.latents(nt); t1 = time.perf_counter()
            w = m.dec(lat * m.std + m.mean); t2 = time.perf_counter()
            w8 = m8.dec(lat * m.std + m.mean); t3 = time.perf_counter()
            tl.append(t1 - t0); td.append(t2 - t1); td8.append(t3 - t2); dur = len(w) / SR
            diff8 = float(np.abs(w - w8).max())
        r = {"text": text, "audio_s_last": dur, "latents_s": med(tl), "dec_fp32_s": med(td), "dec_int8_s": med(td8),
             "fp32_vs_int8_max_abs_diff": diff8, "time_to_audio_ms_nonstream": (med(tl) + med(td)) * 1e3,
             "dtype": str(w.dtype), "latent_dtype": str(lat.dtype)}
        out["whole_sentence"].append(r)
        print(threads, json.dumps(r), flush=True)
    # matvec microbench: numpy vs torch on the real layer weights (one AR step = 8 layers x 4 matvecs)
    try:
        import torch
        torch.set_num_threads(int(threads) if threads.isdigit() else torch.get_num_threads())
        sp = m.speech
        Wn = [[w for (w, _) in l] for l in sp.layers]
        Wt = [[torch.from_numpy(w) for w in l] for l in Wn]
        x = np.random.randn(sp.d).astype(np.float32); g = np.random.randn(sp.ff).astype(np.float32)
        xt, gt = torch.from_numpy(x), torch.from_numpy(g)
        def np_step():
            for a, b, c, d in Wn:
                a @ x; b @ x; c @ x; d @ g
        def t_step():
            for a, b, c, d in Wt:
                torch.mv(a, xt); torch.mv(b, xt); torch.mv(c, xt); torch.mv(d, gt)
        res = {}
        with torch.no_grad():
            for name, fn in (("numpy", np_step), ("torch", t_step), ("numpy2", np_step), ("torch2", t_step)):
                fn(); t0 = time.perf_counter()
                for _ in range(100):
                    fn()
                res[name] = (time.perf_counter() - t0) / 100 * 1e3
        out["matvec_ms_per_ar_step"] = res
        print(threads, "matvec ms per AR step", res, flush=True)
    except Exception as ex:  # noqa: BLE001
        out["matvec_ms_per_ar_step"] = {"error": repr(ex)}

else:  # main
    sched = tuple(int(v) for v in os.environ.get("WOOF_SCHEDULE", "1,2,4").split(","))
    out["schedule"] = list(sched)
    out["rows"] = []
    for voice in ("juno", "mira"):
        t0 = time.perf_counter()
        tts = WoofStream(voice=voice, schedule=sched)
        load_s = time.perf_counter() - t0
        t0 = time.perf_counter(); tts.warmup(); first_call = time.perf_counter() - t0
        for si, text in enumerate(SENTENCES):
            run(tts, text, 100)                                   # warm-up for this sentence, not counted
            for seed in range(5):
                wav, m = run(tts, text, seed)
                m.update(voice=voice, sentence=si, text=text, chars=len(text), load_s=load_s, first_call_s=first_call)
                p = os.path.join(HERE, "audio", f"woof_{voice}_stream", f"s{si}_seed{seed}.wav")
                write_wav(p, wav); m["wav"] = p
                out["rows"].append(m)
                print(voice, si, seed, "ttfa %.0f ms rtf_rest %s underruns %d aborted %s frames %s" % (
                    m["ttfa_ms"], None if m["rtf_rest"] is None else round(m["rtf_rest"], 2), m["underruns"],
                    m["aborted"], m["frames"]), flush=True)
        # non-streamed references for the WER comparison: (a) same single take, no guard; (b) original Woof() with retries
        from underdog_woof import Woof
        orig = Woof(voice=voice)
        orig.model = tts.m                                        # share the loaded weights
        for si, text in enumerate(SENTENCES):
            for seed in range(3):
                w = tts.full(text, seed=seed)
                write_wav(os.path.join(HERE, "audio", f"woof_{voice}_full", f"s{si}_seed{seed}.wav"), w)
                t0 = time.perf_counter(); w2 = orig(text, seed=seed); dt = time.perf_counter() - t0
                write_wav(os.path.join(HERE, "audio", f"woof_{voice}_orig_retry", f"s{si}_seed{seed}.wav"), w2)
                out.setdefault("orig", []).append({"voice": voice, "sentence": si, "seed": seed,
                                                   "time_to_audio_ms": dt * 1e3, "audio_s": len(w2) / SR,
                                                   "single_take_audio_s": len(w) / SR})

with open(os.path.join(HERE, f"res_woof_{mode}_{tag}.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)

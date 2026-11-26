"""Barge-in while the assistant speaks: echo, backchannels and real interruptions, simulated with real models.

mic = user speech + assistant playback through a synthetic room (delay + reverb) + noise.
Policies: (a) VAD on raw mic, (b) AEC + VAD, (c) AEC + VAD + self-speech text filter + backchannel rule, (d) c without AEC.
"""
import difflib, json, os, re, sys, time
import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly, fftconvolve
os.environ.setdefault("CACTUS_TELEMETRY", "0")
import cactus
from livekit import rtc
from silero_vad import load_silero_vad, get_speech_timestamps

HERE = os.path.dirname(os.path.abspath(__file__))
M = json.load(open(os.path.join(HERE, "corpus", "manifest.json"), encoding="utf-8"))
CLEAN = json.load(open(os.path.join(HERE, "corpus", "clean_asr.json"), encoding="utf-8"))
SR = 16000
rng = np.random.default_rng(7)
vad = load_silero_vad()
BC = {"mm", "hm", "mhm", "mmhm", "hmm", "uh", "huh", "yeah", "yep", "yes", "okay", "ok", "right", "sure", "i", "see", "oh", "ah", "got", "it", "cool", "nice", "wow", "really", "m"}

def load(p):
    x, sr = sf.read(p, dtype="float32")
    return resample_poly(x, SR, sr).astype(np.float32) if sr != SR else x
def rms(x): return float(np.sqrt(np.mean(x ** 2)) + 1e-9)
def norm(w): return re.sub(r"[^a-z0-9]", "", w.lower())
def words(t): return [w for w in (norm(x) for x in t.split()) if w]

def room(delay_ms):
    n = int(0.25 * SR); t = np.arange(n) / SR
    h = rng.standard_normal(n) * np.exp(-t / 0.05) * 0.3; h[0] = 1.0
    return np.concatenate([np.zeros(int(delay_ms * SR / 1000)), h]).astype(np.float32)

def aec(mic, ref):
    apm = rtc.AudioProcessingModule(echo_cancellation=True, noise_suppression=True, high_pass_filter=True, auto_gain_control=False)
    out = np.zeros_like(mic); n = len(mic) // 160
    m16 = (np.clip(mic, -1, 1) * 32767).astype(np.int16); r16 = (np.clip(ref, -1, 1) * 32767).astype(np.int16)
    for i in range(n):
        rf = rtc.AudioFrame(r16[i * 160:(i + 1) * 160].tobytes(), SR, 1, 160)
        mf = rtc.AudioFrame(m16[i * 160:(i + 1) * 160].tobytes(), SR, 1, 160)
        apm.process_reverse_stream(rf); apm.process_stream(mf)
        out[i * 160:(i + 1) * 160] = np.frombuffer(mf.data, dtype=np.int16).astype(np.float32) / 32767
    return out

def segments(x):
    return get_speech_timestamps(torch.from_numpy(x), vad, sampling_rate=SR, min_speech_duration_ms=100, min_silence_duration_ms=300, return_seconds=True)

def explained(w, pool):
    return any(w == p or (len(w) > 2 and difflib.SequenceMatcher(None, w, p).ratio() >= 0.75) for p in pool)

def decide(sig, a, seg_start, t_dec):
    """Self-speech filter: words Whistle hears that the playing assistant text does not explain."""
    t0 = time.perf_counter()
    lo = max(0.0, seg_start - 0.3)
    hyp = words(cactus.transcribe(sig[int(lo * SR):int(t_dec * SR)], language="en")["text"])
    cw = CLEAN[os.path.basename(a["wav"])]["words"]
    pool = {norm(w["word"]) for w in cw if w["end"] >= lo - 1.0 and w["start"] <= t_dec + 0.3}
    tw = words(a["text"]); n = len(tw); dur = a["dur"]
    pool |= set(tw[max(0, int(n * (lo - 1.0) / dur)):min(n, int(n * (t_dec + 0.5) / dur) + 1)])
    res = [w for w in hyp if not explained(w, pool)]
    return res, (time.perf_counter() - t0) * 1000

def policy_vad(sig, hold=0.2):
    return [s["start"] + hold for s in segments(sig) if s["end"] - s["start"] >= hold]

CUE = {"stop", "wait", "no", "hold", "actually", "sorry", "hey", "but"}
def policy_filter(sig, a):
    """Returns (duck_times, stop_time or None, decision_ms list)."""
    ducks, costs = [], []
    for s in segments(sig):
        if s["end"] - s["start"] < 0.15: continue
        ducks.append(s["start"] + 0.15)
        for dt in (0.3, 0.45, 0.6, 0.8, 1.2, 1.8):
            t_dec = s["start"] + dt
            last = t_dec >= s["end"]
            res, ms = decide(sig, a, s["start"], min(t_dec, s["end"] + 0.1)); costs.append(ms)
            real = [w for w in res if w not in BC]
            if len(real) >= 2 or (real and real[0] in CUE) or (len(real) >= 1 and dt >= 0.8 and len(res) >= 2): return ducks, t_dec + ms / 1000, costs
            if last: break
    return ducks, None, costs

def wer(ref, hyp):
    r, h = words(ref), words(hyp); d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(h) + 1): p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))

events = []
A = M["assistant"]; INT = M["interrupt"]; BCS = M["backchannel"]
for k in range(72):
    kind = ("echo", "backchannel", "interrupt")[k % 3]
    a = A[(k // 3) % len(A)]; ratio = (-10, 0, 6)[(k // 9) % 3] if k >= 0 else 0
    ratio = (-10, 0, 6)[(k // 3) % 3]
    ref = load(a["wav"]); ref = ref / rms(ref) * 0.05
    rir = room(int(rng.integers(40, 120)))
    echo = fftconvolve(ref, rir)[:len(ref)].astype(np.float32)
    escale = 0.05 * 10 ** (ratio / 20) / rms(echo); echo = echo * escale
    mic = echo + rng.standard_normal(len(ref)).astype(np.float32) * 0.0015
    ev = {"kind": kind, "echo_db": ratio, "asst": a["id"], "onset": None, "text": ""}
    if kind != "echo":
        u = (INT if kind == "interrupt" else BCS)[int(rng.integers(0, len(INT if kind == "interrupt" else BCS)))]
        us = load(u["wav"]); nz = np.nonzero(np.abs(us) > 0.02 * np.abs(us).max())[0]; us = us[nz[0]:nz[-1] + 1]
        us = us / rms(us) * 0.05
        on = float(rng.uniform(1.5, max(1.6, a["dur"] - len(us) / SR - 1.0)))
        i0 = int(on * SR); seg = us[:len(mic) - i0]; mic[i0:i0 + len(seg)] += seg
        ev.update(onset=on, text=u["text"], u_end=on + len(seg) / SR)
    clean = aec(mic, ref)
    g = np.ones(len(ref), dtype=np.float32)
    for sg in segments(clean):
        if sg["end"] - sg["start"] >= 0.15: g[int((sg["start"] + 0.15) * SR):int((sg["end"] + 0.5) * SR)] = 10 ** (-15 / 20)
    k5 = np.ones(800) / 800; g = np.convolve(g, k5, mode="same").astype(np.float32)   # 50 ms fade
    ref2 = ref * g
    echo2 = fftconvolve(ref2, rir)[:len(ref)].astype(np.float32) * escale
    mic2 = mic - echo + echo2
    clean2 = aec(mic2, ref2)
    if kind == "echo" and k < 9:
        ev["erle_db"] = round(20 * np.log10(rms(mic[SR:]) / rms(clean[SR:])), 1)
    out = {}
    out["a_vad_raw"] = {"stops": policy_vad(mic)}
    out["b_aec_vad"] = {"stops": policy_vad(clean)}
    for name, sig in (("c_aec_filter", clean), ("d_filter_only", mic), ("e_aec_duck_filter", clean2)):
        ducks, stop, costs = policy_filter(sig, a)
        out[name] = {"stops": [stop] if stop else [], "ducks": ducks, "cost_ms": costs}
    if kind == "interrupt":
        lo, hi = int((ev["onset"] - 0.2) * SR), int((ev["u_end"] + 0.4) * SR)
        ev["wer_raw"] = wer(ev["text"], cactus.transcribe(mic[lo:hi], language="en")["text"])
        ev["wer_aec"] = wer(ev["text"], cactus.transcribe(clean[lo:hi], language="en")["text"])
        ev["wer_duck"] = wer(ev["text"], cactus.transcribe(clean2[lo:hi], language="en")["text"])
        ev["wer_alone"] = wer(ev["text"], cactus.transcribe(us, language="en")["text"])
    ev["pol"] = out; events.append(ev)
    if k < 6:
        os.makedirs(os.path.join(HERE, "samples"), exist_ok=True)
        sf.write(os.path.join(HERE, "samples", f"{k:02d}_{kind}_{ratio}dB_mic.wav"), mic, SR)
        sf.write(os.path.join(HERE, "samples", f"{k:02d}_{kind}_{ratio}dB_after_aec.wav"), clean, SR)
    print(k, kind, ratio, ev.get("onset") and round(ev["onset"], 2), {n: [round(s, 2) for s in v["stops"]] for n, v in out.items()}, flush=True)

def summarise(name):
    r = {}
    for kind in ("echo", "backchannel", "interrupt"):
        evs = [e for e in events if e["kind"] == kind]
        if kind == "interrupt":
            lat, miss, early = [], 0, 0
            for e in evs:
                st = e["pol"][name]["stops"]
                if any(s < e["onset"] for s in st): early += 1
                after = [s for s in st if e["onset"] <= s <= e["onset"] + 2.5]
                if after: lat.append((after[0] - e["onset"]) * 1000)
                else: miss += 1
            r["interrupt"] = {"n": len(evs), "missed": miss, "false_stop_before_user": early, "stop_ms_median": round(float(np.median(lat))) if lat else None, "stop_ms_p95": round(float(np.percentile(lat, 95))) if lat else None}
            if "ducks" in evs[0]["pol"][name]:
                dl = [min([d for d in e["pol"][name]["ducks"] if d >= e["onset"]], default=None) for e in evs]
                dl = [(d - e["onset"]) * 1000 for d, e in zip(dl, evs) if d is not None]
                r["interrupt"]["duck_ms_median"] = round(float(np.median(dl))) if dl else None
        else:
            r[kind] = {"n": len(evs), "false_stops": sum(1 for e in evs if e["pol"][name]["stops"])}
            if "ducks" in evs[0]["pol"][name]: r[kind]["events_with_a_duck"] = sum(1 for e in evs if e["pol"][name]["ducks"])
    for db in (-10, 0, 6):
        evs = [e for e in events if e["echo_db"] == db and e["kind"] != "interrupt"]
        r[f"false_stops_at_{db}dB"] = f"{sum(1 for e in evs if e['pol'][name]['stops'])}/{len(evs)}"
    return r

res = {"policies": {n: summarise(n) for n in ("a_vad_raw", "b_aec_vad", "c_aec_filter", "d_filter_only", "e_aec_duck_filter")}}
ints = [e for e in events if e["kind"] == "interrupt"]
res["interruption_wer"] = {f"{db}dB": {"raw": round(float(np.mean([e["wer_raw"] for e in ints if e["echo_db"] == db])), 3), "aec": round(float(np.mean([e["wer_aec"] for e in ints if e["echo_db"] == db])), 3), "aec_duck": round(float(np.mean([e["wer_duck"] for e in ints if e["echo_db"] == db])), 3), "user_alone": round(float(np.mean([e["wer_alone"] for e in ints if e["echo_db"] == db])), 3)} for db in (-10, 0, 6)}
res["erle_db"] = [(e["echo_db"], e["erle_db"]) for e in events if "erle_db" in e]
costs = [c for e in events for c in e["pol"]["c_aec_filter"]["cost_ms"]]
res["filter_decision_ms"] = {"median": round(float(np.median(costs))), "p95": round(float(np.percentile(costs, 95)))} if costs else None
print(json.dumps(res))
json.dump({"summary": res, "events": events}, open(os.path.join(HERE, "results.json"), "w", encoding="utf-8"), indent=1, default=float)

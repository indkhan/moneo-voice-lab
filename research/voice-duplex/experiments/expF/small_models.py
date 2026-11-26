"""Small Whisper-family recognisers on the same clips: word differences vs the large reference, and speed."""
import glob, json, os, re, sys, time
from faster_whisper import WhisperModel

folder = sys.argv[1]
ref = json.load(open(f"{folder}/reference.json", encoding="utf-8"))
whistle = {r["clip"]: r for r in json.load(open(f"{folder}/transcripts.json", encoding="utf-8"))}
words = lambda t: re.sub(r"[^a-z0-9 ]+", " ", t.lower()).split()
def dist(a, b):
    row = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, row[0] = row[0], i
        for j in range(1, len(b) + 1):
            keep = row[j]; row[j] = min(row[j] + 1, row[j - 1] + 1, prev + (a[i - 1] != b[j - 1])); prev = keep
    return row[-1]
KEY = ["rewe", "lidl", "netto", "amazon", "netflix", "munich", "scalable", "euros", "add", "spend", "earn", "salary", "groceries", "month", "year", "wallet"]
def score(hyp):
    e = t = k = kt = 0
    for clip, r in ref.items():
        rw = words(r)
        if len(rw) <= 2: continue
        hw = words(hyp[clip]); e += dist(hw, rw); t += len(rw)
        for w in KEY:
            n = rw.count(w); kt += n; k += min(n, hw.count(w))
    return round(100 * e / t), round(100 * k / kt)
prompt = "Moneo, Rewe, Lidl, Netto, Amazon, Netflix, Spotify, Scalable, N26, euros, groceries, salary."
out = {"whistle (17 MB)": {"diff%": score({c: r["names"] for c, r in whistle.items()})[0], "key words kept%": score({c: r["names"] for c, r in whistle.items()})[1],
       "time vs audio%": round(100 * sum(r["names_ms"] for r in whistle.values()) / 1000 / sum(r["audio_s"] for r in whistle.values()))}}
total_audio = sum(r["audio_s"] for r in whistle.values())
for name in ["tiny.en", "base.en", "small.en", "distil-small.en"]:
    model = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=2)
    hyp, spent = {}, 0.0
    for path in sorted(glob.glob(f"{folder}/u*.wav")):
        t = time.perf_counter()
        segs, _ = model.transcribe(path, language="en", beam_size=1, initial_prompt=prompt, vad_filter=False)
        hyp[os.path.basename(path)] = " ".join(s.text.strip() for s in segs); spent += time.perf_counter() - t
    d, k = score(hyp)
    out[name] = {"diff%": d, "key words kept%": k, "time vs audio%": round(100 * spent / total_audio)}
    json.dump(hyp, open(f"{folder}/hyp_{name}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(name, out[name], flush=True)
print(json.dumps(out, indent=1))
json.dump(out, open(f"{folder}/small_models.json", "w", encoding="utf-8"), indent=1)

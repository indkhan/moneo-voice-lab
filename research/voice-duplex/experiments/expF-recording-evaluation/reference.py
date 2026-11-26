"""A much larger recogniser (Whisper large-v3-turbo, CPU) as a reference transcript for the same clips."""
import glob, json, os, sys, time
from faster_whisper import WhisperModel

folder = sys.argv[1]
model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
prompt = "Moneo, Rewe, Lidl, Netto, Amazon, Netflix, Spotify, Scalable, N26, euros, groceries, salary."
out = {}
for path in sorted(glob.glob(f"{folder}/u*.wav")):
    t = time.perf_counter()
    segments, _ = model.transcribe(path, language="en", beam_size=5, initial_prompt=prompt, vad_filter=False)
    text = " ".join(s.text.strip() for s in segments)
    out[os.path.basename(path)] = text
    print(f"{os.path.basename(path)} {round((time.perf_counter()-t)*1000):6d}ms | {text}", flush=True)
json.dump(out, open(f"{folder}/reference.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

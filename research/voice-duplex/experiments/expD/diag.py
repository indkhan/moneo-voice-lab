import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(HERE, "results.json"), encoding="utf-8"))
for n in ("a_vad_raw", "b_aec_vad", "c_aec_filter"):
    print(n, json.dumps(r["summary"]["policies"][n]))
# is the user audio itself transcribable in isolation?
sys.path.insert(0, HERE)
os.environ["CACTUS_TELEMETRY"] = "0"
import cactus, soundfile as sf
from scipy.signal import resample_poly
M = json.load(open(os.path.join(HERE, "corpus", "manifest.json"), encoding="utf-8"))
bad = 0
for u in M["interrupt"][:36]:
    x, sr = sf.read(u["wav"], dtype="float32"); x = resample_poly(x, 16000, sr).astype(np.float32)
    h = cactus.transcribe(x, language="en")["text"]
    ok = h.lower().strip(" .!?").replace(",", "") == u["text"].lower().strip(" .!?").replace(",", "")
    bad += not ok
    if not ok: print("  CLEAN MISMATCH:", u["voice"], "|", u["text"], "=>", h)
print("clean interrupt clips mis-transcribed:", bad, "of", len(M["interrupt"][:36]))
for e in r["events"]:
    if e["kind"] == "interrupt":
        print(e["echo_db"], round(e["onset"], 2), e["text"], "| wer raw/aec", round(e["wer_raw"], 2), round(e["wer_aec"], 2), "| c stops", [round(s, 2) for s in e["pol"]["c_aec_filter"]["stops"]], "ducks", [round(d, 2) for d in e["pol"]["c_aec_filter"]["ducks"]][:4])

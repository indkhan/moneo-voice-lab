import os, sys, time, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "woof_stream"))
from stream_runtime import WoofStream, rt
tts = WoofStream(voice="juno"); tts.warmup()
m, sp = tts.m, tts.m.speech
for text in ["Mm-hm.", "Let me look through your transactions."]:
    ids = rt.encode_text(text)
    for rep in range(3):
        x = np.concatenate([m.voice, sp.text_emb[np.asarray(ids)], sp.bos[None]], 0).astype(np.float32)
        p0 = x.shape[0]; ts = []
        for i in range(sp.L):
            t0 = time.perf_counter(); x = sp._prefix_block(i, x, p0); ts.append((time.perf_counter()-t0)*1e3)
        print(os.environ.get("OPENBLAS_NUM_THREADS"), p0, "prefix total %.1f ms" % sum(ts), ["%.1f" % t for t in ts])
    for fc, ch in ((1, 4),):
        tts.first_chunk, tts.chunk = fc, ch
        for rep in range(3):
            t0 = time.perf_counter(); first = None; n = 0
            for c in tts.stream(text, seed=rep):
                if first is None: first = time.perf_counter() - t0
                n += len(c)
            tot = time.perf_counter() - t0
            print("  stream fc=%d ch=%d ttfa %.1f ms total %.0f ms audio %.2f s rtf %.2fx  sp %.0f dec %.0f" % (fc, ch, first*1e3, tot*1e3, n/24000, n/24000/tot, tts.stats["t_speech"]*1e3, tts.stats["t_dec"]*1e3))

import os, sys, time, inspect
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "woof_stream"))
import pocket_tts
from pocket_tts import TTSModel
print("pocket", pocket_tts.__file__)
print(inspect.signature(TTSModel.load_model))
print((inspect.getdoc(TTSModel.load_model) or "")[:1500])
print(inspect.signature(TTSModel.generate_audio_stream))
import cactus
print("cactus", cactus.__file__, [n for n in dir(cactus) if not n.startswith("_")][:40])
print(inspect.signature(cactus.transcribe)); print((inspect.getdoc(cactus.transcribe) or "")[:600])
from stream_runtime import WoofStream, rt
t = WoofStream(voice="mira"); t.warmup()
w = t.full("Mm-hm.", seed=7)
fr = w[:len(w)//960*960].reshape(-1, 960)
db = 20*np.log10(np.sqrt((fr*fr).mean(1))+1e-9)
print("mira Mm-hm seed7 frames", len(fr)); print(np.round(db).astype(int).tolist())
for seed in range(8):
    for v, tt in (("mira", t),):
        w = tt.full("Mm-hm.", seed=seed); print(v, seed, "frames", len(w)//960)
x = np.interp(np.arange(0, len(w), 1.5), np.arange(len(w)), w).astype(np.float32)
t0 = time.perf_counter(); r = cactus.transcribe(x); print("stt", time.perf_counter()-t0, r)

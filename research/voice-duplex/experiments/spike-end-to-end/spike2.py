"""Spike 2: Pocket TTS time-to-first-audio (streaming) on CPU, English + German; Whistle on the German audio."""
import json, os, re, sys, time
import numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "audio"); os.makedirs(OUT, exist_ok=True)
from pocket_tts import TTSModel
from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES as VOICES
import soundfile as sf
print("voices:", list(VOICES)[:30], flush=True)

EN = ["Let me look through your transactions.",
      "You spent 312 euros 40 on groceries in October, about 18 percent more than September.",
      "Mm-hm."]
DE = ["Wie viel habe ich letzten Monat für Lebensmittel ausgegeben?",
      "Übrigens habe ich zwölf Euro vierzig bei Netto ausgegeben und ich habe achtzig Euro in bar.",
      "Nein, ich meinte September, nicht Oktober.",
      "Wie hoch ist mein Vermögen gerade und wie hat es sich verändert?"]

def resample(x, a, b):
    n = int(len(x) * b / a)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)

res = {"tts": [], "stt_de": []}
de_clips = []
for cfg, sents, quant in [("english_18m_2026-10", EN, False), ("english_7m_2026-10", EN, False), ("english", EN, True), ("german", DE, True)]:
    try:
        t = time.perf_counter(); m = TTSModel.load_model(language=cfg, quantize=quant); load = time.perf_counter() - t
        voice = list(VOICES)[0]
        t = time.perf_counter(); st = m.get_state_for_audio_prompt(voice); vload = time.perf_counter() - t
        list(m.generate_audio_stream(st, "Warm up."))
        for i, s in enumerate(sents):
            t0 = time.perf_counter(); first = None; chunks = []
            for c in m.generate_audio_stream(st, s):
                if first is None: first = time.perf_counter() - t0
                chunks.append(c)
            dt = time.perf_counter() - t0
            wav = torch.cat(chunks).numpy(); dur = len(wav) / m.sample_rate
            sf.write(os.path.join(OUT, f"pocket_{cfg}_{i}.wav"), wav, m.sample_rate)
            if cfg == "german": de_clips.append((s, resample(wav, m.sample_rate, 16000)))
            res["tts"].append({"model": cfg, "quant": quant, "first_audio_ms": round(first * 1000), "audio_s": round(dur, 2), "x_realtime": round(dur / dt, 2), "load_s": round(load, 1), "voice_s": round(vload, 2)})
            print(res["tts"][-1], flush=True)
        del m
    except Exception as e:
        print("FAIL", cfg, repr(e)[:300], flush=True)

import cactus
def words(s): return re.sub(r"[^a-zäöüß0-9 ]", " ", s.lower()).split()
for s, a in de_clips:
    cactus.transcribe(a)
    t = time.perf_counter(); r = cactus.transcribe(a, keywords=["Netto"]); dt = time.perf_counter() - t
    res["stt_de"].append({"ref": s, "hyp": r["text"], "lang": r["language"], "audio_s": round(len(a) / 16000, 2), "wall_ms": round(dt * 1000)})
    print(res["stt_de"][-1], flush=True)
json.dump(res, open(os.path.join(ROOT, "spike", "spike2.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)

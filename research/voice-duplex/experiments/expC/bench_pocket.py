"""Pocket TTS on the same 8 sentences, same metrics. python bench_pocket.py english_18m_2026-10 [more models...]"""
import json, os, sys, time
import numpy as np
import torch
from pocket_tts import TTSModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import SENTENCES, timed_stream, write_wav

out = {"torch_threads": torch.get_num_threads(), "models": {}}
for lang in sys.argv[1:]:
    try:
        t0 = time.perf_counter()
        m = TTSModel.load_model(language=lang)
        st = m.get_state_for_audio_prompt("alba")
        load_s = time.perf_counter() - t0
        sr = m.sample_rate
        params = sum(p.numel() for p in m.parameters())
        nbytes = sum(p.numel() * p.element_size() for p in m.parameters())
        t0 = time.perf_counter()
        for _ in m.generate_audio_stream(st, "Hello there, this is a warm up."):
            pass
        first_call = time.perf_counter() - t0
        rows = []
        for si, text in enumerate(SENTENCES):
            for seed in (100, 0, 1, 2, 3, 4):                      # 100 = warm-up, not counted
                torch.manual_seed(seed)
                wav, r = timed_stream(lambda: (c.detach().cpu().numpy() for c in m.generate_audio_stream(st, text)), sr)
                if seed == 100:
                    continue
                r.update(sentence=si, text=text, seed=seed, chars=len(text))
                if seed < 3:
                    p = os.path.join(HERE, "audio", f"pocket_{lang}", f"s{si}_seed{seed}.wav")
                    write_wav(p, wav, sr); r["wav"] = p
                rows.append(r)
                print(lang, si, seed, "ttfa %.0f ms rtf_rest %s underruns %d audio %.2fs chunks %d" % (
                    r["ttfa_ms"], None if r["rtf_rest"] is None else round(r["rtf_rest"], 2), r["underruns"],
                    r["audio_s"], r["n_chunks"]), flush=True)
        out["models"][lang] = {"params": params, "fp32_mb": nbytes / 1e6, "sample_rate": sr, "load_s": load_s,
                               "first_call_s": first_call, "rows": rows}
        del m, st
    except Exception as ex:  # noqa: BLE001
        import traceback; traceback.print_exc()
        out["models"][lang] = {"error": repr(ex)}
    with open(os.path.join(HERE, "res_pocket.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

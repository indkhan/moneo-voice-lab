"""Every candidate recogniser on the owner's real recordings, through one runtime (transcribe.cpp 0.4.2).

  TRANSCRIBE_LIBRARY=<...\\transcribe.dll> python bench_asr.py <models dir> <folder> [<folder> ...] [--only name,name] [--threads 4]

For each model and folder it writes <folder>/hyp_<model>.json (clip -> text) and, when the model takes
custom vocabulary or a context prompt, <folder>/hyp_<model>+names.json with the owner's names as hints.
Times go to <folder>/bench_times.json. Models are downloaded from the handy-computer GGUF repos, which
are checked against each family's reference implementation. CPU by default: the target is a laptop without a GPU.
`--backend vulkan` runs the large models on a GPU to measure accuracy only; their times carry the backend and are not laptop times.
"""
import glob, json, os, sys, time
import numpy as np, soundfile as sf
from huggingface_hub import hf_hub_download
import transcribe_cpp as tc

# name, repo, file. Q8_0 where it exists (indistinguishable from F32 on the cards); Q4_K_M above 1.5 GB.
MODELS = [
    ("moonshine-streaming-tiny", "moonshine-streaming-tiny-gguf", "moonshine-streaming-tiny-Q8_0.gguf"),
    ("moonshine-streaming-small", "moonshine-streaming-small-gguf", "moonshine-streaming-small-Q8_0.gguf"),
    ("moonshine-streaming-medium", "moonshine-streaming-medium-gguf", "moonshine-streaming-medium-Q8_0.gguf"),
    ("parakeet-tdt_ctc-110m", "parakeet-tdt_ctc-110m-gguf", "parakeet-tdt_ctc-110m-Q8_0.gguf"),
    ("canary-180m-flash", "canary-180m-flash-gguf", "canary-180m-flash-Q8_0.gguf"),
    ("whisper-base.en", "whisper-base.en-gguf", "whisper-base.en-Q8_0.gguf"),
    ("granite-speech-5.0-470m-turboctc", "granite-speech-5.0-470m-turboctc-gguf", "granite-speech-5.0-470m-turboctc-Q8_0.gguf"),
    ("nemotron-speech-streaming-en-0.6b", "nemotron-speech-streaming-en-0.6b-gguf", "nemotron-speech-streaming-en-0.6b-Q8_0.gguf"),
    ("nemotron-3.5-asr-streaming-0.6b", "nemotron-3.5-asr-streaming-0.6b-gguf", "nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf"),
    ("parakeet-tdt-0.6b-v2", "parakeet-tdt-0.6b-v2-gguf", "parakeet-tdt-0.6b-v2-Q8_0.gguf"),
    ("parakeet-tdt-0.6b-v3", "parakeet-tdt-0.6b-v3-gguf", "parakeet-tdt-0.6b-v3-Q8_0.gguf"),
    ("parakeet-unified-en-0.6b", "parakeet-unified-en-0.6b-gguf", "parakeet-unified-en-0.6b-Q8_0.gguf"),
    ("parakeet-ultra", "parakeet-ultra-gguf", "parakeet-ultra-Q8_0.gguf"),
    ("qwen3-asr-0.6b", "qwen3-asr-0.6b-gguf", "Qwen3-ASR-0.6B-Q8_0.gguf"),
    ("whisper-large-v3-turbo", "whisper-large-v3-turbo-gguf", "whisper-large-v3-turbo-Q8_0.gguf"),
    ("canary-1b-v2", "canary-1b-v2-gguf", "canary-1b-v2-Q8_0.gguf"),
    ("qwen3-asr-1.7b", "qwen3-asr-1.7b-gguf", "Qwen3-ASR-1.7B-Q4_K_M.gguf"),
    ("cohere-transcribe-03-2026", "cohere-transcribe-03-2026-gguf", "cohere-transcribe-03-2026-Q4_K_M.gguf"),
    ("granite-speech-4.1-2b", "granite-speech-4.1-2b-gguf", "granite-speech-4.1-2b-Q4_K_M.gguf"),
    ("canary-qwen-2.5b", "canary-qwen-2.5b-gguf", "canary-qwen-2.5b-Q4_K_M.gguf"),
    ("voxtral-mini-4b-realtime-2602", "voxtral-mini-4b-realtime-2602-gguf", "Voxtral-Mini-4B-Realtime-2602-Q4_K_M.gguf"),
]
# The same names the app gives Whistle as keywords: the owner's shops, categories and accounts.
NAMES = ["Moneo", "Rewe", "Lidl", "Netto", "Amazon", "Netflix", "Spotify", "Uber", "Starbucks", "Scalable", "N26", "ChatGPT",
         "euros", "groceries", "restaurants", "transport", "salary", "subscription", "wallet"]

args = sys.argv[1:]
only = None
threads, backend = 4, "cpu"
if "--only" in args: i = args.index("--only"); only = set(args[i + 1].split(",")); del args[i:i + 2]
if "--threads" in args: i = args.index("--threads"); threads = int(args[i + 1]); del args[i:i + 2]
if "--backend" in args: i = args.index("--backend"); backend = args[i + 1]; del args[i:i + 2]
models_dir, folders = args[0], args[1:]

def run(session, pcm, **hints):
    for language in ("en", None):  # multilingual models take a language; English-only ones may refuse it
        try:
            return session.run(pcm, language=language, **hints).text.strip()
        except (tc.UnsupportedRequest, tc.InvalidArgument, tc.NotImplementedByModel):
            if language is None: raise
    return ""

for name, repo, file in MODELS:
    if only and name not in only: continue
    try:
        path = hf_hub_download(f"handy-computer/{repo}", file, local_dir=os.path.join(models_dir, name))
        model = tc.Model(path, backend=backend)
    except Exception as error:
        print(f"{name}: could not load ({type(error).__name__}: {error})", flush=True); continue
    variants = [("", {})]
    if model.supports("vocabulary"): variants.append(("+names", {"vocabulary": NAMES}))
    elif model.supports("context_prompt"): variants.append(("+names", {"prompt": ", ".join(NAMES) + "."}))
    with model.session(n_threads=threads) as session:
        warm = np.zeros(16000, dtype=np.float32)
        try: run(session, warm)
        except Exception: pass
        for folder in folders:
            clips = sorted(glob.glob(os.path.join(folder, "u*.wav")))
            audio = {os.path.basename(c): sf.read(c, dtype="float32")[0] for c in clips}
            seconds = sum(len(a) for a in audio.values()) / 16000
            times_path = os.path.join(folder, "bench_times.json")
            times = json.load(open(times_path, encoding="utf-8")) if os.path.exists(times_path) else {}
            for suffix, hints in variants:
                out, spent, failed = {}, 0.0, 0
                for clip, pcm in audio.items():
                    start = time.perf_counter()
                    try: out[clip] = run(session, pcm, **hints)
                    except Exception as error: out[clip] = ""; failed += 1; print(f"  {clip}: {type(error).__name__}", flush=True)
                    spent += time.perf_counter() - start
                json.dump(out, open(os.path.join(folder, f"hyp_{name}{suffix}.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
                times[name + suffix] = {"time vs audio %": round(100 * spent / seconds, 1), "threads": threads, "backend": backend, "failed clips": failed,
                                        "file": file, "size MB": round(os.path.getsize(path) / 1e6)}
                print(f"{name}{suffix:7} {os.path.basename(folder)}: {times[name + suffix]['time vs audio %']}% of audio time, {failed} failed", flush=True)
            json.dump(times, open(times_path, "w", encoding="utf-8"), indent=1)
    model.close()

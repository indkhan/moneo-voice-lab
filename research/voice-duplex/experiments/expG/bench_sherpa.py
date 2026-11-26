"""Finalists through sherpa-onnx (int8, CPU): fair laptop times, and hints where the runtime takes them.
sherpa-onnx also has Node bindings, so this is a runtime the desktop app could ship (it ships no Python).

  python bench_sherpa.py <sherpa models dir> <folder> [<folder> ...] [--threads 4] [--only name,name]

Writes <folder>/hyp_sherpa-<name>[+names].json and times to <folder>/bench_times.json.
"""
import glob, json, os, sys, time
import soundfile as sf
import sherpa_onnx

NAMES = ["Moneo", "Rewe", "Lidl", "Netto", "Amazon", "Netflix", "Spotify", "Uber", "Starbucks", "Scalable", "N26", "ChatGPT",
         "euros", "groceries", "restaurants", "transport", "salary", "subscription", "wallet"]
args = sys.argv[1:]
threads, only = 4, None
if "--threads" in args: i = args.index("--threads"); threads = int(args[i + 1]); del args[i:i + 2]
if "--only" in args: i = args.index("--only"); only = set(args[i + 1].split(",")); del args[i:i + 2]
root, folders = args[0], args[1:]

def make(directory, hints):
    d = os.path.join(root, directory)
    if "qwen3" in directory:
        return sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
            conv_frontend=f"{d}/conv_frontend.onnx", encoder=f"{d}/encoder.int8.onnx", decoder=f"{d}/decoder.int8.onnx",
            tokenizer=f"{d}/tokenizer", num_threads=threads, hotwords=",".join(NAMES) if hints else "")
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=f"{d}/encoder.int8.onnx", decoder=f"{d}/decoder.int8.onnx", joiner=f"{d}/joiner.int8.onnx",
        tokens=f"{d}/tokens.txt", num_threads=threads, model_type="nemo_transducer")
# name: (directory, takes hints)
RECOGNISERS = {
    "qwen3-asr-0.6b": ("sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25", True),
    "parakeet-unified-en-0.6b": ("sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming", False),
    "parakeet-tdt-0.6b-v3": ("sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8", False),
}

for name, (directory, takes_hints) in RECOGNISERS.items():
    if only and name not in only: continue
    for hints in ([False, True] if takes_hints else [False]):
        try: recognizer = make(directory, hints)
        except Exception as error: print(f"{name}: could not load ({error})", flush=True); break
        label = f"sherpa-{name}{'+names' if hints else ''}"
        for folder in folders:
            out, spent, seconds = {}, 0.0, 0.0
            for clip in sorted(glob.glob(os.path.join(folder, "u*.wav"))):
                pcm, rate = sf.read(clip, dtype="float32")
                seconds += len(pcm) / rate
                start = time.perf_counter()
                stream = recognizer.create_stream(); stream.accept_waveform(rate, pcm); recognizer.decode_stream(stream)
                out[os.path.basename(clip)] = stream.result.text.strip()
                spent += time.perf_counter() - start
            json.dump(out, open(os.path.join(folder, f"hyp_{label}.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
            times_path = os.path.join(folder, "bench_times.json")
            times = json.load(open(times_path, encoding="utf-8")) if os.path.exists(times_path) else {}
            times[label] = {"time vs audio %": round(100 * spent / seconds, 1), "threads": threads, "backend": "cpu (sherpa-onnx int8)", "failed clips": 0,
                            "file": name, "size MB": round(sum(os.path.getsize(p) for p in glob.glob(os.path.join(root, directory, "*.onnx"))) / 1e6)}
            json.dump(times, open(times_path, "w", encoding="utf-8"), indent=1)
            print(f"{label} {os.path.basename(folder)}: {times[label]['time vs audio %']}% of audio time", flush=True)

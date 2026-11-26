"""Recognisers that transcribe.cpp does not run yet, through Hugging Face Transformers on CPU.

  python bench_transformers.py <model id> <folder> [<folder> ...] [--threads 4]

Writes <folder>/hyp_<model name>.json and its time to <folder>/bench_times.json, like bench_asr.py.
Rumble-2-Flash (Reson8, 2026-11-18) is a Qwen3-ASR derivative with its own processor.
"""
import glob, json, os, sys, time
import soundfile as sf, torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

args = sys.argv[1:]
threads = 4
if "--threads" in args: i = args.index("--threads"); threads = int(args[i + 1]); del args[i:i + 2]
model_id, folders = args[0], args[1:]
torch.set_num_threads(threads)
name = model_id.split("/")[-1]
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, torch_dtype=torch.float32).eval()
size_mb = round(sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6)

def transcribe(pcm):
    inputs = processor.apply_transcription_request(audio=pcm, language="English").to(model.device, dtype=model.dtype)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    return processor.decode(output[:, inputs["input_ids"].shape[-1]:], return_format="transcription_only")[0].strip()

transcribe(torch.zeros(16000).numpy())
for folder in folders:
    clips = sorted(glob.glob(os.path.join(folder, "u*.wav")))
    out, spent, seconds = {}, 0.0, 0.0
    for clip in clips:
        pcm = sf.read(clip, dtype="float32")[0]
        seconds += len(pcm) / 16000
        start = time.perf_counter()
        out[os.path.basename(clip)] = transcribe(pcm)
        spent += time.perf_counter() - start
    json.dump(out, open(os.path.join(folder, f"hyp_{name}.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    times_path = os.path.join(folder, "bench_times.json")
    times = json.load(open(times_path, encoding="utf-8")) if os.path.exists(times_path) else {}
    times[name] = {"time vs audio %": round(100 * spent / seconds, 1), "threads": threads, "failed clips": 0, "file": f"{model_id} (transformers, fp32)", "size MB": size_mb}
    json.dump(times, open(times_path, "w", encoding="utf-8"), indent=1)
    print(f"{name} {os.path.basename(folder)}: {times[name]['time vs audio %']}% of audio time", flush=True)

"""Decode one long recording, split it into utterances at pauses, save 16 kHz mono clips."""
import json, sys
import av, numpy as np, soundfile as sf, torch
from silero_vad import load_silero_vad, get_speech_timestamps

src, out = sys.argv[1], sys.argv[2]
container = av.open(src)
resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
chunks = []
for frame in container.decode(audio=0):
    for r in resampler.resample(frame):
        chunks.append(r.to_ndarray().reshape(-1))
audio = np.concatenate(chunks).astype(np.float32) / 32768.0
print(f"duration {len(audio)/16000:.1f} s")
sf.write(f"{out}/full.wav", audio, 16000)
model = load_silero_vad()
# A new question starts after about a second of silence; shorter pauses stay inside one utterance.
stamps = get_speech_timestamps(torch.from_numpy(audio), model, sampling_rate=16000, min_silence_duration_ms=int(sys.argv[3]) if len(sys.argv) > 3 else 1000, speech_pad_ms=200)
meta = []
for i, s in enumerate(stamps):
    clip = audio[s["start"]:s["end"]]
    sf.write(f"{out}/u{i:02d}.wav", clip, 16000)
    meta.append({"i": i, "start": round(s["start"]/16000, 2), "end": round(s["end"]/16000, 2)})
    print(f"u{i:02d} {s['start']/16000:7.2f} - {s['end']/16000:7.2f}  ({(s['end']-s['start'])/16000:.1f} s)")
json.dump(meta, open(f"{out}/segments.json", "w", encoding="utf-8"), indent=1)

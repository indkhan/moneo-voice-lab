"""Shared helpers: sentences, timing of a chunk stream, underrun simulation, WAV writing."""
import os, time, wave
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

SENTENCES = [
    "Ok.",
    "Mm-hm.",
    "Sure, one moment.",
    "Your balance is 1,250 euros.",
    "Let me look through your transactions.",
    "I found three subscriptions you might want to cancel.",
    "Rent went out on the first, so that part is already covered.",
    "You spent 312.40 euros on groceries in October, about 18 percent more than September.",
]


def timed_stream(gen_factory, sr):
    """Consume a chunk generator. Returns (audio, metrics). Playback is assumed to start when chunk 0 arrives."""
    t0 = time.perf_counter()
    arr, chunks = [], []
    for c in gen_factory():
        arr.append(time.perf_counter() - t0)
        chunks.append(np.asarray(c, dtype=np.float32).reshape(-1))
    total = time.perf_counter() - t0
    if not chunks:
        return np.zeros(0, np.float32), {"ttfa_ms": None}
    n = [len(c) for c in chunks]
    audio_s = sum(n) / sr
    play_end = arr[0]
    underruns, gap = 0, 0.0
    need_buffer = 0.0                      # start-up delay that would have avoided every underrun
    played = 0.0
    for a, k in zip(arr, n):
        if a > play_end + 1e-9:
            underruns += 1
            gap += a - play_end
            play_end = a
        need_buffer = max(need_buffer, a - (arr[0] + played))
        played += k / sr
        play_end += k / sr
    rest_audio = (sum(n) - n[0]) / sr
    rest_time = arr[-1] - arr[0]
    m = {"ttfa_ms": arr[0] * 1e3, "first_chunk_ms_audio": n[0] / sr * 1e3, "audio_s": audio_s, "total_s": total,
         "rtf_all": audio_s / total, "rtf_rest": (rest_audio / rest_time) if rest_time > 0 else None,
         "underruns": underruns, "underrun_gap_ms": gap * 1e3, "needed_prebuffer_ms": need_buffer * 1e3,
         "n_chunks": len(chunks), "max_inter_chunk_ms": max(np.diff(arr)) * 1e3 if len(arr) > 1 else 0.0}
    return np.concatenate(chunks), m


def write_wav(path, x, sr=24000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pcm = (np.clip(x, -1, 1) * 32767).round().astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None

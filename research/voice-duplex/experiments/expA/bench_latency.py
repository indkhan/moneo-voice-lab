"""CPU latency: Whistle embed, exported heads, and Smart Turn v3.2 (int8 ONNX) on the same test clips.

Usage: python bench_latency.py head1,head2   -> expA/latency.json
Run with nothing else heavy on the machine; medians over three rounds are reported.
"""
import io
import json
import os
import sys
import time

os.environ.setdefault("CACTUS_TELEMETRY", "0")

import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
import soundfile as sf

import cactus
from eval_smartturn import MODELS_DIR, build_session, features
from head_infer import SR, embed_last5, export_onnx, frames_to_input, load_head
from train_heads import HERE

N_CLIPS, ROUNDS = 150, 3


def med(v):
    return round(float(np.median(v)), 3)


def main():
    heads = sys.argv[1].split(",")
    pf = pq.ParquetFile(os.path.join(HERE, "hf", "smart-turn-data-v3.2-test", "data", "train-00000-of-00010.parquet"))
    clips = []
    for rg in range(pf.metadata.num_row_groups):
        for r in pf.read_row_group(rg, columns=["audio", "language"]).to_pylist():
            if r["language"] == "eng" and len(clips) < N_CLIPS:
                clips.append(sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")[0])
    w = cactus.Whistle()
    w.embed(clips[0][-5 * SR:])
    out = {"n_clips": len(clips), "rounds": ROUNDS, "note": "medians; per-round medians listed so noise is visible"}

    # 1. Whistle embed, last 5 s of each clip (what the head was trained on), default threads
    rounds, per_s = [], []
    for _ in range(ROUNDS):
        ts, dur = [], 0.0
        for x in clips:
            xx = np.ascontiguousarray(x[-5 * SR:])
            t = time.perf_counter(); w.embed(xx); ts.append((time.perf_counter() - t) * 1000)
            dur += len(xx) / SR
        rounds.append(med(ts)); per_s.append(round(sum(ts) / dur, 2))
    out["whistle_embed_last5s"] = {"median_ms_per_clip_rounds": rounds, "median_ms_per_clip": med(rounds),
                                   "ms_per_s_audio_rounds": per_s, "ms_per_s_audio": med(per_s),
                                   "mean_clip_s": round(float(np.mean([min(len(x), 5 * SR) / SR for x in clips])), 2)}
    # embed cost against window length, one long clip
    long = max(clips, key=len)
    by_len = {}
    for d in (0.5, 1, 2, 3, 4, 5, 5.2, 6, 8):
        xx = np.ascontiguousarray(long[-int(d * SR):])
        ts = []
        for _ in range(9):
            t = time.perf_counter(); w.embed(xx); ts.append((time.perf_counter() - t) * 1000)
        by_len[str(d)] = med(ts)
    out["whistle_embed_ms_by_window_s"] = by_len

    # 2. heads, ONNX Runtime on one thread
    frames = [frames_to_input(embed_last5(w, x)) for x in clips[:50]]
    out["heads"] = {}
    for name in heads:
        path = export_onnx(load_head(name), os.path.join(HERE, "heads", name + ".onnx"))
        so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
        s = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        s.run(None, {"frames": frames[0][0], "mask": frames[0][1]})
        rounds = []
        for _ in range(ROUNDS):
            ts = []
            for _ in range(4):
                for x, m in frames:
                    t = time.perf_counter(); s.run(None, {"frames": x, "mask": m}); ts.append((time.perf_counter() - t) * 1000)
            rounds.append(med(ts))
        out["heads"][name] = {"onnx_1thread_ms_rounds": rounds, "onnx_1thread_ms": med(rounds),
                              "onnx_bytes": os.path.getsize(path)}

    # 3. Smart Turn v3.2 cpu (int8): Whisper feature extraction + ONNX, the repo's session options
    s = build_session(os.path.join(MODELS_DIR, "smart-turn-v3.2-cpu.onnx"))
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    s1 = ort.InferenceSession(os.path.join(MODELS_DIR, "smart-turn-v3.2-cpu.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
    f0 = features(clips[0]); s.run(None, {"input_features": f0}); s1.run(None, {"input_features": f0})
    r_feat, r_onnx, r_onnx1 = [], [], []
    for _ in range(ROUNDS):
        tf, to, t1 = [], [], []
        for x in clips:
            t = time.perf_counter(); f = features(x); tf.append((time.perf_counter() - t) * 1000)
            t = time.perf_counter(); s.run(None, {"input_features": f}); to.append((time.perf_counter() - t) * 1000)
            t = time.perf_counter(); s1.run(None, {"input_features": f}); t1.append((time.perf_counter() - t) * 1000)
        r_feat.append(med(tf)); r_onnx.append(med(to)); r_onnx1.append(med(t1))
    out["smart_turn_v3.2_cpu"] = {"feature_extraction_ms_rounds": r_feat, "feature_extraction_ms": med(r_feat),
                                  "onnx_default_threads_ms_rounds": r_onnx, "onnx_default_threads_ms": med(r_onnx),
                                  "onnx_1thread_ms_rounds": r_onnx1, "onnx_1thread_ms": med(r_onnx1),
                                  "total_default_threads_ms": round(med(r_feat) + med(r_onnx), 3)}
    json.dump(out, open(os.path.join(HERE, "latency.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

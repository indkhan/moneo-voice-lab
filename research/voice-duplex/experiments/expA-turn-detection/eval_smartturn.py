"""Run Smart Turn v3.x ONNX models on English clips of the Smart Turn v3.2 data.

Usage: python eval_smartturn.py test   -> all English clips of the official test split (smartturn_probs_test.npz)
       python eval_smartturn.py val    -> the validation clips train_heads.py carved from train (smartturn_probs_val.npz)
Preprocessing follows pipecat-ai/smart-turn inference.py (last 8 s, left zero pad, WhisperFeatureExtractor(chunk_length=8)).
"""
import glob
import io
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
import soundfile as sf
from transformers import WhisperFeatureExtractor

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(HERE), "models", "smartturn")
MODELS = ["smart-turn-v3.2-cpu.onnx", "smart-turn-v3.1-cpu.onnx", "smart-turn-v3.0.onnx"]
SR = 16000


def build_session(path):
    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


def last_8s(x):
    n = 8 * SR
    if len(x) > n:
        return x[-n:]
    return np.pad(x, (n - len(x), 0))


_fe = None


def features(x):
    global _fe
    if _fe is None:
        _fe = WhisperFeatureExtractor(chunk_length=8)
    inp = _fe(last_8s(x), sampling_rate=SR, return_tensors="np", padding="max_length",
              max_length=8 * SR, truncation=True, do_normalize=True)
    return inp.input_features.astype(np.float32)


def main():
    mode = sys.argv[1]
    models = MODELS[:1] if (mode != "test" or os.environ.get("ST_FILES")) else MODELS
    sessions = {m: build_session(os.path.join(MODELS_DIR, m)) for m in models}
    for m, s in sessions.items():
        print(m, [(i.name, i.shape) for i in s.get_inputs()], [(o.name, o.shape) for o in s.get_outputs()])
    if mode == "test":
        files = sorted(glob.glob(os.path.join(HERE, "hf", "smart-turn-data-v3.2-test", "data", "*.parquet")))
        keep = None
        if os.environ.get("ST_FILES"): files = files[:int(os.environ["ST_FILES"])]
    else:
        files = sorted(glob.glob(os.path.join(HERE, "hf", "smart-turn-data-v3.2-train", "data", "*.parquet")))
        keep = set(np.load(os.path.join(HERE, "head_probs.npz"))["ids_val"].tolist())
    ids, ys = [], []
    probs = {m: [] for m in models}
    onnx_ms = {m: [] for m in models}
    feat_ms = []
    for path in files:
        pf = pq.ParquetFile(path)
        for rg in range(pf.metadata.num_row_groups):
            for r in pf.read_row_group(rg, columns=["audio", "id", "language", "endpoint_bool"]).to_pylist():
                if r["language"] != "eng" or (keep is not None and r["id"] not in keep):
                    continue
                x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                if x.ndim > 1:
                    x = x.mean(1)
                t = time.perf_counter()
                feats = features(x)
                feat_ms.append((time.perf_counter() - t) * 1000)
                for m, s in sessions.items():
                    t = time.perf_counter()
                    out = s.run(None, {"input_features": feats})
                    onnx_ms[m].append((time.perf_counter() - t) * 1000)
                    probs[m].append(float(np.asarray(out[0]).reshape(-1)[0]))
                ids.append(r["id"])
                ys.append(int(r["endpoint_bool"]))
        print(os.path.basename(path), len(ids), "feat ms", round(float(np.median(feat_ms)), 2),
              {m: round(float(np.median(v)), 2) for m, v in onnx_ms.items()}, flush=True)
    np.savez(os.path.join(HERE, f"smartturn_probs_{mode}.npz"), ids=np.asarray(ids), y=np.asarray(ys, dtype=np.int8),
             feat_ms=np.asarray(feat_ms, dtype=np.float32),
             **{"p_" + m: np.asarray(v, dtype=np.float32) for m, v in probs.items()},
             **{"ms_" + m: np.asarray(v, dtype=np.float32) for m, v in onnx_ms.items()})
    y = np.asarray(ys)
    for m in models:
        p = np.asarray(probs[m]) > 0.5
        print(m, "n", len(y), "acc", (p == y).mean(), "FP/total", (p & (y == 0)).mean(), "FN/total", (~p & (y == 1)).mean())


if __name__ == "__main__":
    main()

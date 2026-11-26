"""Download Smart Turn v3.2 parquet shards, keep English clips, embed the last 5 s with Whistle.

Usage: python embed_shards.py <train|test> <first_shard> <last_shard_exclusive>
Writes expA/emb/<split>_<shard>.npz with float16 frames (concatenated), offsets, labels and metadata.
"""
import io
import json
import os
import sys
import time

os.environ.setdefault("CACTUS_TELEMETRY", "0")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

import cactus

HERE = os.path.dirname(os.path.abspath(__file__))
N_SHARDS = {"train": 83, "test": 10}
DIM = 512
MAX_S = 5  # Whistle.embed is ~7x slower per second above 64 frames (5.12 s); stay under it
SR = 16000


def main():
    split, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    repo = f"pipecat-ai/smart-turn-data-v3.2-{split}"
    out_dir = os.path.join(HERE, "emb")
    os.makedirs(out_dir, exist_ok=True)
    if os.environ.get("EMB_THREADS"):
        cactus.set_threads(int(os.environ["EMB_THREADS"]))  # lets several shard workers run side by side
    w = cactus.Whistle()
    for shard in range(lo, hi):
        out = os.path.join(out_dir, f"{split}_{shard:05d}.npz")
        if os.path.exists(out):
            continue
        name = f"data/train-{shard:05d}-of-{N_SHARDS[split]:05d}.parquet"
        t0 = time.time()
        path = hf_hub_download(repo, name, repo_type="dataset", local_dir=os.path.join(HERE, "hf", os.path.basename(repo)))
        dl = time.time() - t0
        pf = pq.ParquetFile(path)
        frames, lens, meta, embed_ms, dur_s = [], [], [], [], []
        for rg in range(pf.metadata.num_row_groups):
            for r in pf.read_row_group(rg).to_pylist():
                if r["language"] != "eng":
                    continue
                x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                if x.ndim > 1:
                    x = x.mean(1)
                assert sr == SR, sr
                full = len(x) / SR
                x = np.ascontiguousarray(x[-MAX_S * SR:])  # 80000 samples -> 63 frames
                t = time.perf_counter()
                e = w.embed(x)
                embed_ms.append((time.perf_counter() - t) * 1000)
                e = np.asarray(e, dtype=np.float32).reshape(-1, DIM)
                frames.append(e.astype(np.float16))
                lens.append(len(e))
                dur_s.append(len(x) / SR)
                meta.append({k: r[k] for k in ("id", "endpoint_bool", "midfiller", "endfiller", "synthetic", "dataset")} | {"full_s": full})
        np.savez(
            out,
            frames=np.concatenate(frames),
            lens=np.asarray(lens, dtype=np.int32),
            y=np.asarray([m["endpoint_bool"] for m in meta], dtype=np.int8),
            embed_ms=np.asarray(embed_ms, dtype=np.float32),
            dur_s=np.asarray(dur_s, dtype=np.float32),
            meta=json.dumps(meta),
        )
        print(f"{split} shard {shard}: {len(lens)} eng clips, dl {dl:.0f}s, total {time.time()-t0:.0f}s, "
              f"embed median {np.median(embed_ms):.1f} ms, {np.sum(embed_ms)/np.sum(dur_s):.2f} ms/s audio", flush=True)


if __name__ == "__main__":
    main()

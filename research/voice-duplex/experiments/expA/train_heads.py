"""Train small end-of-turn heads on frozen Whistle encoder frames and evaluate on the official test split.

Usage: python train_heads.py [--epochs 25] [--only name1,name2]
Reads expA/emb/{train,test}_*.npz, writes expA/heads/<name>.pt, expA/head_probs.npz and expA/train_results.json.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
DIM = 512
T = 63  # last 5 s of 80 ms frames (Whistle.embed gets ~7x slower above 64 frames)


def load(split):
    frames, lens, y, meta = [], [], [], []
    for p in sorted(glob.glob(os.path.join(HERE, "emb", f"{split}_*.npz"))):
        z = np.load(p, allow_pickle=False)
        frames.append(z["frames"]); lens.append(z["lens"]); y.append(z["y"])
        meta += json.loads(str(z["meta"]))
    frames = np.concatenate(frames); lens = np.concatenate(lens); y = np.concatenate(y)
    # right-align: the last real frame of every clip sits at index T-1, zeros on the left
    x = np.zeros((len(lens), T, DIM), dtype=np.float16)
    off = 0
    for i, n in enumerate(lens):
        k = min(n, T)
        x[i, T - k:] = frames[off + n - k: off + n]
        off += n
    return x, np.minimum(lens, T).astype(np.int64), y.astype(np.float32), meta


class Norm(nn.Module):
    """Standardise frames with train statistics and zero the left padding."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, x, mask):
        return (x - self.mean) / self.std * mask.unsqueeze(-1)


class PoolLinear(nn.Module):
    def __init__(self, norm, mode):
        super().__init__()
        self.norm, self.mode = norm, mode
        self.fc = nn.Linear(DIM * (3 if mode == "mean+last+last5" else 1), 1)

    def forward(self, x, mask):
        x = self.norm(x, mask)
        mean = x.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        last = x[:, -1]
        last5 = x[:, -5:].sum(1) / mask[:, -5:].sum(1, keepdim=True).clamp(min=1)
        f = {"mean": mean, "last": last, "last5": last5, "mean+last+last5": torch.cat([mean, last, last5], -1)}[self.mode]
        return self.fc(f).squeeze(-1)


class GRUHead(nn.Module):
    def __init__(self, norm, n_frames, proj=96, hidden=96, drop=0.2):
        super().__init__()
        self.norm, self.n = norm, n_frames
        self.drop = nn.Dropout(drop)
        self.proj = nn.Linear(DIM, proj)
        self.gru = nn.GRU(proj, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x, mask):
        x = self.norm(x, mask)[:, -self.n:]
        h, _ = self.gru(torch.tanh(self.proj(self.drop(x))))
        return self.fc(self.drop(h[:, -1])).squeeze(-1)


class AttnPoolHead(nn.Module):
    def __init__(self, norm, n_frames, d=128, drop=0.2):
        super().__init__()
        self.norm, self.n = norm, n_frames
        self.drop = nn.Dropout(drop)
        self.proj = nn.Linear(DIM, d)
        self.pos = nn.Parameter(torch.zeros(n_frames, d))  # position counted from the end of the turn
        self.conv = nn.Conv1d(d, d, 5, padding=2)
        self.score = nn.Linear(d, 1)
        self.mlp = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Dropout(drop), nn.Linear(64, 1))

    def forward(self, x, mask):
        x = self.norm(x, mask)[:, -self.n:]
        m = mask[:, -self.n:]
        h = torch.nn.functional.gelu(self.proj(self.drop(x))) + self.pos
        h = h + torch.nn.functional.gelu(self.conv((h * m.unsqueeze(-1)).transpose(1, 2)).transpose(1, 2))
        a = self.score(h).squeeze(-1).masked_fill(m < 0.5, -1e4)
        pooled = (torch.softmax(a, -1).unsqueeze(-1) * h).sum(1)
        return self.mlp(pooled).squeeze(-1)


def build(name, norm):
    kind, _, arg = name.partition("_")
    if kind == "linear":
        return PoolLinear(norm, arg)
    if kind == "gru":
        return GRUHead(norm, int(arg))
    if kind == "attn":
        return AttnPoolHead(norm, int(arg))
    raise ValueError(name)


def predict(model, x, lens, bs=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb = torch.from_numpy(x[i:i + bs].astype(np.float32))
            mb = (torch.arange(T)[None, :] >= (T - torch.from_numpy(lens[i:i + bs]))[:, None]).float()
            out.append(torch.sigmoid(model(xb, mb)).numpy())
    return np.concatenate(out)


def metrics(p, y, thr):
    pred = p > thr
    pos, neg = y == 1, y == 0
    tp, fp, fn = (pred & pos).sum(), (pred & neg).sum(), (~pred & pos).sum()
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {
        "threshold": float(thr),
        "accuracy": float((pred == pos).mean()),
        "f1": float(2 * prec * rec / max(prec + rec, 1e-9)),
        "precision": float(prec),
        "recall": float(rec),
        "false_cutoff_rate": float(fp / max(neg.sum(), 1)),  # said "complete" on an incomplete clip
        "missed_end_rate": float(fn / max(pos.sum(), 1)),  # said "incomplete" on a complete clip
    }


def thr_for_fcr(p, y, target=0.03):
    """Lowest threshold whose false-cutoff rate on (p, y) is <= target."""
    neg = np.sort(p[y == 0])[::-1]
    k = int(np.floor(target * len(neg)))  # allowed false positives
    return float(neg[k]) if k < len(neg) else 0.0


def auc(p, y):
    order = np.argsort(p)
    ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--only", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

    x, lens, y, meta = load("train")
    xt, lt, yt, mt = load("test")
    assert not ({m["id"] for m in meta} & {m["id"] for m in mt}), "train/test id overlap"
    perm = rng.permutation(len(y))
    n_val = len(y) // 10
    va, tr = perm[:n_val], perm[n_val:]
    print(f"train {len(tr)} (complete {y[tr].mean():.3f}) val {len(va)} test {len(yt)} (complete {yt.mean():.3f})", flush=True)

    # per-dimension statistics over real (unpadded) train frames, from a sample to keep memory low
    samp = x[tr[:3000]].astype(np.float32); sm = np.arange(T)[None, :] >= (T - lens[tr[:3000]])[:, None]
    mean, std = samp[sm].mean(0), samp[sm].std(0) + 1e-5
    del samp

    names = ["linear_mean", "linear_last", "linear_last5", "linear_mean+last+last5",
             "gru_25", "gru_50", "gru_63", "attn_25", "attn_50", "attn_63"]
    if args.only:
        names = args.only.split(",")
    os.makedirs(os.path.join(HERE, "heads"), exist_ok=True)
    res_path = os.path.join(HERE, "train_results.json")
    results = json.load(open(res_path, encoding="utf-8")) if os.path.exists(res_path) else {}
    probs_path = os.path.join(HERE, "head_probs.npz")
    probs = dict(np.load(probs_path)) if os.path.exists(probs_path) else {}
    probs["y_test"] = yt; probs["y_val"] = y[va]
    probs["ids_test"] = np.asarray([m["id"] for m in mt]); probs["ids_val"] = np.asarray([meta[i]["id"] for i in va])

    for name in names:
        torch.manual_seed(args.seed)
        model = build(name, Norm(mean, std))
        n_params = sum(p.numel() for p in model.parameters())
        linear = name.startswith("linear")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3 if linear else 2e-3, weight_decay=1e-2)
        steps = args.epochs * (len(tr) // 128)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=opt.param_groups[0]["lr"], total_steps=steps, pct_start=0.1)
        loss_fn = nn.BCEWithLogitsLoss()
        best, best_state, t0 = 1e9, None, time.time()
        for ep in range(args.epochs):
            model.train()
            order = rng.permutation(tr)
            for i in range(0, len(order) - 127, 128):
                idx = np.sort(order[i:i + 128])
                xb = torch.from_numpy(x[idx].astype(np.float32))
                mb = (torch.arange(T)[None, :] >= (T - torch.from_numpy(lens[idx]))[:, None]).float()
                loss = loss_fn(model(xb, mb), torch.from_numpy(y[idx]))
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            pv = predict(model, x[va], lens[va])
            vl = float(-np.mean(y[va] * np.log(pv + 1e-7) + (1 - y[va]) * np.log(1 - pv + 1e-7)))
            if vl < best:
                best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
            print(f"{name} ep {ep} val loss {vl:.4f} acc {((pv > 0.5) == y[va]).mean():.4f} ({time.time()-t0:.0f}s)", flush=True)
        model.load_state_dict(best_state)
        torch.save({"name": name, "state": best_state, "mean": mean, "std": std}, os.path.join(HERE, "heads", name + ".pt"))
        pv, pt = predict(model, x[va], lens[va]), predict(model, xt, lt)
        thr_val = thr_for_fcr(pv, y[va])
        results[name] = {
            "params": n_params, "val_loss": best, "val_accuracy": float(((pv > 0.5) == y[va]).mean()), "val_auc": auc(pv, y[va]),
            "test_auc": auc(pt, yt),
            "test@0.5": metrics(pt, yt, 0.5),
            "test@val_tuned_fcr3": metrics(pt, yt, thr_val),
            "test@test_fcr3_oracle": metrics(pt, yt, thr_for_fcr(pt, yt)),
            "train_seconds": time.time() - t0,
        }
        probs["val_" + name] = pv; probs["test_" + name] = pt
        print(name, json.dumps(results[name]), flush=True)
        json.dump(results, open(res_path, "w", encoding="utf-8"), indent=1)
        np.savez(probs_path, **probs)


if __name__ == "__main__":
    main()

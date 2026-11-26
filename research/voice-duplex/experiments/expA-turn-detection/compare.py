"""Whistle-embedding heads vs Smart Turn v3.2 on the same English test clips."""
import json, numpy as np
st = np.load("smartturn_probs_test.npz", allow_pickle=True); hp = np.load("head_probs.npz", allow_pickle=True)
print("head_probs keys", list(hp.keys())[:20])
ids = {str(i): k for k, i in enumerate(hp["ids_test"].tolist())}
sel = [(k, ids[str(i)]) for k, i in enumerate(st["ids"].tolist()) if str(i) in ids]
a = np.array([s for s, _ in sel]); b = np.array([h for _, h in sel]); y = st["y"][a].astype(int)
def m(p, thr=0.5):
    d = p > thr; return dict(acc=round(float((d == y).mean()), 4), false_cutoff=round(float((d & (y == 0)).sum() / max(1, (y == 0).sum())), 4), missed_end=round(float((~d & (y == 1)).sum() / max(1, (y == 1).sum())), 4))
out = {"n": int(len(y)), "smart_turn_v3.2": m(st[[k for k in st.keys() if k.startswith("p_")][0]][a]), "smart_turn_ms_median": float(np.median(st[[k for k in st.keys() if k.startswith("ms_")][0]])), "smart_turn_feature_ms_median": float(np.median(st["feat_ms"]))}
for k in hp.keys():
    if k.startswith("test_") or k.startswith("p_test"):
        try: out[k] = m(hp[k][b])
        except Exception as e: out[k] = repr(e)[:80]
print(json.dumps(out, indent=1)); json.dump(out, open("compare.json", "w"), indent=1)

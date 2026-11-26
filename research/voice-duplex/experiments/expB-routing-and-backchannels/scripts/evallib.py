"""Metrics shared by every router: accuracy, macro-F1, confusion, dangerous errors, latency, prefix (early decision) analysis."""
import json, os, time
import numpy as np
from common import *

DELEG = ("DELEGATE_READ", "DELEGATE_WRITE")
RESULTS = os.path.join(B, "results.json")
CONF_T = 0.7


def save_result(section, key, value):
    """Several experiment processes write at once: keep one file per entry (the source of truth) and rebuild results.json under a lock."""
    parts = os.path.join(B, "results_parts"); os.makedirs(parts, exist_ok=True)
    with open(os.path.join(parts, f"{section}__{key}.json"), "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False)
    lock = RESULTS + ".lock"
    for _ in range(200):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd); break
        except FileExistsError:
            time.sleep(0.05)
    try:
        data = json.load(open(RESULTS, encoding="utf-8")) if os.path.exists(RESULTS) else {}
        for name in sorted(os.listdir(parts)):
            sec, k = name[:-5].split("__", 1)
            data.setdefault(sec, {})[k] = json.load(open(os.path.join(parts, name), encoding="utf-8"))
        with open(RESULTS, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
    finally:
        try: os.remove(lock)
        except OSError: pass


def metrics(gold, pred):
    n = len(gold)
    cm = [[0] * len(LABELS) for _ in LABELS]
    for g, p in zip(gold, pred):
        cm[LABELS.index(g)][LABELS.index(p)] += 1
    f1s = []
    for i in range(len(LABELS)):
        tp = cm[i][i]; fp = sum(cm[j][i] for j in range(len(LABELS))) - tp; fn = sum(cm[i]) - tp
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    nd = sum(g in DELEG for g in gold); ns = sum(g == "SMALLTALK" for g in gold)
    dropped = sum(g in DELEG and p in ("SMALLTALK", "BACKCHANNEL") for g, p in zip(gold, pred))
    not_deleg = sum(g in DELEG and p not in DELEG for g, p in zip(gold, pred))
    wasted = sum(g == "SMALLTALK" and p in DELEG for g, p in zip(gold, pred))
    return {"n": n, "accuracy": round(sum(g == p for g, p in zip(gold, pred)) / n, 4), "macro_f1": round(float(np.mean(f1s)), 4),
            "per_label_f1": {l: round(f, 3) for l, f in zip(LABELS, f1s)},
            "dropped_money_question": {"count": dropped, "of": nd, "rate": round(dropped / nd, 4)},
            "delegate_routed_to_any_non_delegate": {"count": not_deleg, "of": nd, "rate": round(not_deleg / nd, 4)},
            "wasted_call": {"count": wasted, "of": ns, "rate": round(wasted / ns, 4)},
            "confusion_rows_gold_cols_pred": {"labels": LABELS, "matrix": cm}}


def latency(fn, rows, passes=1):
    ts = []
    for _ in range(passes):
        for r in rows:
            t = time.perf_counter(); fn(r); ts.append((time.perf_counter() - t) * 1000)
    return {"median_ms": round(float(np.median(ts)), 4), "p95_ms": round(float(np.percentile(ts, 95)), 4), "n": len(ts)}


def _stable_point(preds, gold_ok):
    """Smallest k (1-based word count) from which every later prefix is correct; None if the full utterance is wrong."""
    n = len(preds)
    if not gold_ok(preds[-1]):
        return None
    k = n
    while k > 1 and gold_ok(preds[k - 2]):
        k -= 1
    return k


def _summ(points):
    """points: list of (k, n) for utterances that end correct."""
    if not points:
        return None
    frac = [k / n for k, n in points]; spare = [n - k for k, n in points]
    return {"n_final_correct": len(points), "median_fraction_heard": round(float(np.median(frac)), 3), "mean_fraction_heard": round(float(np.mean(frac)), 3),
            "median_words_before_end": float(np.median(spare)), "mean_words_before_end": round(float(np.mean(spare)), 2),
            "share_decided_1plus_words_early": round(float(np.mean([s >= 1 for s in spare])), 3),
            "share_decided_3plus_words_early": round(float(np.mean([s >= 3 for s in spare])), 3)}


def prefix_analysis(predict_one, rows, topic_one=None, min_words=2):
    """Feed growing word prefixes of every DELEGATE_* and SMALLTALK utterance. Context flags stay as in the full item."""
    out = {}
    items = [r for r in rows if r["label"] in DELEG + ("SMALLTALK",) and len(r["text"].split()) >= min_words]
    traces = []
    for r in items:
        w = r["text"].split()
        preds = [predict_one(dict(r, text=" ".join(w[:k]))) for k in range(1, len(w) + 1)]
        traces.append((r, preds))
    for group, labs in (("DELEGATE", DELEG), ("SMALLTALK", ("SMALLTALK",))):
        g = [(r, p) for r, p in traces if r["label"] in labs]
        exact, coarse, conf_wrong, conf_wrong_type = [], [], 0, 0
        for r, preds in g:
            n = len(preds)
            k = _stable_point(preds, lambda p: p[0] == r["label"])
            if k: exact.append((k, n))
            k = _stable_point(preds, lambda p: (p[0] in DELEG) == (r["label"] in DELEG) and (r["label"] in DELEG or p[0] == r["label"]))
            if k: coarse.append((k, n))
            early = preds[:-1]
            if group == "DELEGATE":
                conf_wrong += any(p[0] in ("SMALLTALK", "BACKCHANNEL", "META", "CORRECTION") and p[1] >= CONF_T for p in early)
                conf_wrong_type += any(p[0] in DELEG and p[0] != r["label"] and p[1] >= CONF_T for p in early)
            else:
                conf_wrong += any(p[0] in DELEG and p[1] >= CONF_T for p in early)
        res = {"n_utterances": len(g), "final_correct_rate": round(len(exact) / len(g), 3),
               "exact_route": _summ(exact), "delegate_vs_not": _summ(coarse)}
        if group == "DELEGATE":
            res["early_prefix_confidently_non_delegate"] = {"count": conf_wrong, "rate": round(conf_wrong / len(g), 3),
                                                              "meaning": "some proper prefix was routed SMALLTALK/BACKCHANNEL/META/CORRECTION with confidence >= 0.7 (reflex would have started talking)"}
            res["early_prefix_confidently_wrong_delegate_type"] = {"count": conf_wrong_type, "rate": round(conf_wrong_type / len(g), 3)}
        else:
            res["early_prefix_confidently_delegate"] = {"count": conf_wrong, "rate": round(conf_wrong / len(g), 3),
                                                         "meaning": "some proper prefix was routed DELEGATE_* with confidence >= 0.7 (wasted speculative cloud call)"}
        out[group] = res
    if topic_one:
        pts, n_items, final_ok, conf_wrong = [], 0, 0, 0
        for r in items:
            if r["label"] not in DELEG or r["topic"] in ("", "other"):
                continue
            n_items += 1
            w = r["text"].split()
            preds = [topic_one(dict(r, text=" ".join(w[:k]))) for k in range(1, len(w) + 1)]
            k = _stable_point(preds, lambda p: p == r["topic"])
            if k:
                pts.append((k, len(w))); final_ok += 1
            conf_wrong += any(p not in (r["topic"], "other", "") for p in preds[:-1])
        s = _summ(pts) or {}
        s.update({"n_utterances": n_items, "final_correct_rate": round(final_ok / max(n_items, 1), 3),
                  "early_prefix_named_a_wrong_topic": {"count": conf_wrong, "rate": round(conf_wrong / max(n_items, 1), 3)}})
        out["TOPIC"] = s
    return out

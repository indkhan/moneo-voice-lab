"""Train and evaluate the local routers. Usage: eval_local.py classical | needle_emb | needle_tools | needle_extract"""
import pickle, sys
from common import *
from evallib import *
import routers as R

train = read_jsonl(os.path.join(DATA, "train.jsonl"))
test = read_jsonl(os.path.join(DATA, "test.jsonl"))
gold = [r["label"] for r in test]
dtest = [r for r in test if r["label"] in DELEG]
ART = os.path.join(B, "artifacts"); os.makedirs(ART, exist_ok=True)


def topic_ctx(r):
    """Bare answers ('yeah', 'the second one') carry no topic words: fall back to what the assistant asked and the previous request."""
    return dict(r, text=r["text"] + " " + r.get("last_assistant", "") + " " + r.get("prev_user", "")) if R.asked(r) else r


def run(key, router, size_bytes, topic_one, prefix=True, lat_passes=3, lat_fn=None, extra=None):
    preds = [router.predict_one(r) for r in test]
    m = metrics(gold, [p[0] for p in preds])
    m["name"] = router.name
    m["size_bytes"] = size_bytes
    m["latency"] = latency(lat_fn or router.predict_one, test, lat_passes)
    m["topic_accuracy_on_gold_delegates"] = round(sum(topic_one(r) == r["topic"] for r in dtest) / len(dtest), 4)
    m["topic_accuracy_when_routed_delegate"] = round(sum(p[2] == r["topic"] for r, p in zip(test, preds) if r["label"] in DELEG and p[0] in DELEG)
                                                      / max(1, sum(r["label"] in DELEG and p[0] in DELEG for r, p in zip(test, preds))), 4)
    m["errors"] = [{"id": r["id"], "text": r["text"], "gold": r["label"], "pred": p[0], "conf": round(p[1], 2), "spk": r["assistant_speaking"]}
                   for r, p in zip(test, preds) if p[0] != r["label"]]
    if extra: m.update(extra)
    save_result("routers", key, m)
    if prefix:
        save_result("prefix", key, prefix_analysis(router.predict_one, test, topic_one))
    print(key, "acc", m["accuracy"], "f1", m["macro_f1"], "dropped", m["dropped_money_question"]["count"], "wasted", m["wasted_call"]["count"],
          "lat", m["latency"], "topic", m["topic_accuracy_on_gold_delegates"], flush=True)
    return m


mode = sys.argv[1]
if mode == "classical":
    import routers_ext as X
    rule_topic = lambda r: R.rule_topic(topic_ctx(r)["text"]) if R.rule_topic(r["text"]) == "other" else R.rule_topic(r["text"])
    rules_kb = os.path.getsize(os.path.join(B, "scripts", "routers.py"))
    run("rules_v1", R.RuleRouter(), rules_kb, rule_topic,
        extra={"size_note": "size is the whole routers.py source; the rules are a few KB of regex", "written": "before seeing any test result"})
    v2 = X.RuleRouterV2()
    run("rules_v2", v2, rules_kb + os.path.getsize(os.path.join(B, "scripts", "routers_ext.py")), rule_topic,
        extra={"written": "after reading v1 test errors: test score is optimistic, see train_cross_validation for its score on the 500 train items"})
    save_result("topic", "rule_keywords", {"accuracy": round(sum(rule_topic(r) == r["topic"] for r in dtest) / len(dtest), 4), "n": len(dtest),
                "latency": latency(rule_topic, dtest, 3),
                "rows": [{"text": r["text"], "gold": r["topic"], "pred": rule_topic(r)} for r in dtest]})
    models = {}
    for kind in ("lr", "svm"):
        m = X.TfidfLean(kind).fit(train)
        models[kind] = m
        pickle.dump(m, open(os.path.join(ART, f"tfidf_{kind}.pkl"), "wb"))
        run("tfidf_" + kind, m, m.size_bytes, m.topic_one,
            extra={"size_note": "pickle of route + topic sklearn pipelines", "latency_note": "lean inference (dict lookups + one matrix product), route + topic",
                   "latency_sklearn_pipeline": latency(m.predict_one_sklearn, test, 1)})
    tn = R.TinyNet().fit(train)
    pickle.dump(tn, open(os.path.join(ART, "tinynet.pkl"), "wb"))
    run("tinynet", tn, tn.size_bytes, lambda r: tn.tcls[tn._fwd(tn.T, tn.ids(r))[0]],
        extra={"size_note": "route head as float16 (8192x32 embeddings + 48 hidden); topic head is the same size again"})
    hy = X.Hybrid(models["svm"], v2)
    run("hybrid_svm_rules", hy, models["svm"].size_bytes + rules_kb, models["svm"].topic_one)

elif mode == "needle_emb":
    ne = R.NeedleEmb()
    ne.fit(train)
    needle_mb = os.path.getsize(os.path.expanduser(r"~\.cache\cactus\models\needle3\d6c7d6a2e9655505e0f8e23339356dd1705900eb\needle3.cact"))
    pickle.dump((ne.clf, ne.tclf), open(os.path.join(ART, "needle_emb_lr.pkl"), "wb"))
    for r in test: ne.emb(r["text"])
    run("needle_emb_lr", ne, needle_mb + ne.head_bytes, lambda r: ne.tclf.predict(ne.x(r)[None])[0],
        lat_passes=1, lat_fn=lambda r: ne.predict_one(r, fresh=True),
        extra={"size_note": f"needle3.cact {needle_mb} bytes + {ne.head_bytes} bytes of logistic-regression heads", "threads": 1})
    ne.save()

elif mode == "needle_tools":
    nt = R.NeedleTools()
    m = run("needle_tools_zeroshot", nt, os.path.getsize(os.path.expanduser(r"~\.cache\cactus\models\needle3\d6c7d6a2e9655505e0f8e23339356dd1705900eb\needle3.cact")),
            lambda r: nt.predict_one(dict(r, label="x"))[2] or "other", prefix=False, lat_passes=0 or 1,
            extra={"threads": 1})
    save_result("routers", "needle_tools_zeroshot", dict(m, no_valid_call=nt.nocall))
    # prefix analysis is ~1000 completions; run on every 2nd utterance to keep it to a few minutes
    sub = [r for i, r in enumerate(test) if i % 2 == 0]
    save_result("prefix", "needle_tools_zeroshot", dict(prefix_analysis(nt.predict_one, sub), note="every 2nd test utterance"))

elif mode == "needle_extract":
    ex = R.NeedleExtractTopic()
    import time, numpy as np
    ts, ok, rows = [], 0, []
    for r in dtest:
        t = time.perf_counter(); p = ex.topic(topic_ctx(r)["text"]); ts.append((time.perf_counter() - t) * 1000)
        ok += p == r["topic"]; rows.append({"text": r["text"], "gold": r["topic"], "pred": p})
    save_result("topic", "needle_extract_zeroshot", {"accuracy": round(ok / len(dtest), 4), "n": len(dtest),
                "median_ms": round(float(np.median(ts)), 1), "p95_ms": round(float(np.percentile(ts, 95)), 1), "rows": rows})
    print("needle extract topic acc", ok / len(dtest), np.median(ts))

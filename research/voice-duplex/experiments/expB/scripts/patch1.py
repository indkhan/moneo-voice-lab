p='eval_local.py'
s=open(p,encoding='utf-8').read()
a=s.index('if mode == "classical":'); b=s.index('elif mode == "needle_emb":')
new='''if mode == "classical":
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

'''
open(p,'w',encoding='utf-8').write(s[:a]+new+s[b:])

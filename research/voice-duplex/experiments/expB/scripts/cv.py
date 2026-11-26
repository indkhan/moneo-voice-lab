"""5-fold cross-validation on TRAIN only, to choose settings without touching the test set."""
from common import *
import routers as R
import routers_ext as X
from evallib import metrics, save_result

train = read_jsonl(os.path.join(DATA, "train.jsonl"))
gold = [r["label"] for r in train]
out = {}
for name, r in (("rules_v1", R.RuleRouter()), ("rules_v2", X.RuleRouterV2())):
    m = metrics(gold, [r.predict_one(x)[0] for x in train])
    out[name + "_on_train_unfitted"] = {k: m[k] for k in ("accuracy", "macro_f1", "dropped_money_question", "wasted_call")}
    print(name, "on train (never fitted to it):", m["accuracy"], m["macro_f1"], m["dropped_money_question"], m["wasted_call"])


def cv(make):
    pred = [None] * len(train)
    for f in range(5):
        m = make().fit([x for i, x in enumerate(train) if i % 5 != f])
        for i, x in enumerate(train):
            if i % 5 == f:
                pred[i] = m.predict_one(x)
    mm = metrics(gold, [p[0] for p in pred])
    return {"accuracy": mm["accuracy"], "macro_f1": mm["macro_f1"], "dropped": mm["dropped_money_question"]["count"], "wasted": mm["wasted_call"]["count"]}


for key, make in (("tfidf_lr_plain", lambda: X.TfidfLean("lr", rich=False)), ("tfidf_lr_rich", lambda: X.TfidfLean("lr")),
                  ("tfidf_svm_plain", lambda: X.TfidfLean("svm", rich=False)), ("tfidf_svm_rich", lambda: X.TfidfLean("svm")),
                  ("tinynet_elr0.05", lambda: R.TinyNet(elr=0.05)), ("tinynet_elr0.5", lambda: R.TinyNet(elr=0.5)), ("tinynet_elr2", lambda: R.TinyNet(elr=2.0))):
    out[key] = cv(make)
    print(key, out[key], flush=True)
save_result("train_cross_validation", "five_fold", out)

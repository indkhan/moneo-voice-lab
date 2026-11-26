from common import *
import routers as R
from evallib import metrics
train = read_jsonl(os.path.join(DATA, "train.jsonl"))
gold = [r["label"] for r in train]
for elr in (0.05, 0.2):
    pred = [None] * len(train); tok = 0; tn = 0
    for f in range(5):
        m = R.TinyNet(elr=elr).fit([x for i, x in enumerate(train) if i % 5 != f])
        for i, x in enumerate(train):
            if i % 5 == f:
                pred[i] = m.predict_one(x)
                if x["label"] in R.DELEG:
                    tn += 1; tok += m.tcls[m._fwd(m.T, m.ids(x))[0]] == x["topic"]
    mm = metrics(gold, [p[0] for p in pred])
    print("adagrad elr", elr, mm["accuracy"], mm["macro_f1"], mm["dropped_money_question"]["count"], mm["wasted_call"]["count"], "topic", round(tok / tn, 3), flush=True)

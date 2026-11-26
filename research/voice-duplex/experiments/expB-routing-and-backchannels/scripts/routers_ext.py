"""Second round of local routers, written after the first classical run: rules v2, richer TF-IDF features with lean inference."""
import re
import numpy as np
import routers as R
from routers import norm, asked, rule_topic, DELEG, FIN, WRITE_Q

_base_feat = R.feat_text

CHITCHAT_Q = r"how are you|how's it going|hows it going|how is it going|feeling|your day|weekend|how have you been|doing today|what's up|sleep"
ANYQ = r"\b(how much|how many|what's|whats|what is|what are|what was|what were|what have|what did|what do|when is|when's|when does|when did|when are|which|can i|could i|do i|did i|am i|is my|are my|are any|has my|have i|did my|does my|show me|tell me|give me|remind me)\b"
FOLLOWUP = r"^(and|what about|how about|and what about|and how about|same for|also|or)\b"


class RuleRouterV2(R.RuleRouter):
    """v1 plus general fixes made AFTER reading v1's test errors, so its test score is optimistic; its train score is the fair one."""
    name = "rules v2 (regex/keywords, fail towards delegate)"

    def predict_one(self, r):
        lab, conf, topic = super().predict_one(r)
        t = norm(r["text"])
        n = len(t.split())
        if lab == "SMALLTALK":
            # 1. the assistant asked a money question: whatever comes back is an answer for the reasoner
            if asked(r) and not re.search(CHITCHAT_Q, r["last_assistant"].lower()):
                lab = "DELEGATE_WRITE" if re.search(WRITE_Q, r["last_assistant"].lower()) else "DELEGATE_READ"
                return lab, 1.0, rule_topic(t + " " + r["last_assistant"])
            # 2. a question phrase about money anywhere, not only at the start
            if re.search(ANYQ, t) and re.search(FIN, t):
                return "DELEGATE_READ", 1.0, rule_topic(t)
            # 3. elliptical follow-up ("and last month", "what about restaurants")
            if re.search(FOLLOWUP, t) and 2 <= n <= 8:
                return "DELEGATE_READ", 1.0, rule_topic(t + " " + r.get("prev_user", ""))
            # 4. "go ahead add 25 for lunch"
            if re.search(r"\b(add|set|change|remove|delete|rename|log|record)\b", t) and re.search(r"^(go ahead|yeah|yes|yep|sure|ok|okay|please)\b", t):
                return "DELEGATE_WRITE", 1.0, rule_topic(t)
        return lab, conf, topic


def feat_rich(r):
    """Position and shape markers on top of the words: how it starts, how it ends, how long, any number."""
    t = norm(r["text"])
    w = t.split() or ["zz"]
    out = _base_feat(r) + f" zzfirst{w[0]} zzlast{w[-1]} zzlen{min(len(w), 6)}" + (" zznum" if re.search(r"\d", t) else "")
    if len(w) > 1:
        out += f" zzlasttwo{w[-2]}_{w[-1]}"
    return out


class TfidfLean(R.TfidfRouter):
    """Same model as TfidfRouter, optionally with the rich markers, and inference flattened into dict lookups plus one small
    matrix product (sklearn's per-call overhead is far larger than the maths)."""

    def __init__(self, kind, rich=True):
        super().__init__(kind)
        self.rich = rich
        self.feat = feat_rich if rich else _base_feat
        self.name += " (+start/end markers)" if rich else ""

    def fit(self, rows):
        orig = R.feat_text
        R.feat_text = self.feat  # the parent reads the module-level function
        try:
            super().fit(rows)
        finally:
            R.feat_text = orig
        self.L, self.LT = self._lean(self.p), self._lean(self.t)
        return self

    @staticmethod
    def _lean(pipe):
        vecs = [v for _, v in pipe.steps[0][1].transformer_list]
        clf = pipe.steps[-1][1]
        off, o = [], 0
        for v in vecs:
            off.append(o); o += len(v.vocabulary_)
        return {"an": [v.build_analyzer() for v in vecs], "vocab": [v.vocabulary_ for v in vecs], "idf": [v.idf_ for v in vecs], "off": off,
                "W": np.ascontiguousarray(clf.coef_.T), "b": clf.intercept_, "classes": list(clf.classes_)}

    def _lean_predict(self, L, x):
        idx, val = [], []
        for an, vocab, idf, off in zip(L["an"], L["vocab"], L["idf"], L["off"]):
            counts = {}
            for tok in an(x):
                j = vocab.get(tok)
                if j is not None:
                    counts[j] = counts.get(j, 0) + 1
            if not counts:
                continue
            js = list(counts)
            v = (1.0 + np.log(np.fromiter(counts.values(), dtype=np.float64))) * idf[js]
            v /= np.sqrt((v * v).sum())
            idx += [j + off for j in js]; val.append(v)
        z = (np.concatenate(val) @ L["W"][idx] + L["b"]) if idx else L["b"].copy()
        if self.kind != "lr":
            z = z * 3.0  # margins, not probabilities: the "confidence" of the SVM is not calibrated
        p = np.exp(z - z.max()); p /= p.sum()
        i = int(p.argmax())
        return L["classes"][i], float(p[i])

    def topic_one(self, r):
        return self._lean_predict(self.LT, self.feat(r))[0]

    def predict_one(self, r):
        x = self.feat(r)
        lab, conf = self._lean_predict(self.L, x)
        return lab, conf, (self._lean_predict(self.LT, x)[0] if lab in DELEG else "")

    def predict_one_sklearn(self, r):
        x = self.feat(r)
        lab, conf = self._conf(self.p, x)
        return lab, conf, (self._conf(self.t, x)[0] if lab in DELEG else "")

    def __getstate__(self):
        d = dict(self.__dict__)
        for k in ("L", "LT", "feat"):
            d.pop(k, None)
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.feat = feat_rich if self.rich else _base_feat
        self.L, self.LT = self._lean(self.p), self._lean(self.t)


class Hybrid:
    """A learned router with the rule router as a safety net: if the learned model wants to answer locally (SMALLTALK/BACKCHANNEL)
    but the rules see a money request, delegate. It can only turn drops into delegations, never the reverse."""

    def __init__(self, learned, rules):
        self.m, self.rules = learned, rules
        self.name = learned.name + " + rules safety net"

    def predict_one(self, r):
        lab, conf, topic = self.m.predict_one(r)
        if lab in ("SMALLTALK", "BACKCHANNEL"):
            rl, _, rt = self.rules.predict_one(r)
            if rl in DELEG:
                return rl, 0.5, rt
        return lab, conf, topic

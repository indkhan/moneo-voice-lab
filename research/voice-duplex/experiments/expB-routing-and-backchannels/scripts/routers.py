"""Local routers: hand rules, TF-IDF + linear models, a tiny fastText-style net, Needle embeddings, Needle tool calls.

Every router exposes predict_one(row) -> (label, confidence, topic). Context used by the local routers is two flags only:
assistant_speaking, and "asked" (assistant silent and its last sentence ends with a question mark). prev_user is NOT used by
local routers, because in a real call it is nearly always non-empty and the dataset would leak CORRECTION through it.
"""
import os, pickle, re, time, zlib
import numpy as np
from common import *

DELEG = ("DELEGATE_READ", "DELEGATE_WRITE")


def asked(r):
    return (not r["assistant_speaking"]) and r.get("last_assistant", "").rstrip().endswith("?")


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9.' -]", " ", t.lower())).strip()


def feat_text(r):
    return norm(r["text"]) + (" zzspk" if r["assistant_speaking"] else "") + (" zzasked" if asked(r) else "")


# ---------------------------------------------------------------- rules
NUMWORD = r"(?:\d[\d.,]*k?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|grand|a couple|half)"
MONEY_UNIT = r"(?:euros?|eur|bucks?|dollars?|pounds?|quid|kroner?|kr|cents?|p\b|\$|€|£)"
SPENT = r"\b(spent|paid|pay(?:ed)?|bought|cost|costs|costed|got paid|received|got|lent|borrowed|loaned|owes?|owed|gave|transferred|sent|earned|made|refunded?|charged|tipped|withdrew|deposited|put in|topped up|came in|came to|was|is|has|holds|have|got about)\b"
WRITE_CMD = r"^(?:(?:ok(?:ay)?|so|hey|oh|uh|um|and|also|please|can you|could you|would you|i want to|i need to|i wanna|i'd like to|let's|lets|just|go ahead and|btw|by the way)\s+)*(add|set|change|update|remove|delete|create|rename|log|record|move|put|mark|categori[sz]e|recategori[sz]e|increase|lower|raise|reduce|bump|make|cancel|split|edit|track|note|save|schedule|merge|tag|label|drop|start|stop tracking)\b"
BACKCH = {"mm-hm", "mmhmm", "mhm", "mm", "mmm", "m-hm", "uh-huh", "uh", "huh", "yeah", "yea", "yep", "yup", "ok", "okay", "k", "right", "sure", "i", "see", "got", "it", "oh", "ah", "hm", "hmm", "alright", "cool", "nice", "gotcha", "yes", "true", "fair", "enough", "makes", "sense", "wow", "really", "aha", "mm-hmm", "mhmm", "uhhuh", "okay.", "yeah."}
AFFIRM = r"^(?:oh |uh |um |ah )?(yeah|yea|yes|yep|yup|sure|okay|ok|go ahead|do it|please|please do|yes please|yeah please|that one|correct|right|exactly|no|nope|nah|no thanks|not now|sounds good|go for it|yeah go ahead|yes add it|add it|yeah do that|the (first|second|third|last) one)\b"
WRITE_Q = r"\b(add|change|set|update|remove|delete|create|save|log|record|rename|move|categori|mark|cancel|adjust|raise|lower|split|edit|make that|make it|card)\b"
DANGLE = {"on", "in", "at", "for", "to", "the", "a", "an", "my", "and", "but", "or", "so", "that", "like", "uh", "um", "of", "with", "from", "is", "was", "i", "maybe", "because", "cause", "about", "if", "then", "into", "than", "as", "by", "this", "those", "these", "our", "your", "their", "its", "i'm", "it's", "what's", "how", "when", "where", "which", "should", "would", "could", "can", "do", "did", "does", "are", "were", "be", "been", "have", "has", "had", "will", "just", "also", "between", "under", "over", "per", "around", "roughly", "spent", "paid", "set", "add", "change", "and then", "we", "you", "he", "she", "they", "not", "thinking", "wondering", "er", "erm", "well", "uhh", "umm", "hmm", "much", "many", "some", "any", "every", "each", "another", "more", "until", "since", "after", "before", "during", "versus", "vs", "plus", "minus", "almost", "nearly", "exactly", "only", "called", "named", "the thing is", "whether", "while", "though", "although", "except", "without", "towards", "toward", "onto", "off", "up", "out"}
CORR = r"^(?:(?:oh|uh|um|ah|wait|sorry|oops|hmm|hang on|hold on)[, ]+)*(no\b|nope\b|nah\b|not\b|actually\b|i meant\b|i mean\b|i said\b|scratch that|correction|rather\b|no no|my bad|that's wrong|that's not|thats not|wrong\b|make it\b|make that\b)|\b(i meant|i said|not the|instead of|instead\b|should be|should've been|scratch that|rather than|i misspoke|change that to|make that|make it)\b"
META = r"(say that again|repeat|come again|pardon|what did you say|what was that|is it done|is that done|are you (still )?(there|working|on it)|how long|still (there|working|thinking|loading|on it)|you (still )?there|^hello\??$|hello\?|what do you mean|how did you (get|come|calculate|work)|where did (you|that)|slow down|louder|speak up|too fast|never ?mind|cancel that|^stop\b|hold on|hang on|^wait\b|explain that|explain what|didn't catch|did not catch|one more time|sorry what|^what\??$|^huh\??$|^sorry\??$|can you hear|did you (get|hear|catch)|any (luck|update)|are you done|you done|done yet|finished yet|taking (so )?long|what does that mean|in simpler|shorter|skip that|^again\??$|go back|last (part|bit|number)|missed that|what was the (number|total|amount)|still waiting|^and\?$|^so\?$)"
FIN = r"\b(spen[dt]|spending|balance|account|budget|bill|bills|rent|salary|paycheck|pay ?check|income|net worth|savings?|saved|invest\w*|stocks?|etfs?|portfolio|subscription\w*|groceries|grocery|transactions?|expenses?|money|afford|owe|debt|loan|mortgage|cash|credit|card|bank|payment|payments|paid|tax\w*|insurance|budgeted|left over|leftover|category|categories|netflix|spotify|restaurants?|eating out|takeaway|transport|fuel|gas|electricity|utilities|forecast|goal|goals|interest|dividends?|refund|charges?|fees?|overdraft|pension|wealth|cost|costs|price|worth|euros?|dollars?|pounds?|bucks|kroner?|import|sync|export|statement|app|moneo|widget|chart|report|breakdown|month|monthly|last week|this week|this year|last year|per month|average|total|how much|how many|biggest|largest|most)\b"
QUESTION = r"^(?:(?:ok(?:ay)?|so|hey|oh|uh|um|and|also|well|right|hi|hello|yeah|quick question|one more thing|by the way|btw)[, ]+)*(how|what|what's|whats|when|which|where|why|who|do|did|does|can|could|am|is|are|was|were|have|has|will|would|should|show|tell|give|list|compare|break|check|pull|find|look|remind|any|whose)\b"

TOPIC_KW = [
    ("net worth", r"net worth|networth|worth overall|total wealth|wealth"),
    ("groceries", r"grocer|supermarket|netto|lidl|aldi|rewe|tesco|walmart|kroger|fotex|føtex|coles|woolworths|costco|food shop|edeka|albert heijn|sainsbury|billa|spar\b|hofer|loblaws|no frills|heb\b|h-e-b|trader joe"),
    ("eating out", r"restaurant|eating out|takeaway|take ?out|lunch|dinner|brunch|breakfast|coffee|cafe|café|pizza|burger|sushi|kebab|deliveroo|uber eats|doordash|wolt|just eat|bar tab|drinks|pub\b|beers?\b|starbucks|mcdonald|tim hortons|ate out"),
    ("rent", r"\brent\b|landlord|mortgage"),
    ("subscriptions", r"subscription|netflix|spotify|disney|hbo|prime\b|icloud|youtube premium|membership|gym"),
    ("bills", r"\bbills?\b|electric|utilities|utility|internet|phone bill|water bill|heating|gas bill|energy|wifi|broadband"),
    ("transport", r"transport|fuel|diesel|petrol|\bgas\b|uber|taxi|cab\b|train|bus\b|metro|parking|toll|car payment|truck payment|bike|lyft|ticket"),
    ("travel", r"travel|flight|hotel|airbnb|holiday|vacation|trip\b"),
    ("health", r"pharmacy|doctor|dentist|health|medicine|prescription|therap"),
    ("insurance", r"insurance|premium"),
    ("taxes", r"\btax"),
    ("investments", r"invest|stocks?|shares?|etf|portfolio|scalable|broker|dividend|crypto|bitcoin|fund\b|msci"),
    ("savings", r"savings?|saved|saving|emergency fund|set aside|put away"),
    ("income", r"salary|paycheck|pay ?check|income|got paid|wage|payday|earn|freelance|invoice|tips\b|client paid|pension"),
    ("debts", r"\blent\b|borrow|owes?\b|owed|loan|debt|pay (me|him|her|them) back|paid (me|him|her) back|iou"),
    ("transfers", r"transfer|moved? .* to (my|the)|sent .* to|mobilepay|tikkie|venmo|e-?transfer|wire"),
    ("cash", r"\bcash\b|atm|withdrew|withdraw"),
    ("budget", r"budget"),
    ("goals", r"\bgoals?\b|saving for|save up|target|afford|plan\b|forecast|retire|on track"),
    ("account balance", r"balance|how much (do i have|is in|money do i have)|in my (account|checking|current)|checking|current account|account (has|holds)|have in"),
    ("shopping", r"amazon|shopping|clothes|shoes|ikea|zara|h&m|bought|electronics|order(ed)?\b"),
    ("entertainment", r"cinema|movie|concert|games?\b|steam|playstation|tickets|night out|party"),
    ("transactions", r"transaction|payment|charge|purchase|refund|categor"),
    ("app", r"\bapp\b|moneo|import|sync|connect|export|statement|widget|dark mode|settings|how do i|can you (do|even)|what can you"),
    ("spending", r"spend|spent|spending|expenses?"),
]
TOPIC_RX = [(t, re.compile(p)) for t, p in TOPIC_KW]


def rule_topic(text):
    t = norm(text)
    for topic, rx in TOPIC_RX:
        if rx.search(t):
            return topic
    return "other"


class RuleRouter:
    name, size_bytes = "rules (regex/keywords)", None

    def predict_one(self, r):
        t = norm(r["text"])
        words = t.split()
        has_num = bool(re.search(r"\b" + NUMWORD + r"\b", t)) and bool(re.search(r"\d|" + MONEY_UNIT + r"|hundred|thousand|grand|fifty|twenty|forty|thirty|sixty|seventy|eighty|ninety|fifteen|twelve|ten\b", t))
        if r["assistant_speaking"] and len(words) <= 4 and all(w.strip(".,") in BACKCH for w in words):
            return "BACKCHANNEL", 1.0, ""
        if asked(r) and len(words) <= 5 and re.search(AFFIRM, t):
            lab = "DELEGATE_WRITE" if re.search(WRITE_Q, r["last_assistant"].lower()) else "DELEGATE_READ"
            return lab, 1.0, rule_topic(r["last_assistant"])
        if re.search(META, t) and not has_num and len(words) <= 12:
            return "META", 1.0, ""
        last = words[-1].strip(".,?") if words else ""
        if last in DANGLE or (len(words) >= 2 and " ".join(words[-2:]) in DANGLE) or t.endswith(("...", "-")):
            return "INCOMPLETE", 1.0, ""
        if re.search(CORR, t) and len(words) <= 14:
            return "CORRECTION", 1.0, ""
        if re.search(WRITE_CMD, t) or (has_num and re.search(SPENT, t) and not re.search(QUESTION, t)):
            return "DELEGATE_WRITE", 1.0, rule_topic(t)
        if re.search(QUESTION, t) and re.search(FIN, t):
            return "DELEGATE_READ", 1.0, rule_topic(t)
        if has_num and re.search(FIN, t):
            return "DELEGATE_WRITE", 1.0, rule_topic(t)
        if re.search(FIN, t) and re.search(r"\b(my|i|me|we|our)\b", t) and re.search(r"\b(show|tell|give|check|want|need|wonder|curious|see|know|look)\b", t):
            return "DELEGATE_READ", 1.0, rule_topic(t)
        return "SMALLTALK", 1.0, ""


# ---------------------------------------------------------------- TF-IDF + linear
class TfidfRouter:
    def __init__(self, kind):
        self.kind = kind
        self.name = "TF-IDF word+char + " + ("logistic regression" if kind == "lr" else "linear SVM")

    def fit(self, rows):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_pipeline, make_union
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import LinearSVC

        def pipe(clf):
            return make_pipeline(make_union(
                TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"[^ ]+"),
                TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)), clf)
        mk = (lambda: LogisticRegression(C=20, max_iter=2000, class_weight="balanced")) if self.kind == "lr" else (lambda: LinearSVC(C=1.0, class_weight="balanced"))
        self.p = pipe(mk()).fit([feat_text(r) for r in rows], [r["label"] for r in rows])
        d = [r for r in rows if r["label"] in DELEG]
        self.t = pipe(mk()).fit([feat_text(r) for r in d], [r["topic"] for r in d])
        self.size_bytes = len(pickle.dumps((self.p, self.t)))
        return self

    def _conf(self, model, x):
        if self.kind == "lr":
            p = model.predict_proba([x])[0]
        else:
            z = model.decision_function([x])[0] * 3.0
            p = np.exp(z - z.max()); p /= p.sum()
        i = int(p.argmax())
        return model.classes_[i], float(p[i])

    def predict_one(self, r):
        x = feat_text(r)
        lab, conf = self._conf(self.p, x)
        topic = self._conf(self.t, x)[0] if lab in DELEG else ""
        return lab, conf, topic


# ---------------------------------------------------------------- tiny fastText-style net (numpy)
class TinyNet:
    """Hashed word uni/bigrams and char trigrams -> mean of 32-d embeddings -> 48 ReLU -> softmax. Stored as float16."""
    name = "tiny fastText-style net (hashed n-grams, numpy)"

    def __init__(self, buckets=8192, dim=32, hidden=48, seed=0, elr=0.1):
        self.B, self.D, self.H, self.seed, self.elr = buckets, dim, hidden, seed, elr

    def ids(self, r):
        t = feat_text(r)
        w = t.split()
        grams = w + [a + "_" + b for a, b in zip(w, w[1:])] + ["^" + w[0], w[-1] + "$"] if w else ["<empty>"]
        s = " " + t + " "
        grams += [s[i:i + 3] for i in range(len(s) - 2)]
        return np.fromiter((zlib.crc32(g.encode("utf-8")) % self.B for g in grams), dtype=np.int64)

    def _fit_head(self, X, y, n_cls, epochs, rng):
        E = rng.normal(0, 0.1, (self.B, self.D)).astype(np.float32)
        W1 = rng.normal(0, 0.2, (self.D, self.H)).astype(np.float32); b1 = np.zeros(self.H, np.float32)
        W2 = rng.normal(0, 0.2, (self.H, n_cls)).astype(np.float32); b2 = np.zeros(n_cls, np.float32)
        P = [E, W1, b1, W2, b2]
        M = [np.zeros_like(p) for p in P]; V = [np.zeros_like(p) for p in P]
        lr, step = 0.01, 0
        G = np.zeros_like(E)
        for ep in range(epochs):
            for i in rng.permutation(len(X)):
                ids = X[i]
                if rng.random() < 0.5 and len(ids) > 4:  # n-gram dropout
                    ids = ids[rng.random(len(ids)) > 0.15]
                h0 = E[ids].mean(0); a = h0 @ W1 + b1; h1 = np.maximum(a, 0)
                z = h1 @ W2 + b2; p = np.exp(z - z.max()); p /= p.sum()
                dz = p; dz[y[i]] -= 1
                gW2 = np.outer(h1, dz); gb2 = dz
                da = (W2 @ dz) * (a > 0); gW1 = np.outer(h0, da); gb1 = da
                dh0 = (W1 @ da) / len(ids)
                step += 1
                for k, g in ((1, gW1), (2, gb1), (3, gW2), (4, gb2)):
                    M[k] = 0.9 * M[k] + 0.1 * g; V[k] = 0.999 * V[k] + 0.001 * g * g
                    P[k] -= lr * M[k] / (np.sqrt(V[k]) + 1e-8)
                u, c = np.unique(ids, return_counts=True)  # sparse Adagrad on the embedding rows that were used
                g = c[:, None] * dh0[None, :]
                G[u] += g * g
                E[u] -= self.elr * g / (np.sqrt(G[u]) + 1e-6)
        return [p.astype(np.float16) for p in P]

    def fit(self, rows):
        rng = np.random.default_rng(self.seed)
        X = [self.ids(r) for r in rows]
        self.cls = LABELS
        self.P = self._fit_head(X, [LABELS.index(r["label"]) for r in rows], len(LABELS), 40, rng)
        d = [i for i, r in enumerate(rows) if r["label"] in DELEG]
        self.tcls = sorted({rows[i]["topic"] for i in d})
        self.T = self._fit_head([X[i] for i in d], [self.tcls.index(rows[i]["topic"]) for i in d], len(self.tcls), 60, rng)
        self.size_bytes = sum(p.nbytes for p in self.P)  # route head only; the topic head is the same size again
        self.P = [p.astype(np.float32) for p in self.P]; self.T = [p.astype(np.float32) for p in self.T]
        return self

    @staticmethod
    def _fwd(P, ids):
        E, W1, b1, W2, b2 = P
        z = np.maximum(E[ids].mean(0) @ W1 + b1, 0) @ W2 + b2
        p = np.exp(z - z.max()); p /= p.sum()
        return int(p.argmax()), float(p.max())

    def predict_one(self, r):
        ids = self.ids(r)
        i, conf = self._fwd(self.P, ids)
        lab = self.cls[i]
        return lab, conf, (self.tcls[self._fwd(self.T, ids)[0]] if lab in DELEG else "")


# ---------------------------------------------------------------- Needle
_needle_cache_path = os.path.join(B, "needle_emb_cache.pkl")
_needle = {}


def needle_agent(key, **kw):
    import cactus
    cactus.set_threads(1)  # the shared CPU is saturated; one thread was the fastest and steadiest setting measured
    if key not in _needle:
        _needle[key] = cactus.Needle(stateless=True, auto_date=False, **kw)
    return _needle[key]


class NeedleEmb:
    """Logistic regression on Needle 3 text embeddings (3072-d) plus the two context flags."""
    name = "Needle 3 embeddings + logistic regression"

    def __init__(self):
        self.cache = pickle.load(open(_needle_cache_path, "rb")) if os.path.exists(_needle_cache_path) else {}
        self.agent = needle_agent("emb")
        self.last_embed_ms = 0.0

    def emb(self, text):
        t = norm(text)
        if t not in self.cache:
            self.cache[t] = np.asarray(self.agent.embed(t), dtype=np.float32)
        return self.cache[t]

    def save(self):
        pickle.dump(self.cache, open(_needle_cache_path, "wb"))

    def x(self, r, fresh=False):
        e = np.asarray(self.agent.embed(norm(r["text"])), dtype=np.float32) if fresh else self.emb(r["text"])
        e = e / (np.linalg.norm(e) + 1e-9)
        return np.concatenate([e, [float(r["assistant_speaking"]), float(asked(r))]])

    def fit(self, rows):
        from sklearn.linear_model import LogisticRegression
        X = np.stack([self.x(r) for r in rows])
        self.clf = LogisticRegression(C=50, max_iter=3000, class_weight="balanced").fit(X, [r["label"] for r in rows])
        d = [i for i, r in enumerate(rows) if r["label"] in DELEG]
        self.tclf = LogisticRegression(C=50, max_iter=3000, class_weight="balanced").fit(X[d], [rows[i]["topic"] for i in d])
        self.head_bytes = len(pickle.dumps((self.clf, self.tclf)))
        self.save()
        return self

    def predict_one(self, r, fresh=False):
        x = self.x(r, fresh)[None]
        p = self.clf.predict_proba(x)[0]
        lab = self.clf.classes_[int(p.argmax())]
        return lab, float(p.max()), (self.tclf.predict(x)[0] if lab in DELEG else "")


def needle_tools():
    topic = {"type": "string", "enum": TOPICS, "description": "what it is about"}
    def t(name, desc, with_topic=False):
        return {"name": name, "description": desc,
                "parameters": {"type": "object", "properties": {"topic": topic} if with_topic else {}, "required": ["topic"] if with_topic else []}}
    return [
        t("delegate_read", "The user asks a question about their own money, spending, balances, bills, budgets, investments, plans or the app.", True),
        t("delegate_write", "The user states money they spent, received, lent or borrowed, or what an account holds, or asks to add, change or remove something.", True),
        t("correction", "The user corrects or replaces their previous request, e.g. 'no, I meant last month', 'actually make it 15'."),
        t("meta", "The user asks about progress, or to repeat or explain the last answer, e.g. 'is it done?', 'say that again'."),
        t("smalltalk", "Greeting, chit-chat, thanks or goodbye with no request and no fact about their own money."),
        t("backchannel", "A lone 'mm-hm', 'yeah', 'okay' or 'right' while the assistant is speaking."),
        t("incomplete", "The utterance is cut off mid-thought and is not a finished sentence, e.g. 'how much did I spend on'."),
    ]


class NeedleTools:
    """Zero-shot: the 7 routes as Needle tools, the picked tool is the route."""
    name = "Needle 3 zero-shot (routes as tools)"

    def __init__(self):
        self.agent = needle_agent("tools", tools=needle_tools(),
                                  system="Route what the user said on a voice call with a personal-finance assistant. Call exactly one tool.")
        self.nocall = 0

    def predict_one(self, r):
        q = r["text"]
        if r["assistant_speaking"]:
            q = "(said while the assistant is speaking) " + q
        elif asked(r):
            q = f"(answering the assistant's question: {r['last_assistant']}) " + q
        res = self.agent.complete(q, max_new_tokens=40)
        calls = res.get("function_calls") or res.get("suppressed_calls") or []
        if not calls:
            self.nocall += 1
            return "SMALLTALK", 0.0, ""
        lab = str(calls[0].get("name", "")).upper()
        if lab not in LABELS:
            self.nocall += 1
            return "SMALLTALK", 0.0, ""
        topic = str((calls[0].get("arguments") or {}).get("topic", "")).lower() if lab in DELEG else ""
        return lab, float(res.get("confidence") or 0.0), topic if topic in TOPICS else ("other" if lab in DELEG else "")


class NeedleExtractTopic:
    """Zero-shot topic slot with Needle extraction over a one-field schema constrained to the topic list."""

    def __init__(self):
        schema = {"name": "money_topic", "description": "The money topic the user is talking about.",
                  "parameters": {"type": "object", "properties": {"topic": {"type": "string", "enum": TOPICS}}, "required": ["topic"]}}
        self.agent = needle_agent("extract", tools=[schema], system="Extract the money topic of what the user said.")

    def topic(self, text):
        res = self.agent.complete(text, max_new_tokens=24)
        calls = res.get("function_calls") or res.get("suppressed_calls") or []
        t = str((calls[0].get("arguments") or {}).get("topic", "")).lower() if calls else ""
        return t if t in TOPICS else "other"

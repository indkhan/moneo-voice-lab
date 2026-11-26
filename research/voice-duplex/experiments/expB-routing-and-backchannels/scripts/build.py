"""Clean, dedupe and split the raw generations into train.jsonl (500) and test.jsonl (200)."""
import glob, os, random, re
from collections import Counter, defaultdict
from common import *

TRAIN_TARGET = {"DELEGATE_READ": 110, "DELEGATE_WRITE": 110, "CORRECTION": 62, "META": 52, "SMALLTALK": 70, "BACKCHANNEL": 36, "INCOMPLETE": 60}
TEST_TARGET = {"DELEGATE_READ": 44, "DELEGATE_WRITE": 44, "CORRECTION": 24, "META": 20, "SMALLTALK": 30, "BACKCHANNEL": 14, "INCOMPLETE": 24}


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9. ]", "", t.lower())).strip()


def clean(x):
    if not isinstance(x.get("text"), str) or x.get("label") not in LABELS:
        return None
    text = x["text"].strip()
    if not text or len(text.split()) > 45:
        return None
    topic = (x.get("topic") or "").strip().lower()
    if x["label"] not in ("DELEGATE_READ", "DELEGATE_WRITE"):
        topic = ""
    elif topic not in TOPICS:
        topic = "other"
    return {"text": text, "label": x["label"], "assistant_speaking": bool(x.get("assistant_speaking")),
            "last_assistant": str(x.get("last_assistant") or "").strip(), "prev_user": str(x.get("prev_user") or "").strip(),
            "topic": topic, "gen": x["_gen"], "focus": x["_focus"], "persona": x["_persona"]}


def pick(rows, target, rng, seen):
    """Dedupe on normalised text (+context for very short ones), then sample to the per-label target, hard cases first."""
    by = defaultdict(list)
    for r in rows:
        short = len(r["text"].split()) <= 3
        key = (norm(r["text"]), r["label"], r["assistant_speaking"], norm(r["last_assistant"])[:40] if short else "")
        if key in seen:
            continue
        seen.add(key)
        by[r["label"]].append(r)
    out = []
    for lab, n in target.items():
        pool = by[lab]
        rng.shuffle(pool)
        pool.sort(key=lambda r: r["focus"] != "MIXED")  # keep the boundary cases
        out += pool[:n]
        if len(pool) < n:
            print("short of", lab, len(pool), "/", n)
    rng.shuffle(out)
    return out


if __name__ == "__main__":
    raw = []
    for p in sorted(glob.glob(os.path.join(DATA, "raw", "*.jsonl"))):
        split = os.path.basename(p).split("-")[0]
        for x in read_jsonl(p):
            c = clean(x)
            if c:
                c["split"] = split
                raw.append(c)
    print("raw clean", len(raw), Counter((r["split"], r["label"]) for r in raw))
    rng = random.Random(7)
    seen = set()
    train = pick([r for r in raw if r["split"] == "train"], TRAIN_TARGET, rng, seen)
    # test items whose text already appears in train are dropped unless they are short (backchannels and bare answers repeat by nature)
    train_texts = {norm(r["text"]) for r in train}
    test_pool = [r for r in raw if r["split"] == "test" and (norm(r["text"]) not in train_texts or len(r["text"].split()) <= 3)]
    test = pick(test_pool, TEST_TARGET, rng, set())
    # Barge-in: the generators only ever set assistant_speaking for backchannels, corrections and meta, which would teach
    # "assistant speaking => not a request". Flip the flag on ~15% of longer requests and chit-chat (the label does not change).
    for rows in (train, test):
        for r in rows:
            r["barge_in"] = False
            if (r["label"] in ("DELEGATE_READ", "DELEGATE_WRITE", "SMALLTALK") and len(r["text"].split()) >= 5
                    and not r["assistant_speaking"] and not r["last_assistant"].rstrip().endswith("?") and rng.random() < 0.2):
                r["assistant_speaking"], r["barge_in"] = True, True
    for i, r in enumerate(train):
        r["id"] = f"tr{i:03d}"
    for i, r in enumerate(test):
        r["id"] = f"te{i:03d}"
    write_jsonl(os.path.join(DATA, "train.jsonl"), train)
    write_jsonl(os.path.join(DATA, "test_unreviewed.jsonl"), test)
    print("train", len(train), Counter(r["label"] for r in train))
    print("test", len(test), Counter(r["label"] for r in test))
    print("test short overlap with train:", sum(norm(r["text"]) in train_texts for r in test))
    for name, rows in (("train", train), ("test", test)):
        print(name, "assistant_speaking by label", Counter((r["label"], r["assistant_speaking"]) for r in rows))

"""Prompted generative routers: small local GGUF models through llama-cpp-python, and gpt-6-luna as the ceiling.
Usage: eval_llm.py local <key> <gguf path> [prefix]   |   eval_llm.py luna"""
import random, sys, time
import numpy as np
from common import *
from evallib import *

train = read_jsonl(os.path.join(DATA, "train.jsonl"))
test = read_jsonl(os.path.join(DATA, "test.jsonl"))
gold = [r["label"] for r in test]


def item(r):
    return (f'assistant_speaking: {"true" if r["assistant_speaking"] else "false"}\nlast_assistant: "{r["last_assistant"]}"\n'
            f'prev_user: "{r["prev_user"]}"\nuser: "{r["text"]}"')


def fewshot(n_per=2, seed=3):
    rng = random.Random(seed)
    out = []
    for lab in LABELS:
        pool = [r for r in train if r["label"] == lab]
        for r in rng.sample(pool, n_per):
            out.append(item(r) + f"\n=> {lab}|{r['topic'] or 'none'}")
    rng.shuffle(out)
    return "\n\n".join(out)


SYS = (GUIDE + "\n\nYou are the router. For the item given, answer with exactly one line: LABEL|topic  (topic is 'none' unless the label is DELEGATE_READ or DELEGATE_WRITE). No other text.\n\nExamples:\n\n" + fewshot())


def parse(text):
    line = text.strip().splitlines()[0] if text.strip() else ""
    lab, _, topic = line.partition("|")
    lab = lab.strip().upper().replace(" ", "_")
    topic = topic.strip().lower()
    return (lab if lab in LABELS else None), (topic if topic in TOPICS else ("other" if lab in ("DELEGATE_READ", "DELEGATE_WRITE") else ""))


mode = sys.argv[1]
if mode == "local":
    key, path = sys.argv[2], sys.argv[3]
    from llama_cpp import Llama, LlamaGrammar
    gbnf = ('root ::= label "|" topic\nlabel ::= ' + " | ".join(f'"{l}"' for l in LABELS) +
            "\ntopic ::= " + " | ".join(f'"{t}"' for t in TOPICS + ["none"]))
    grammar = LlamaGrammar.from_string(gbnf, verbose=False)
    llm = Llama(model_path=path, n_ctx=4096, n_threads=4, n_batch=512, verbose=False)
    bad = [0]

    def predict_one(r):
        out = llm.create_chat_completion(messages=[{"role": "system", "content": SYS}, {"role": "user", "content": item(r)}],
                                         max_tokens=16, temperature=0.0, grammar=grammar)
        lab, topic = parse(out["choices"][0]["message"]["content"] or "")
        if lab is None:
            bad[0] += 1
            lab = "SMALLTALK"
        return lab, 1.0, topic if lab in DELEG else ""

    predict_one(test[0])  # first call pays for the system prompt once; later calls reuse it from the KV cache
    ts, preds = [], []
    for r in test:
        t = time.perf_counter(); preds.append(predict_one(r)); ts.append((time.perf_counter() - t) * 1000)
    m = metrics(gold, [p[0] for p in preds])
    dtest = [(r, p) for r, p in zip(test, preds) if r["label"] in DELEG and p[0] in DELEG]
    m.update({"name": f"{key} prompted (guide + 14 examples, grammar-constrained)", "size_bytes": os.path.getsize(path),
              "latency": {"median_ms": round(float(np.median(ts)), 1), "p95_ms": round(float(np.percentile(ts, 95)), 1), "n": len(ts)},
              "threads": 4, "unparseable": bad[0], "prompt_tokens_system": len(llm.tokenize(SYS.encode("utf-8"))),
              "topic_accuracy_when_routed_delegate": round(sum(p[2] == r["topic"] for r, p in dtest) / max(1, len(dtest)), 4),
              "errors": [{"id": r["id"], "text": r["text"], "gold": r["label"], "pred": p[0]} for r, p in zip(test, preds) if p[0] != r["label"]]})
    save_result("routers", key, m)
    print(key, m["accuracy"], m["macro_f1"], m["dropped_money_question"], m["wasted_call"], m["latency"], flush=True)
    if len(sys.argv) > 4:
        sub = [r for i, r in enumerate(test) if i % 3 == 0]
        save_result("prefix", key, dict(prefix_analysis(predict_one, sub), note="every 3rd test utterance; no confidence from this router, every answer counts as confident"))

elif mode == "luna":
    MODEL = "gpt-6-luna"
    preds = {}
    for b in range(0, len(test), 25):
        chunk = test[b:b + 25]
        user = "Route each item. Answer with one line per item, in order: id|LABEL|topic\n\n" + "\n\n".join(f"id: {r['id']}\n" + item(r) for r in chunk)
        res = call(MODEL, SYS.replace("For the item given, answer with exactly one line: LABEL|topic", "For each item given, answer with one line: id|LABEL|topic"),
                   user, effort="low", tag="luna-router-batch")
        for line in res["text"].splitlines():
            parts = [x.strip() for x in line.split("|")]
            if len(parts) >= 2 and parts[0] in {r["id"] for r in chunk}:
                lab = parts[1].upper()
                topic = parts[2].lower() if len(parts) > 2 else ""
                preds[parts[0]] = (lab if lab in LABELS else "SMALLTALK", 1.0, topic if topic in TOPICS else "")
        print("batch", b, len(preds), flush=True)
    missing = [r["id"] for r in test if r["id"] not in preds]
    plist = [preds.get(r["id"], ("SMALLTALK", 0.0, "")) for r in test]
    m = metrics(gold, [p[0] for p in plist])
    dt = [(r, p) for r, p in zip(test, plist) if r["label"] in DELEG and p[0] in DELEG]
    # single-item calls: what one live routing decision would cost
    rng = random.Random(5)
    firsts, totals, agree = [], [], 0
    sample = rng.sample(test, 15)
    for r in sample:
        res = call(MODEL, SYS, item(r), effort="low", tag="luna-router-single")
        firsts.append(res["first_ms"]); totals.append(res["total_ms"])
        agree += parse(res["text"])[0] == r["label"]
    m.update({"name": "gpt-6-luna (ChatGPT plan, effort low) prompted: ceiling", "size_bytes": None, "missing_in_batch_output": missing,
              "latency": {"median_ms": float(np.median(totals)), "p95_ms": float(np.percentile(totals, 95)), "n": len(totals),
                          "first_token_median_ms": float(np.median(firsts)), "first_token_p95_ms": float(np.percentile(firsts, 95)),
                          "note": "15 single-item calls over the network; accuracy above is from 8 batched calls of 25 items"},
              "single_item_accuracy_on_15": round(agree / len(sample), 3),
              "topic_accuracy_when_routed_delegate": round(sum(p[2] == r["topic"] for r, p in dt) / max(1, len(dt)), 4),
              "errors": [{"id": r["id"], "text": r["text"], "gold": r["label"], "pred": p[0]} for r, p in zip(test, plist) if p[0] != r["label"]]})
    save_result("routers", "gpt6_luna", m)
    print("luna", m["accuracy"], m["macro_f1"], m["dropped_money_question"], m["wasted_call"], m["latency"])

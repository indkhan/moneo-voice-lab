"""Talker test: tiny local chat models answer 30 small-talk utterances; gpt-6-luna writes the reference and judges blind.
Usage: talker.py local <key> <gguf> [threads]   |   talker.py luna   |   talker.py judge"""
import random, re, sys, time
import numpy as np
from common import *
from evallib import save_result

test = read_jsonl(os.path.join(DATA, "test.jsonl"))
small = [r for r in test if r["label"] == "SMALLTALK"][:30]
OUT = os.path.join(B, "talker"); os.makedirs(OUT, exist_ok=True)
SYS = ("You are Moneo, a relaxed friend on a voice call. Reply to what the user just said in ONE short, warm, casual spoken sentence "
       "(under 15 words). Never state any money fact, amount, price or balance. Never say you did, will do or are doing anything. "
       "No emojis, no lists, no questions about money.")


def user_msg(r):
    return r["text"]


mode = sys.argv[1]
if mode == "local":
    key, path = sys.argv[2], sys.argv[3]
    threads = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    from llama_cpp import Llama
    llm = Llama(model_path=path, n_ctx=1024, n_threads=threads, verbose=False)
    llm.create_chat_completion(messages=[{"role": "system", "content": SYS}, {"role": "user", "content": "hi"}], max_tokens=4)  # warm up + cache the system prompt
    rows = []
    for r in small:
        t0 = time.perf_counter(); first = None; n = 0; text = ""
        for ch in llm.create_chat_completion(messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user_msg(r)}],
                                             max_tokens=40, temperature=0.3, top_p=0.9, seed=1, stream=True):
            d = ch["choices"][0]["delta"].get("content")
            if d:
                if first is None: first = time.perf_counter()
                n += 1; text += d
        end = time.perf_counter()
        rows.append({"id": r["id"], "user": r["text"], "reply": text.strip(), "first_ms": round(((first or end) - t0) * 1000, 1),
                     "total_ms": round((end - t0) * 1000, 1), "tokens": n, "tok_s": round((n - 1) / (end - first), 1) if first and n > 1 and end > first else None})
        print(rows[-1], flush=True)
    write_jsonl(os.path.join(OUT, f"{key}_t{threads}.jsonl"), rows)
    tps = [x["tok_s"] for x in rows if x["tok_s"]]
    save_result("talker", f"{key}_t{threads}", {"model": key, "threads": threads, "size_bytes": os.path.getsize(path),
                "first_token_median_ms": float(np.median([x["first_ms"] for x in rows])), "first_token_p95_ms": float(np.percentile([x["first_ms"] for x in rows], 95)),
                "full_reply_median_ms": float(np.median([x["total_ms"] for x in rows])),
                "tok_s_median": float(np.median(tps)) if tps else None, "median_tokens": float(np.median([x["tokens"] for x in rows]))})

elif mode == "luna":
    rows = []
    for r in small:
        res = call("gpt-6-luna", SYS, user_msg(r), effort="low", tag="talker-luna")
        rows.append({"id": r["id"], "user": r["text"], "reply": res["text"].strip(), "first_ms": res["first_ms"], "total_ms": res["total_ms"]})
        print(rows[-1], flush=True)
    write_jsonl(os.path.join(OUT, "gpt6_luna.jsonl"), rows)
    save_result("talker", "gpt6_luna", {"model": "gpt-6-luna (effort low)", "size_bytes": None,
                "first_token_median_ms": float(np.median([x["first_ms"] for x in rows])), "first_token_p95_ms": float(np.percentile([x["first_ms"] for x in rows], 95)),
                "full_reply_median_ms": float(np.median([x["total_ms"] for x in rows]))})

elif mode == "judge":
    files = sys.argv[2:]  # reply files to compare, e.g. lfm25_350m_q8_t4 gemma3_270m_q8_t4 gpt6_luna
    replies = {f: {x["id"]: x["reply"] for x in read_jsonl(os.path.join(OUT, f + ".jsonl"))} for f in files}
    rng = random.Random(11)
    letters = "ABCDEFGH"[:len(files)]
    keymap, scores = {}, {f: [] for f in files}
    JSYS = ("You judge replies from a voice assistant called Moneo (a personal-finance app) to small talk on a phone call. The assistant should sound like a relaxed friend: "
            "one short, warm, casual spoken sentence. It must never state a money fact (amount, price, balance, spending) and must never claim it did, is doing or will do an action.\n"
            "For every reply give: nat = naturalness as spoken conversation 1-5, app = appropriateness to what the user said 1-5, "
            "unsafe = true if the reply states any money fact or claims an action was/will be done, else false. Judge each reply on its own; you do not know which system wrote which.\n"
            "Output JSON Lines only, one per reply: {\"u\": <utterance number>, \"r\": \"<letter>\", \"nat\": n, \"app\": n, \"unsafe\": true|false}")
    for b in range(0, len(small), 10):
        chunk = small[b:b + 10]
        parts = []
        for i, r in enumerate(chunk):
            order = files[:]; rng.shuffle(order)
            keymap.update({(b + i, letters[j]): f for j, f in enumerate(order)})
            parts.append(f"Utterance {b + i}: \"{r['text']}\"\n" + "\n".join(f"  {letters[j]}: \"{replies[f].get(r['id'], '')}\"" for j, f in enumerate(order)))
        res = call("gpt-6-luna", JSYS, "\n\n".join(parts), effort="low", tag="talker-judge")
        for x in parse_jsonl_text(res["text"]):
            f = keymap.get((x.get("u"), x.get("r")))
            if f:
                scores[f].append({"u": x["u"], "nat": x.get("nat"), "app": x.get("app"), "unsafe": bool(x.get("unsafe"))})
        print("judged", b, {f: len(v) for f, v in scores.items()}, flush=True)
    summary = {}
    for f, v in scores.items():
        summary[f] = {"n_judged": len(v), "naturalness_mean": round(float(np.mean([x["nat"] for x in v])), 2), "appropriateness_mean": round(float(np.mean([x["app"] for x in v])), 2),
                      "unsafe_flags": sum(x["unsafe"] for x in v), "unsafe_utterances": [x["u"] for x in v if x["unsafe"]],
                      "over_one_sentence": sum(len(re.findall(r"[.!?]+(\s|$)", t)) > 1 for t in replies[f].values()),
                      "empty_replies": sum(not t for t in replies[f].values())}
    pairs = [{"user": r["text"], **{f: replies[f].get(r["id"], "") for f in files}} for r in small]
    save_result("talker", "judge", {"judge": "gpt-6-luna, blind, replies shuffled per utterance", "summary": summary, "pairs": pairs})
    print(json.dumps(summary, indent=1))

"""End-to-end reflex loop on recorded speech, paced in real time.

Speculative reflex cascade (ours) vs a classic endpoint-then-go cascade built from the same parts:
  Whistle STT (re-run on the growing buffer), TF-IDF router, Smart Turn v3.2 end-of-turn check, streaming Woof TTS,
  gpt-6-luna on the ChatGPT plan as the thinking model (real network calls).
Ours: prepares the acknowledgement audio while the user is still talking, starts the thinking model at the first
160 ms of silence, and commits as soon as the turn detector says "done". Classic: waits 600 ms of silence first.
"""
import json, os, re, sys, threading, time
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("CACTUS_TELEMETRY", "0")
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("spike", "expA", os.path.join("expB", "scripts"), os.path.join("expC", "woof_stream")): sys.path.insert(0, os.path.join(ROOT, p))
import cactus
from llm import stream as llm_stream
from eval_smartturn import build_session, features
from routers import TfidfRouter, rule_topic, DELEG
from stream_runtime import WoofStream

SR, FRAME = 16000, 1280          # 80 ms frames
SPEC_SIL, CLASSIC_SIL, FALLBACK_SIL = 0.16, 0.60, 1.20
REASONER = "You are Moneo's finance assistant on a voice call. This is a latency test without tools: invent plausible figures. Answer in one or two short spoken sentences."

def rows(path): return [json.loads(l) for l in open(path, encoding="utf-8")]
def nw(t): return re.sub(r"[^a-z0-9 ]", "", t.lower()).split()
def resample(x, a, b):
    n = int(len(x) * b / a); return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)

# ---- pick utterances and synthesize the "user" with a different TTS and voice
test = rows(os.path.join(ROOT, "expB", "data", "test.jsonl"))
def pick(label, n): return [r for r in test if r["label"] == label and 5 <= len(r["text"].split()) <= 16 and not r["assistant_speaking"]][:n]
utts = [dict(text=r["text"], label=r["label"], pause=None) for r in pick("DELEGATE_READ", 10) + pick("DELEGATE_WRITE", 6) + pick("SMALLTALK", 6)]
for r in pick("DELEGATE_READ", 14)[10:14]:
    w = r["text"].split(); k = len(w) // 2
    utts.append(dict(text=r["text"], label=r["label"], pause=(" ".join(w[:k]), " ".join(w[k:]))))
if os.environ.get("E2E_LIMIT"): utts = [utts[0], utts[10], utts[16], utts[22]][:int(os.environ["E2E_LIMIT"])]
import torch
from pocket_tts import TTSModel
ptts = TTSModel.load_model(language="english", quantize=True); pst = ptts.get_state_for_audio_prompt("marius")
def say(t): return resample(torch.cat(list(ptts.generate_audio_stream(pst, t))).numpy(), ptts.sample_rate, SR)
def trim(x):
    nz = np.nonzero(np.abs(x) > 0.03 * np.abs(x).max())[0]; return x[max(0, nz[0] - 800):nz[-1] + 800]
for u in utts:
    if u["pause"]:
        a, b = trim(say(u["pause"][0] + ",")), trim(say(u["pause"][1] + "."))
        u["audio"] = np.concatenate([np.zeros(4000, np.float32), a, np.zeros(int(0.7 * SR), np.float32), b]); u["pause_at"] = (4000 + len(a)) / SR
    else:
        u["audio"] = np.concatenate([np.zeros(4000, np.float32), trim(say(u["text"] + ("?" if u["label"] == "DELEGATE_READ" else ".")))])
    u["speech_end"] = len(u["audio"]) / SR - 0.05
    u["audio"] = np.concatenate([u["audio"], np.zeros(int(12 * SR), np.float32)])
del ptts, pst
torch.set_num_threads(1)

router = TfidfRouter("lr").fit(rows(os.path.join(ROOT, "expB", "data", "train.jsonl")))
st = build_session(os.path.join(ROOT, "models", "smartturn", "smart-turn-v3.2-cpu.onnx")); st_in = st.get_inputs()[0].name
def turn_done(buf):
    t = time.perf_counter(); p = float(np.ravel(st.run(None, {st_in: features(buf)})[0])[0]); return p, (time.perf_counter() - t) * 1000
tts = WoofStream(voice="juno", schedule=(1, 2, 4, 8)); tts.warmup()
tts_lock = threading.Lock()
cactus.transcribe(utts[0]["audio"][:SR * 3], language="en"); turn_done(utts[0]["audio"][:SR * 3])
KEYWORDS = ["net worth", "Netto", "Rewe", "groceries", "savings"]

def first_chunk_ms(text):
    with tts_lock:
        t = time.perf_counter()
        for _ in tts.stream(text, seed=1): return (time.perf_counter() - t) * 1000
    return None

class Reasoner(threading.Thread):
    """One speculative call to the thinking model; records when the first sentence is complete."""
    def __init__(self, text, t0):
        super().__init__(daemon=True); self.text, self.t0 = text, t0; self.started = time.perf_counter() - t0
        self.first_token = self.first_sentence = None; self.sentence = ""; self.cancelled = False; self.error = None
    def run(self):
        try:
            acc = ""
            for _, d in llm_stream("gpt-6-luna", REASONER, self.text, "low"):
                if self.cancelled: return
                if self.first_token is None: self.first_token = time.perf_counter() - self.t0
                acc += d
                m = re.search(r"[.!?](\s|$)", acc)
                if m and len(acc.split()) >= 4:
                    self.sentence = acc[:m.end()].strip(); self.first_sentence = time.perf_counter() - self.t0; return
            self.sentence = acc.strip(); self.first_sentence = time.perf_counter() - self.t0
        except Exception as e: self.error = repr(e)[:200]

def run(u):
    a = u["audio"]; t0 = time.perf_counter(); now = lambda: time.perf_counter() - t0
    log = dict(text=u["text"], label=u["label"], paused=bool(u["pause"]), speech_end=round(u["speech_end"], 2), wasted_reasoner=0)
    last_voice, in_speech, partial, last_stt_frame = 0.0, False, "", -99
    streak, ack, ack_text, reasoners, spec, commit, classic_commit = 0, {}, None, [], None, None, None
    stt_ms, thr = [], 0.02 * float(np.abs(a).max())
    i = 0
    while True:
        i += 1; arrive = i * FRAME / SR
        if i * FRAME > len(a): break
        d = arrive - now()
        if d > 0: time.sleep(d)
        buf = a[:i * FRAME]; fr = a[(i - 1) * FRAME:i * FRAME]
        voiced = float(np.sqrt(np.mean(fr ** 2))) > thr
        if voiced:
            last_voice = arrive
            if commit is not None and arrive < u["speech_end"]:                      # we answered into a mid-sentence pause
                log["false_commit"] = log.get("false_commit", 0) + 1; commit = None
            if classic_commit is not None and arrive < u["speech_end"]:
                log["classic_false_commit"] = log.get("classic_false_commit", 0) + 1; classic_commit = None
            if spec and not spec.cancelled and commit is None and arrive - spec.started > 0.05:
                spec.cancelled = True; log["wasted_reasoner"] += 1; spec = None       # user kept talking: cancel
            in_speech = True
        sil = arrive - last_voice
        behind = now() - arrive > 0.12
        if in_speech and commit is None and ((i - last_stt_frame >= 3 and not behind) or (sil >= SPEC_SIL and spec is None)):
            t = time.perf_counter(); partial = cactus.transcribe(buf, language="en", keywords=KEYWORDS)["text"].strip(); stt_ms.append((time.perf_counter() - t) * 1000); last_stt_frame = i
            lab, conf, _ = router.predict_one(dict(text=partial, assistant_speaking=False, last_assistant="", prev_user=""))
            streak = streak + 1 if (lab in DELEG and conf >= 0.6 and len(partial.split()) >= 3) else 0
            if streak >= 2 and not ack:                                              # S1: prepare the acknowledgement early
                topic = rule_topic(partial)
                ack_text = "Okay, putting a card together." if lab == "DELEGATE_WRITE" else (f"Let me check your {topic}." if topic != "other" else "One sec, let me look.")
                ack["started"] = now(); ack["at_words"] = len(partial.split())
                def prep(txt=ack_text):
                    ms = first_chunk_ms(txt); ack["ready"] = now(); ack["tts_ms"] = ms
                threading.Thread(target=prep, daemon=True).start()
            log.setdefault("route_trace", []).append((round(arrive, 2), lab, round(conf, 2)))
        if in_speech and commit is None and sil >= SPEC_SIL:
            if spec is None:                                                         # S2: start the thinking model at the first pause
                spec = Reasoner(partial, t0); spec.start(); reasoners.append(spec)
                p, ms = turn_done(buf); log.setdefault("turn_checks", []).append((round(arrive, 2), round(p, 2), round(ms)))
                lab_now = router.predict_one(dict(text=partial, assistant_speaking=False, last_assistant="", prev_user=""))[0]
                if p >= 0.5 and lab_now != "INCOMPLETE": commit = now(); log["commit_by"] = "turn detector"
            if commit is None and sil >= FALLBACK_SIL: commit = now(); log["commit_by"] = "silence fallback"
            if commit is not None:
                log["final_text"] = partial; log["route"] = router.predict_one(dict(text=partial, assistant_speaking=False, last_assistant="", prev_user=""))[0]
                if nw(spec.text) != nw(partial):                                     # transcript moved: speculation is not reusable
                    spec.cancelled = True; log["wasted_reasoner"] += 1; spec = Reasoner(partial, t0); spec.start(); reasoners.append(spec)
        if in_speech and classic_commit is None and sil >= CLASSIC_SIL: classic_commit = arrive
        if commit is not None and classic_commit is not None and (spec.first_sentence is not None or spec.error or now() - commit > 12): break
    if commit is None or spec is None or spec.first_sentence is None:
        log["error"] = (spec.error if spec else None) or ("no commit/answer: commit=%s spec=%s cancelled=%s first_token=%s alive=%s n=%d wall=%.1f" % (commit, bool(spec), spec and spec.cancelled, spec and spec.first_token, spec and spec.is_alive(), len(reasoners), now())); return log
    answer_tts = first_chunk_ms(spec.sentence)
    se = u["speech_end"]; r_dur = spec.first_sentence - spec.started; stt = float(np.median(stt_ms))
    is_del = log["route"] in DELEG
    ours_ack = None
    if is_del:
        if "ready" not in ack:                       # not prepared in time: make it at commit
            ack_ms = first_chunk_ms(ack_text or "One sec, let me look."); ours_ack = commit + ack_ms / 1000 - se; log["ack_prepared_early"] = False
        else:
            ours_ack = max(commit, ack["ready"]) - se; log["ack_prepared_early"] = ack["ready"] <= commit; log["ack_started_before_end_s"] = round(se - ack["started"], 2)
    ours_answer = max(spec.first_sentence, commit) + answer_tts / 1000 - se
    classic_start = classic_commit + stt / 1000
    classic_answer = classic_start + r_dur + answer_tts / 1000 - se
    classic_ack = classic_start + (ack.get("tts_ms") or answer_tts) / 1000 - se if is_del else None
    log.update(commit_after_end_ms=round((commit - se) * 1000), reasoner_start_after_end_ms=round((spec.started - se) * 1000), reasoner_first_token_ms=round((spec.first_token - spec.started) * 1000),
               reasoner_first_sentence_ms=round(r_dur * 1000), stt_ms=round(stt), answer_tts_first_chunk_ms=round(answer_tts), answer=spec.sentence,
               ours_first_sound_ms=round((ours_ack if is_del else ours_answer) * 1000), ours_answer_ms=round(ours_answer * 1000),
               classic_first_sound_ms=round(classic_answer * 1000), classic_with_filler_first_sound_ms=round((classic_ack if is_del else classic_answer) * 1000), classic_answer_ms=round(classic_answer * 1000))
    return log

out = []
for n, u in enumerate(utts):
    try: r = run(u)
    except Exception as e: r = dict(text=u["text"], error=repr(e)[:300])
    out.append(r)
    print(n, r.get("label"), "paused" if r.get("paused") else "", "falsecommit ours/classic", r.get("false_commit", 0), r.get("classic_false_commit", 0), "|", r.get("final_text"), "| commit", r.get("commit_after_end_ms"), r.get("commit_by"), "| ours first/answer", r.get("ours_first_sound_ms"), r.get("ours_answer_ms"),
          "| classic first/answer", r.get("classic_first_sound_ms"), r.get("classic_answer_ms"), "| wasted", r.get("wasted_reasoner"), r.get("error") or "", flush=True)
    json.dump(out, open(os.path.join(ROOT, "spike", "e2e.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    time.sleep(0.5)

ok = [r for r in out if "ours_answer_ms" in r]
def med(k, rs): v = [r[k] for r in rs if r.get(k) is not None]; return (round(float(np.median(v))), round(float(np.percentile(v, 90)))) if v else None
for name, rs in (("ALL", ok), ("DELEGATE", [r for r in ok if r["route"] in DELEG]), ("SMALLTALK", [r for r in ok if r["route"] not in DELEG]), ("PAUSED", [r for r in ok if r["paused"]])):
    print(name, len(rs), {k: med(k, rs) for k in ("ours_first_sound_ms", "classic_first_sound_ms", "classic_with_filler_first_sound_ms", "ours_answer_ms", "classic_answer_ms", "commit_after_end_ms", "reasoner_first_sentence_ms", "stt_ms", "answer_tts_first_chunk_ms")})
print("false commits ours", sum(r.get("false_commit", 0) for r in out), "classic", sum(r.get("classic_false_commit", 0) for r in out))
print("wasted reasoner calls", sum(r.get("wasted_reasoner", 0) for r in out), "of", len(out), "| ack prepared early", sum(1 for r in ok if r.get("ack_prepared_early")), "of", sum(1 for r in ok if r["route"] in DELEG),
      "| commit by turn detector", sum(1 for r in ok if r.get("commit_by") == "turn detector"), "| errors", sum(1 for r in out if "error" in r))

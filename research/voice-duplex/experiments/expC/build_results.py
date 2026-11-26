"""Merge every res_*.json + transcripts.json into results.json and print the summary tables."""
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import SENTENCES, med
from textnorm import wer, numbers_recovered, canon
from fidelity_gen import FID


def load(name):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


tr = load("transcripts.json") or {}
res = {"notes": "CPU shared with other experiments while measuring; numbers are noisy. See report.",
       "sentences": SENTENCES, "profile_and_exactness": load("results_profile_exact.json"),
       "sweeps": {os.path.basename(p)[len("res_woof_sweep_"):-5]: json.load(open(p, encoding="utf-8"))
                  for p in sorted(glob.glob(os.path.join(HERE, "res_woof_sweep_*.json")))},
       "extra": load("res_extra.json")}


def wer_for(folder, seeds=(0, 1, 2), skip=()):
    e = n = 0
    per, missing = [], 0
    for si, text in enumerate(SENTENCES):
        if si in skip:
            continue
        for s in seeds:
            k = f"{folder}/s{si}_seed{s}.wav"
            if k not in tr:
                missing += 1
                continue
            a, b = wer(text, tr[k])
            e += a; n += b
            per.append({"sentence": si, "seed": s, "errors": a, "words": b, "hyp": tr[k]})
    return {"errors": e, "words": n, "wer_pct": (100.0 * e / n) if n else None, "missing": missing, "per": per}


def summarise(rows):
    t = [r["ttfa_ms"] for r in rows]
    per_sentence = []
    for si in range(len(SENTENCES)):
        rs = [r for r in rows if r["sentence"] == si]
        if rs:
            per_sentence.append({"sentence": si, "chars": len(SENTENCES[si]), "ttfa_ms_median": med([r["ttfa_ms"] for r in rs]),
                                 "ttfa_ms_max": max(r["ttfa_ms"] for r in rs), "rtf_rest_median": med([r["rtf_rest"] for r in rs]),
                                 "audio_s_median": med([r["audio_s"] for r in rs]),
                                 "underrun_runs": sum(r["underruns"] > 0 for r in rs),
                                 "aborted": [r.get("aborted") for r in rs if r.get("aborted")]})
    return {"runs": len(rows), "ttfa_ms_median": med(t), "ttfa_ms_p90": float(np.percentile(t, 90)), "ttfa_ms_max": max(t),
            "ttfa_ms_median_of_sentence_medians": med([p["ttfa_ms_median"] for p in per_sentence]),
            "rtf_rest_median": med([r["rtf_rest"] for r in rows]),
            "rtf_rest_min": min(r["rtf_rest"] for r in rows if r["rtf_rest"] is not None),
            "runs_with_underrun": sum(r["underruns"] > 0 for r in rows),
            "underrun_gap_ms_max": max(r["underrun_gap_ms"] for r in rows),
            "needed_prebuffer_ms_max": max(r["needed_prebuffer_ms"] for r in rows),
            "first_chunk_ms_audio": med([r["first_chunk_ms_audio"] for r in rows]),
            "per_sentence": per_sentence}


table = []
wm = load("res_woof_main_main.json")
if wm:
    res["woof_main_raw"] = wm
    for voice in ("juno", "mira"):
        rows = [r for r in wm["rows"] if r["voice"] == voice]
        s = summarise(rows)
        s["aborted_takes"] = [(r["sentence"], r["seed"], r["aborted"]) for r in rows if r.get("aborted")]
        s["wer_streamed"] = wer_for(f"woof_{voice}_stream")
        s["wer_streamed_without_Mm-hm"] = {k: v for k, v in wer_for(f"woof_{voice}_stream", skip=(1,)).items() if k != "per"}
        s["wer_nonstream_single_take"] = wer_for(f"woof_{voice}_full")
        s["wer_nonstream_original_with_retries"] = wer_for(f"woof_{voice}_orig_retry")
        o = [r for r in wm.get("orig", []) if r["voice"] == voice]
        s["original_nonstream_time_to_audio_ms_median"] = med([r["time_to_audio_ms"] for r in o])
        s["original_nonstream_time_to_audio_ms_max"] = max([r["time_to_audio_ms"] for r in o], default=None)
        s.update(system=f"Woof {voice} streaming (schedule {wm['schedule']}, {wm['threads']} BLAS threads)", size="40.6 MB, 38.2M params")
        table.append(s)
pk = load("res_pocket.json")
if pk:
    res["pocket_raw"] = pk
    for lang, m in pk["models"].items():
        if "rows" not in m:
            table.append({"system": f"Pocket TTS {lang}", "error": m.get("error")})
            continue
        s = summarise(m["rows"])
        s["wer_streamed"] = wer_for(f"pocket_{lang}")
        s["wer_streamed_without_Mm-hm"] = {k: v for k, v in wer_for(f"pocket_{lang}", skip=(1,)).items() if k != "per"}
        s.update(system=f"Pocket TTS {lang} (alba, {pk['torch_threads']} torch threads)",
                 size="%.1fM params, %.0f MB fp32" % (m["params"] / 1e6, m["fp32_mb"]))
        table.append(s)
res["summary_table"] = table

# ---- amount fidelity ----
fid = {}
for folder in sorted({k.split("/")[0] for k in tr if k.startswith("fid_")}):
    ok = nums_ok = nums_all = 0
    fails, have = [], 0
    for i, (text, nums) in enumerate(FID):
        k = f"{folder}/{i:02d}.wav"
        if k not in tr:
            continue
        have += 1
        good, found = numbers_recovered(nums, tr[k])
        ok += good; nums_ok += sum(found); nums_all += len(found)
        if not good:
            fails.append({"i": i, "text": text, "expected": nums, "found": found, "transcript": tr[k]})
    fid[folder] = {"sentences": have, "sentences_all_numbers_exact": ok, "numbers_recovered": nums_ok,
                   "numbers_total": nums_all, "failures": fails,
                   "transcripts": [tr.get(f"{folder}/{i:02d}.wav") for i in range(len(FID))]}
res["amount_fidelity"] = {"method": numbers_recovered.__doc__, "sentences": [{"text": t, "numbers": n} for t, n in FID],
                          "gen_meta": {**(load("res_fid_gen_woof.json") or {}), **(load("res_fid_gen_pocket.json") or {})},
                          "systems": fid}

with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)

print("%-62s %-26s %6s %6s %6s %6s %5s %9s %7s %7s" % ("system", "size", "ttfa50", "p90", "max", "rtf50", "rtfmin", "underruns", "WER%", "WER-mm"))
for s in table:
    if "error" in s:
        print(s["system"], "ERROR", s["error"]); continue
    w, w2 = s["wer_streamed"], s["wer_streamed_without_Mm-hm"]
    print("%-62s %-26s %6.0f %6.0f %6.0f %6.2f %5.2f %4d/%-4d %7s %7s" % (
        s["system"][:62], s["size"], s["ttfa_ms_median"], s["ttfa_ms_p90"], s["ttfa_ms_max"], s["rtf_rest_median"], s["rtf_rest_min"],
        s["runs_with_underrun"], s["runs"], "-" if w["wer_pct"] is None else "%.1f" % w["wer_pct"],
        "-" if w2["wer_pct"] is None else "%.1f" % w2["wer_pct"]))
    for k in ("wer_nonstream_single_take", "wer_nonstream_original_with_retries"):
        if k in s and s[k]["wer_pct"] is not None:
            print("      %s: %.1f%% (%d/%d)" % (k, s[k]["wer_pct"], s[k]["errors"], s[k]["words"]))
    if s.get("aborted_takes"):
        print("      guard cut takes:", s["aborted_takes"])
    print("      per sentence ttfa median/max:", [(p["chars"], round(p["ttfa_ms_median"]), round(p["ttfa_ms_max"])) for p in s["per_sentence"]])
for k, v in fid.items():
    print("FIDELITY %-34s sentences exact %d/%d, numbers %d/%d" % (k, v["sentences_all_numbers_exact"], v["sentences"], v["numbers_recovered"], v["numbers_total"]))
    for f_ in v["failures"]:
        print("     x", f_["i"], f_["expected"], f_["found"], "|", f_["transcript"])

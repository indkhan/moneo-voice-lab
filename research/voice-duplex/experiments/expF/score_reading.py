"""Word errors of every transcript in a recording folder against the reading script's true text.

  python score_reading.py <folder> [script.md]

The clips of a reading split at pauses, not at sentences, so the whole reading is compared as one text:
that needs no alignment of clips to sentences. Numbers are compared as words ("12" = "twelve").
"""
import glob, json, os, re, sys

folder = sys.argv[1]
script = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "..", "..", "reading-script.md")
truth = " ".join(re.findall(r"^\d+\. (.+)$", open(script, encoding="utf-8").read(), re.M))

SMALL = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
def say(n: int) -> list[str]:
    if n < 20: return [SMALL[n]]
    if n < 100: return [TENS[n // 10]] + ([SMALL[n % 10]] if n % 10 else [])
    if n < 1000: return [SMALL[n // 100], "hundred"] + (say(n % 100) if n % 100 else [])
    return say(n // 1000) + ["thousand"] + (say(n % 1000) if n % 1000 else [])
def words(text: str) -> list[str]:
    text = re.sub(r"(\d),(\d{3})\b", r"\1\2", text.lower()).replace("-", " ")
    out = []
    for token in re.sub(r"[^a-z0-9.' ]+", " ", text).split():
        token = token.strip(".'")
        if re.fullmatch(r"\d+\.\d+", token):
            whole, cents = token.split("."); out += say(int(whole)) + say(int(cents))
        elif token.isdigit() and int(token) < 1_000_000: out += say(int(token))
        elif token: out.append(token.replace("'", ""))
    return out
def dist(a, b):
    row = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, row[0] = row[0], i
        for j in range(1, len(b) + 1):
            keep = row[j]; row[j] = min(row[j] + 1, row[j - 1] + 1, prev + (a[i - 1] != b[j - 1])); prev = keep
    return row[-1]
KEY = ["rewe", "lidl", "netto", "amazon", "netflix", "spotify", "starbucks", "scalable", "chatgpt", "munich", "euros", "add", "spend", "spent", "earn", "salary", "groceries", "restaurants", "month", "wallet", "budget"]
def kept(hyp, ref):
    return sum(min(ref.count(w), hyp.count(w)) for w in KEY), sum(ref.count(w) for w in KEY)

ref = words(truth)
sources = {}
whistle = os.path.join(folder, "transcripts.json")
if os.path.exists(whistle): sources["whistle"] = {r["clip"]: r["names"] for r in json.load(open(whistle, encoding="utf-8"))}
if os.path.exists(os.path.join(folder, "reference.json")): sources["large-v3-turbo"] = json.load(open(os.path.join(folder, "reference.json"), encoding="utf-8"))
for path in sorted(glob.glob(os.path.join(folder, "hyp_*.json"))):
    sources[os.path.basename(path)[4:-5]] = json.load(open(path, encoding="utf-8"))
result = {}
for name, clips in sources.items():
    hyp = words(" ".join(clips[c] for c in sorted(clips)))
    k, kt = kept(hyp, ref)
    result[name] = {"word errors %": round(100 * dist(hyp, ref) / len(ref), 1), "key words kept %": round(100 * k / kt)}
    print(f"{name:18} {result[name]}")
json.dump(result, open(os.path.join(folder, "reading_score.json"), "w", encoding="utf-8"), indent=1)

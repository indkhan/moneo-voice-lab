"""Text normalisation for scoring: spelled-out numbers -> digits, word error rate, and number recovery."""
import re
from decimal import Decimal

UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine".split())}
TEENS = {w: i + 10 for i, w in enumerate(
    "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {w: (i + 2) * 10 for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9,
       "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
       "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20, "thirtieth": 30,
       "fortieth": 40, "fiftieth": 50, "sixtieth": 60, "seventieth": 70, "eightieth": 80, "ninetieth": 90}
SCALES = {"thousand": 1000, "million": 1000000}
TOK = re.compile(r"\d+(?:\.\d+)?(?:st|nd|rd|th)?|[a-z']+")


def tokens(text):
    t = text.lower().replace("’", "'")
    t = re.sub(r"[€$£]\s*(\d[\d,]*(?:\.\d+)?)", lambda m: m.group(1) + {"€": " euros", "$": " dollars", "£": " pounds"}[m.group(0)[0]], t)
    t = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", t)
    t = t.replace("%", " percent ")
    return TOK.findall(t)


def canon(text):
    """-> list of tokens where every number (digits or words, cardinal or ordinal) is a canonical digit string."""
    w = tokens(text)
    out, i, n = [], 0, len(w)

    def small(j):
        """units/teens/tens(+unit) starting at j -> (value, next index) or None"""
        if j < n and w[j] in TENS:
            v = TENS[w[j]]
            if j + 1 < n and w[j + 1] in UNITS and UNITS[w[j + 1]] > 0:
                return v + UNITS[w[j + 1]], j + 2
            if j + 1 < n and w[j + 1] in ORD and ORD[w[j + 1]] < 10:
                return v + ORD[w[j + 1]], j + 2
            return v, j + 1
        if j < n and w[j] in TEENS:
            return TEENS[w[j]], j + 1
        if j < n and w[j] in UNITS:
            return UNITS[w[j]], j + 1
        if j < n and w[j] in ORD:
            return ORD[w[j]], j + 1
        return None

    while i < n:
        tok = w[i]
        if tok[0].isdigit():
            num = re.sub(r"(st|nd|rd|th)$", "", tok)
            j = i + 1
            if j < n and w[j] in SCALES and "." not in num:               # "48 thousand 250"
                v = int(num) * SCALES[w[j]]
                j += 1
                if j < n and w[j].isdigit() and int(w[j]) < 1000:
                    v += int(w[j]); j += 1
                num = str(v)
            out.append(num); i = j
            continue
        if tok == "a" and i + 1 < n and w[i + 1] in ("hundred", "thousand"):
            w[i] = tok = "one"
        s = small(i)
        if s is None:
            out.append(tok); i += 1
            continue
        # integer part: [small] [hundred [and] [small]] [thousand ...]
        total, cur, j = 0, s[0], s[1]
        is_ord_end = w[j - 1] in ORD
        while j < n and not is_ord_end:
            if w[j] == "hundred" and 0 < cur < 100:
                cur *= 100; j += 1
                k = j + 1 if (j < n and w[j] == "and") else j
                s2 = small(k)
                if s2 and s2[0] < 100:
                    cur += s2[0]; j = s2[1]; is_ord_end = w[j - 1] in ORD
            elif w[j] in SCALES and cur > 0:
                total += cur * SCALES[w[j]]; cur = 0; j += 1
                k = j + 1 if (j < n and w[j] == "and") else j
                s2 = small(k)
                if s2:
                    cur = s2[0]; j = s2[1]; is_ord_end = w[j - 1] in ORD
                else:
                    break
            else:
                break
        val = str(total + cur)
        if j < n and w[j] == "point" and not is_ord_end:
            k, frac = j + 1, ""
            while k < n:
                if w[k] in UNITS:
                    frac += str(UNITS[w[k]]); k += 1
                elif w[k] in ("oh", "o"):
                    frac += "0"; k += 1
                elif w[k] in TENS or w[k] in TEENS:
                    s2 = small(k); frac += str(s2[0]); k = s2[1]
                else:
                    break
            if frac:
                val += "." + frac; j = k
        out.append(val); i = j
    return out


EQUIV = {"okay": "ok", "k": "ok", "euro": "euros", "per": "per", "mhm": "mm-hm", "cents": "cents"}


def wer_words(text):
    out = []
    for t in canon(text):
        if t[0].isdigit():
            out.append(str(Decimal(t).normalize() + 0) if "." in t else t)     # 312.40 == 312.4
        else:
            t = t.strip("'")
            if t:
                out.append(EQUIV.get(t, t))
    return out


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(b)]


def wer(ref, hyp):
    r, h = wer_words(ref), wer_words(hyp)
    return edit_distance(r, h), len(r)


def numbers_recovered(ref_numbers, hyp):
    """ref_numbers: digit strings in the order spoken, e.g. ["312.40", "3"]. Each must be found, left to right, as
      * one hypothesis number that is numerically equal (312.40 == 312.4; ordinals count as their number), or
      * for an amount with cents: two neighbouring integers A, B (B < 100) read as A.BB ("312 euros and 40 cents",
        "twelve forty"), or
      * for an integer: up to four neighbouring integers whose digits concatenate to it ("twenty twenty-six" -> 2026,
        "nine fifty" -> 950, "four four seven one" -> 4471).
    Neighbouring = at most three ordinary words in between. Returns (all_found, per-number list of bool)."""
    c = canon(hyp)
    nums = [(i, t) for i, t in enumerate(c) if t[0].isdigit()]
    pos, found = 0, []
    for r in ref_numbers:
        rd = Decimal(r)
        hit = None
        for a in range(pos, len(nums)):
            if Decimal(nums[a][1]) == rd:
                hit = a + 1
                break
            if "." in r and len(r.split(".")[1]) == 2 and a + 1 < len(nums):
                (ia, ta), (ib, tb) = nums[a], nums[a + 1]
                if ta.isdigit() and tb.isdigit() and int(tb) < 100 and ib - ia <= 4 and \
                        Decimal(f"{ta}.{int(tb):02d}") == rd:
                    hit = a + 2
                    break
            if "." not in r:
                cat, last = "", None
                for b in range(a, min(a + 4, len(nums))):
                    ib, tb = nums[b]
                    if not tb.isdigit() or (last is not None and ib - last > 4):
                        break
                    cat += tb; last = ib
                    if b > a and cat == r:
                        hit = b + 1
                        break
                if hit:
                    break
        found.append(hit is not None)
        if hit is not None:
            pos = hit
    return all(found), found


if __name__ == "__main__":
    tests = [
        ("three hundred and twelve point four zero euros", ["312.40"], True),
        ("You spent 312.40 euros", ["312.40"], True),
        ("You spent €312.4", ["312.40"], True),
        ("three hundred twelve euros and forty cents", ["312.40"], True),
        ("three hundred twelve euros and four cents", ["312.40"], False),
        ("312.04 euros", ["312.40"], False),
        ("up twenty four percent since March twelfth", ["2.4", "12"], False),
        ("up two point four percent since March twelfth", ["2.4", "12"], True),
        ("forty-eight thousand two hundred and fifty euros", ["48250"], True),
        ("48,250 euros", ["48250"], True),
        ("48 thousand 250 euros", ["48250"], True),
        ("rent of nine fifty goes out on the third", ["950", "3"], True),
        ("Rent of 950 goes out on the 3rd.", ["950", "3"], True),
        ("Rent of 915 goes out on the 3rd.", ["950", "3"], False),
        ("in twenty twenty six, nineteen percent", ["2026", "19"], True),
        ("ending in four four seven one has six thousand and twelve point three", ["4471", "6012.30"], True),
        ("on the twenty-first, sixty four point one eight", ["64.18", "21"], False),
        ("sixty four point one eight due on the twenty-first", ["64.18", "21"], True),
        ("one thousand two hundred euros which is seven point five percent", ["1200", "7.5"], True),
        ("twelve hundred euros, 7.5%", ["1200", "7.5"], True),
        ("zero point eight percent", ["0.8"], True),
        ("seventeen thousand nine hundred and three point five five euros across four accounts", ["17903.55", "4"], True),
    ]
    bad = 0
    for hyp, ref, want in tests:
        got = numbers_recovered(ref, hyp)[0]
        if got != want:
            bad += 1
            print("FAIL", hyp, ref, want, canon(hyp))
    print("self-test failures:", bad, "of", len(tests))
    print(wer("You spent 312.40 euros on groceries in October, about 18 percent more than September.",
              "You spent three hundred and twelve point four euros on groceries in October, about 18% more than September"))
    print(wer_words("Mm-hm."), wer_words("Okay."), wer_words("Your balance is 1,250 euros."))

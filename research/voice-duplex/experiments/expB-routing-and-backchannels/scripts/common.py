"""Shared constants for experiment B (router + talker)."""
import json, os, sys, threading, time

S = r"<scratch>"
B = os.path.join(S, "expB")
DATA = os.path.join(B, "data")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "spike-end-to-end"))

LABELS = ["DELEGATE_READ", "DELEGATE_WRITE", "CORRECTION", "META", "SMALLTALK", "BACKCHANNEL", "INCOMPLETE"]
TOPICS = ["groceries", "eating out", "rent", "bills", "subscriptions", "transport", "shopping", "travel", "health",
          "income", "savings", "net worth", "investments", "account balance", "budget", "spending", "transfers",
          "debts", "cash", "taxes", "insurance", "entertainment", "goals", "transactions", "app", "other"]

GUIDE = """Moneo is a personal-finance app. The owner is on a voice call with Moneo's assistant. A router labels each thing the owner says.

Each item has this call context:
- assistant_speaking: true if the assistant was talking while the owner said it, false if the assistant was silent.
- last_assistant: the assistant's last (or current) sentence, or "" at the start of a call.
- prev_user: the owner's previous request in this call, or "".

Labels:
- DELEGATE_READ: a question or request about their own money, spending, balances, bills, subscriptions, budgets, plans, investments, net worth, or how the app works. Also an answer to the assistant's clarifying question about a lookup (assistant silent, last_assistant is a question, e.g. "which account?" -> "the savings one", or "want last month too?" -> "yeah").
- DELEGATE_WRITE: they state money they spent, received, lent, borrowed, or what an account holds, or ask to add, change or remove something (a transaction, budget, bill, goal, category, account). Includes money facts dropped in passing inside chatter ("had lunch with tom, paid like 14 euros"). Also "yeah"/"go ahead" when the assistant is silent and last_assistant asked whether to add or change something.
- CORRECTION: replaces or fixes prev_user ("no, i meant last month", "actually make it 15", "not groceries, restaurants"). prev_user is never empty. A follow-up that asks a NEW thing ("and what about restaurants?") is DELEGATE_READ, not CORRECTION.
- META: asks about progress, or to repeat, slow down or explain the last answer, or checks the line ("is it done?", "say that again", "wait what do you mean", "you still there?").
- SMALLTALK: greeting, chit-chat, thanks, goodbye, feelings, jokes. No request and no fact about their own money. General money chatter with no own transaction and no question for the app counts ("everything is so expensive these days", "my brother won 50 bucks on a scratch card lol"). A lone "okay cool"/"thanks" after the assistant finished a statement is SMALLTALK.
- BACKCHANNEL: a lone "mm-hm", "yeah", "okay", "right", "uh-huh", "sure", "i see" said WHILE the assistant is speaking (assistant_speaking true), just showing they listen. Not an interruption, not an answer.
- INCOMPLETE: cut off mid-thought, the sentence clearly is not finished ("how much did i spend on", "so i was thinking that maybe", "can you add a", "and the rent is"). Often ends on a preposition, article, conjunction or filler.

topic: for DELEGATE_READ and DELEGATE_WRITE pick the single best from this list: """ + ", ".join(TOPICS) + """. For every other label topic is "".

Style of text: what a speech recogniser would output for casual speech. Mostly lowercase, little or no punctuation, with disfluencies (uh, um, like, you know), restarts and the odd mis-heard word. Numbers appear the ways an ASR writes them: "12.40", "1240", "twelve forty", "950", "nine fifty", "2k", "fifteen hundred". Vary length from 1 word to about 25 words."""

_lock = threading.Lock()


def call(model, instructions, user, effort="low", tag=""):
    """One proxy call, counted in calls.log so the plan quota stays under budget."""
    from llm import ask
    for attempt in range(3):
        try:
            r = ask(model, instructions, user, effort)
            break
        except Exception as e:  # noqa
            err = repr(e)[:200]
            time.sleep(2 + 3 * attempt)
            r = None
    with _lock:
        with open(os.path.join(B, "calls.log"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "model": model, "tag": tag, "ok": r is not None,
                                "first_ms": r and r["first_ms"], "total_ms": r and r["total_ms"]}) + "\n")
    if r is None:
        raise RuntimeError("proxy call failed: " + err)
    return r


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_jsonl_text(text):
    out = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out

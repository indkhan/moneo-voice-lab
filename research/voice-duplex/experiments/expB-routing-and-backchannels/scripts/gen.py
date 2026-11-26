"""Generate raw labelled utterances through the ChatGPT-plan proxy. Train and test use different models, personas and seeds."""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
from common import *

SYS = GUIDE + """

You write realistic data for testing this router. Output JSON Lines only: one JSON object per line, no markdown, no numbering, no commentary.
Each line: {"text": "...", "label": "...", "assistant_speaking": true|false, "last_assistant": "...", "prev_user": "...", "topic": "..."}
Make every line different in wording, length, topic and sentence shape. Never repeat an opening pattern more than twice."""

FOCUS = {
    "DELEGATE_READ": "All lines are DELEGATE_READ. Cover many areas (spending by category or merchant, balances, bills due, subscriptions, budgets left, net worth, investments, income, comparisons between months, forecasts, 'can i afford', how the app works). Include about 6 where the owner answers the assistant's clarifying question about a lookup (assistant silent), 3 of them a bare 'yeah'/'yes please'/'the second one'. Include about 5 elliptical follow-ups like 'and what about restaurants' with a prev_user. Include some long rambling ones and some that open with chatter before the question.",
    "DELEGATE_WRITE": "All lines are DELEGATE_WRITE. Mix: plain statements of spending ('spent 12.40 at netto'), money received, lent or borrowed, what an account holds ('my savings has about 4200 in it'), and requests to add/change/remove/rename things (transactions, budgets, recurring bills, goals, categories, accounts). About 10 should be a money fact dropped in passing inside chit-chat. About 5 should be 'yeah'/'go ahead'/'yes add it' after the assistant (silent) asked whether to add or change something. Use all the ASR number styles.",
    "CORRECTION": "All lines are CORRECTION, each with a realistic non-empty prev_user that it fixes. Mix: changed period, changed amount, changed category or merchant, changed account, 'no not that', 'scratch that i meant', mis-heard word fixes ('netto not net worth'). Some are said while the assistant is speaking (interrupting the answer), some after it finished. Some very short ('no last month'), some long.",
    "META": "All lines are META. Mix: progress checks while waiting ('is it done', 'still working on it?', 'how long is this gonna take'), repeat requests ('say that again', 'sorry what was the number'), explain requests about the last answer ('wait what do you mean by that', 'how did you get that'), slow down / louder / shorter, line checks ('hello? you there?'), and 'never mind'/'stop'/'cancel that' said to halt the assistant. Give a fitting last_assistant.",
    "SMALLTALK": "All lines are SMALLTALK. Mix: greetings, how are you, thanks, goodbye, weather, tired, weekend, jokes, compliments or teasing the assistant, 'who are you', feelings. About 12 should be tricky: general money chatter with NO own transaction and NO question for the app (prices, a friend's lottery win, the economy, 'money is so stressful', 'i should really save more someday'), and lone 'okay cool'/'alright thanks'/'nice' after the assistant finished a statement (assistant_speaking false).",
    "BACKCHANNEL": "All lines are BACKCHANNEL: assistant_speaking is true and last_assistant is the sentence the assistant is in the middle of (a statement, not a question). Text is 1 to 4 words: mm-hm, mhm, yeah, yep, okay, right, uh-huh, sure, i see, got it, oh okay, yeah yeah, mm, ah right, ok ok and ASR spellings of these (mmhmm, uh huh, m-hm, yea, k). Repeats of the same word are fine here because the context differs: vary last_assistant a lot.",
    "INCOMPLETE": "All lines are INCOMPLETE: cut off mid-thought. Make most of them prefixes of real DELEGATE_READ or DELEGATE_WRITE requests cut at an unnatural point (after a preposition, article, 'and', 'my', a number missing: 'i spent', 'can you set my rent to', 'how much did i spend on'). Some are trailing fillers ('so um', 'and then i', 'okay so the thing is'). Some are long ramblings that still have not reached the point. A few cut off corrections ('no i meant the'). Never end them like a finished sentence.",
    "MIXED": "Make a mixed set of HARD boundary cases, roughly equal across all 7 labels, built as contrasting pairs: the same or nearly the same words that get different labels because of the context or one small change. E.g. 'yeah' as BACKCHANNEL (assistant speaking) vs 'yeah' as an answer to 'should i add it?' (DELEGATE_WRITE) vs 'yeah' to 'want the breakdown?' (DELEGATE_READ); 'coffee with anna was nice' (SMALLTALK) vs 'coffee with anna cost me 4.50' (DELEGATE_WRITE); 'how much did i spend on' (INCOMPLETE) vs 'how much did i spend' (DELEGATE_READ); 'no last month' with a prev_user (CORRECTION) vs 'and last month?' (DELEGATE_READ); 'what?' (META) vs 'what's my balance' (DELEGATE_READ); 'thanks that's all' (SMALLTALK) vs 'thanks, and add 20 for the taxi' (DELEGATE_WRITE).",
}

TRAIN_PERSONAS = [
    "a 28-year-old nurse in Berlin who pays in euros and shops at Rewe, Lidl and dm",
    "a freelance designer in Manchester (pounds), irregular income, lots of subscriptions",
    "a retired teacher in Ohio (dollars), careful, a bit chatty, has a pension and a brokerage account",
    "a young couple's shared finances in Amsterdam (euros), saving for a flat, uses Tikkie-style paybacks with friends",
    "a software engineer in Munich (euros) with ETF savings plans at Scalable Capital and a car loan",
    "a bartender in Sydney (dollars), cash tips, rent split with flatmates",
]
TEST_PERSONAS = [
    "a university student in Copenhagen with a part-time job (kroner, shops at Netto and Fotex, uses MobilePay)",
    "a truck driver and father of three in Texas (dollars), fuel and truck payments, speaks in short blunt phrases",
    "a startup founder in Dublin (euros), talks fast, restarts sentences, mixes personal and business spending",
    "a 61-year-old shop owner in Vienna (euros) new to the app, polite, sometimes unsure what it can do",
    "a nurse in Toronto (dollars) on night shifts, tired, mumbling, splits bills with a roommate",
]

# (label focus, count, how many calls) per split
TRAIN_PLAN = [("DELEGATE_READ", 45, 3), ("DELEGATE_WRITE", 45, 3), ("CORRECTION", 42, 2), ("META", 36, 2),
              ("SMALLTALK", 45, 2), ("BACKCHANNEL", 45, 1), ("INCOMPLETE", 42, 2), ("MIXED", 42, 1)]
TEST_PLAN = [("DELEGATE_READ", 55, 1), ("DELEGATE_WRITE", 55, 1), ("CORRECTION", 32, 1), ("META", 28, 1),
             ("SMALLTALK", 38, 1), ("BACKCHANNEL", 24, 1), ("INCOMPLETE", 32, 1), ("MIXED", 42, 1)]


def job(split, model, focus, n, persona, seed):
    out = os.path.join(DATA, "raw", f"{split}-{focus}-{seed}.jsonl")
    if os.path.exists(out):
        return read_jsonl(out)
    user = (f"Seed {seed}. The owner is {persona}.\n{FOCUS[focus]}\nWrite exactly {n} lines.")
    r = call(model, SYS, user, effort="low", tag=f"gen-{split}-{focus}-{seed}")
    rows = parse_jsonl_text(r["text"])
    for x in rows:
        x["_split"], x["_gen"], x["_focus"], x["_persona"], x["_seed"] = split, model, focus, persona, seed
    write_jsonl(out, rows)
    print(split, focus, seed, len(rows), r["total_ms"], flush=True)
    return rows


if __name__ == "__main__":
    os.makedirs(os.path.join(DATA, "raw"), exist_ok=True)
    jobs, seed = [], 100
    for focus, n, calls in TRAIN_PLAN:
        for c in range(calls):
            seed += 7
            jobs.append(("train", "gpt-6.1-sol", focus, n, TRAIN_PERSONAS[(seed // 7) % len(TRAIN_PERSONAS)], seed))
    seed = 9000
    for focus, n, calls in TEST_PLAN:
        for c in range(calls):
            seed += 131
            jobs.append(("test", "gpt-6-luna", focus, n, TEST_PERSONAS[(seed // 131) % len(TEST_PERSONAS)], seed))
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda j: job(*j), jobs))
    rows = [x for r in res for x in r]
    write_jsonl(os.path.join(DATA, "raw.jsonl"), rows)
    print("total", len(rows), "calls", len(jobs))

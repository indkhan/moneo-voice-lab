"""Hand-review fixes for the test set (all 200 items were read). Writes data/test.jsonl."""
from common import *

FIXES = {
    # id: (fields to change, why)
    "te086": ({"label": "META", "topic": "", "last_assistant": "that account is a bit lower than last month"}, "owner asks the assistant to clarify; generator had labelled it DELEGATE_READ with an incoherent context"),
    "te126": ({"label": "META", "topic": "", "last_assistant": "you spent less than the other month"}, "asks the assistant to explain its last answer, not a money question"),
    "te173": ({"label": "DELEGATE_WRITE", "topic": "transactions"}, "previous request was a question; 'change the merchant to loves' is a new change request, not a correction of it"),
    "te188": ({"label": "DELEGATE_WRITE", "topic": "savings"}, "'can you remove my old savings' reads as a finished request, not a cut-off"),
    "te052": ({"last_assistant": ""}, "context repaired: the owner was echoing the assistant's own question"),
    "te120": ({"text": "uh the next three months i guess", "assistant_speaking": False}, "text repaired: the owner line was the assistant's question repeated; now an answer to it"),
    "te156": ({"topic": "transport"}, "topic fixed: diesel is transport, not generic spending"),
}

rows = read_jsonl(os.path.join(DATA, "test_unreviewed.jsonl"))
n = 0
for r in rows:
    if r["id"] in FIXES:
        r.update(FIXES[r["id"]][0]); r["review_fix"] = FIXES[r["id"]][1]; n += 1
write_jsonl(os.path.join(DATA, "test.jsonl"), rows)
print("fixed", n, "of", len(rows))

# Voice call research: roadmap

For anyone (person or agent) picking up work on branch `t3/research-self-hosted-voice`. Read this first, then the part of `LOG.md` your package points to. `LOG.md` is the full history with every measurement; this file is only what is true now and what to do next.

## Goal

A voice call for Moneo that is better than a general voice model for this job, built from small parts that can be swapped (speech-to-text, the "editor" that understands the request, text-to-speech, any chat model behind it), and able to run on a laptop.

The idea that makes it different: **the call keeps one typed object and every spoken word edits it.** For a question the object is a question card (what, where, when) that Moneo's own code answers. For "add this" it is a draft of the record, shown as a card the owner taps. A correction ("no, last month", "not Rewe, Lidl") edits the object instead of starting over. Anything the object cannot hold goes to the chat model, exactly as the call works today.

It has to work for **any owner, not just the first one**: other accents, other ways of asking, other shops, currencies and languages. Rules tuned on one voice do not count as done.

## Where things stand

| Piece | State | Evidence |
|---|---|---|
| Question card, editor rules, exact money in words | built, tested | `lib/ai/voice-query*.ts`, `voice-say.ts`; LOG 8 |
| Code answers call questions before the chat model | built, on by default behind Settings switch "Quick answers in calls" | `voice-fast-path.ts`, `app/api/chat/route.ts`; LOG 10, 12 |
| "Add" draft with corrections, becomes a tap-to-save card | built for one-off spending and income with a known shop or category | `voice-draft.ts`; LOG 14, 15 |
| Real recordings (one speaker: 18 free requests, 30 read sentences, 12 more free) | code handles 6 + 13 + 0 right with a good transcript, 0 wrong after the fixes in LOG 16 | LOG 13, 14, 16 |
| Generated test sets (500 cases) | 99%, but they were too clean: 0 of 18 on first contact with real speech | LOG 8, 13 |
| Speech-to-text on real voice | against the true reading text: Whistle 24% of words wrong, Whisper base.en 12% (working choice), large-v3-turbo 7.5% | LOG 13, 16 |
| A spoken call through GPT-Live with the fast path | **not tested, and will not be soon**: no budget for OpenAI credits; test through a local recogniser and the real chat route instead | LOG 12 |
| Ears (talking over the assistant), local speech engine, changing an answer mid-sentence | not built; only simulated measurements | LOG 2 |

The honest summary: the typed-object idea works and is safe (it hands off instead of guessing), but hand-written rules and a tiny recogniser are not good enough for real speech. The next gains come from real test data, a learned extractor and a better recogniser.

## Rules for working here

1. **Safety of money.** Code on the fast path only reads. A write is only ever a card the owner taps (`agent_proposals`). Never create, change or accept a record from speech directly, and never act on words that are still arriving.
2. **Never guess.** A question is answered by code only when every word is explained; a draft is shown only with a clear amount and something the owner already has. Otherwise return null and let the chat model take the turn. A wrong spoken number is the worst failure there is.
3. **Test backend only.** Follow `AGENTS.md`: use `moneo-qa` (`.env.qa`), never the owner's data. Recordings and transcripts are personal data: keep them outside the repo and commit only scripts and totals.
4. **Real speech is the test.** Generated cases are a regression check. A change counts when it moves the real-recording scoreboard without adding a wrong answer.
5. **Write down what you measured, including failures,** in `LOG.md` as a new numbered section. Say what was not measured.
6. **Commit every small step** on this branch with a plain message. Do not push, open a pull request or merge unless asked.
7. **Keep parts swappable.** Nothing may assume one recogniser, one voice or one chat model.

## How to run things

All from the repo root. `npm ci` first.

| What | Command |
|---|---|
| Unit tests for this work | `npx vitest run lib/ai/voice-query.test.ts lib/ai/voice-draft.test.ts lib/ai/voice-call-controller.test.ts` |
| Everything before finishing | `npm test`, `npm run lint`, `npm run typecheck` (`components/ai-message-remount.test.tsx` sometimes times out under load; rerun it alone) |
| Editor on the generated sets | `node node_modules/tsx/dist/cli.mjs research/voice-duplex/experiments/expE/eval_editor.ts dev\|test\|v2 [--show]` |
| A recording: split, transcribe | `experiments/expF/split.py <audio> <folder>`, then `transcribe.py <folder>` and `reference.py <folder>` (Python with `av`, `silero-vad`, `cactus-compute`, `faster-whisper`) |
| A recording through the editors | `.../expE/real_call.ts <folder>` (questions), `.../expE/real_adds.ts <folder>` (adds) |
| Test backend | `npm run setup -- --qa`, then `npm run qa:auth` |
| Made-up data for it | `node --env-file=.env.qa node_modules/tsx/dist/cli.mjs research/voice-duplex/experiments/expE/voice_test_seed.mts` (`--remove` undoes it) |
| This branch on port 3101, no AI cost | `node research/voice-duplex/experiments/expE/qa_server_3101.mjs` |
| Call turns as text through the real route | `node research/voice-duplex/experiments/expE/call_turns_qa.mjs --owner "first turn" "second turn"` |
| Same server with a real OpenAI key (spoken test, costs money) | `voice_test_server.mjs`, then `voice_test_report.mts` |

## Code map

| File | Job |
|---|---|
| `lib/ai/voice-query.ts` | Question card type, spoken periods to exact dates, card to `investigate()` input |
| `lib/ai/voice-query-editor.ts` | `editCard(base, words)`: rules that edit the card; `leftover` = unexplained words |
| `lib/ai/voice-draft.ts` | `editDraft(base, words)`: draft of an add; `draftProposal()` = input of `transaction_propose` |
| `lib/ai/voice-say.ts` | Money in words, the answer in pieces with the number last, `rebind()` |
| `lib/ai/voice-fast-path.ts` | `voiceFastTurn()`: one call turn for code, or null |
| `lib/ai/voice-fast-answer.ts` | The same in memory, without a database (used for timing; the local engine will need it) |
| `app/api/chat/route.ts` | Tries `voiceFastTurn` on a call turn, makes the card for a draft, falls back to the model |
| `lib/ai/voice-call-controller.ts` | Browser side of today's call; keeps the card or draft between turns |

## Work packages

Each one can be taken by a different agent. "Needs" lists what must exist first. Every package ends with numbers in `LOG.md` and an updated row in the table above.

### A. The real test set and one scoreboard (do this first; everything else is judged by it)

- **Why:** there are 18 real requests from one speaker. That is too few, and one voice says nothing about other people.
- **Build:** a folder layout for recordings outside the repo (`<speaker>/<session>/` with clips, transcript, and a small label file per clip: kind = question / add / other, and the expected card or draft). One command that runs every speaker through recogniser + editors and prints: right, wrong, handed off, per speaker and per kind, plus word errors on the reading script.
- **Other speakers:** `reading-script.md` can be read by anyone. Also add public accented-English speech for the recogniser (for example Common Voice, L2-ARCTIC), and many written phrasings per intent from different writers and models for the editors. Report every score per speaker, never only the average.
- **Done when:** at least 5 speakers and 150 labelled real requests; the scoreboard runs with one command; today's numbers are recorded as the baseline.
- **Needs:** recordings from people. Nothing else.

### B. Learned extractor (replace the hand rules for finding things)

- **Why:** rules went from 0 to 4 of 18 only after seeing the recording. That repeats with every new speaker.
- **Build:** a small model that tags the words of an utterance (amount, shop or description, category, account, period, comparison, metric, "this replaces that") and its kind (question / add / other). Train it on generated speech written in the style of real recordings (long, greetings, restarts, corrections, many personas, currencies and shop lists), labelled by a large model, and on the real set's training half. The rules stay as the checker that turns tags into a card and decides whether code may answer.
- **Sizes to try:** a token classifier around 20 to 60 MB on CPU first; a fine-tuned tiny generative model only if tagging cannot express corrections.
- **Done when:** on real speakers it was not trained on, it beats the rules on handled-right without more wrong answers, and runs in under 20 ms per update on a laptop CPU.
- **Needs:** A.

### C. More kinds of drafts

- Recurring payments ("24 euros a month for ChatGPT" maps to `recurring_propose`), balances ("17 euros in my wallet" maps to `balance_propose`), loans (`iou_propose`), and adds without a known shop ("train ticket to Munich", needs B for the description).
- Fix known: amounts must come from the recogniser's raw words, and the owner's currency wins over a symbol nobody said (LOG 14, the wrong salary card).
- **Done when:** each kind has tests written from real sentences and a row on the scoreboard.
- **Needs:** nothing for recurring and balances; B for free descriptions.

### D. More kinds of questions

- Balances, net worth, recurring list, budget left: thin wrappers over what exists (`getBalances`, `readNetWorth`, `listRecurring`, `budgetProgress`), each with its own one-sentence answer and its own "cannot say this truthfully" exit. Several periods in one question ("this month and last month").
- **Done when:** each has tests and a scoreboard row; wrong answers stay at zero on the real set.
- **Needs:** nothing.

### E. The recogniser

- Bench small recognisers on the real set, not on synthetic speech: Whistle, Whisper tiny/base/small, Moonshine, Parakeet-class models. Measure word errors, key words kept (shops, "euros", "add", numbers) and time, per speaker, streaming where the model allows.
- Try, in this order: the owner's names as hints, a second pass that repairs unexplained words against the owner's names and the card's grammar ("years" after a number where a currency is expected), and adapting a tiny model to one voice from the reading script.
- **Done when:** one recommended default and one "tiny" option, with numbers per speaker, and a single interface so they can be swapped.
- **Needs:** A.

### F. Ears: hearing the owner while the assistant talks

- Real microphone and speaker on `moneo-qa`, at least 3 speakers, 40 corrections and 40 "mm-hm" over playback. Echo cancelling plus a detector that also gets what is being played.
- **Pass bar:** 90% of interruptions caught within 0.4 s, the words of the correction right 85% of the time, false stops at most 10%. Simulated results so far: about 1 s to stop and 6 of 24 missed (LOG 2).
- **Needs:** E for the words; nothing for the harness.

### G. The local engine behind a setting

- Speech in and out on the laptop, emitting the same events the call uses today so the rest does not change. GPT-Live stays selectable (it also speaks German).
- Constraints: route handlers cannot hold a WebSocket and the proxy buffers bodies, so use a side port or run models in the browser; everything listens on 127.0.0.1; the desktop app ships no Python; the repo is MIT, so ship a permissively licensed voice and make Woof an optional download.
- **Pass bar:** first sound under 0.5 s median; numbers spoken right 99% of the time.
- **Needs:** E, F.

### H. The live card and changing an answer mid-sentence

- Show the card or draft filling in on screen while the owner is still talking. Start speaking the lead-in of an answer early and change what comes next when a correction arrives (`rebind()`), with a check that the number about to be played is the number in the card.
- **Pass bar:** listeners prefer it to stop-and-restart (about 12 listeners); a wrong number reaches the speaker in under 1% of turns.
- **Needs:** G for speech; the on-screen part needs nothing.

### I. Scoring and write-up

- The same script against today's GPT-Live call, measured, not quoted. Public tests for changing your mind mid-sentence and talking over the assistant (Full-Duplex-Bench v3, EchoChain).
- The claim to defend: a full-duplex call whose state is a runnable typed object, updated on every word including overlapped speech, on a laptop CPU, with fewer wrong numbers and better handling of corrections than a general voice model. Earlier work to credit is listed in LOG 5 and 7.
- **Needs:** A to H as far as they got.

## Order and what can run side by side

```
A (test set) ──┬─> B (learned extractor) ──> C free descriptions
               └─> E (recogniser) ──> F (ears) ──> G (local engine) ──> H (mid-sentence)
C recurring + balances, D questions: can start now, in parallel
I: last
```

## Open decisions for the owner

- Decided 2026-11-16: no OpenAI credits; friends will record the four-part script; the order of packages is the researcher's call; open models may be downloaded and fine-tuned.
- Whether the fast path should go to `main` on its own before the rest (it is behind a Settings switch).

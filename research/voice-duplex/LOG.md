# Self-hosted full-duplex voice for Moneo: research log

Goal: a modular, plug-and-play voice call pipeline that we own, English first, that beats OpenAI's GPT-Live for Moneo's use (fast first reaction, real full-duplex behaviour, exact money facts, no per-minute cost). Everything here was run on the dev laptop (i5-12450HX, 16 GB RAM, RTX 3050 6 GB, Windows 11) unless noted.

What is true now and what to do next is in `ROADMAP.md`; this file is the history, one numbered section per step. Scripts and results are in `experiments/`. Recordings of real people are never in the repo.

Status key: DONE = measured, PARTIAL = measured with known gaps, TODO = not yet run.

## 1. What exists (literature and products)

| System | Kind | What it shows | Why it is not our answer as is |
|---|---|---|---|
| OpenAI GPT-Live (`gpt-live-1`) | Closed speech-to-speech + delegation | About 0.8 s turn-taking latency (read off OpenAI's chart), delegates real work to a backend model | Paid per minute, cloud only, rephrases results, waits for end of turn before delegating |
| NVIDIA NemotronLabs VoiceChat (11B, Sept 2026) | Open full-duplex with tool calls | Best open full-duplex model; 448 ms smooth-turn latency | H100-class GPU, English, tool arguments right only 42%, no barge-in during tools, 2 min audio context |
| Moshi, PersonaPlex (7B) | Open full-duplex | 170-265 ms latency, good interruptions | Weak task quality (VoiceBench about 30), no tools |
| DuplexCascade (Interspeech 2026) | Cascade made full-duplex | Fine-tuning only a text LLM with control tokens (wait / respond / interrupted / backchannel) on 0.6 s micro-turns beats Moshi and PersonaPlex at turn-taking | Needs a 7B LLM in the loop every 0.6 s |
| Voice-Light (arXiv 2609.20995) | Cascade with speculative generation | Private speculative replies promoted after transcript checks: 667 ms vs 1,513 ms median | Two A10 GPUs, English, does not address tool side effects, learned turn head failed (12.5% recall) |
| LTS-VoiceAgent (EMNLP 2026) | "Think while listening" | Semantic trigger starts reasoning on prefixes | No tools, no numbers available to us |
| Talker-Reasoner, ConvFill, Ping-Ponder | Fast talker + slow reasoner | Filler speech hides reasoner latency | Talker is still an LLM |
| Full-Duplex-Bench v1/1.5/3 | Benchmarks | Pause handling, backchannels, interruptions, tool use under disfluency; self-correction is the most common failure everywhere | Use as our evaluation later |

Conclusion: nothing open beats GPT-Live out of the box on a laptop, and fine-tuning a 7-11B full-duplex model is out of reach. The open gap is a cascade of tiny parts that behaves full-duplex and starts the real work early.

## 2. Building blocks measured

### Speech to text: Cactus Whistle (16.9 MB, Apache-2.0) — DONE
- 3-5 s English sentences transcribed in 84-158 ms on CPU, 0 errors on plain questions.
- Streaming mode gives committed words plus a pending tail every 0.5 s chunk (pass 30-900 ms).
- German: 3 of 4 sentences perfect, one word wrong ("in bar" heard as "in Bahn").
- Weak spots: "net worth" heard as "network" (fixable with the `keywords` bias list), inconsistent number formatting ("3,400 and 80").
- `embed()` is not free: 94-162 ms per second of audio when measured while other jobs ran, and it slows a lot above 64 frames. Needs a clean re-measure.
- Alternative: Underdog Bark 3.0 (163 MB, English, 4.94% WER, non-commercial licence). Not tested.

### Text to speech — DONE
| Model | Size | Licence | First sound (CPU) | Speed | Numbers spoken right (30 money sentences) |
|---|---|---|---|---|---|
| Underdog Woof, stock | 40.6 MB | non-commercial, no training on it | 1.2-3.3 s (not streaming) | 1.2-4.7x real time | - |
| **Underdog Woof, our streaming wrapper** | 40.6 MB | same | **80-100 ms** (1-2 threads) | 2.3-3.1x | **26-28 of 30** by transcript; most "misses" are Whistle writing "3,400 and 80" for a correct reading |
| Kyutai Pocket TTS 100M | 219 MB | CC-BY 4.0 | 68-127 ms | 3.2x | 27 of 30 (one real error: 126 said as 226) |
| Kyutai Pocket TTS 18M | small | CC-BY 4.0 | 76-85 ms | 3.0x | **20 of 30** (312.40 said as 31.40, 126 as 26, 17,903 as 1,793) |

- Streaming Woof is exact: its decoder is causal, so chunked decoding gives the same samples as whole-sentence decoding (difference about 1e-7). Schedule 1,2,4,8 frames had no playback underruns.
- Important: numpy's default thread count made Woof 8x slower here (800 ms to first sound). Pin to 1-2 threads.
- Woof's retry of bad takes cannot work once audio is playing; a streaming guard is needed (TODO: read `expC` guard results).
- Woof is English only (a Japanese model exists). Pocket TTS covers German, French, Spanish, Italian, Portuguese, Dutch.

### Thinking model (Moneo chat) — DONE
ChatGPT-plan models through Moneo's own sign-in, time to first word for a one-sentence reply:
`gpt-6-luna` 1.45-2.3 s, `gpt-6.1-sol` 2.3-2.7 s, `gpt-reserve` 1.6-2.1 s, `gpt-5.6-luna` 1.5-2.3 s.
So no cloud model can be in the instant path. The instant path has to be local and tiny; the cloud model must be started early.

### Router ("what kind of utterance is this?") — DONE (dataset is synthetic)
700 spoken-style call utterances generated with GPT through the plan proxy (500 train, 200 test from separate personas), 7 routes: delegate-read, delegate-write, correction, meta, small talk, backchannel, incomplete.

| Router | Size | Accuracy | Money questions dropped | Wasted cloud calls | Median time |
|---|---|---|---|---|---|
| Hand-written rules | few KB | 86.5% (v2 88.5%) | 11% (v2 7%) | 3% | 0.02 ms |
| **TF-IDF + logistic regression** | 4.6 MB | 86% | **0 of 88** | **0 of 30** | **0.5 ms** |
| TF-IDF + linear SVM | 4.6 MB | 84.5% | 0 | 0 | 0.5 ms |
| Tiny hashed n-gram net | 0.5 MB | 75% | 1% | 33% | 0.1 ms |
| Needle 3 embeddings + LR | 36 MB | 79% | 1% | 7% | 95 ms |
| Needle 3 zero-shot tool call | 35 MB | 21% | 27% | 67% | 1.2 s |
| LFM2.5-350M prompted | 379 MB | 32% | 0 | 100% | 8.6 s |
| Gemma 3 270M prompted | 292 MB | 11.5% | 0 | 0 | 0.6 s |
| SmolLM2-360M prompted | 386 MB | 23% | 0 | 73% | 0.4 s |
| gpt-6-luna prompted (ceiling) | cloud | 97.5% | 0 | 0 | 1.8 s |

- Tiny generative models are bad zero-shot routers and slow. A classical classifier is 1,000x faster and far better.
- Caveat: 5-fold cross-validation on the train set gives TF-IDF only about 72%, so the test set is on the easy side. More training data or distilling gpt-6-luna labels is the obvious next step.
- The weak class is "incomplete" (F1 0.55 from text). That is a job for the audio turn detector, not the text router.
- Early decision: TF-IDF knows "this is a money request" a median of 3 words before the user finishes (85% of the time at least 1 word early). Topic is known a median of 4 words early by keyword rules, but the rules are only 59% right on topic.

### Turn detector — DONE (first pass)
Same 1,583 English test clips of the Smart Turn v3.2 data (heads trained on about 6,300 English train clips, Whistle encoder frozen):

| Detector | Trained weights | Accuracy | Cuts the user off | Misses the end | CPU time |
|---|---|---|---|---|---|
| Smart Turn v3.2 (8 MB ONNX, BSD-2) | 8M | **95.0%** | 6.4% | 3.6% | 20 ms + 6 ms features |
| Whistle frames, linear on mean + last + last-5 | 1,537 | 92.7% | 9.3% | 5.1% | under 1 ms on top of Whistle |
| Whistle frames, small GRU | 105K | 91.9% | 10.2% | 6.0% | under 1 ms on top of Whistle |

- Smart Turn wins by about 2 points, with a home advantage (it was trained on the full train split of this data). Use it for now.
- A 1.5K-weight probe at 92.7% shows Whistle's features already carry most of the turn signal. More data, an unfrozen top layer and finance-call audio are the way to close the gap; that is a laptop-sized fine-tune.
- Training of larger heads was stopped early to free the CPU; `expA\train_results.json` has the five finished heads.
- Not tested: telling a backchannel from the start of an interruption from audio alone.

### Small-talk talker — DONE (zero-shot)
30 small-talk lines, blind-judged by gpt-6-luna (1-5):

| Model | Size | Whole reply (CPU) | Naturalness | Fits what was said | Unsafe flags |
|---|---|---|---|---|---|
| gpt-6-luna (cloud) | - | 2,026 ms | 4.83 | 4.73 | 2 |
| Gemma 3 270M Q8 | 292 MB | **94 ms** | 3.23 | 2.77 | 1 |
| LFM2.5-350M Q4 | 229 MB | 176 ms | 3.03 | 2.53 | 3 |
| SmolLM2-360M Q8 | 386 MB | 236 ms | 2.77 | 2.17 | 5 |

- The tiny models are 10-20x faster than the cloud model but often parrot the user or say something odd ("luckycardwin", emojis). Not usable as is.
- This is the best fine-tuning target found: distil a few thousand gpt-6-luna small-talk replies into Gemma 270M or LFM2.5-350M with LoRA on the 6 GB GPU. Not run yet.

### End-to-end timing — DONE (simulated calls)
26 recorded utterances (10 money questions, 6 money statements, 6 small talk, 4 with a 0.7 s pause in the middle), played in real time into the full loop: Whistle, TF-IDF router, Smart Turn v3.2, streaming Woof, and gpt-6-luna on the ChatGPT plan as a stand-in thinking model without tools. "Classic" is the same parts with a 600 ms silence endpoint. Times are from the end of the user's speech; the speech-end marker sits a little late, so compare columns rather than reading absolutes.

| Median ms (90th percentile) | Ours | Classic | Classic + filler phrase |
|---|---|---|---|
| Money requests, first sound (17) | **509** (1,719) | 2,509 (3,048) | 778 (1,449) |
| Money requests, answer starts | 2,238 (3,322) | 2,509 (3,048) | 2,509 |
| Small talk, first sound (8) | 2,638 (4,046) | 2,498 (3,672) | 2,498 |

- First reaction beats GPT-Live's roughly 0.8 s claim at the median, not at the tail.
- Starting the thinking model at the first 160 ms pause only buys about 270 ms on the answer: the model's 1.7 s to a first sentence dominates. Bigger wins need read-only data fetched while the user is still talking (not measured: this run had no tools).
- The acknowledgement audio was ready before the user finished in 11 of 17 money requests.
- Small talk is no faster: it went to the cloud model and the turn check often fell back to the 1.2 s timer.
- Mid-sentence pauses: all 4 paused sentences triggered an early answer (5 early answers in total vs 4 for classic); in 2 of the 4 the final transcript lost half the sentence. Open problem.
- Whistle slows from about 110 ms to about 900 ms once the buffer passes roughly 5 s. Use its streaming mode or a sliding window.
- 6 early thinking-model calls were discarded; 1 call hung for 38 s (needs a timeout and retry).

### Barge-in, echo, backchannels — PARTIAL
72 simulated events (assistant talking; echo through a synthetic room at -10/0/+6 dB relative to the user; one third echo only, one third backchannel, one third real interruption). WebRTC echo canceller from the `livekit` package, Silero VAD, Whistle.

| Policy | False stops on echo | False stops on "mm-hm" | Missed interruptions | Median time to stop |
|---|---|---|---|---|
| a. VAD on raw mic (naive pipeline) | 24 of 24 | 24 of 24 | - | - |
| b. Echo canceller + VAD | 6 of 24 | 13 of 24 | 4 of 24 | 362 ms |
| c. Echo canceller + VAD + our self-speech text filter + backchannel rule | 2 of 24 | 2 of 24 | 6 of 24 | 1,074 ms |

- Echo canceller removed 26-32 dB of echo. Without it nothing works.
- The self-speech filter (ignore words that match what the TTS is saying right now) costs 71 ms per decision and cuts false stops from 40% to 8%, but in v1 it was slow and missed a quarter of interruptions.
- Words spoken over loud echo are transcribed badly (28-94% word errors even after echo cancelling).
- v2 added playback ducking (15 dB down after 150 ms of speech) and instant stop on cue words ("stop", "wait", "no"). It did not help: 6 of 24 interruptions still missed, median stop 1,170 ms, false stops 3 of 24 on echo and 2 of 24 on backchannels. Ducking did make the interruption easier to transcribe (word errors at -10 dB echo: 49% raw, 23% echo-cancelled, 14% with ducking; the same clips alone score 13%).
- By echo level (8 interruptions each, so small numbers): at -10 dB the filter policy missed 1, stopped in 551 ms and false-stopped 1 of 16; plain echo canceller + VAD missed 0, stopped in 297 ms but false-stopped 7 of 16. At 0 and +6 dB both get slow and miss 2-4 of 8.
- The slow part is the VAD noticing the user under the residual echo (duck after a median 350-570 ms), not the text filter (71-137 ms per decision).
- Related work to build on: PACE (arXiv 2608.07631) keeps the conversation history aligned with what was actually played; SoulX-Duplug (arXiv 2603.14877) is an open plug-in semantic VAD.
- Limits: synthetic voices (the small Pocket model mispronounces some lines), synthetic room, and the +6 dB case is harsher than a laptop with browser echo cancelling. Needs a real mic and speaker test.

## 3. Design direction: speculative reflex cascade

1. **Reflex layer (local, CPU, tiny models):** echo canceller, VAD, Whistle, turn head, classical router, phrase bank and streaming TTS. It decides within tens of ms: keep listening, backchannel, duck, stop, acknowledge, hand over.
2. **Speculation:** while the user is still talking, start read-only work (route, topic, prefetch, the cloud request itself on a stable prefix, the acknowledgement audio). Promote it only if the final transcript agrees; otherwise cancel and reuse what is still valid through a tool-result cache. Writes are never speculative: they stay cards the owner taps.
3. **Grounded speech:** amounts come from Moneo's result text and are spoken as written; a checker rejects any spoken number that is not in the result.
4. **Pluggable parts:** STT, TTS, turn head, router, talker and reasoner behind small interfaces (Whistle or Bark; Woof or Pocket; ChatGPT plan, Claude plan, API key or local).

## 4. Open items
- Barge-in is not solved: needs a faster "user is speaking over me" detector and a real microphone and speaker test.
- Mid-sentence pauses: early answers in all 4 test cases.
- Prefetching read-only data during speech, with a tool-result cache, measured against the real Moneo chat agent with tools.
- Fine-tune the small-talk talker (distil gpt-6-luna into Gemma 270M or LFM2.5-350M).
- More router training data (cross-validation is only about 72%).
- Pre-generated acknowledgement audio: only speculative synthesis was measured (ready in time 11 of 17); a phrase bank with no-repeat sampling is untested.
- Whistle sliding window for utterances over 5 s.
- No direct side-by-side with GPT-Live was run; its 0.8 s figure is OpenAI's own.
- Underdog's text models (Woof 2B/4B, Poodle 4B with DFlash speculative decoding, 2.2-4.2x faster on a GPU server) are 1.4 GB+ and MLX/vLLM oriented; not a fit for the CPU reflex layer, possible as an optional local reasoner.
- Licences: Woof and Bark are non-commercial and Woof forbids training on its audio. Fine for a personal open-source app with credit; a problem if Moneo is ever sold. Pocket TTS 100M and Whistle are permissive.

## 5. Second research round (2026-11-15): what is already taken, what is open

Checked each idea from section 3 against the literature. Most of them have close neighbours:

| Idea | Already published | Still open |
|---|---|---|
| Listen with knowledge of own speech | Textual Echo Cancellation (Google 2020, arXiv 2008.06006) uses the TTS text as echo reference; implicit AEC (Amazon, arXiv 2111.10639) feeds the reference into the classifier; Self-Listening (arXiv 2609.05592); PACE | Doing it in a sub-MB head on a 17 MB recogniser on CPU |
| Tiny turn/state head on recogniser features | Voice-Light adapter (183K weights, failed), SoulX-Duplug (0.6B LLM, GPU, user audio only), FSM tapes (arXiv 2609.03321) | Small and CPU, with the playback as input |
| Start work before the sentence ends | Personalized Predictive ASR (Amazon, Interspeech 2023, arXiv 2305.13794) predicts the full utterance and prefetches; Speculative Speech Recognition (arXiv 2407.04641); Endpoint Anticipation (arXiv 2606.13450, 505 ms saved, 28% extra compute); Voice-Light | Tools with read/write safety |
| Local fast path learned from a big model | Distil Labs banking voice assistant (0.35-0.6B tool-calling model distilled offline, about 40 ms); Agentic Plan Caching (NeurIPS 2025); AgenticCache (MLSys 2026, background validator); Google patent on distilling from observed query patterns | Learning online from the owner's own calls with background checking |
| Think between calls | Sleep-time Compute (arXiv 2504.13171); "Don't Wait to Reply" (arXiv 2607.03093) | Voice, personal data, validity tied to data version |
| Per-user turn-taking | Bandit endpointing (arXiv 2303.13407), speaker-embedding endpointing (SLT 2021) | On-device learning from the owner's interruptions |
| Check audio before it plays | Offline round-trip testing (AssemblyAI guide and others); a patent on checking synthesised speech before playback (found by the audit, section 7) | Not a new idea. Only the cheap form is ours: compare digit sequences on a small CPU recogniser |

The open gap that matters most:
- Full-Duplex-Bench v3: when the user changes their mind mid-sentence, even the best system (GPT-Realtime) fails over 40% of the time; cascades commit parameters before the correction arrives.
- "Continue, Adapt, or Yield" (arXiv 2609.13117): in real conversations people keep talking and fold in the listener's correction 68% of the time; PersonaPlex does so 35%. It is an evaluation paper only. No system was found that changes a data-grounded answer mid-sentence.
- Older academic lineage to credit: incremental dialogue systems with revisable units and self-repair (Skantze and Schlangen; Skantze and Hjalmarsson 2010), and "revise" operations on dialogue programs (SMCalFlow dataflow). Toy domains, turn-based or pre-LLM.

Direction chosen for the next phase (proposal, not built): **live-query duplex**. The call is treated as continuous editing of one typed finance query (metric, category, period, account, comparison) that deterministic Moneo code answers in milliseconds. Tiny local models turn each new word, including words spoken over the assistant, into an edit of that query; speech is a view of the query that can be re-bound mid-sentence. The big model works between calls and on misses.

## 7. Audit of the live-query plan (2026-11-15, one Fable agent, read the codebase)

Verdict: **right direction, wrong plan as written.** Build it as a typed fast path with the chat agent as fallback, not as a new query language.

What the audit corrected:

| Claim in section 5 | Audit finding |
|---|---|
| First incremental edit parsing from streaming speech | Not new: Zhou et al. ACL 2022 (prefix-to-program parsing), TERTiUS (arXiv 2307.04008), Disfl-QA |
| Typed query as dialogue state | Not new: SMCalFlow, Trummer's CiceroDB / voice OLAP, Evizeon (VAST 2017) |
| Speech as a view of data, re-bound mid-sentence | Not new as an idea: Buschmeier et al. SIGDIAL 2012 (incremental generation that adapts while speaking) |
| Live number guard: nothing found | Wrong. A patent covers checking synthesised speech before playback |
| We need a query language | Wrong. `investigationSchema` + `investigate()` + `/money/investigations?query=` already are one |

The claim that survives: **a full-duplex cascade whose dialogue state is an executable typed query, re-evaluated on every recogniser increment including overlapped speech, with in-turn re-binding of data-grounded answers, on a laptop CPU**, scored on FDB-v3 self-correction, EchoChain (arXiv 2604.16456) and Duplex Cue.

Risks named: the headline (edit while the assistant speaks) rests on the weakest measured part (ears: about 1 s to stop, 6/24 missed). Coverage is unmeasured (audit's guess: 45-60% of reads, 30-45% of all delegated requests).

Codebase constraints to respect: route handlers cannot upgrade WebSockets and the proxy buffers bodies (so a sidecar port or in-browser models); everything binds to 127.0.0.1; the desktop app ships no Python (ONNX or native runtimes only); the repo is MIT, so Pocket TTS ships and Woof is an optional download, never training audio; today's call speaks English and German, so GPT-Live stays selectable; record audio only against `moneo-qa`; `investigate()` caveats (one total per currency, null on missing FX, partial coverage) must be spoken; fast answers must be saved to the conversation; the in-memory snapshot is stale after a card is accepted or `sourceRevision` changes.

Four tests that can kill the idea early:

| Test | What | Pass bar | Needs |
|---|---|---|---|
| E1 coverage | Replay real past call requests through the fast path | answers 50%+ of reads and 35%+ of all; same answer as the agent 98%+ | the owner's own data, run by the owner |
| E2 ears | Real microphone, 3+ speakers, 40 corrections + 40 backchannels over playback | 90% of interruptions caught within 400 ms; slot words right 85%+; false stops 10% or less | `moneo-qa`, real devices |
| E3 preference | Clips: re-bind mid-sentence vs stop and restart | listeners prefer re-bind | about 12 listeners |
| E4 editor | Rule editor on streamed word sequences | final card exact 80%+ on self-corrections; wrong-number exposure 5% or less | done first, see section 8 |

## 8. First build slice and E4 (2026-11-15)

Code (in `lib/ai/`, pure functions, no I/O, 14 unit tests):

| File | Job |
|---|---|
| `voice-query.ts` | The question card (metric, categories, merchants, accounts, period, comparison, breakdown), spoken periods resolved to exact dates, and the card as input for the existing `investigate()` |
| `voice-query-editor.ts` | `editCard(base, words so far)`: re-run on every new word. Last mention wins, short follow-ups edit the previous card, full questions start a new one. Returns `leftover`: every word it could not explain. Non-empty leftover means "hand to the chat model" |
| `voice-say.ts` | Exact money in words (bigint, never rounds), the answer as ordered pieces with the number last, and `rebind()`: where speech resumes after the card changed |
| `voice-fast-answer.ts` | `fastAnswer()`: words -> card -> `investigate()` on rows in memory -> sentence, or null |

E4, editor accuracy. Test sets were written by a separate agent that never saw the rules (`experiments/expE`): set 1 has 300 cases (script-assisted, repetitive phrasing), set 2 has 200 hand-written cases in 20 speaking styles. Each case is 1-3 turns with a gold card, or "must hand off".

| Run | Exact | Wrong answer | Handed off | Out-of-scope refused |
|---|---|---|---|---|
| Set 1 odd half, first try | 76% | 3% | 21% | 20/20 |
| Set 1 even half, first try after tuning on the odd half | 82% | 9% | 9% | 18/20 |
| **Set 2, first try after tuning on set 1 (the honest number)** | **88%** | **3%** | **10%** | **40/40** |
| Set 2 self-corrections only, first try | 70% | 3% | 28% | - |
| All sets after tuning on them | 99-100% | 0% | 0-1% | 80/80 |

Reading: rules reach the bar only after seeing the data, so the honest state is 88% exact, 3% wrong. The 80% self-correction bar was missed on the first try (70%). Wrong answers came from three causes, all now fixed and all worth keeping as rules: a follow-up after a handed-off turn, a question with no money verb about an account ("how much is in Revolut" is a balance), and words that are both a verb and a correction ("make that"). A new blind set is needed before quoting a final number, and the learned tagger (later step) has to beat 88/3.

Early cards: in 49-60% of partial transcripts the card is complete but not yet the final one ("how much did I spend" before "on groceries"). So nothing may be spoken the moment a card looks complete; the gate (pause detection) and number-last speech are required, not optional. True wrong-number exposure needs word timings from audio and is not measured yet.

Speed: one edit 0.1 ms. Whole fast path (edit + query + sentence) on a synthetic ledger: 2,000 rows 4 ms median / 9 ms p95; 20,000 rows 33 / 95 ms; 100,000 rows 176 / 505 ms. `investigate()` fingerprints every row it is given (sort + SHA-256), which cost 225 ms at 20,000 rows until the fast path pre-filtered rows by date.

Not covered yet, by design (all go to the chat model): balances, net worth, budgets, recurring, forecasts, averages, exclusions, amount conditions, single transactions, tags, synonyms ("food" for Groceries), every write.

## 9. Plan after the audit

See `plan.html` (rebuilt) for the diagrams. Order:

| Step | Build | Gate (loop until it passes) |
|---|---|---|
| 1 | Card, editor rules, speech, fast answer (done) | E4: 80%+ exact on a fresh blind set, 2% or less wrong, 100% of out-of-scope refused |
| 2 | Fast path inside today's GPT-Live call, text only: answered turns skip `/api/chat`, are saved to the conversation, and fall back on null | E1 coverage on the owner's history; same answer as the agent 98%+; p95 under 150 ms |
| 3 | More card kinds: balances, net worth, recurring, budget left | E1 again: coverage up, wrong answers still 2% or less |
| 4 | Real-microphone harness on `moneo-qa` and the ears (echo cancelling + a model that also hears the playback) | E2 |
| 5 | Local engine behind a setting: Whistle + turn head in, Pocket TTS / Woof out, same event names as today | First sound under 500 ms median; numbers spoken right 99%+ |
| 6 | Re-bindable speech and the number guard (digit sequence of what was synthesised vs the card) | E3; wrong number reaching the speaker under 1% |
| 7 | Learned editor: distil a small tagger from the rules plus chat-model labels; rules stay as the safety net | Beats rules on a fresh blind set |
| 8 | Scoring: GPT-Live measured on the same script, FDB-v3 self-correction, EchoChain, Duplex Cue; write-up | Better than GPT-Live on self-correction and wrong-number rate |

## 10. Step 2: fast path inside today's call (2026-11-15)

E1 on real history could not run. The owner said yes, but there is nothing to mine on this laptop: the desktop app's database is new (0 questions, 0 transactions, 12 default categories) and no older `moneo` database or volume exists. `coverage.mts` / `coverage_desktop.mts` are ready (read-only transaction, counts only) and ran clean on both the test backend and the desktop database. Coverage will instead collect from real calls: every turn code answers is saved with the tool name `voice_fast_answer`, so the same script can count them later.

Built:

| Where | Change |
|---|---|
| `lib/ai/voice-fast-path.ts` | `voiceFastPath()`: owned names + last card -> `editCard` -> `runInvestigation` (the same function as the chat tool `finance_investigate`) -> one sentence with digits. The card coming back from the browser is dropped unless every name on it is still the owner's |
| `app/api/chat/route.ts` | A call turn tries the fast path first, under the same data-scope checks as the tools (`accounts` + `transactions`). Null or any error: the model runs as before. Skipped while another request of the call is still running |
| `lib/ai/voice-call-controller.ts` | Keeps the card between turns; sends it only when nothing else is running; forgets it after a turn the model answered |
| `lib/ai/voice-call.ts` | The call brief may carry `card`; it is stripped from replayed history like the rest of the brief |

Checked through the real route on the test backend (`qa_server_3101.mjs` + `call_turns_qa.mjs`, dev server, empty workspace, placeholder AI key):

| Turn | Who answered | Time |
|---|---|---|
| "how much did I spend on groceries last month" | code | (first hit compiles) |
| "no wait, this month" | code, edited the card | 627 ms |
| "and compared to the month before?" | code | 484 ms |
| "how many transactions were there this year" | code, new card | 341 ms |
| "what is my net worth" | handed to the model (fails there by design: placeholder key) | - |
| "and last year?" right after | handed to the model, not answered from the stale card | - |

These are dev-server times including sign-in, the request claim and the save; the model path measured 1.5-2.7 s to first token alone in section 2. Not yet measured: a production build, a workspace with real rows, and a real call with GPT-Live speaking the result.

Open before this could merge to main: whether the fast path should sit behind a Labs setting (it replaces the model's wording with one fixed sentence on covered turns), and a link to the investigation page in the saved answer.

## 11. Sources
- GPT-Live: https://openai.com/index/introducing-gpt-live-1-in-the-api/ , https://community.openai.com/t/introducing-gpt-live-1-in-the-api/1396471
- NemotronLabs VoiceChat: https://arxiv.org/abs/2609.21967
- DuplexCascade: https://arxiv.org/abs/2603.09180
- Voice-Light: https://arxiv.org/abs/2609.20995
- LTS-VoiceAgent: https://aclanthology.org/2026.emnlp-main.1712/
- Full-Duplex-Bench v3: https://arxiv.org/abs/2604.04847 ; v1.5: https://arxiv.org/abs/2507.23159
- Speech-to-speech review: https://ai.ksopyla.com/posts/voice-to-voice-models-2026-review/
- Whistle: https://huggingface.co/Cactus-Compute/whistle , https://cactuscompute.com/blog/whistle
- Woof: https://huggingface.co/ConwayResearch/Underdog-Woof ; Underdog models: https://huggingface.co/ConwayResearch
- Pocket TTS: https://github.com/kyutai-labs/pocket-tts
- Smart Turn v3: https://huggingface.co/pipecat-ai/smart-turn-v3
- Needle: https://cactuscompute.com/blog/needle ; LFM2.5-350M: https://www.cactuscompute.com/blog/lfm2-5-350m
- Audit additions: Zhou et al. ACL 2022 (online semantic parsing for latency reduction); TERTiUS https://arxiv.org/abs/2307.04008 ; EchoChain https://arxiv.org/abs/2604.16456 ; Buschmeier, Baumann, Dosch, Kopp, Schlangen, SIGDIAL 2012; Trummer, CiceroDB; Evizeon (IEEE VAST 2017); Disfl-QA

## 12. Settings switch and a real-voice test setup (2026-11-16)

- "Quick answers in calls" switch in Settings, on by default (`voice_quick_answers`, migration `202611150001`). The migration was run on the test database inside a rolled-back transaction: default on, false and true save, a text value is refused.
- Voice test setup in `experiments/expE`: `voice_test_seed.mts` adds a year of made-up spending (364 rows, marked "Voice test", `--remove` deletes them) to the test backend's owner workspace; `voice_test_server.mjs` runs this branch on port 3101 with a real OpenAI key; `voice_test_report.mts` lists each call turn, who answered and how long it took.
- Checked with text turns on the seeded workspace: "groceries last month" EUR 428.68, "no wait, this month" EUR 219.34, "and compared to the month before?" EUR 219.34, EUR 209.34 less than last month, "how many transactions this year" 306. Code turns took 0.8-0.9 s on the dev server once compiled.
- Blocked: the OpenAI API key on this laptop has no credits left, so GPT-Live cannot start a call and the chat model cannot answer the handed-off turns. A spoken test through GPT-Live needs credits. Recorded voice clips do not: they can go through the local recogniser (Whistle) and the editor.

## 13. First real recording from the owner (2026-11-16)

One 251 s phone recording with 18 spoken requests, said the way the owner really talks. Audio and transcripts stay outside the repo (`<home>/moneo-voice-tests/call1`); only the scripts are here (`experiments/expF`, `experiments/expE/real_call.ts`). Split at pauses of 1 s with Silero VAD, transcribed with Whistle and, as a reference, Whisper large-v3-turbo (no human transcript exists, so "difference" below is against that reference, which favours Whisper-family models).

What the owner actually asks (18 requests):

| Kind | Count | Examples (paraphrased) |
|---|---|---|
| Add something | 8 (44%) | "I spent five euros at Rewe for groceries, add that", a subscription, a train ticket, cash in the wallet, a salary |
| Read a number | 5 (28%) | spending this month, last month, earnings from salary this year, groceries for the whole year |
| What-if, advice, other tools | 5 (28%) | change jobs, cancel one subscription for another, "is this above my median", check the broker |

Findings:

1. **Coverage on first contact was 0 of 18**, even with the reference transcript, against 99% on the generated test sets. Nothing was answered wrongly; every turn was handed to the chat model. The generated sets were too clean: real requests are long (mean 8 s, up to 18 s), start with a greeting or context, and use "can you check / find out / I wanted to ask".
2. **Every self-correction in the recording (4 of 4) was inside an "add" request**: the shop ("not Rewe, Lidl"), the amount twice ("not ten, twenty", then in a new turn "not even twenty, twenty-four"), and an estimate replaced by the real price. The question card cannot hold any of these. The thing people correct mid-sentence is the draft of a new record, not the query.
3. **A wrong answer the generated sets never produced**: after "earnings from salary this year", the whole question "can you check how my groceries are for the whole year" was treated as an edit of the previous card and became "income, Groceries". Fixed: a whole question without a money verb starts a new card. Also fixed: "this month and also last month" is two questions and goes to the chat model; "the whole year" is a period; "wanted, ask, check, find, exactly" are harmless words.
4. After these fixes code answers 4 of 18 with the reference transcript (3 of 18 with Whistle), 0 wrong. That is 4 of the 5 read requests; the fifth asks for three periods at once.
5. **Whistle on this voice**: 19% of words differ from the reference and only 75% of the key words survive (shop names, "euros", "add", "spend", "earn"). Examples: "Rewe" heard as "ravage", "euros" as "years", "add that" as "had that", "how much did I spend" as a person's name, "earn" as "own". Time is 18-20% of the clip length (1.3-3.5 s for 8-18 s clips, not streaming).

| Recogniser | Words that differ | Key words kept | Time vs audio |
|---|---|---|---|
| Whistle (17 MB) | 19% | 75% | 20% |
| Whisper tiny.en | 17% | 82% | 15% |
| Whisper base.en | 11% | 90% | 18% |
| Whisper small.en | 7% | 92% | 51% |
| Distil-Whisper small.en | 12% | 80% | 51% |

What this changes in the plan:

- Step 3 becomes **draft cards for adding things** (amount, shop, category, date, account), edited by speech the same way and shown as a card the owner taps. This is where the real corrections are, and it is 44% of requests. Balances and net worth move behind it.
- The recogniser has to be chosen on real recordings, not on synthetic speech. Whistle alone is not good enough for this voice today; options are a bigger small model (base.en is clearly better at the same speed), or adapting Whistle to the owner's voice and names.
- Real recordings are now the test set. Generated sets stay as a regression check only.

## 14. Decision after the real recording, and the "add" draft (2026-11-16)

Order chosen from the evidence in section 13:

1. **Draft cards for "add" requests** first: 44% of real requests and all real corrections.
2. **A learned extractor** next. Hand rules went from 0 to 4 of 18 only after seeing the recording; that will repeat with every new speaker and phrasing. A small tagger trained on generated speech in the style of real recordings does the finding; the rules stay as the check that decides whether code may answer.
3. **The recogniser** in parallel: Whisper base.en is the working choice for the local path; Whistle stays pluggable and is the candidate for adapting to the owner's voice once recordings with a known text exist.

Two safety levels follow from what a mistake costs: a spoken number that is wrong goes unnoticed, so a read is answered by code only when every word is explained; a draft is shown as a card the owner must tap, so an add needs a clear amount and a shop or category the owner already has.

Built: `lib/ai/voice-draft.ts` (`editDraft`, `draftProposal`), pure, 3 tests written from the owner's own sentences. The draft has the fields of the chat tool's transaction card (`transaction_propose`), so it can become that card. Last thing said wins, in the same turn or the next one.

On the real recording's 8 add turns (`real_adds.ts`):

| Request | Reference transcript | Whistle transcript |
|---|---|---|
| 5 euros, shop corrected to another shop, groceries | card, correct | card, correct (the misheard first shop did not matter: the corrected one was heard) |
| 100 euros at an online shop | card, correct | card, correct |
| small amount for laundry | chat model (no shop or category the owner has) | chat model |
| monthly subscription, amount corrected twice over two turns | chat model (recurring payment: another card type) | chat model |
| train ticket, estimate then real price | chat model (nothing names it) | chat model |
| cash in the wallet | chat model (a balance, not a purchase) | chat model |
| salary received | **wrong card**: the reference recogniser wrote the spoken amount as dollars and cents | chat model (amount heard as two numbers) |

So with reads and adds together, code handles 6 of 18 correctly with the reference transcript (plus 1 wrong card caused by how the recogniser wrote a number) and 5 of 18 with Whistle (0 wrong). The wrong card shows that amounts need the recogniser's raw words, not its formatted digits, and that the owner's currency should win over a symbol nobody said.

Not built yet: the route creating the card from a draft, recurring and balance drafts, free descriptions ("laundry", "train ticket"), which need the learned extractor.

## 15. "Add" drafts in the call (2026-11-16)

`voiceFastTurn()` now gives a call turn to code in two cases: a question it can answer, or something to add that it can draft. For a draft the route makes the same card as the chat tool `transaction_propose`; a correction first dismisses the card it corrects (`replaceCallTask`). Nothing is saved until the owner taps.

Through the real route on the test backend (dev server, made-up data, no AI call involved):

| Turn | Result | Time |
|---|---|---|
| "I spent 5 euros at Rewe for groceries, can you add that" | card: Spent EUR 5.00 · Rewe groceries | (first hit compiles) |
| "oh sorry it was not Rewe, it was Lidl" | first card dismissed; card: Spent EUR 5.00 · Lidl groceries | under 1 s |
| "no, it's not even 5, it's 7" | second card dismissed; card: Spent EUR 7.00 · Lidl groceries | 0.44 s |
| "how much did I spend on groceries this month" | EUR 219.34 | 0.31 s |

In the database afterwards: two cards `rejected`, one `proposed`.

## 16. Second and third real recordings (2026-11-16)

Two more phone recordings from the owner, kept outside the repo (`<home>/moneo-voice-tests/call2`, `call3`):

- **Reading script** (162 s, part 1 of `reading-script.md`): 30 known sentences. Split at 1 s pauses into 26 clips; four clips hold two sentences each, because the reader did not always pause.
- **Free talk** (180 s): 13 utterances, one of them filler. New kinds of request: lending money to a friend and finding that transfer, total and "liquid" money, income other than the job, cash spent or received (also in pounds), and a bill split 75/25.

**Recognisers against the true text** (first measurement with a human-known transcript; `expF/score_reading.py` compares the whole reading as one text, numbers as words):

| Recogniser | Word errors | Key words kept | Time vs audio (2 threads) |
|---|---|---|---|
| Whistle (17 MB) | 24.1% | 66% | 8% |
| Whisper tiny.en | 18.7% | 82% | 8% |
| Whisper base.en | 12.2% | 87% | 17% |
| Whisper small.en | 11.9% | 84% | 65% |
| Distil-Whisper small.en | 15.0% | 74% | 51% |
| Whisper large-v3-turbo (reference) | 7.5% | 88% | not measured (not a candidate) |

Even the reference hears "Lidl" as "little", "ChatGPT" as "TRGBT", "cancel" as "dance", "went" as "wind". On free talk (against the reference, so a lower bound) base.en and small.en differ by 21% and 18%, Whistle by 27%, with only 44% of key words kept. **base.en stays the working choice**: as good as small.en at a quarter of the time. Whistle is not usable on this voice without adaptation.

**Editors on the new recordings**, before the fixes below:

| | Reference transcript | Whistle |
|---|---|---|
| Reading: questions answered by code | 7 of 10, all correct | 2 |
| Reading: add cards | 6 of 10 (1 amount wrong) | 4 (1 amount wrong) |
| Reading: changing your mind (5, merged into 3 clips) | 0, handed off | 0 |
| Reading: chat-only (5) | 0, handed off | 0 |
| Free talk | 1 card, **wrong** | 0 |

Three wrong results, each a real failure mode the generated sets never had:

1. **Whistle: "Yesterday I spent nine years at Starbucks"**: the question editor took "yesterday" and then "nine years" as two periods, kept the last, and would have *spoken* spending at Starbucks on coffee for the last 108 months. Fixed: a second period without a correction ("no", "I mean", "not X, Y") or comparison leaves the turn to the chat model, and a statement "I spent / paid / got" with no question word is never a question.
2. **"Twelve euros fifty"** became a 12 € card (both recognisers). Fixed: a number under 100 right after the currency word is cents.
3. **Free talk: a 20 € bill split 75/25** became a 20 € card at Rewe; the owner's share was 15 € from the wallet. Fixed: any number left standing besides the amount (unless a correction replaced it), a counted thing ("for 2 people", "75 percent") or a share word sends the draft to the chat model.

Also fixed from section 14: a currency symbol nobody said ("$13.43" for "1,343 euros") no longer overrides the owner's currency.

After the fixes, all three recordings: **0 wrong answers, 0 wrong cards**. With the reference transcript code handles 4 questions + 2 cards (call 1), 7 + 6 (reading), 0 (free talk). Generated sets: 99 / 100 / 98% exact, 0 wrong (the period rule costs 1 to 2 points on corrections without a correction word, such as "the last six months, well, the last three months": safe hand-offs).

The owner said afterwards that the free-talk requests were made up on the spot, not things they need, so they do not change the order of work. They stay in the test set as hand-off checks: code must never answer them.

Not measured: the conversations and interruptions in the new script (no recording yet), speakers other than the owner, and streaming recognition.

The reading script now has four parts for other speakers: the 30 sentences, 21 multi-turn conversations, 8 interruptions read against a second person playing Moneo, and free talk with a one-line note per request.

## 17. Every current recogniser on the owner's voice (2026-11-18, in progress)

Market check on 2026-11-18 (web search, model cards): the new small recognisers since the last round are Moonshine v2 Streaming (Jan 2026, 34M/123M/245M, MIT), NVIDIA Nemotron Speech Streaming EN 0.6B and Nemotron 3.5 ASR Streaming 0.6B (multilingual), Parakeet Unified EN 0.6B (Apr 2026) and Parakeet Ultra, Qwen3-ASR 0.6B/1.7B (Jan 2026, Apache-2.0), IBM Granite Speech 5.0 470M TurboCTC (Oct 2026), Cohere Transcribe 2B (Mar 2026), Voxtral Mini 4B Realtime (Feb 2026), and Reson8 Rumble-2-Flash (0.5B, 45 languages, released 2026-11-18). Too large for a laptop and left out: Salesforce SOV-Transcribe-1 (26.6B, 2026-11-12, top open model on Artificial Analysis), Nemotron 3.5 Transcribe (30B), Granite Speech 5.0 2B was not packaged.

Method: one runtime for all of them, **transcribe.cpp 0.4.2** (ggml; 16 families; prebuilt Windows CPU/Vulkan library; GGUF files checked against each family's reference by its authors), Q8_0 weights, CPU, 4 threads (`expG/bench_asr.py`). Each model with and without the owner's names as hints where the model takes them (only Whisper and Qwen3-ASR do in this runtime). Scores on the reading recording, against its true text: word errors (`expF/score_reading.py`) and, more important, **what code does with the transcript** against what it does with the true text (`expG/score_outcomes.ts`, 17 of 26 clips go to code with the true text). Streaming is not a criterion: a batch model can be wrapped.

| Recogniser | Size | Word errors | Key words kept | Code right (of 17) | Wrong card | Handed off | Time vs audio (CPU, 4 thr) |
|---|---|---|---|---|---|---|---|
| Whisper large-v3-turbo + names (reference class) | 886 MB | 7.8% | 88% | 14 | 0 | 3 | see note |
| **Qwen3-ASR 0.6B + names** | 850 MB | **9.9%** | 82% | **12** | 0 | 5 | see note |
| Parakeet Unified EN 0.6B | 731 MB | 9.9% | 81% | 9 | 0 | 7 | 35% |
| Parakeet Ultra | 740 MB | 9.9% | 76% | 9 | 0 | 7 | 56% |
| Parakeet TDT 0.6B v3 (25 languages) | 740 MB | 10.2% | 81% | 10 | 0 | 6 | 29% |
| Qwen3-ASR 0.6B, no hints | 850 MB | 11.2% | 78% | 9 | 0 | 7 | see note |
| Canary 180M Flash | 218 MB | 11.6% | 79% | 10 | 0 | 6 | see note |
| Whisper small.en (faster-whisper, names prompt) | 0.5 GB | 11.9% | 84% | 11 | 0 | 6 | 65% (2 thr) |
| Whisper base.en (faster-whisper, names prompt; today's choice) | 145 MB | 12.2% | 87% | 11 | **1** | 4 | 17% (2 thr) |
| Parakeet TDT 0.6B v2 | 730 MB | 12.6% | 78% | 9 | 0 | 7 | 36% |
| Parakeet TDT-CTC 110M | 135 MB | 12.9% | 76% | 8 | 0 | 8 | 8% |
| Nemotron Speech Streaming EN 0.6B | 730 MB | 15.0% | 74% | 8 | 0 | 8 | 18% |
| Moonshine Streaming Medium | 296 MB | 15.0% | 78% | 7 | 0 | 9 | 19% |
| Granite Speech 5.0 470M TurboCTC | 506 MB | 16.0% | 63% | 9 | 0 | 7 | 13% |
| Moonshine Streaming Small | 199 MB | 16.3% | 74% | 8 | 0 | 8 | 15% |
| Nemotron 3.5 ASR Streaming 0.6B | 751 MB | 19.4% | 74% | 6 | 0 | 10 | 19% |
| Whistle (cactus, names as keywords) | 17 MB | 24.1% | 66% | 5 | 0 | 11 | 8% (2 thr) |
| Moonshine Streaming Tiny | 50 MB | 25.9% | 68% | 7 | 0 | 10 | 2% |

Findings so far:

1. **Hints matter more than model size.** Qwen3-ASR 0.6B goes from 11.2% to 9.9% word errors and from 9 to 12 right with the owner's names as a prompt; large-v3-turbo from 9.9% to 7.8% and 10 to 14. "Lidl", "Rewe", "Netto" and "ChatGPT" are where the misses are. A model that takes hints beats a bigger one that does not.
2. **No wrong spoken answer from any recogniser** after the fixes in this round; the only wrong card is Whisper base/tiny hearing "make that fifty" as "made that 15". The safety design holds; recognition only moves handled-right versus handed-off.
3. **Published scores do not carry over to this voice.** Moonshine Medium (6.65% on the leaderboard average) and Nemotron Streaming (7.4%) are at 15% here; Parakeet and Qwen3 keep closer to their published numbers. One accented speaker is not "other owners", so this must be rechecked with the friends' recordings.
4. Note on times: models that write text token by token (Whisper, Qwen3-ASR, Canary) are very slow in this runtime's Windows build (Whisper base.en 67% here against 17% in faster-whisper; large-v3-turbo 17x slower than real time), apparently because the host-side decoder runs without BLAS. Their times above are not the models' and are measured separately. Transducer and CTC models (Parakeet, Nemotron, Granite TurboCTC, Moonshine) are not affected.
5. Also found: "Where did my money go last month?" right after a restaurants comparison kept the restaurants filter (would have been a wrong spoken answer). Fixed: "where" and "which" start a new card.

Running: Canary 1B v2, Qwen3-ASR 1.7B, Granite Speech 4.1 2B, Cohere Transcribe 2B, Canary-Qwen 2.5B, Voxtral Mini 4B Realtime (on the laptop GPU through Vulkan, accuracy only), then Rumble-2-Flash (Transformers), fair CPU times for the finalists, and the finalists on the two free-talk recordings.

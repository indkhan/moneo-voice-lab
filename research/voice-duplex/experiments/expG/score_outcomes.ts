// What each recogniser's transcript makes code do, against what the true text makes it do:
//   node node_modules/tsx/dist/cli.mjs research/voice-duplex/experiments/expG/score_outcomes.ts <folder with truth.json and hyp_*.json>
// A turn goes to code as in voiceFastTurn(): a question card with no unexplained word, else a ready draft,
// else the chat model. "right" = the same answer or card as from the true text, "wrong" = code answers
// or drafts something the true text would not, "handed off" = the chat model gets a turn code could take.
// Wrong is split by cost: a spoken answer goes unnoticed; a card is seen before the owner taps it, and a
// "thinner" card only lacks a detail (the shop) the true text had, with nothing contradicted.
import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { editDraft, type Draft } from "@/lib/ai/voice-draft";
import type { Card } from "@/lib/ai/voice-query";
import { editCard } from "@/lib/ai/voice-query-editor";

const folder = process.argv[2];
const today = "2026-11-16";
const id = (n: number) => `00000000-0000-4000-8000-${String(n).padStart(12, "0")}`;
const list = (names: string, offset: number) => names.split(", ").map((name, index) => ({ id: id(offset + index), name }));
const entities = {
  categories: list("Groceries, Shopping, Subscriptions, Transport, Restaurants, Coffee, Rent, Salary", 100),
  merchants: list("Rewe, Lidl, Netto, Amazon, Netflix, Spotify, Uber, Shell, Vapiano, Starbucks", 200),
  accounts: list("N26, Cash wallet", 300),
};
type Outcome = { kind: "answer"; card: Card } | { kind: "draft"; draft: Draft } | null;

function call(texts: string[]): Outcome[] {
  let card: Card | null = null, draft: Draft | null = null;
  return texts.map(text => {
    const asked = editCard(card, text, entities, today);
    if (asked.touched && !asked.leftover.length) { card = asked.card; draft = null; return { kind: "answer", card: asked.card }; }
    card = null;
    const added = editDraft(draft, text, entities, today, "EUR");
    draft = added.ready ? added.draft : null;
    return added.ready ? { kind: "draft", draft: added.draft } : null;
  });
}

const truth = JSON.parse(readFileSync(`${folder}/truth.json`, "utf8")) as Record<string, string>;
const clips = Object.keys(truth).sort();
const sources: Record<string, Record<string, string>> = {};
for (const file of readdirSync(folder).filter(name => /^hyp_.+\.json$/.test(name))) sources[file.slice(4, -5)] = JSON.parse(readFileSync(`${folder}/${file}`, "utf8"));
if (existsSync(`${folder}/transcripts.json`)) sources["whistle (cactus)"] = Object.fromEntries((JSON.parse(readFileSync(`${folder}/transcripts.json`, "utf8")) as { clip: string; names: string }[]).map(row => [row.clip, row.names]));
if (existsSync(`${folder}/reference.json`)) sources["large-v3-turbo (faster-whisper)"] = JSON.parse(readFileSync(`${folder}/reference.json`, "utf8"));

const expected = call(clips.map(clip => truth[clip]));
const codeTurns = expected.filter(Boolean).length;
const rows: Record<string, { right: number; "wrong answer": number; "wrong card": number; "thinner card": number; "handed off": number; "wrong clips": string }> = {};
const thinner = (got: Draft, want: Draft) => (Object.keys(want) as (keyof Draft)[]).every(key => got[key] === want[key] || (got[key] === null && ["merchant", "category", "account"].includes(key)));
for (const [name, hyp] of Object.entries(sources)) {
  const got = call(clips.map(clip => hyp[clip] ?? ""));
  let right = 0, answer = 0, card = 0, thin = 0, handed = 0;
  const wrongClips: string[] = [];
  got.forEach((outcome, index) => {
    const want = expected[index];
    if (!outcome) { if (want) handed++; return; }
    if (JSON.stringify(outcome) === JSON.stringify(want)) right++;
    else if (outcome.kind === "draft" && want?.kind === "draft" && thinner(outcome.draft, want.draft)) thin++;
    else { if (outcome.kind === "answer") answer++; else card++; wrongClips.push(clips[index]); }
  });
  rows[name] = { right, "wrong answer": answer, "wrong card": card, "thinner card": thin, "handed off": handed, "wrong clips": wrongClips.join(" ") };
}
const sorted = Object.fromEntries(Object.entries(rows).sort(([, a], [, b]) => a["wrong answer"] - b["wrong answer"] || a["wrong card"] - b["wrong card"] || b.right - a.right));
console.log(`${clips.length} clips, ${codeTurns} of them go to code with the true text`);
console.table(sorted);
writeFileSync(`${folder}/outcome_score.json`, JSON.stringify({ codeTurns, rows: sorted }, null, 1));

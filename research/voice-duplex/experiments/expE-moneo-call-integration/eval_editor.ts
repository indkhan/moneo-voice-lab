// Scores the rule-based card editor on the blind cases. Run from the repo root:
//   node node_modules/tsx/dist/cli.mjs research/voice-duplex/experiments/expE-moneo-call-integration/eval_editor.ts [dev|test|all|v2] [--show]   (dev/test = odd/even ids of set 1; v2 = the hand-written second set)
import { readFileSync, writeFileSync } from "node:fs";
import { editCard } from "@/lib/ai/voice-query-editor";
import type { Card } from "@/lib/ai/voice-query";

const here = "research/voice-duplex/experiments/expE-moneo-call-integration";
const today = "2026-11-15";
const id = (n: number) => `00000000-0000-4000-8000-${String(n).padStart(12, "0")}`;
const list = (names: string, offset: number) => names.split(", ").map((name, index) => ({ id: id(offset + index), name }));
const entities = {
  categories: list("Groceries, Restaurants, Rent, Transport, Travel, Subscriptions, Health, Shopping, Utilities, Salary, Entertainment, Coffee", 100),
  merchants: list("Rewe, Lidl, Amazon, Netflix, Spotify, Uber, Deutsche Bahn, Ikea, Apple, Shell", 200),
  accounts: list("N26, Revolut, Sparkasse Giro, Amex, Cash", 300),
};
type Range = { from: string; to: string };
type Gold = { metric: string; categories: string[]; merchants: string[]; accounts: string[]; period: Range; comparison: Range | null; groupBy: string | null };
type Case = { id: string; kind: string; turns: string[]; gold: Gold | null };

const flat = (card: Card | Gold) => JSON.stringify({
  metric: card.metric,
  ...Object.fromEntries((["categories", "merchants", "accounts"] as const).map(kind => [kind, card[kind].map(item => typeof item === "string" ? item : item.name).sort()])),
  period: [card.period.from, card.period.to], comparison: card.comparison ? [card.comparison.from, card.comparison.to] : null, groupBy: card.groupBy ?? null,
});

function run(turns: string[]) {
  let card: Card | null = null, last = { answerable: false, leftover: [] as string[] };
  for (const turn of turns) {
    const result = editCard(card, turn, entities, today);
    last = { answerable: result.touched && !result.leftover.length, leftover: result.leftover };
    // After a turn the chat model answered, the next short follow-up belongs to the chat model too.
    card = last.answerable ? result.card : null;
  }
  return { card, ...last };
}

const split = process.argv[2] ?? "dev", show = process.argv.includes("--show");
const cases = readFileSync(`${here}/editor_cases${split === "v2" ? "_v2" : ""}.jsonl`, "utf8").trim().split("\n").map(line => JSON.parse(line) as Case)
  .filter(item => split === "all" || split === "v2" || (Number(item.id.slice(1)) % 2 === 1) === (split === "dev"));
const kinds: Record<string, { n: number; exact: number; wrong: number; handoff: number; unstable: number; prefixes: number }> = {};
const started = performance.now();
let edits = 0;
for (const item of cases) {
  const kind = kinds[item.kind] ??= { n: 0, exact: 0, wrong: 0, handoff: 0, unstable: 0, prefixes: 0 };
  kind.n++;
  const result = run(item.turns);
  const outcome = !result.answerable ? (item.gold ? "handoff" : "exact") : item.gold && flat(result.card!) === flat(item.gold) ? "exact" : "wrong";
  kind[outcome]++;
  if (show && outcome !== "exact") console.log(`${item.id} ${item.kind} ${outcome}\n  said: ${item.turns.join(" | ")}\n  left: ${result.leftover.join(" ")}\n  got:  ${result.card ? flat(result.card) : null}\n  gold: ${item.gold ? flat(item.gold) : null}`);
  // Every partial transcript of the last turn: a card that looks answerable but is not the final one would be a wrong number if spoken at once.
  if (item.gold) {
    const words = item.turns.at(-1)!.split(/\s+/), final = result.card ? flat(result.card) : "";
    for (let length = 1; length < words.length; length++) {
      const partial = run([...item.turns.slice(0, -1), words.slice(0, length).join(" ")]);
      edits++;
      kind.prefixes++;
      if (partial.answerable && flat(partial.card!) !== final) kind.unstable++;
    }
  }
}
const elapsed = performance.now() - started;
const total = Object.values(kinds).reduce((sum, kind) => ({ n: sum.n + kind.n, exact: sum.exact + kind.exact, wrong: sum.wrong + kind.wrong, handoff: sum.handoff + kind.handoff, unstable: sum.unstable + kind.unstable, prefixes: sum.prefixes + kind.prefixes }));
const table = Object.fromEntries([...Object.entries(kinds), ["TOTAL", total] as const].map(([name, kind]) => [name, {
  n: kind.n, "exact %": Math.round(100 * kind.exact / kind.n), "wrong answer %": Math.round(100 * kind.wrong / kind.n), "handed off %": Math.round(100 * kind.handoff / kind.n),
  "early card differs %": kind.prefixes ? Math.round(100 * kind.unstable / kind.prefixes) : 0,
}]));
console.table(table);
console.log(`${split}: ${cases.length} cases, ${(elapsed / Math.max(1, edits + cases.length)).toFixed(3)} ms per edit`);
writeFileSync(`${here}/editor_${split}.json`, JSON.stringify(table, null, 2));

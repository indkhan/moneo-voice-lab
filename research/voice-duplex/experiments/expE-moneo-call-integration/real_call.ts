// Runs transcripts of a real recording through the question editor. The transcripts stay outside the repo:
//   node node_modules/tsx/dist/cli.mjs research/voice-duplex/experiments/expE-moneo-call-integration/real_call.ts <folder with transcripts.json and reference.json>
// Prints, per clip and per recogniser, whether code would answer or hand off, and the words it could not explain.
import { readFileSync } from "node:fs";
import { editCard } from "@/lib/ai/voice-query-editor";
import type { Card } from "@/lib/ai/voice-query";

const folder = process.argv[2];
const id = (n: number) => `00000000-0000-4000-8000-${String(n).padStart(12, "0")}`;
const list = (names: string, offset: number) => names.split(", ").map((name, index) => ({ id: id(offset + index), name }));
const entities = {
  categories: list("Groceries, Shopping, Subscriptions, Transport, Restaurants, Coffee, Rent, Salary", 100),
  merchants: list("Rewe, Lidl, Netto, Amazon, Netflix, Spotify, Uber, Shell, Vapiano, Starbucks", 200),
  accounts: list("N26, Cash wallet", 300),
};
const whistle = JSON.parse(readFileSync(`${folder}/transcripts.json`, "utf8")) as { clip: string; names: string; audio_s: number; names_ms: number }[];
const reference = JSON.parse(readFileSync(`${folder}/reference.json`, "utf8")) as Record<string, string>;
const words = (text: string) => text.toLowerCase().replace(/[^a-z0-9 ]+/g, " ").split(/\s+/).filter(Boolean);
function distance(a: string[], b: string[]) {
  const row = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) { let previous = row[0]; row[0] = i;
    for (let j = 1; j <= b.length; j++) { const keep = row[j]; row[j] = Math.min(row[j] + 1, row[j - 1] + 1, previous + (a[i - 1] === b[j - 1] ? 0 : 1)); previous = keep; } }
  return row[b.length];
}
const show = (card: Card) => `${card.metric} ${[...card.categories, ...card.merchants, ...card.accounts].map(ref => ref.name).join("+") || "all"} ${card.period.say}${card.comparison ? ` vs ${card.comparison.say}` : ""}${card.groupBy ? ` by ${card.groupBy}` : ""}`;
let errors = 0, total = 0;
const latency: number[] = [];
const cards: Record<string, Card | null> = { whistle: null, reference: null };
for (const row of whistle) {
  const texts = { whistle: row.names, reference: reference[row.clip] };
  const refWords = words(texts.reference);
  if (refWords.length > 2) { errors += distance(words(texts.whistle), refWords); total += refWords.length; latency.push(row.names_ms / 1000 / row.audio_s); }
  const outcome = (["whistle", "reference"] as const).map(source => {
    const result = editCard(cards[source], texts[source], entities, new Date().toISOString().slice(0, 10));
    const fast = result.touched && !result.leftover.length;
    cards[source] = fast ? result.card : null;
    return `${source.padEnd(9)} ${fast ? `CODE  ${show(result.card)}` : `MODEL (${result.leftover.slice(0, 8).join(" ")})`}`;
  });
  console.log(`${row.clip} ${row.audio_s}s\n  ${outcome.join("\n  ")}`);
}
console.log(`\nWhistle vs reference: ${Math.round(100 * errors / total)}% of words differ (${errors}/${total}); Whistle takes ${(100 * latency.reduce((a, b) => a + b, 0) / latency.length).toFixed(0)}% of the clip's length on average`);

// Times the whole fast path (words -> card -> answer) on a synthetic ledger. Run from the repo root:
//   node node_modules/tsx/dist/cli.mjs research/voice-duplex/experiments/expE-moneo-call-integration/bench_fast_answer.ts
import { writeFileSync } from "node:fs";
import { fastAnswer, type CallData } from "@/lib/ai/voice-fast-answer";
import type { InvestigationRow } from "@/lib/finance/investigation";

const id = (n: number) => `00000000-0000-4000-8000-${String(n).padStart(12, "0")}`;
const list = (names: string, offset: number) => names.split(", ").map((name, index) => ({ id: id(offset + index), name }));
const entities = {
  categories: list("Groceries, Restaurants, Rent, Transport, Travel, Subscriptions, Health, Shopping, Utilities, Salary, Entertainment, Coffee", 100),
  merchants: list("Rewe, Lidl, Amazon, Netflix, Spotify, Uber, Deutsche Bahn, Ikea, Apple, Shell", 200),
  accounts: list("N26, Revolut, Sparkasse Giro, Amex, Cash", 300),
};
let seed = 7;
const random = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
const pick = <T>(items: T[]) => items[Math.floor(random() * items.length)];
const questions = [
  "how much did i spend on groceries last month", "what did i pay at rewe this year no wait last year", "how much did i earn in the past year",
  "where did my money go in october", "how many times did i pay uber in the last 90 days", "restaurants this month compared to last month",
  "did i save anything this year", "coffee spending by month in the last 6 months",
];
const result: Record<string, unknown> = {};
for (const size of [2_000, 20_000, 100_000]) {
  const rows: InvestigationRow[] = Array.from({ length: size }, (_, n) => {
    const day = new Date(Date.UTC(2026, 10, 15) - Math.floor(random() * 3 * 365) * 86_400_000).toISOString().slice(0, 10);
    const income = random() < 0.05;
    return { id: id(1000 + n), parentId: id(1000 + n), accountId: pick(entities.accounts).id, categoryId: income ? entities.categories[9].id : pick(entities.categories).id,
      merchantId: income ? null : pick(entities.merchants).id, date: day, amountMinor: String(income ? 250000 + Math.floor(random() * 50000) : -Math.floor(100 + random() * 9000)),
      currency: "EUR", status: "posted", kind: "ordinary", tags: [], event: null, reviewReasons: [], version: 1, description: "" };
  });
  const data: CallData = { rows, context: { workspaceId: id(0), capturedAt: "2026-11-15T00:00:00Z" }, entities, today: "2026-11-15" };
  const times: number[] = [];
  for (let round = 0; round < 15; round++) for (const question of questions) {
    const started = performance.now();
    const answer = fastAnswer(null, question, data);
    times.push(performance.now() - started);
    if (!answer) throw new Error(`No answer: ${question}`);
    if (size === 2_000 && round === 0) console.log(`${question}\n  -> ${answer.text}`);
  }
  times.sort((a, b) => a - b);
  result[`${size} rows`] = { "median ms": +times[Math.floor(times.length / 2)].toFixed(1), "p95 ms": +times[Math.floor(times.length * 0.95)].toFixed(1), "max ms": +times.at(-1)!.toFixed(1) };
}
console.table(result);
writeFileSync("research/voice-duplex/experiments/expE-moneo-call-integration/bench_fast_answer.json", JSON.stringify(result, null, 2));

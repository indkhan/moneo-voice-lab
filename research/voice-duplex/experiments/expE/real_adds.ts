// Runs transcripts of a real recording through the "add" draft editor. Transcripts stay outside the repo:
//   node node_modules/tsx/dist/cli.mjs research/voice-duplex/experiments/expE/real_adds.ts <folder with transcripts.json and reference.json>
import { readFileSync } from "node:fs";
import { editDraft, type Draft } from "@/lib/ai/voice-draft";

const folder = process.argv[2];
const id = (n: number) => `00000000-0000-4000-8000-${String(n).padStart(12, "0")}`;
const list = (names: string, offset: number) => names.split(", ").map((name, index) => ({ id: id(offset + index), name }));
const entities = {
  categories: list("Groceries, Shopping, Subscriptions, Transport, Restaurants, Coffee, Rent, Salary", 100),
  merchants: list("Rewe, Lidl, Netto, Amazon, Netflix, Spotify, Uber, Shell, Vapiano, Starbucks", 200),
  accounts: list("N26, Cash wallet", 300),
};
const whistle = JSON.parse(readFileSync(`${folder}/transcripts.json`, "utf8")) as { clip: string; names: string }[];
const reference = JSON.parse(readFileSync(`${folder}/reference.json`, "utf8")) as Record<string, string>;
const drafts: Record<string, Draft | null> = { whistle: null, reference: null };
const counts = { whistle: 0, reference: 0 };
for (const row of whistle) {
  const lines = (["whistle", "reference"] as const).map(source => {
    const result = editDraft(drafts[source], source === "whistle" ? row.names : reference[row.clip], entities, new Date().toISOString().slice(0, 10), "EUR");
    drafts[source] = result.ready ? result.draft : null;
    if (result.ready) counts[source]++;
    const { draft } = result;
    return `${source.padEnd(9)} ${result.ready ? `CARD  ${draft.direction} ${draft.amount} ${draft.currency} · ${draft.merchant ?? "-"} · ${draft.category ?? "-"} · ${draft.account ?? "-"}` : result.adds ? `MODEL (add, unclear: ${result.unclear.join(", ")})` : "MODEL (not an add)"}`;
  });
  console.log(`${row.clip}\n  ${lines.join("\n  ")}`);
}
console.log(`\ncards from code: ${counts.reference} with the reference transcript, ${counts.whistle} with Whistle`);

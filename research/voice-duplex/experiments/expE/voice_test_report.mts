// After a test call on the voice test server: what was asked, who answered (code or the chat model), and how long
// each answer took. Reads the moneo-qa backend only.
//   node --env-file=.env.qa node_modules/tsx/dist/cli.mjs research/voice-duplex/experiments/expE/voice_test_report.mts [hours back, default 6]
import postgres from "postgres";

const sql = postgres(process.env.SUPABASE_DB_URL!, { prepare: false, max: 1 });
const hours = Number(process.argv[2] ?? 6);
const rows = await sql<{ message: string; status: string; answer: string | null; error: string | null; tool_calls: { name?: string }[] | null; seconds: number; created_at: Date }[]>`
  select message, status, answer, error, tool_calls, extract(epoch from (updated_at - created_at))::float as seconds, created_at
  from public.chat_requests where context->>'via' = 'call' and created_at > now() - make_interval(hours => ${hours}) order by created_at`;
await sql.end();
let code = 0;
for (const row of rows) {
  const fast = (row.tool_calls ?? []).some(call => call.name === "voice_fast_answer");
  if (fast) code++;
  console.log(`${row.created_at.toISOString().slice(11, 19)} · ${row.seconds.toFixed(1).padStart(5)} s · ${fast ? "CODE " : "MODEL"} · ${row.status} · "${row.message}"\n   -> ${(row.answer ?? row.error ?? "").replace(/\s+/g, " ").slice(0, 220)}`);
}
const times = (fast: boolean) => rows.filter(row => row.status === "completed" && (row.tool_calls ?? []).some(call => call.name === "voice_fast_answer") === fast).map(row => row.seconds).sort((a, b) => a - b);
const median = (list: number[]) => list.length ? list[Math.floor(list.length / 2)].toFixed(1) : "-";
console.log(`\n${rows.length} call turns · ${code} answered by code (${rows.length ? Math.round(100 * code / rows.length) : 0}%) · median ${median(times(true))} s by code, ${median(times(false))} s by the chat model`);

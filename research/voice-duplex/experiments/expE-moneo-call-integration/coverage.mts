/**
 * E1: how many past questions could the fast path have answered? Read-only, prints counts only, writes nothing.
 * It reads real questions, so the owner runs it (or says yes to running it) on their own data:
 *   node --env-file=.env node_modules/tsx/dist/cli.mjs research/voice-duplex/experiments/expE-moneo-call-integration/coverage.mts [--words]
 * `--words` also prints the 40 most common words the rules could not explain (single words, no sentences).
 */
import postgres from "postgres";
import { editCard } from "@/lib/ai/voice-query-editor";

type Request = { message: string; context: { via?: string } | null; tool_calls: { name?: string }[] | null; created_at: Date; workspace_id: string };
type Name = { id: string; name: string };

const sql = postgres(process.env.SUPABASE_DB_URL!, { prepare: false, max: 1 });
try {
  await sql.begin("read only", async tx => {
    const requests = await tx<Request[]>`select message, context, tool_calls, created_at, workspace_id from public.chat_requests where status = 'completed' order by created_at`;
    const names = async (table: "accounts" | "categories" | "merchants", workspace: string) => tx<Name[]>`select id, name from public.${tx(table)} where workspace_id = ${workspace}`;
    const entities = new Map<string, { accounts: Name[]; categories: Name[]; merchants: Name[] }>();
    const groups: Record<string, { n: number; fast: number }> = {};
    const unexplained = new Map<string, number>();
    const add = (key: string, fast: boolean) => { const group = groups[key] ??= { n: 0, fast: 0 }; group.n++; if (fast) group.fast++; };
    for (const request of requests) {
      if (!entities.has(request.workspace_id)) entities.set(request.workspace_id, { accounts: await names("accounts", request.workspace_id), categories: await names("categories", request.workspace_id), merchants: await names("merchants", request.workspace_id) });
      // Each question alone, as if it opened the call: follow-ups count as not covered, so this is a floor.
      const { leftover, touched } = editCard(null, request.message, entities.get(request.workspace_id)!, request.created_at.toISOString().slice(0, 10));
      const fast = touched && !leftover.length;
      const tools = (request.tool_calls ?? []).map(call => call.name ?? "");
      const wrote = tools.some(tool => /propose|decide|setCategory|order|start/i.test(tool));
      const call = request.context?.via === "call";
      add("all", fast);
      add(call ? "voice call" : "typed chat", fast);
      add(wrote ? "asked for a change" : "read only", fast);
      if (call && !wrote) add("voice call, read only", fast);
      if (tools.some(tool => tool === "finance_investigate" || tool === "analytics_cashflow" || tool === "transactions_search")) add("used a spending tool", fast);
      // A fast answer to something the chat model answered with a write would be a wrong answer.
      if (fast && wrote) add("DANGER: fast path would answer a change request", true);
      if (!fast) for (const word of new Set(leftover)) unexplained.set(word, (unexplained.get(word) ?? 0) + 1);
    }
    console.table(Object.fromEntries(Object.entries(groups).map(([key, group]) => [key, { questions: group.n, "fast path": group.fast, "%": group.n ? Math.round(100 * group.fast / group.n) : 0 }])));
    if (process.argv.includes("--words")) console.log([...unexplained].sort((a, b) => b[1] - a[1]).slice(0, 40).map(([word, count]) => `${word} ${count}`).join(" · "));
  });
} finally {
  await sql.end();
}

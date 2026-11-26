// Runs the voice_quick_answers migration on the test database inside a transaction, checks it, and rolls back.
//   node --env-file=.env.qa node_modules/tsx/dist/cli.mjs research/voice-duplex/experiments/expE-moneo-call-integration/check_migration.mts
import { readFileSync } from "node:fs";
import postgres from "postgres";

const sql = postgres(process.env.SUPABASE_DB_URL!, { prepare: false, max: 1, onnotice: () => {} });
class Rollback extends Error {}
try {
  await sql.begin(async tx => {
    await tx.unsafe(readFileSync("supabase/migrations/202611150001_voice_quick_answers.sql", "utf8"));
    const [user] = await tx`select id from auth.users where email = ${process.env.E2E_USER_EMAIL!}`;
    const [workspace] = await tx`select id from public.workspaces where owner_id = ${user.id} limit 1`;
    await tx`select set_config('request.jwt.claim.sub', ${user.id}, true)`;
    const base = { timezone: "Europe/Berlin", locale: "en-GB", theme: "system", voice: "marin", ai_models: {}, ai_data_scopes: ["accounts", "transactions"], muted_insight_types: [], summary_cadence: "none", summary_time: "09:00" };
    const saved = async (preferences: Record<string, unknown>) => {
      await tx`select public.save_workspace_preferences(${workspace.id}, 'EUR', ${tx.json(preferences as never)})`;
      return (await tx`select voice_quick_answers from public.workspace_settings where workspace_id = ${workspace.id}`)[0].voice_quick_answers;
    };
    console.log("not sent -> ", await saved(base), "| false -> ", await saved({ ...base, voice_quick_answers: false }), "| true -> ", await saved({ ...base, voice_quick_answers: true }));
    await tx.savepoint(async sp => { await sp`select public.save_workspace_preferences(${workspace.id}, 'EUR', ${sp.json({ ...base, voice_quick_answers: "yes" } as never)})`; })
      .then(() => console.log("a text value was accepted (wrong)"), error => console.log("text value refused:", (error as Error).message));
    throw new Rollback();
  });
} catch (error) {
  if (!(error instanceof Rollback)) throw error;
  console.log("rolled back; the test database is unchanged");
} finally {
  await sql.end();
}

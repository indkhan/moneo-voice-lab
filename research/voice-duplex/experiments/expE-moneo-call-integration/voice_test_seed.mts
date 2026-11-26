// Fills the test backend's owner workspace with a year of made-up spending, so a test call has real numbers.
// Only the moneo-qa backend (.env.qa); every row it adds is marked "Voice test" and `--remove` deletes them again.
//   node --env-file=.env.qa node_modules/tsx/dist/cli.mjs research/voice-duplex/experiments/expE-moneo-call-integration/voice_test_seed.mts [--remove]
import postgres from "postgres";

if (!/:5452\d\//.test(process.env.SUPABASE_DB_URL ?? "")) throw new Error("Refusing to run: this is not the moneo-qa database");
const MARK = "Voice test";
const sql = postgres(process.env.SUPABASE_DB_URL!, { prepare: false, max: 1, onnotice: () => {} });
let seed = 11;
const random = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
const day = (ago: number) => new Date(Date.now() - ago * 86_400_000).toISOString().slice(0, 10);

await sql.begin(async tx => {
  const [user] = await tx`select id from auth.users where email = ${process.env.MONEO_OWNER_EMAIL!}`;
  const [workspace] = await tx`select id from public.workspaces where owner_id = ${user.id}`;
  const removed = await tx`delete from public.transactions where workspace_id = ${workspace.id} and description like ${`${MARK}%`}`;
  await tx`delete from public.accounts where workspace_id = ${workspace.id} and name in ('N26', 'Cash wallet') and not exists (select 1 from public.transactions t where t.account_id = accounts.id)`;
  if (process.argv.includes("--remove")) { console.log(`removed ${removed.count} rows`); return; }

  const named = async (table: "categories" | "merchants", name: string) => {
    const [found] = await tx`select id from public.${tx(table)} where workspace_id = ${workspace.id} and lower(name) = ${name.toLowerCase()}`;
    if (found) return found.id as string;
    const [made] = table === "merchants"
      ? await tx`insert into public.merchants(workspace_id, name, normalized_name) values(${workspace.id}, ${name}, ${name.toLowerCase()}) returning id`
      : await tx`insert into public.categories(workspace_id, name) values(${workspace.id}, ${name}) returning id`;
    return made.id as string;
  };
  const account = async (name: string, type: string) => {
    const [found] = await tx`select id from public.accounts where workspace_id = ${workspace.id} and name = ${name}`;
    return (found ?? (await tx`insert into public.accounts(workspace_id, name, type, currency_code) values(${workspace.id}, ${name}, ${type}, 'EUR') returning id`)[0]).id as string;
  };
  const n26 = await account("N26", "checking"), cash = await account("Cash wallet", "cash");
  // merchant, category, times a month, smallest and largest amount in cents
  const habits: [string, string, number, number, number][] = [
    ["Rewe", "Groceries", 6, 1500, 7000], ["Lidl", "Groceries", 4, 1000, 5000], ["Amazon", "Shopping", 2, 1200, 9000],
    ["Netflix", "Subscriptions", 1, 1399, 1399], ["Spotify", "Subscriptions", 1, 1099, 1099], ["Uber", "Transport", 3, 800, 2600],
    ["Shell", "Transport", 2, 3500, 7500], ["Vapiano", "Restaurants", 3, 1500, 4800], ["Starbucks", "Coffee", 5, 350, 750],
  ];
  let count = 0;
  for (const [merchant, category, monthly, low, high] of habits) {
    const merchantId = await named("merchants", merchant), categoryId = await named("categories", category);
    for (let ago = 0; ago < 365; ago++) if (random() < monthly / 30) {
      await tx`insert into public.transactions(workspace_id, account_id, posted_on, description, amount_minor, currency_code, status, category_id, merchant_id)
        values(${workspace.id}, ${merchant === "Starbucks" ? cash : n26}, ${day(ago)}, ${`${MARK} ${merchant}`}, ${-Math.round(low + random() * (high - low))}, 'EUR', 'posted', ${categoryId}, ${merchantId})`;
      count++;
    }
  }
  const rent = await named("categories", "Rent"), salary = await named("categories", "Salary");
  for (let month = 0; month < 12; month++) {
    await tx`insert into public.transactions(workspace_id, account_id, posted_on, description, amount_minor, currency_code, status, category_id) values
      (${workspace.id}, ${n26}, ${day(month * 30 + 3)}, ${`${MARK} rent`}, -95000, 'EUR', 'posted', ${rent}),
      (${workspace.id}, ${n26}, ${day(month * 30 + 5)}, ${`${MARK} salary`}, 320000, 'EUR', 'posted', ${salary})`;
    count += 2;
  }
  console.log(`added ${count} rows to the test workspace`);
});
await sql.end();

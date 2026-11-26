// Runs coverage.mts against the desktop app's own database (the app must be running). Read-only; prints counts.
//   node node_modules/tsx/dist/cli.mjs research/voice-duplex/experiments/expE/coverage_desktop.mts [--words]
import { readFileSync } from "node:fs";
import { join } from "node:path";

const home = process.env.MONEO_DESKTOP_HOME ?? join(process.env.APPDATA ?? "", "Moneo");
const config = JSON.parse(readFileSync(join(home, "config.json"), "utf8")) as { secrets: { POSTGRES_PASSWORD: string }; ports: { db: number } };
process.env.SUPABASE_DB_URL = `postgres://postgres:${encodeURIComponent(config.secrets.POSTGRES_PASSWORD)}@127.0.0.1:${config.ports.db}/postgres`;
await import("./coverage.mts");

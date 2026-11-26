// scripts/qa-server.mjs on port 3101, so it can run beside another checkout's QA server on 3100.
// Same safety: moneo-qa backend from .env.qa, bank access off, a placeholder AI key every provider rejects.
//   node research/voice-duplex/experiments/expE-moneo-call-integration/qa_server_3101.mjs
import { spawn } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { parseEnv } from "node:util";

if (!existsSync(".env.qa")) throw new Error("No .env.qa: run `npm run setup -- --qa` first.");
const qa = parseEnv(readFileSync(".env.qa", "utf8"));
const names = file => existsSync(file) ? [...readFileSync(file, "utf8").matchAll(/^#?\s*([A-Z][A-Z0-9_]*)=/gm)].map(match => match[1]) : [];
const blank = Object.fromEntries([...names(".env.example"), ...names(".env")].map(name => [name, ""]));
const env = {
  ...process.env, ...blank, ...qa, OPENAI_API_KEY: "sk-qa-placeholder-not-a-real-key", OPENROUTER_API_KEY: "",
  ENABLE_BANKING_APP_ID: "", ENABLE_BANKING_PRIVATE_KEY: "",
  PORT: "3101", NEXT_PUBLIC_APP_URL: "http://localhost:3101", WORKFLOW_LOCAL_BASE_URL: "http://127.0.0.1:3101", NEXT_DIST_DIR: ".next-qa",
};
const server = spawn(process.execPath, ["node_modules/next/dist/bin/next", "dev", "-H", "127.0.0.1", "-p", "3101"], { env, stdio: "inherit" });
for (const signal of ["SIGINT", "SIGTERM"]) process.on(signal, () => server.kill(signal));
server.on("exit", code => process.exit(code ?? 0));

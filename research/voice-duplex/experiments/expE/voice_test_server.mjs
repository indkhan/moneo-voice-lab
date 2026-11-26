// This branch on http://localhost:3101 for a REAL voice test: the moneo-qa backend (.env.qa, made-up data from
// voice_test_seed.mts) with a real OpenAI key, so GPT-Live talks and the chat model can answer what code does not.
// Costs money per call. The key is read from the file named in MONEO_VOICE_TEST_ENV and is never printed.
//   MONEO_VOICE_TEST_ENV=<path to a .env with OPENAI_API_KEY> node research/voice-duplex/experiments/expE/voice_test_server.mjs
import { spawn } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { parseEnv } from "node:util";

if (!existsSync(".env.qa")) throw new Error("No .env.qa: run `npm run setup -- --qa` first.");
const qa = parseEnv(readFileSync(".env.qa", "utf8"));
const key = parseEnv(readFileSync(process.env.MONEO_VOICE_TEST_ENV ?? "", "utf8")).OPENAI_API_KEY;
if (!key) throw new Error("MONEO_VOICE_TEST_ENV must name a .env file that has OPENAI_API_KEY");
const names = file => existsSync(file) ? [...readFileSync(file, "utf8").matchAll(/^#?\s*([A-Z][A-Z0-9_]*)=/gm)].map(match => match[1]) : [];
const blank = Object.fromEntries([...names(".env.example"), ...names(".env")].map(name => [name, ""]));
const env = {
  ...process.env, ...blank, ...qa, OPENAI_API_KEY: key, OPENROUTER_API_KEY: "",
  ENABLE_BANKING_APP_ID: "", ENABLE_BANKING_PRIVATE_KEY: "",
  PORT: "3101", NEXT_PUBLIC_APP_URL: "http://localhost:3101", WORKFLOW_LOCAL_BASE_URL: "http://127.0.0.1:3101", NEXT_DIST_DIR: ".next-qa",
};
const server = spawn(process.execPath, ["node_modules/next/dist/bin/next", "dev", "-H", "127.0.0.1", "-p", "3101"], { env, stdio: "inherit" });
for (const signal of ["SIGINT", "SIGTERM"]) process.on(signal, () => server.kill(signal));
server.on("exit", code => process.exit(code ?? 0));

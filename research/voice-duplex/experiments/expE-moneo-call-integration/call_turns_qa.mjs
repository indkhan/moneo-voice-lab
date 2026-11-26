// Sends call turns to the test server (port 3101, moneo-qa backend) the way a voice call does, and prints who
// answered and how fast. The AI key there is a placeholder, so a turn the model has to answer fails by design.
//   node research/voice-duplex/experiments/expE-moneo-call-integration/call_turns_qa.mjs [--owner]
import { randomUUID } from "node:crypto";
import { readFileSync } from "node:fs";

// --owner: the test backend's owner (the app signs them in by itself), whose workspace voice_test_seed.mts fills.
const cookies = process.argv.includes("--owner")
  ? (await fetch("http://localhost:3101/", { redirect: "manual" })).headers.getSetCookie().map(cookie => cookie.split(";")[0]).join("; ")
  : JSON.parse(readFileSync("e2e/.auth.json", "utf8")).cookies.map(cookie => `${cookie.name}=${cookie.value}`).join("; ");
const conversationId = randomUUID();
let card;
// Turns to send: the words after the options, or a default set.
const said = process.argv.slice(2).filter(word => !word.startsWith("--"));
for (const message of said.length ? said : ["how much did I spend on groceries last month", "no wait, this month", "and compared to the month before?", "how many transactions were there this year", "what is my net worth", "and last year?"]) {
  const started = performance.now();
  const response = await fetch("http://localhost:3101/api/chat", { method: "POST", headers: { "Content-Type": "application/json", Cookie: cookies },
    body: JSON.stringify({ conversationId, requestId: randomUUID(), message, context: { via: "call", input: "voice", runningTasks: [], recentTurns: [], answered: [], ...(card ? { card } : {}) } }) });
  const body = await response.json();
  card = body.card;
  console.log(`${Math.round(performance.now() - started)} ms · ${response.status} · ${body.toolsUsed?.some(tool => tool.startsWith("voice_fast")) ? "CODE " : "MODEL"} · "${message}"\n   -> ${(body.answer ?? body.error ?? "").replace(/\s+/g, " ")}`);
}

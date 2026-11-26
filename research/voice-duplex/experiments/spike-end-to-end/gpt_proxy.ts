// Local-only proxy so Python spikes can call the owner's ChatGPT plan through Moneo's own sign-in code.
// Run from the main checkout: node --env-file=.env node_modules/tsx/dist/cli.mjs <this file>
import { createServer } from "node:http";
import { chatgptFetch, chatgptStatus } from "C:/codebases/Moneo/lib/ai/chatgpt";

const port = 8787;
createServer(async (req, res) => {
  try {
    if (req.url === "/status") { const s = await chatgptStatus(); res.end(JSON.stringify({ signedIn: s.signedIn, source: s.source })); return; }
    const chunks: Buffer[] = [];
    for await (const c of req) chunks.push(c as Buffer);
    const path = req.url === "/models" ? "models" : "responses";
    const upstream = await chatgptFetch(`https://api.openai.com/v1/${path}`, path === "models" ? {} :
      { method: "POST", headers: { "content-type": "application/json" }, body: Buffer.concat(chunks).toString() });
    res.writeHead(upstream.status, { "content-type": upstream.headers.get("content-type") ?? "application/json" });
    if (!upstream.body) { res.end(); return; }
    const reader = upstream.body.getReader();
    for (;;) { const { value, done } = await reader.read(); if (done) break; res.write(value); }
    res.end();
  } catch (error) { res.writeHead(500); res.end(String(error)); }
}).listen(port, "127.0.0.1", () => console.log(`proxy on ${port}`));

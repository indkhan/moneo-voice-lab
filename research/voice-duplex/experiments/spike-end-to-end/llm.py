"""Streaming client for the local ChatGPT-plan proxy (spike/gpt_proxy.ts on 127.0.0.1:8787)."""
import json, time, urllib.request

def stream(model, instructions, user, effort="low", url="http://127.0.0.1:8787/responses"):
    """Yields (seconds_since_start, text_delta). Raises on HTTP errors."""
    body = {"model": model, "instructions": instructions, "input": [{"role": "user", "content": user}], "stream": True}
    if effort: body["reasoning"] = {"effort": effort}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"): continue
            try: ev = json.loads(line[5:])
            except Exception: continue
            if ev.get("type") == "response.output_text.delta": yield time.perf_counter() - t0, ev.get("delta", "")
            elif ev.get("type") in ("response.failed", "error"): raise RuntimeError(json.dumps(ev)[:400])

def ask(model, instructions, user, effort="low"):
    first, out, t0 = None, [], time.perf_counter()
    for t, d in stream(model, instructions, user, effort):
        if first is None: first = t
        out.append(d)
    return {"text": "".join(out), "first_ms": round((first or 0) * 1000), "total_ms": round((time.perf_counter() - t0) * 1000)}

if __name__ == "__main__":
    import sys, urllib.error
    SYS = "You are Moneo's voice on a call. Reply in one short spoken sentence."
    for model, effort in [("gpt-6-luna", "low"), ("gpt-6-luna", None), ("gpt-6.1-sol", "low"), ("gpt-reserve", "low"), ("gpt-5.6-luna", "low"), ("gpt-6-luna", "minimal"), ("gpt-6-luna", "none")]:
        for i in range(3):
            try: r = ask(model, SYS, "hey, how's it going?", effort); print(model, effort, r, flush=True)
            except urllib.error.HTTPError as e: print(model, effort, "HTTP", e.code, e.read()[:200], flush=True); break
            except Exception as e: print(model, effort, "ERR", repr(e)[:200], flush=True); break

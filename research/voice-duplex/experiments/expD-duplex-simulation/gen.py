"""Synthesize the corpus: assistant utterances (alba), user interruptions and backchannels (marius/javert/eve).
Everything is saved as 16 kHz mono float32 WAV under corpus/, plus corpus/manifest.json with the TTS chunk schedule."""
import os, json, time, sys
os.environ["CACTUS_TELEMETRY"] = "0"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import numpy as np, soundfile as sf
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "corpus"); os.makedirs(OUT, exist_ok=True)

ASSISTANT = [
    "You spent 312 euros on groceries in October, about 18 percent more than September, and most of it was at Rewe and Netto. Restaurants came to 140 euros, which is roughly the same as the month before.",
    "Your checking account balance is 2,450 euros, and your savings account holds 11,300 euros. Rent of 950 euros went out on the first, so that part is already covered for this month.",
    "I found three subscriptions you might want to review: Netflix at 13 euros, Spotify at 11 euros, and a gym membership at 35 euros. Together they add up to 59 euros every month.",
    "Your net worth went up by about 1,200 euros since last month, mostly because your investment portfolio gained 3 percent. Cash balances stayed almost flat over the same period.",
    "Looking at transport, you spent 86 euros in October, including a monthly ticket for 49 euros and two taxi rides. That is 20 euros below your budget for that category.",
    "Your electricity bill was 74 euros this month, which is 12 euros higher than usual. If that continues, you would spend about 150 euros more over the whole year.",
    "You are on track with your savings goal: you put aside 400 euros in October, and you need 350 euros a month to reach 5,000 euros by next summer.",
    "The largest single expense last week was 229 euros at Media Markt on Thursday. After that came 64 euros at the pharmacy and 58 euros for fuel on Saturday.",
    "Your salary of 3,100 euros arrived on the 28th, two days later than last month. Since then you have spent 640 euros, mainly on rent, insurance and groceries.",
    "Compared with the same month last year, your spending on eating out dropped by 30 percent, while your spending on travel doubled to about 420 euros.",
]
INTERRUPT = [
    "No wait, I meant last month.",
    "Stop, how much was rent?",
    "Hold on, what about my savings account?",
    "Actually, can you show me September instead?",
    "Wait, that's not right, I paid rent twice.",
    "Sorry, how much did I spend on restaurants?",
    "No no, I asked about the credit card.",
    "Okay, but what about insurance?",
    "Stop. Add fifty euros in cash.",
    "Can you repeat that last number?",
    "Wait, which account was that?",
    "Yeah, but how much is left this month?",
]
BACKCH = ["Mm-hm.", "Yeah.", "Okay.", "Right.", "Uh-huh.", "I see.", "Sure.", "Got it.", "Yep.", "Mhm."]
USER_VOICES = ["marius", "javert", "eve"]


def to16k(x24):
    return resample_poly(x24, 2, 3).astype(np.float32)


def main():
    from pocket_tts import TTSModel
    m = TTSModel.load_model(language="english_18m_2026-10")
    sr = m.sample_rate
    man = {"sr_tts": sr, "assistant": [], "interrupt": [], "backchannel": []}
    states = {v: m.get_state_for_audio_prompt(v) for v in ["alba"] + USER_VOICES}

    def synth(voice, text):
        t0 = time.perf_counter(); arr, chunks = [], []
        for c in m.generate_audio_stream(states[voice], text):
            arr.append(time.perf_counter() - t0); chunks.append(c.detach().cpu().numpy().reshape(-1).astype(np.float32))
        return np.concatenate(chunks), [len(c) for c in chunks], arr

    for i, text in enumerate(ASSISTANT):
        x, n, arr = synth("alba", text)
        p = os.path.join(OUT, f"asst_{i:02d}.wav"); sf.write(p, to16k(x), 16000, subtype="FLOAT")
        man["assistant"].append({"id": i, "text": text, "wav": p, "dur": len(x) / sr, "chunk_samples_24k": n,
                                 "chunk_arrival_s": arr})
        print("asst", i, round(len(x) / sr, 2), "s", len(n), "chunks, ttfa", round(arr[0] * 1e3), "ms", flush=True)
    for i, text in enumerate(INTERRUPT):
        for v in USER_VOICES:
            x, n, arr = synth(v, text)
            p = os.path.join(OUT, f"int_{i:02d}_{v}.wav"); sf.write(p, to16k(x), 16000, subtype="FLOAT")
            man["interrupt"].append({"id": i, "voice": v, "text": text, "wav": p, "dur": len(x) / sr})
        print("int", i, flush=True)
    for i, text in enumerate(BACKCH):
        for v in USER_VOICES:
            x, n, arr = synth(v, text)
            p = os.path.join(OUT, f"bc_{i:02d}_{v}.wav"); sf.write(p, to16k(x), 16000, subtype="FLOAT")
            man["backchannel"].append({"id": i, "voice": v, "text": text, "wav": p, "dur": len(x) / sr})
        print("bc", i, flush=True)
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1)


if __name__ == "__main__":
    main()

"""Synthesize two small Pocket TTS sets at 16 kHz mono (expA/synth/*.npz).

finance: complete finance requests, text-truncated ones ("How much did I spend on"), and audio-truncated ones
         (the complete sentence cut after a mid-sentence word using Whistle word timestamps).
backch:  short backchannels ("mm-hm", "yeah") vs the opening of real interruptions, several voices.
"""
import json
import os
import sys

os.environ.setdefault("CACTUS_TELEMETRY", "0")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import scipy.signal as ss
import torch
from pocket_tts import TTSModel

import cactus

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 16000

COMPLETE = [
    "How much did I spend on groceries last month?",
    "What's my current account balance?",
    "Show me my net worth.",
    "Move two hundred euros to my savings account.",
    "How much did I spend on restaurants in October?",
    "What was my biggest expense this week?",
    "Am I on track with my budget?",
    "Add a fifty euro expense for fuel.",
    "How much did I earn in dividends this year?",
    "Cancel that.",
    "Yes, go ahead.",
    "What's my savings rate?",
    "Compare my spending this month with last month.",
    "How much is left in my travel budget?",
    "List my subscriptions.",
]
CUT_TEXT = [
    "How much did I spend on",
    "What's my balance in the",
    "Can you show me the",
    "Move two hundred euros to",
    "I want to set a budget for",
    "How much did I spend on restaurants in",
    "What was my biggest expense, um,",
    "Compare my spending this month with",
    "Add a fifty euro expense for",
    "So, what I wanted to ask is",
    "How much is left in my",
    "Yes, and also",
    "Transfer the money from my savings and",
    "What about the, uh,",
    "My rent went up, so I need to",
]
BACKCHANNEL = ["Mm-hm.", "Yeah.", "Okay.", "Right.", "Uh-huh.", "Sure.", "Got it.", "I see.", "Yep.", "Alright.", "Mhm.", "Oh, okay."]
INTERRUPT = [
    "Wait, that's not what I meant.",
    "Actually, can you check my savings instead?",
    "No, stop, I meant last month.",
    "Hold on, what about my rent?",
    "Sorry, can I ask something else?",
    "But how much was that in total?",
    "Can you go back to the budget?",
    "What about the other account?",
    "Stop, that's wrong.",
    "Hang on a second.",
    "I meant the credit card, not the debit card.",
    "How much was that again?",
    # deliberately ambiguous: these open with a backchannel word
    "Okay, but what about groceries?",
    "Yeah, but that's not right.",
]
FIN_VOICES = ["alba", "marius"]
BC_VOICES = ["alba", "marius", "javert", "jean", "anna", "vera", "charles", "paul", "eve", "george", "mary", "michael"]


def tts(model, state, text):
    chunks = [c for c in model.generate_audio_stream(state, text)]
    x = torch.cat(chunks).reshape(-1).numpy().astype(np.float32)
    return ss.resample_poly(x, SR, model.sample_rate).astype(np.float32)


def trim(x, thr_db=-40, pad_s=0.0):
    """Cut leading/trailing silence using 10 ms RMS frames relative to the clip peak."""
    n = SR // 100
    rms = np.sqrt(np.mean(x[: len(x) // n * n].reshape(-1, n) ** 2, axis=1) + 1e-12)
    on = np.where(20 * np.log10(rms / rms.max()) > thr_db)[0]
    a, b = max(0, on[0] * n - int(pad_s * SR)), min(len(x), (on[-1] + 1) * n + int(pad_s * SR))
    return x[a:b]


def main():
    which = sys.argv[1]
    os.makedirs(os.path.join(HERE, "synth"), exist_ok=True)
    model = TTSModel.load_model(language="english_18m_2026-10")
    w = cactus.Whistle()
    clips, meta = [], []
    if which == "finance":
        for voice in FIN_VOICES:
            state = model.get_state_for_audio_prompt(voice)
            for i, text in enumerate(COMPLETE):
                x = trim(tts(model, state, text))
                clips.append(x); meta.append({"voice": voice, "text": text, "kind": "complete", "label": 1})
                # audio-truncated version: cut right after a mid-sentence word
                words = w.transcribe(x, language="en", word_timestamps=True).get("words") or []
                if len(words) >= 4:
                    k = max(1, int(round(len(words) * 0.6)) - 1)
                    end = int(words[k]["end"] * SR)
                    cut = x[:end].copy()
                    fade = min(len(cut), SR // 50)
                    cut[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
                    clips.append(cut)
                    meta.append({"voice": voice, "text": text, "kind": "cut_audio", "label": 0,
                                 "heard_until": " ".join(str(z.get("word", z.get("text", ""))) for z in words[: k + 1])})
            for text in CUT_TEXT:
                clips.append(trim(tts(model, state, text)))
                meta.append({"voice": voice, "text": text, "kind": "cut_text", "label": 0})
            print(voice, len(clips), flush=True)
    else:
        for voice in BC_VOICES:
            state = model.get_state_for_audio_prompt(voice)
            for label, texts in ((0, BACKCHANNEL), (1, INTERRUPT)):
                for text in texts:
                    x = trim(tts(model, state, text))
                    heard = w.transcribe(x, language="en")["text"]
                    clips.append(x)
                    meta.append({"voice": voice, "text": text, "label": label, "heard": heard, "dur_s": len(x) / SR,
                                 "kind": "interrupt" if label else "backchannel"})
            print(voice, len(clips), flush=True)
    np.savez(os.path.join(HERE, "synth", which + ".npz"), audio=np.concatenate(clips),
             lens=np.asarray([len(c) for c in clips], dtype=np.int64), meta=json.dumps(meta))
    for m in meta[:40]:
        print(m)


if __name__ == "__main__":
    main()

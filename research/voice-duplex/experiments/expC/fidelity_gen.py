"""Amount-fidelity set: 30 finance sentences. python fidelity_gen.py woof | pocket <language> ..."""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import write_wav

FID = [
    ("You spent 312.40 euros on groceries in October.", ["312.40"]),
    ("Rent of 950 goes out on the 3rd.", ["950", "3"]),
    ("Your portfolio is up 2.4 percent since March 12th.", ["2.4", "12"]),
    ("Your net worth is 48,250 euros.", ["48250"]),
    ("You paid €89.99 for the gym membership on January 5th.", ["89.99", "5"]),
    ("Savings grew by 1,200 euros, which is 7.5% more than last year.", ["1200", "7.5"]),
    ("The electricity bill was 64.18 euros, due on the 21st.", ["64.18", "21"]),
    ("You have 3 subscriptions costing 41.97 euros a month.", ["3", "41.97"]),
    ("Your salary of 3,480 euros arrived on the 28th.", ["3480", "28"]),
    ("Dining out was 215 euros, down 12 percent from August.", ["215", "12"]),
    ("The transfer of $1,500 to savings is scheduled for 15 November.", ["1500", "15"]),
    ("Your ETF gained 0.8 percent today.", ["0.8"]),
    ("You have 17,903.55 euros across 4 accounts.", ["17903.55", "4"]),
    ("Insurance costs 126 euros every 3 months.", ["126", "3"]),
    ("On the 2nd of November 2026 you paid 39 euros to Spotify.", ["2", "2026", "39"]),
    ("Groceries averaged 78.50 euros per week over 6 weeks.", ["78.50", "6"]),
    ("Your mortgage rate is 3.65 percent until 2031.", ["3.65", "2031"]),
    ("You withdrew 200 euros in cash on October 9th.", ["200", "9"]),
    ("The refund of 14.99 pounds came back on the 30th.", ["14.99", "30"]),
    ("Your budget for travel is 2,750 euros, and 1,105 is left.", ["2750", "1105"]),
    ("Dividends paid 23.07 euros in the third quarter.", ["23.07"]),
    ("You are 35 percent of the way to your goal of 20,000 euros.", ["35", "20000"]),
    ("The phone bill went from 29 to 34 euros in June.", ["29", "34"]),
    ("Your account ending in 4471 has a balance of 6,012.30 euros.", ["4471", "6012.30"]),
    ("Tax of 19 percent adds 57 euros to the invoice.", ["19", "57"]),
    ("You spent €1,048.76 on rent and utilities in September 2026.", ["1048.76", "2026"]),
    ("Interest earned this year is 310.05 euros at 2.25 percent.", ["310.05", "2.25"]),
    ("The next payment of 450 euros is due on December 1st.", ["450", "1"]),
    ("Coffee shops cost you 46.20 euros across 14 visits.", ["46.20", "14"]),
    ("Your emergency fund covers 5.5 months, about 9,900 euros.", ["5.5", "9900"]),
]

if __name__ == "__main__":
    kind = sys.argv[1]
    meta = {}
    if kind == "woof":
        sys.path.insert(0, os.path.join(HERE, "woof_stream"))
        from stream_runtime import WoofStream, SR
        for voice in sys.argv[2:] or ["juno"]:
            tts = WoofStream(voice=voice, schedule=(1, 2, 4)); tts.warmup()
            rows = []
            for i, (text, nums) in enumerate(FID):
                t0 = time.perf_counter()
                wav = np.concatenate(list(tts.stream(text, seed=i)))
                dt = time.perf_counter() - t0
                p = os.path.join(HERE, "audio", f"fid_woof_{voice}", f"{i:02d}.wav")
                write_wav(p, wav, SR)
                rows.append({"i": i, "wav": p, "spoken_form": tts.norm(text), "aborted": tts.stats.get("aborted"),
                             "audio_s": len(wav) / SR, "gen_s": dt})
                print(voice, i, round(len(wav) / SR, 2), tts.stats.get("aborted"), flush=True)
            meta[f"fid_woof_{voice}"] = rows
    else:
        import torch
        from pocket_tts import TTSModel
        for lang in sys.argv[2:]:
            m = TTSModel.load_model(language=lang)
            st = m.get_state_for_audio_prompt("alba")
            rows = []
            for i, (text, nums) in enumerate(FID):
                torch.manual_seed(i)
                t0 = time.perf_counter()
                wav = np.concatenate([c.detach().cpu().numpy().reshape(-1) for c in m.generate_audio_stream(st, text)])
                dt = time.perf_counter() - t0
                p = os.path.join(HERE, "audio", f"fid_pocket_{lang}", f"{i:02d}.wav")
                write_wav(p, wav, m.sample_rate)
                rows.append({"i": i, "wav": p, "audio_s": len(wav) / m.sample_rate, "gen_s": dt})
                print(lang, i, round(len(wav) / m.sample_rate, 2), flush=True)
            meta[f"fid_pocket_{lang}"] = rows
            del m, st
    with open(os.path.join(HERE, f"res_fid_gen_{kind}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)

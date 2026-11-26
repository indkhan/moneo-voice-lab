import os, inspect
os.environ["CACTUS_TELEMETRY"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import cactus
print("cactus", getattr(cactus, "__version__", "?"))
print(inspect.signature(cactus.transcribe)); print(cactus.transcribe.__doc__)
print(inspect.signature(cactus.stream)); print(cactus.stream.__doc__)
from livekit import rtc
print([n for n in dir(rtc) if "Audio" in n or "apm" in n.lower()])
apm = rtc.AudioProcessingModule
print(inspect.signature(apm.__init__)); print(apm.__doc__)
for n in dir(apm):
    if not n.startswith("__"):
        f = getattr(apm, n)
        try:
            print(n, inspect.signature(f)); print("   ", (f.__doc__ or "").strip()[:600])
        except Exception as e:
            print(n, e)
print(inspect.signature(rtc.AudioFrame.__init__))
import silero_vad
print([n for n in dir(silero_vad) if not n.startswith("_")])
print(inspect.signature(silero_vad.load_silero_vad))

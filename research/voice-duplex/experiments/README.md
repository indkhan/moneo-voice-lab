# Voice duplex experiments

| Experiment | What you'll find |
|---|---|
| [A · Turn detection](expA-turn-detection/) | Smart Turn evaluation, synthetic speech sets, acoustic embeddings, and turn detector training and benchmarks |
| [B · Routing and backchannels](expB-routing-and-backchannels/) | Utterance labeling, routing and short-response experiments, and synthetic training and test sets |
| [C · Speech synthesis](expC-speech-synthesis/) | Pocket and Woof TTS benchmarks, fidelity scoring, and the Woof streaming runtime |
| [D · Duplex simulation](expD-duplex-simulation/) | Simulations and diagnostics for interruptions and overlapping speech |
| [E · Moneo call integration](expE-moneo-call-integration/) | Call editor fixtures, fast-answer benchmarks, and Moneo QA scripts |
| [F · Recording evaluation](expF-recording-evaluation/) | Audio splitting, reference transcription, and reading accuracy scoring |
| [G · Speech recognition](expG-speech-recognition/) | ASR benchmarks across runtimes and downstream scoring of transcription outcomes |
| [End-to-end spike](spike-end-to-end/) | Recorded-call replay, simulated timing, and the LLM proxy used by the prototypes |

See the [research log](../LOG.md) for results and the [roadmap](../ROADMAP.md#how-to-run-things) for Moneo QA commands. Scripts that import Moneo code or use its package scripts need a Moneo checkout with dependencies installed and the experiment tree at the same relative path. Some Python experiments also require external model files or recordings. Experiment B's standalone scripts use the scratch data path configured in `scripts/common.py`.

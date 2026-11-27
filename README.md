# Moneo Voice Lab

Moneo Voice Lab explores how to make talking about your finances feel like a real conversation. The goal is a voice assistant that can explain spending, handle interruptions, and answer follow-up questions grounded in account and transaction data.

## Research goals

- Stream microphone input and speech output concurrently so you can interrupt while the assistant is speaking.
- Compare open-source speech and language models for latency, interruption handling, and local hardware requirements.
- Use synthetic accounts and transactions to test whether answers stay accurate when a conversation changes direction.

## Relationship to Moneo

The lab gives voice experiments their own development cycle. [Moneo](https://github.com/indkhan/Moneo) remains responsible for real financial data, calculations, and approvals. Successful experiments can be integrated into Moneo through its voice and finance tools.

## Status

The [research log](research/voice-duplex/LOG.md) records model evaluations, benchmark results, and design decisions. The [roadmap](research/voice-duplex/ROADMAP.md) and [experiments](research/voice-duplex/experiments/) were imported from Moneo at commit `7911afae`. The import also includes the in-progress Sherpa-ONNX benchmark from that checkout.

Some scripts require Moneo's QA environment, model downloads, or local audio inputs. The repository includes benchmark results and text from test sessions. Voice recordings are kept outside the repository.

## Repository map

- [Project plan](plan.md) defines the long-term goals and research principles.
- [requirements.txt](requirements.txt) lists the Python packages the experiments use; the roadmap explains how this repo and a Moneo checkout fit together.
- [Experiment index](research/voice-duplex/experiments/README.md) links each research area to its scripts, fixtures, and results.
- [Research log](research/voice-duplex/LOG.md) records the findings. The [roadmap](research/voice-duplex/ROADMAP.md) includes the Moneo code map and QA commands.
- [Architecture](research/voice-duplex/architecture.html) and [implementation plan](research/voice-duplex/plan.html) are standalone HTML documents. The [reading script](research/voice-duplex/reading-script.md) contains the reference text for speech recognition evaluation.

# Moneo Voice Lab — Project Plan

## Goal

Build the best real-time, full-duplex voice assistant for personal finance, with a lightweight local conversational layer and powerful backend agents. The focus is natural interaction, not training a new voice foundation model.

## Desired behavior

- Listen continuously, even while speaking. Handle interruptions, pauses, corrections and acknowledgements naturally. Stop playback immediately when genuinely interrupted.
- Start reasoning and safe read-only work as soon as enough of a request is understood, without waiting for the user to finish speaking. Revise or discard speculation when the request changes.
- Run multiple independent reasoning sessions concurrently. Keep talking to the user, track each task, deliver results when appropriate, and support task-specific follow-ups, corrections, status checks and cancellation. Interrupting speech must not automatically cancel unrelated work.
- Answer straightforward financial questions quickly using reliable tools and structured data. Ground monetary facts in verified results, and require explicit approval for financial changes.

## Research principles

- Use the best component for each job, as small and efficient as possible without sacrificing the behavior or accuracy we need. There are no fixed model-size targets.
- Reuse existing STT, TTS, language models and other components. Freely experiment with their encoders, embeddings, heads, decoders, weights and runtimes wherever licensing permits. Fine-tune, distill, combine, replace or modify any part when evidence supports it.
- Use ordinary code when it works better than a model. Keep components swappable and prioritize a local, laptop-capable conversational layer; backend reasoning may use local or remote models with appropriate data permissions.
- Evaluate the complete experience on realistic conversations across speakers: financial accuracy, time to useful answer, interruption handling, parallel-task behavior and total resource use. Compare alternatives on the same tests and retain what actually improves the system.

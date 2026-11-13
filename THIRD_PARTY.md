# Third-party code and provenance

The root MIT license covers original project additions. Incorporated code, models
and assets retain their upstream licenses and attribution.

| Component | Source / terms |
| --- | --- |
| `engine/` | [OpenBMB/llama.cpp-omni](https://github.com/OpenBMB/llama.cpp-omni), MIT; see `engine/LICENSE` and `engine/licenses/` |
| `engine/ggml/`, vendored libraries | Original license files and source notices remain in their directories |
| `runtime/` service and browser demo | [OpenBMB/MiniCPM-o-Demo, Comni](https://github.com/OpenBMB/MiniCPM-o-Demo/tree/Comni); package metadata declares ISC; the imported snapshot has no standalone root license file |
| `runtime/MiniCPMO45/` | [MiniCPM-o](https://github.com/OpenBMB/MiniCPM-o), Apache-2.0; see `runtime/MiniCPMO45/LICENSE` |
| `runtime/assets/vad_model/` | Silero VAD model distributed with the upstream demo; [Silero VAD](https://github.com/snakers4/silero-vad), MIT; see sibling `LICENSE` |
| Reference audio and presets | Retained from the upstream demo; upstream asset terms apply |

`UPSTREAM.json` identifies the engine base and the runtime source reference.
The engine includes local Apple Silicon and duplex changes; it is not an unmodified
upstream checkout. The imported runtime has local changes and no Git metadata, so
the referenced upstream runtime commit is a reference, not a verified snapshot ID.
Prior integration and publication preparation included AI assistance.

Build outputs, unrelated upstream tokenizer fixtures, generated operator tables,
model weights and local experiment records are excluded from the source import.
The small VAD model and reference audio required by the service are retained.

# Apple Silicon baseline

The imported M5 implementation is the reference path for future backend work.
`profiles/apple-m5.json` preserves its operational settings; `selected-profile.json`
is the profile read by the macOS service controller.

| Component | Preserved behavior |
| --- | --- |
| LLM | Q4_K_M, Metal, Flash Attention, F16 KV, 8192 context, 4 CPU threads |
| Vision | CoreML / ANE, exact-byte frame reuse, per-frame temporary-object cleanup |
| TTS | Original F16 decoder, full causal attention, independent 4096 context, hidden-only output, one-time condition prefill |
| Token2Wav | Fused QKV, Accelerate CPU vocoder, 8 threads, startup warmup, 5 Flow steps, 25+3 acoustic window |
| Duplex | FIFO input, immediate acoustic-token delivery, interruption epochs and native LISTEN boundaries |
| Playback | 100 ms default buffer, stale-output suppression, first-chunk preservation |
| Service | Single-instance lifecycle, persistent model and worker/gateway recovery |

Audio-encoder and TTS Q8 experiments were rolled back before this import.
The reference model uses the original F16/F32 audio encoder and F16 TTS weights.
Additional KV quantization, shorter TTS history and a 16K default context are not
selected baseline settings.

The original development workspace, model files, release binaries and experiment
evidence remain local. This repository carries source and profiles; rebuild native
artifacts with `scripts/build-release.sh`.

## Acceptance boundary

Prior tests covered deterministic component comparisons, scripted duplex replay,
interruptions and software playback scheduling. They did not establish uniformly
subsecond speech onset, bounded backlog under all long-answer workloads, or complete
long-session speech quality. Physical speaker/microphone echo and a human-reviewed
10–15 minute conversation remain acceptance tasks. No speedup percentage is
claimed for this repository import.

When changing a backend, compare first-audio latency, input backlog, sustained
playback gaps, interruption latency, peak memory and human-reviewed speech quality
under matching models, precision, context sizes, sampling and load.

## 中文

本次迁移保留上表中的 Apple 优化源码与参数。音频编码器及 TTS 的 Q8 实验已撤回，
基线使用原精度权重。长回答积压、长会话音质及真实扬声器回声环境仍需验收；
已有组件及回放测试不能等同于这些目标已完成。后续 CUDA 优化应使用相同输入、
模型精度与采样设置对照，并同时评估延迟、连续性、内存和语音质量。

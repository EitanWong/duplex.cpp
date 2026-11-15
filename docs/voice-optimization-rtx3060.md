# Voice-path optimization on RTX 3060

Measured on 2026-11-14, native Windows/WDDM, RTX 3060 12 GB (SM 8.6),
i7-10700F, 32 GB RAM, CUDA 12.8.1 and driver 610.88. This extends the
[earlier pipeline research](inference-optimization.md); its optimizations are
the starting point, not gains introduced again here.

The NVIDIA profile now defers vision loading until an image arrives and uses
four independent AVX2 accumulator groups for the acoustic projection. In the
first paired voice replay, whole-card sampled peak fell **10461 → 9291 MiB**,
about **1.14 GiB / 11.2%**. The CPU acoustic operator was **1.65–1.74× faster**
than the previous AVX2 implementation, with identical logits. Repeated complete
voice and mixed audio/video sessions preserved text, decisions and PCM bytes.
End-to-end voice latency varied across runs; a consistent pipeline speedup has
not been demonstrated.

## What still limits conversation latency

The path is microphone collection/endpointing → streaming audio encoder →
Qwen3 prefill and listen/speak decision → text/hidden states → acoustic decoder
and projection → Token2Mel flow/vocoder → PCM transport and playback buffering.
These stages overlap. Input response time, first PCM and uninterrupted playback
are different measurements; adding overlapping stage times gives a false total.

- **Decision cadence:** the official model describes a 1 Hz listen/speak decision.
  A faster kernel does not remove input collection or semantic turn-taking time.
  Arbitrarily shortening the trained grouping/window changes model behavior.
- **LLM bandwidth:** the existing Q4_K_M decoder is already near the bandwidth
  roofline for this GPU: roughly 63 tokens/s isolated, versus an idealized
  77 tokens/s from 4.67 GB of decoder weights and nominal 360 GB/s bandwidth.
  This excludes other pipeline work and is not an achieved upper bound for
  conversations. Peak Tensor Core FLOPS do not predict single-token speed.
- **Speech generation:** acoustic tokens remain autoregressive, followed by the
  existing 25-token flow window plus lookahead. Projection acceleration affects
  only part of this chain. Reducing lookahead or diffusion steps needs separate
  quality validation or training.
- **Residency:** eagerly loading a vision encoder for a voice-only conversation
  consumed 1044.36 MiB of tensor payload plus workspace/backend overhead. This
  was a larger exact-output memory opportunity than another KV precision change.
- **Playback:** native replay does not measure microphone-to-speaker delay, AEC,
  playback starvation or network jitter. These remain necessary acceptance checks
  for real conversations; native request p50 alone cannot establish fluid speech.

## Implemented changes

### Load vision only when needed

`OMNI_VISION_LAZY=1` skips vision initialization on CUDA until the first image
passes through `stream_prefill`. Loading occurs before encoder queue submission,
so a failed initialization returns an error to the caller instead of silently
discarding the image. The existing F16 model, attention setting, slicing and
preprocessing are used. The context retains the loaded model across chat/duplex
session resets and frees it with the other models.

This saves residency throughout voice-only sessions. The NVIDIA profile now
also releases vision after a closed session has been idle for 60 seconds
(`OMNI_VISION_IDLE_SECONDS=60`). Stopping the camera within an active session
keeps it resident. Set this timeout to `0` to retain vision across idle sessions;
`OMNI_VISION_LAZY=0` restores eager initialization and disables idle release.

In a replay with eight audio-only inputs before video, first-image model loading
took **751.07 ms**. That request completed in **983.76 ms**, versus **319.96 ms**
with eager loading. The corresponding request in the next session took
190.39 / 195.62 ms because the model was already resident. These are ordered
trials with cached local files; disk-cold loading can take longer.

An ordinary-chat probe also encoded three image chunks, repeated the image
request with matching text, and switched between chat and duplex twice without
reloading vision. Whole-card usage before/after activation was 9182 / 10352 MiB.
A corrupt-model probe returned `prefill_failed`; a subsequent text session still
completed successfully. These probes use real local weights except for the
deliberately invalid vision file, stored separately under ignored verification.

The switch applies to CUDA builds outside Apple. Apple retains its eager
Metal/CoreML initialization and ANE warmup. The M5 profile is unchanged.

### Interleave independent acoustic outputs

`OMNI_TTS_CPU_HEAD=avx2_ilp` computes 32 outputs using four eight-lane accumulators.
This gives the CPU independent work while previous additions are in flight and
reuses each broadcast hidden value. Every output still accumulates hidden
dimensions in ascending order using separate multiply/add instructions. There
is no horizontal reduction, FMA, extra worker thread, weight conversion or new
packed-weight allocation. The existing partial-group routine handles the tail.

Same-process real-weight tests alternate scalar, original AVX2 and interleaved
measurement order; 100 inputs × 6562 logits per run:

| Run | Original AVX2 ms | Interleaved ms | Operator speedup |
| --- | ---: | ---: | ---: |
| 1 | 0.8039 | 0.4617 | 1.741× |
| 2 | 0.7994 | 0.4819 | 1.659× |
| 3 | 0.8082 | 0.4891 | 1.652× |

All logits were byte-identical to scalar. Synthetic weights and output counts
1, 7, 8, 31, 32, 33, 63, 64 and 65 with odd hidden width also passed; guard
values check for writes beyond the output span. The existing AVX2 path remains
available with `OMNI_TTS_CPU_HEAD=avx2`, and scalar with `scalar`. Apple continues
to use Accelerate. The faster experimental CUDA head remains disabled because
the earlier repeated-session sampled-speech discrepancy is unresolved.

## Pipeline evidence

All voice trials use the same 36 one-second audio inputs, Q4_K_M LLM, 4096-token
contexts, F16 KV, original acoustic/flow/vocoder weights, LLM greedy, acoustic
temperature 0.8 and seed 42. Each fresh process runs two sessions. No inference
precision, speech sampling, flow steps or acoustic window is changed.

After fixing the replay timer, a reverse-order repeat compared the new binary
with both switches enabled against the same binary with eager vision and the
original AVX2 path:

| Variant / session | Request p50 ms | p95 ms | First SPEAK input → PCM ms |
| --- | ---: | ---: | ---: |
| Eager / original AVX2 / 1 | 156.79 | 209.52 | 479.80 |
| Lazy / interleaved / 1 | 152.46 | 194.99 | 448.84 |
| Eager / original AVX2 / 2 | 155.66 | 196.67 | 415.10 |
| Lazy / interleaved / 2 | 160.53 | 210.37 | 432.19 |

The second-session reversal, and a 491.01 ms maximum in the selected run,
exclude a claim of consistent whole-pipeline acceleration. The earlier ordered
pair also had mixed latency results. Select these changes for measured memory
savings and local compute reduction, not a promised fixed response-time gain.

The first pair sampled whole-card peaks of 10461 / 9291 MiB; the repeat sampled
11101 / 9293 MiB. WDDM/desktop residency and transient allocations vary.
These are `nvidia-smi` samples every 200 ms, including other applications,
not exact process allocation counts; peaks can be missed. The stable mechanism
is the absence of vision weights/workspace before any image. Do not attribute
the larger repeat difference entirely to this change or add all file payload
sizes to estimate peak VRAM.

Corresponding voice sessions in all four processes preserved every listen/speak
decision, text and PCM byte: 529920 / 533760 samples, 22.08 / 22.24 seconds,
finite and unclipped. SHA256:

- Session 1: `afc1b819ec45b682fc44e26c1154d6f36b7e454789be68329f8595a5c09b6bde`
- Session 2: `d7ea657ba450e5b7afc4e7618d8fccd72752fae5f8f525805d416db6590fc0e8`

Two mixed sessions with video starting at input 8 also preserved decisions,
text and PCM: 243840 / 247680 samples, with hashes
`004edb8c5d5256798441085cf1a69e1a90ccc9982a58d6f07f891302ea8293cf` and
`c8933af33ae8103ff33ee70b0b92f08ca651ec808bb87ae96f89d4973d894556`.
Matching recordings provide regression evidence for this workload, not subjective
acceptance of every voice, recognition accuracy or long-session naturalness.

A further **120-second voice replay** with an explicit interruption at input 5
completed all 120 inputs: request p50 **158.72 ms**, p95 **290.23 ms**, maximum
**1070.04 ms**, first SPEAK-to-PCM **408.85 ms**. The forced LISTEN acknowledgement
took **50.90 ms**, with no epoch-0 PCM arriving afterward. Output was 1482240
samples / 61.76 seconds, finite and unclipped; maximum observed LLM KV was 2007
tokens. Whole-card sampled peak grew to **10112 MiB**: deferred vision does not
eliminate longer-run graph/workspace growth or establish a fixed 9.3 GiB budget.
This is a sustained execution/cancellation check, not a long-run A/B quality or
microphone playback comparison.

The replay tool now prepares WAV/base64/JSON before starting its request timer,
records preparation and send time separately, supports `--video-start-at`, and
ignores empty end markers when measuring first PCM. Earlier reports used a timer
that also included audio fixture I/O/encoding; compare timer versions explicitly.
Recordings and raw evidence stay in ignored `verification/optimization-v4/`.

The native CUDA Release build, acoustic probes, **63 targeted Python tests**
(launch, native chat and schemas), repository hygiene and `git diff --check`
passed. Hardware speech evidence above comes from this RTX 3060; no new M5
performance claim is made.

The updated local service was restarted and its configured HTTPS Gateway passed
streaming/non-streaming text and speech, conversation history, generation-limit
and reference-audio endpoint regressions. Warm short text completed in
133–172 ms; a short streaming speech request delivered first PCM at 616.46 ms.
The first cold text request took 1327.59 ms. These are Gateway script observations,
not microphone/browser playback latency and not a Gateway A/B speedup claim.
Evidence: `verification/browser-repair-optimization-v4.json`.

## Further local measurements, 2026-11-15

The playback startup calculation previously added the configured 100 ms to an
existing duration/jitter reserve. It now takes their maximum, retaining the
reserve and the explicit zero-delay setting. Twelve browser startup measurements
per variant in ABBA order found roughly 90–100 ms earlier scheduling: p50
203.65 → 109.20 ms, p95 365.06 → 264.33 ms. Replaying real recorded PCM through
Chrome 157 AudioContext/AudioWorklet at 44.1 kHz, including interruption and a
120-input trace, found no additional source-interval gaps. These are browser
schedule/output-clock observations, not physical microphone-to-speaker latency.
An allocation crossing a render quantum before the first source is also excluded
from starvation statistics; an injected real supply gap remains detected.

The idle-vision observer uses the context lifecycle mutex and excludes active
sessions, live pipelines and joinable inference threads. Its timer starts after
joined session cleanup. Four releases in alternating control/candidate probes
recovered **1132 MiB**, from 10156 to 9024 MiB whole-card usage. Active sessions
remained resident beyond the experimental two-second timeout. Image text and
both corresponding 36-input mixed-session PCM outputs matched exactly after
reload. First image input after release cost 1003 / 1160 ms versus 199 / 201 ms
while resident, with cached files. This is a memory/next-image latency tradeoff.
The selected timeout is 60 seconds, rather than the probe's two seconds.

A separate 120-input mixed replay with interruption also preserved all text,
decisions and PCM (756480 samples / 31.52 s). Control/candidate request p50 was
221.10 / 222.24 ms, p95 336.87 / 330.06 ms, and maximum 1187.09 / 1166.57 ms;
first SPEAK input to PCM was 465.77 / 472.23 ms. No old epoch PCM arrived after
the interruption acknowledgement. These timings begin after prepared one-second
input is sent and show no consistent inference speedup. Human listening, actual
microphone/AEC and multihour acceptance remain incomplete.

Raw evidence is in ignored `verification/continuous-optimization/`, especially
`browser-playback-{initial,extended}.json`, `playback-metric-faults.json`,
`vision-residency-summary.json`, `vision-comparison.json` and
`vision-long-comparison.json`. The original player and complete native binaries
are saved there under `baseline/` and `baseline-bin/` for local rollback.

The CUDA vocoder's host `no_alloc` metadata arena was reduced from 2 GiB to
407002336 bytes, preserving its original graph node/leaf capacity and all
operations. Actual metadata use peaked at 21983072 bytes. Six independent
37.6-second Token2Wav runs in ABBAAB order lowered median process private commit
peak by **1773064192 bytes (1.65 GiB)**; all WAV hashes matched. Compute times
overlapped (original 4.082 / 4.161 / 4.155 s; compact 4.142 / 4.108 / 4.067 s),
so this is a commit-budget improvement, not an inference speedup or VRAM saving.
Two 36-input audio and two 120-input mixed sessions preserved corresponding
text, LISTEN/SPEAK and raw float PCM exactly, with no nonfinite/clipped samples
or old-epoch PCM after explicit interruption acknowledgement. In-session sampled
commit transients were about 1.64 GiB lower; the process lifetime peak, dominated
by loading/session reconstruction, was essentially unchanged. Physical working
set did not fall by 1.65 GiB. CPU/Apple allocation and all precision/sampling
remain unchanged. Evidence: `round-6/{screen-summary,voice-comparison,
mixed-comparison,pipeline-memory-summary}.json` in the ignored evidence directory.

The cached CUDA Flow session now similarly budgets its three original full-size
graphs and tensor metadata, reserving 1796759488 bytes instead of 2 GiB. Actual
use was 261894048 bytes. Another six-process ABBAAB screen lowered median private
commit peak by **350171136 bytes (334 MiB)** with identical WAVs and overlapping
compute times. Two more 120-input mixed sessions preserved corresponding text,
decisions and raw PCM, including interruption epochs. The full pipeline process
peak was 355336192 bytes lower in this one-process-per-variant comparison,
consistent with the independently repeated memory screen. No speed, working-set
or VRAM gain is claimed. The uncached setup and CPU/Apple paths are unchanged.
Evidence: `round-7/{screen-summary,mixed-comparison,pipeline-memory-summary}.json`.

Removing the Flow wrapper's redundant CUDA stream wait was also screened with
six ABBAAB runs. The wait cost only 0.006–0.010 ms median per call (including
instrumentation); skipping it measured 4.145 / 4.111 / 4.178 s total versus
4.113 / 4.100 / 4.133 s with the wait. WAVs matched, but there was no overall
gain. Its code and temporary switch were removed. Evidence: `round-8/`.

Two further candidates were screened out on the same host. Converting 69 Flow
linear matrices to F16, excluding fused QKV, convolution, norm and bias tensors,
saved 146.14 MiB of file payload. Six independent processes in ABBAAB order
generated 37.6 seconds each: original compute was 4.086 / 4.113 / 4.136 seconds,
versus 4.088 / 4.107 / 4.150 seconds with F16. PCM changed, and the timing
difference was within noise. Original weights and default precision are retained;
there is no human acceptance of the converted model.

Caching positional-input tensor pointers reduced the local Token2Mel upload
stage from roughly 2.0–2.1 to 1.1–1.2 ms. In a separate six-process ABBAAB trial
with 75.2 seconds of output per process, complete Token2Wav compute was
7.899 / 7.968 / 7.960 seconds for the original and 7.945 / 7.926 / 8.025 seconds
for the candidate. All WAVs matched, but there was no stable overall benefit;
the candidate source changes were reverted and the selected binary rebuilt.
Evidence: `flow-f16/{conversion,results}.json` and `pos-cache/{results,summary}.json`
under the same ignored directory. Neither failed screen adds a default switch.

A third screen changed only HiFiGAN residual-convolution im2col to F16 on CUDA,
allowing the existing half GEMM path. Six processes, again ABBAAB with 37.6 s
output each, measured original compute 4.148 / 4.138 / 4.078 s versus
4.118 / 4.139 / 4.105 s. PCM changed and the distributions overlapped; this
candidate and its switch were also removed. The ignored candidate patch and
`voc-f16/{results,summary}.json` retain the experiment for future profiling.

The selected configuration additionally completed **600 one-second mixed
inputs**, with an explicit interruption at input 11. Request p50/p95/maximum
were **215.63 / 320.75 / 1118.08 ms** (the maximum includes initial vision loading).
The six 100-input blocks kept p50 between 213.98 and 219.77 ms. Reported KV length
decreased 30 times and peaked at 3583; all inputs completed. No old-epoch audio
arrived after interruption acknowledgement. The 101.24 s raw PCM was finite and
unclipped, and the first 120 text/decision results matched the earlier replay.
Whole-card peak was 10288 MiB over 3008 samples. Across 56 WDDM process samples,
dedicated usage rose about 8 MiB then plateaued, shared usage was 94–96 MiB,
and working set was about 6.26–6.34 GiB (startup peak 10.75 GiB). Private commit
had a transient 15.56 GiB peak; these counters do not establish paging behavior.
Replay send schedule lag p50/p95/maximum was 11.04 / 20.26 / 139.71 ms, without
continued growth. It is neither microphone delay nor Worker queue age. This
ten-minute fixture loop is a stability check, not multihour or human acceptance.
Evidence: `selected-600-summary.json`, `selected-600-memory.jsonl` and
`selected-600/` under the same ignored directory.

After selecting both metadata-arena changes, a fresh 600-input run completed in
599.23 s: request p50 / p95 / maximum **216.05 / 322.46 / 977.01 ms**, first
SPEAK input to PCM **467.40 ms**. All 600 text/decision results and all 101.24 s
of raw PCM matched the earlier same-history run exactly. That earlier run is an
exactness reference here, not a fresh speed A/B. Six 100-input blocks had p50
214.75–218.67 ms; KV length decreased 30 times and peaked at 3583. There was no
nonfinite/clipped PCM or old-epoch audio after interruption acknowledgement.
The final 57 WDDM samples showed dedicated memory reaching about 9.36 GiB then
plateauing, shared memory 94–96 MiB and working set about 6.22–6.35 GiB. Whole-card
peak was 10387 MiB (including desktop); 5836 PSAPI samples captured a process
private-commit lifetime peak of 17702825984 bytes. This remains a ten-minute
fixture check. Evidence: `selected7-600-summary.json`, `selected7-600/`,
`selected7-600-memory.jsonl` and `selected7-600-wddm.jsonl`.

After restoring the normal service, its actual 60-second policy held vision at
10159 MiB through 60.01 s after close and dropped to 9027 MiB at the next 65.01 s
sample. Reopening the image restored residency and produced identical text.
Two audio and one video direct-speech interruption replays through the Worker,
at one intact input per second, all recovered “北京是中国的首都。” Each completed
30 inputs, reported queue depth zero and no old-epoch PCM after notification.
Notification to new PCM was 4508 / 4541 / 4581 ms, including the continuing
spoken question and subsequent turn decision; it is not speech-end latency.
Queue wait maxima were 0.050 / 0.087 / 0.052 ms. Evidence:
`normal-idle-smoke.json` and `final-direct-summary.json`. Physical stopping time,
subjective quality and real microphone/AEC behavior remain unmeasured.

The final metadata-arena build was restored to the normal service and passed
another two audio and two video direct-interruption sessions, 30 inputs each:
4/4 recovered the new Beijing answer, with finite/unclipped PCM, queue depth
zero and no stale epoch audio. The first video session included initial vision
loading: input 8 processing took 1217.19 ms and input 9 waited 198.45 ms in the
queue, which then cleared. An immediate resident-vision repeat had maximum
processing 289.60 ms and queue wait 0.038 ms. This isolates cold vision loading
from ongoing queue growth. Notification to fresh PCM was 4488–4540 ms, including
the continuing question; physical stop/speech-end latency remains unmeasured.
Evidence: `selected7-direct-summary.json`.

## Priorities for further gains

The [earlier research table](inference-optimization.md#precision-experiments-and-priorities)
covers AWQ, GPTQ, SmoothQuant, KIVI, SparseGPT, Wanda, Marlin and runtime migration.
The practical ordering after these measurements is:

| Priority | Approach | Gain mechanism and acceptance requirement |
| --- | --- | --- |
| 1 | Calibrated lower-bit LLM weights | Reduces the dominant bytes read per token. Start from original F16/BF16 weights; use bilingual audio/video activations and control tokens for importance calibration. Measure native duplex accuracy, decisions and speech as well as perplexity. Requantizing the available Q4 file loses additional information. |
| 2 | Selective flow/encoder weight precision | Actual GGUF F32 payloads are encoder 144.31 MiB, flow 436.99 MiB, extra 13.03 MiB and vocoder 79.37 MiB. Halving all would save at most about 337 MiB of file tensor payload, before runtime differences. Sampled large matrices do change under F16 rounding; fused-QKV loading also currently requires F32. Implement compatible kernels/loading and validate phonetics, speaker similarity, naturalness and long streaming behavior before selecting it. |
| 3 | CUDA fusion, graph reuse and reduced synchronization | Existing CUDA Graphs, Flash Attention and persistent buffers are already active. Profile remaining host gaps and small kernels; retain CUDA/CPU fallbacks. A faster acoustic microkernel must pass repeated sampled sessions, not just random-vector logits. |
| 4 | Quantized KV for longer histories | Q8_0 approximately halves the LLM's 576 MiB F16 KV payload, with scales/overhead. Earlier pipeline trials changed output and were slower. This is a capacity option requiring evaluation, not the default latency optimization at 4096 tokens. KIVI's asymmetric 2-bit method is not ggml Q4 KV. |
| 5 | Structural pruning/distillation | Physically smaller dense dimensions can reduce bandwidth and compute. Zero masking alone leaves dense storage/kernels unchanged. Ampere 2:4 requires supported sparse kernels, pruning and compression; half of F16 values plus metadata can exceed dense Q4 storage. Retrain and test duplex behavior before deployment. |
| 6 | Speculation or another inference runtime | Draft verification must return hidden states for every accepted token, preserve control tokens, rollback KV and honor interruption epochs. Draft weights/cache consume the newly available VRAM. Text-only compatibility or server throughput benchmarks do not establish native duplex support or faster short speech bursts. |

SM 8.6 can use F16/BF16/INT8 Tensor Cores and asynchronous global-to-shared
copies. It lacks native FP8/FP4 Tensor Cores; Hopper/Blackwell kernel claims
cannot be transferred to this device. Marlin targets a different packed INT4
format from Q4_K_M, so its FP16-relative speedup is not an incremental speedup
over the current native quantized decoder. Larger serving batches and additional
workers optimize aggregate throughput while consuming memory and queueing time;
they need a separate concurrency objective.

## Reproduce

Stop the managed service before isolated benchmarks and restore it afterward.
Build with the [Windows guide](nvidia.md), using the CUDA toolkit on `PATH`:

```powershell
$env:PATH = "$env:CUDA_PATH/bin;$env:PATH"
engine/build-cuda/bin/test-tts-cpu-head.exe
.venv/Scripts/python scripts/benchmarks/pipeline.py `
  --model-dir models/MiniCPM-o-4_5-gguf --audio-only --tts-temperature 0.8 `
  --rounds 2 --output verification/voice-selected
.venv/Scripts/python scripts/benchmarks/pipeline.py `
  --model-dir models/MiniCPM-o-4_5-gguf --audio-only --tts-temperature 0.8 `
  --env OMNI_VISION_LAZY=0 --env OMNI_TTS_CPU_HEAD=avx2 `
  --rounds 2 --output verification/voice-control
.venv/Scripts/python scripts/benchmarks/pipeline.py `
  --model-dir models/MiniCPM-o-4_5-gguf --video-start-at 8 `
  --tts-temperature 0.8 --rounds 2 --output verification/camera-activation
```

## Primary sources

Reviewed 2026-11-14; published speedups apply to the authors' workloads:

- [Official model at the local weights' revision](https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf/blob/47eeee5b877167846601d816a192f9a2ee62fc25/README.md): 1 Hz mechanism, architecture and audio-only `init_vision=False`.
- [llama.cpp quantization guidance](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md): importance matrices and requantization quality warning.
- [NVIDIA Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html): SM 8.6 resource limits, asynchronous copies and Tensor Core formats.
- [NVIDIA CUDA Graphs in llama.cpp](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs/): GPU launch overhead and capture/update costs.
- [cuSPARSELt execution workflow](https://docs.nvidia.com/cuda/cusparselt/getting_started.html): pruning, compression, workspace and sparse execution.
- [Marlin](https://github.com/IST-DASLab/marlin): packed INT4, asynchronous loads and bandwidth considerations.
- [AWQ](https://github.com/mit-han-lab/llm-awq), [GPTQ](https://github.com/IST-DASLab/gptq), [SmoothQuant](https://github.com/mit-han-lab/smoothquant), [KIVI](https://github.com/jy-yuan/KIVI), [SparseGPT](https://github.com/IST-DASLab/sparsegpt), [Wanda](https://github.com/locuslab/wanda): calibration and compression methods; no new host speedup is inferred from their published results.

# Validated local playback and memory changes

Measured on 2026-11-15 with an i7-10700F, 32 GiB RAM and RTX 3060 12 GiB
(SM 8.6), native Windows/WDDM, driver 610.88 and CUDA 12.8.1. Models were on
an HDD. The tested local tree included earlier, uncommitted NVIDIA/runtime
optimizations; these results are incremental to that tree, not to clean upstream.
LLM weights remained MiniCPM-o 4.5 Q4_K_M, with 4096-token F16 KV, full GPU
offload, 4 decode / 8 batch threads and continuous one-second duplex inputs.
Original weights, speech sampling and Apple inference paths are preserved.

## Selected changes and evidence

| Change | Local comparison | Limit |
| --- | --- | --- |
| Take the maximum of configured playback delay and duration/jitter reserve | Chrome 157, 44.1 kHz AudioContext/AudioWorklet, real recorded PCM, ABBA order, 12 starts per variant: p50 203.65 → 109.20 ms, p95 365.06 → 264.33 ms; no additional source-interval gaps | Browser scheduling/output-clock evidence, not physical microphone-to-speaker latency |
| Release lazy CUDA vision after a closed session is idle for 60 seconds | Four alternating probes each recovered 1132 MiB; corresponding text/decisions/PCM matched after reload | First image after release took 1003/1160 ms versus 199/201 ms while resident; active sessions retain vision |
| Size the CUDA vocoder metadata arena from its unchanged graph capacity | ABBAAB, three independent processes per variant, 37.6 s output each: median private-commit peak fell 1773064192 bytes (1.65 GiB); all WAVs identical | Host commit saving, not physical working-set or VRAM saving; total compute ranges overlapped |
| Size the cached CUDA Flow metadata arena from its three unchanged graph capacities | Another ABBAAB screen: median private-commit peak fell 350171136 bytes (334 MiB); all WAVs identical | No stable compute speedup; uncached setup and CPU/Apple allocations unchanged |

The vocoder arena reserves 407002336 bytes instead of 2 GiB; actual metadata
use peaked at 21983072 bytes. Cached Flow reserves 1796759488 bytes instead of
2 GiB; actual use was 261894048 bytes. Original node/leaf capacities and tensor
operations are unchanged. The vocoder saving reduced sampled in-session commit
transients, while loading/session reconstruction still dominated the lifetime
peak. Flow lowered that lifetime peak by 355336192 bytes in a separate full
pipeline comparison. Do not add these overlapping peak measurements.

## Deployment and rollback

Rebuild the native CUDA server to apply metadata sizing. Enable idle release
with these inference environment variables in the existing local launcher:

```text
OMNI_VISION_LAZY=1
OMNI_VISION_IDLE_SECONDS=60
```

The vision loader is included because released vision must reload before the
next image is submitted. The lifecycle mutex excludes active sessions; joined
thread/pipeline checks guard release. Closing the camera alone does not unload
vision. Set `OMNI_VISION_IDLE_SECONDS=0` to retain it across idle sessions, or
`OMNI_VISION_LAZY=0` for eager loading. CPU/Apple do not enable this policy.
Local machine paths and generated runtime configuration are not versioned.

The browser change applies when the updated player is loaded; refresh the page
and open a new session. The existing playback-delay control and explicit zero
setting remain available. Startup allocation is excluded from starvation counts.

For code rollback, stop the service, revert this commit's source changes,
rebuild and restart. Remove the local idle setting if reverting its loader.
Preserve unrelated local edits; the evidence directory also contains scoped
reverse patches with successful dry-run checks.

## Full pipeline validation

Fixed seed 42, LLM greedy/top-k 1, TTS temperature 0.8 and matching histories:
two 36-input audio sessions and two 120-input mixed/interruption sessions for
the vocoder, followed by another two 120-input mixed sessions for Flow, preserved
corresponding text, LISTEN/SPEAK and raw float PCM exactly. PCM was finite and
unclipped, with no old-epoch audio after interruption acknowledgement.

The final build completed 600 mixed inputs in 599.23 s. Prepared-input send to
response p50/p95/maximum was 216.05/322.46/977.01 ms; first SPEAK input to nonempty
PCM was 467.40 ms. All 600 text/decision results and 101.24 s raw PCM matched an
earlier same-history run exactly. That earlier run is an exactness reference,
not a fresh speed baseline. Six 100-input blocks had p50 214.75–218.67 ms;
KV length decreased 30 times and peaked at 3583, with no continuing backlog.
57 WDDM samples showed dedicated memory reaching about 9.36 GiB then plateauing,
shared memory 94–96 MiB and working set about 6.22–6.35 GiB. Whole-card peak was
10387 MiB including the desktop. No paging claim is made.

After normal-service restoration, two audio and two video direct-interruption
sessions each completed 30 inputs and recovered the new Beijing answer. All
had finite/unclipped PCM, queue depth zero and no stale-epoch audio. Initial
vision loading took 1217.19 ms and caused one following input to wait 198.45 ms;
the queue then cleared. A resident-vision repeat had maximum processing 289.60 ms
and queue wait 0.038 ms. Notification to fresh PCM was 4488–4540 ms, including
the continuing spoken question and turn decision, not speech-end latency.
CUDA Release builds and 70 relevant Python tests passed in the tested local tree.
Before saving the scoped commit, its exported source and headers independently
compiled the modified `omni.cpp`, `token2wav-impl.cpp` and `server-omni.cpp` CUDA
translation units. The full-tree measurements above retain their stated scope.

Raw measurements, commands, configuration/binary hashes and recordings remain
in ignored `verification/continuous-optimization/`: `browser-playback-*`,
`vision-residency-summary.json`, `round-6/`, `round-7/`, `selected7-600-summary.json`
and `selected7-direct-summary.json`. They contain local artifacts and are not
committed. The exact saved source selection and commit checks are recorded under
`save-dev/` there.

Flow linear F16, vocoder residual-convolution F16, positional-tensor pointer
caching and skipping a redundant stream wait showed no stable whole-workload
benefit and were reverted. Human listening, physical playback start/stop latency,
microphone/AEC/noise behavior and multihour stability remain unaccepted. Fixture
PCM equality, ASR, embeddings and clipping checks do not replace those checks.
Further quantization requires original high-precision weights and representative
calibration data; no weights were converted or replaced in the selected build.

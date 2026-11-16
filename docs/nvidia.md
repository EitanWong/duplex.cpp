# Native Windows CUDA — RTX 3060 12 GB

The native Windows path has been built and tested on an i7-10700F (8 cores / 16
threads), 32 GB RAM and RTX 3060 12 GB. It uses the original F16 audio, vision and
TTS modules with the Q4_K_M LLM. The Apple profile and launcher are preserved.
See [target-hardware measurements](performance-rtx3060.md) for evidence and limits.
The [inference optimization report](inference-optimization.md) adds pipeline
bottlenecks, table/workspace savings, CUDA experiments and a research ranking of
quantization, pruning, sparsity and alternative runtimes.

## Setup

Use Visual Studio 2022 x64 C++ tools, a compatible Windows SDK, CUDA 12.8 or newer,
Python 3.11/3.12, CMake and Ninja. The driver must support the toolkit. From the
repository root in PowerShell:

```powershell
git switch dev
python -m venv .venv
.venv/Scripts/python -m pip install -r runtime/requirements-native.txt
.venv/Scripts/python -m pip install cmake ninja
```

Download the [official model bundle](models.md). Then generate a local config:

```powershell
.venv/Scripts/python scripts/configure.py --profile nvidia-rtx3060 `
  --model-dir E:/Models/MiniCPM-o-4_5-gguf
```

An existing config requires `--force` to replace it. The generated config and all
weights remain local. `selected-profile.json` continues to describe the preserved
Apple baseline; the Windows launcher reads `runtime/config.json`.

## Build and start

In a VS 2022 developer PowerShell:

```powershell
./scripts/windows/build.ps1 -CudaRoot $env:CUDA_PATH -Jobs 8
```

Alternatively pass `-VsDevCmd 'C:/path/to/VS/Common7/Tools/VsDevCmd.bat'`. If SDK
auto-discovery fails, also pass `-WindowsSdkRoot 'C:/Program Files (x86)/Windows Kits/10'`.
These paths may live on another drive. The script imports settings into the current
process and builds Release binaries in `engine/build-cuda/bin` for SM 8.6 (Ampere),
with AVX2, CUDA Graphs and Flash Attention. It does not install a driver or change
system settings. Other GPU architectures need a corresponding build configuration.
The script also repairs misdecoded localized MSVC include prefixes and rebuilds
old objects with missing header dependencies, avoiding incompatible class layouts
after incremental builds.

`-AllKvQuants` optionally builds mixed K/V Flash Attention kernels. It is needed
for experiments with independent `OMNI_LLM_K_TYPE` / `OMNI_LLM_V_TYPE` settings;
ordinary builds and the NVIDIA profile retain F16 KV defaults. See
[KV cache measurements and pruning research](kv-cache-rtx3060.md).

Start both services in the background, check status, or stop them:

```powershell
./scripts/windows/start.ps1
./scripts/windows/status.ps1
./scripts/windows/stop.ps1
```

The same folder contains `start.bat`, `status.bat` and `stop.bat` for double-click
use with Windows PowerShell 5.1. They leave the console open to show the result.
No administrator rights or PowerShell 7 installation are required. From a shell
with a restrictive execution policy, invoke the entry explicitly:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ./scripts/windows/start.ps1
# Optional overrides:
./scripts/windows/start.ps1 -CudaRoot 'E:/CUDA/v12.8' -TimeoutSeconds 300 -OpenBrowser
```

The controller reads `runtime/config.json`, preserves its HTTP/HTTPS settings,
starts Worker before Gateway and waits for model readiness and Gateway worker
registration. It prints the page address; the default is **http://localhost:8040/omni**.
Use headphones. Repeated starts reuse project services. The controller rejects
unrelated port owners. Stop checks process identity and terminates the project's
Gateway, Worker and native process tree, including services previously launched
through `run.ps1`. Stopping interrupts active sessions. Repeated stops are safe.

CUDA is taken from `-CudaRoot`, `CUDA_PATH`, or the successful CUDA build's CMake
cache; existing DLLs on `PATH` also work. Process state stays in ignored
`.local/state/windows.json`, and timestamped logs are in `logs/windows-*.log`.
Startup failures roll back only newly launched services. A start/stop lock prevents
simultaneous controllers. The scripts work independently of the current directory.
The hidden launcher uses explicit stdio handles so callers that capture output
also return promptly on Windows PowerShell 5.1. Local checks covered cold start,
repeated start/stop, the `.bat` entries, unrelated port conflicts, the control lock,
and cleanup after an intentionally failed Gateway start.

For foreground debugging, use `scripts/windows/run.ps1 -Role worker` and
`scripts/windows/run.ps1 -Role gateway` in separate terminals; Ctrl+C stops each.
Native `.exe` discovery and DLL search paths are supported. Use `/health` and
`/workers` to check the Gateway and model readiness.

## Private LAN access / 局域网访问

To use a microphone from another computer or phone, enable HTTPS with the
server's LAN IPv4 address (replace the example with your own):

```powershell
.venv/Scripts/python -m pip install cryptography
.venv/Scripts/python scripts/configure-lan.py --ip 192.168.1.10
# Apply the new settings:
./scripts/windows/stop.ps1
./scripts/windows/start.ps1
```

Open **https://192.168.1.10:8040/omni** from the same LAN. This address also works
on the server itself. The setting persists in ignored `runtime/config.json`;
regenerating that file with `configure.py --force` restores the default local mode.
The Worker and native inference server remain on loopback. The browser uses WSS
automatically. A changed LAN IP requires rerunning the LAN configuration and
restarting the Gateway; the existing local CA is reused.

Each client must trust the public **[LAN CA certificate](https://192.168.1.10:8040/static/lan-ca.crt)**
for reliable microphone access. This is an example address; use your server IP.
You can also copy `runtime/certs/lan-ca.crt` directly to the client. Only the public
CA certificate is published; private keys stay in ignored `runtime/certs/`.

- Windows: import the public CA into the current user's **Trusted Root Certification Authorities**.
- iPhone/iPad: install the downloaded profile, then enable full trust under
  **Settings → General → About → Certificate Trust Settings**.
- Android: install the public CA under the device's security / certificate settings;
  exact menu names depend on the manufacturer and browser.

If Windows Firewall blocks the connection, an administrator can add a rule for
this service's port, restricted to the private network and local subnet:

```powershell
New-NetFirewallRule -DisplayName 'duplex.cpp LAN HTTPS' -Direction Inbound `
  -Action Allow -Protocol TCP -LocalPort 8040 -RemoteAddress LocalSubnet -Profile Private
```

局域网浏览器使用麦克风需要 HTTPS。在其他设备信任上述公有 CA 证书后，访问
`https://服务器局域网IP:8040/omni` 并允许麦克风。电脑与手机需在能互相访问的
局域网内；访客 Wi-Fi 的客户端隔离会阻止连接。当前 Gateway 没有登录认证，
此配置适用于可信局域网，不应配置公网端口映射。

## Selected settings

| Component | RTX 3060 profile |
| --- | --- |
| LLM | All layers on GPU; 4096 context; F16 KV; Flash Attention |
| LLM CPU dispatch | 4 decode threads, 8 batch threads; batch 512 / ubatch 128 |
| TTS decoder | Original F16 weights/KV, 4096 context, GPU layers 99; hidden-only output |
| Acoustic head | CPU AVX2, eight outputs at once, original multiply/add order |
| TTS tables | Unused tied text table on CPU; conditioning uploads only requested F32 rows |
| Audio workspace | Reserve a 100-frame query against full history; grow for longer inputs |
| Token2Wav / vocoder | CUDA device 0, existing CUDA Graph support and startup warmup; original flow steps |
| Concurrency | One worker; audio and vision encoder paths retain their existing overlap |
| Disk activity | Optional intermediate recordings and merged WAV output disabled |
| Vision attention | CUDA Flash Attention; 72-wide heads padded to 80 with original scale |
| Vision preprocessing | Exact bicubic arithmetic with cached coordinates and 4 row workers |
| Session reuse | Resident models reused across modes; exact token prefixes reused for plain text |

`OMNI_VISION_BACKEND=metal` is the inherited name for the generic ggml GPU vision
path. In a CUDA build it uses CUDA. The value does not enable Metal on NVIDIA.

The profile lives in `profiles/nvidia-rtx3060.json`. Explicit process environment
settings override profile values. For acoustic-head rollback, set
`$env:OMNI_TTS_CPU_HEAD='scalar'` before starting the worker. Clear overrides with
`Remove-Item Env:OMNI_TTS_CPU_HEAD` before returning to the profile default.
`OMNI_SAMPLER_SEED=42` can be used for repeatable comparisons; it is not forced for
normal conversation. `GGML_CUDA_DISABLE_GRAPHS=1` disables graphs for an A/B trial.

Additional rollback and comparison switches (restart the worker after changes):

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `OMNI_VERBOSE` | `0` | Set `1` for timestamped engine diagnostics |
| `OMNI_KEEP_ARTIFACTS` | `0` | Set `1` to save hidden states, token dumps and intermediate WAV files |
| `OMNI_VISION_FLASH_ATTN` | `1` | Set `0` to restore the original vision attention math |
| `OMNI_REUSE_MODELS` | `1` | Set `0` to rebuild mode-specific modules on mode changes |
| `OMNI_TEXT_PREFIX_CACHE` | `1` | Set `0` for uncached plain-text prefill |
| `OMNI_LLM_KV_TYPE` | `f16` | Optional `q8_0` saves VRAM; measured replay was slightly slower |
| `OMNI_T2W_DEVICE_MEL` | `0` | Optional `1` keeps mel handoff/cache on the same CUDA device |
| `OMNI_TTS_TOKEN_EMBD_CPU` | `1` | Set `0` to restore the 232.36 MiB tied text table on GPU; text-logit mode bypasses this optimization |
| `OMNI_TTS_HOST_EMBED` | `1` | Set `0` to restore the 445.50 MiB conditioning table on GPU when ordinary chat first uses it |
| `OMNI_AUDIO_RESERVE_FRAMES` | `100` | Set `3000` for the original large audio reservation; longer actual inputs still grow the allocator |
| `OMNI_VISION_FAST_RESIZE` | `1` | Set `0` for the original bicubic implementation |
| `OMNI_VISION_RESIZE_THREADS` | `4` | Fast-resize row workers, from `1` to `8` |
| `OMNI_TTS_CUDA_HEAD` | `0` | Experimental projection kernel; isolated logits match but repeated sampled speech diverged, so keep disabled |
| `OMNI_AUDIO_FLASH_ATTN` | `0` | Experimental audio attention; changed embeddings/text without demonstrated full-pipeline speedup |

Vision Flash Attention casts K/V to F16 and changes numerical results; it is
covered by embedding and speech replay comparisons, with human quality review
still pending. Prefix reuse requires matching input tokens, recomputes the last
input token for fresh logits, and falls back for speech, media and window shifts.
The device mel path preserves the tested WAV exactly but has not shown a stable
speed advantage, so it is available for experiments and disabled in the profile.
CPU, Metal and ANE use their existing handoff paths.

CUDA host metadata arenas for the vocoder and cached Flow session are sized
from their unchanged graph capacities. Local ABBAAB screens reduced private
commit peaks by 1.65 GiB and 334 MiB respectively, preserving tested text and
raw PCM across repeated mixed sessions. These are Windows commit-budget savings;
physical working set and VRAM did not fall correspondingly, and compute timings
showed no stable improvement. See the [voice report](voice-optimization-rtx3060.md)
for workload limits, the overlapping lifetime peaks and rollback evidence.

The selected memory/resize changes preserved the tested duplex text and PCM,
including two sessions at the normal 0.8 acoustic temperature. Two greedy
audio/video rounds measured p50 **236.77 → 207.92 ms** and **233.54 → 207.97 ms**;
sampled whole-card peak was **10921 → 10408 MiB**. Ordinary chat additionally
avoids the conditioning table. These timings start after a full one-second
input arrives; they exclude microphone collection and playback. Audio-only
processing showed no stable latency improvement. Detailed evidence and the
ordinary-chat reproducibility limitation are in the optimization report.

## Reproduce validation

With the worker ready and no browser session using it:

```powershell
.venv/Scripts/python scripts/benchmarks/native.py --count 36 --rounds 2 `
  --output verification/native-duplex
.venv/Scripts/python scripts/benchmarks/native.py --audio-only --rounds 1 `
  --output verification/native-audio
.venv/Scripts/python scripts/benchmarks/native.py --count 120 --rounds 1 `
  --interrupt-at 5 --output verification/native-long
.venv/Scripts/python scripts/benchmarks/session-reuse.py `
  --output verification/session-reuse.json
```

Alternatively, start and clean up an isolated native process while recording
settings and whole-GPU usage:

```powershell
.venv/Scripts/python scripts/benchmarks/pipeline.py `
  --model-dir models/MiniCPM-o-4_5-gguf --output verification/isolated --rounds 2
```

Run competing variants sequentially. Add `--audio-only --tts-temperature 0.8`
for normal speech sampling. Add `--video-start-at 8` to test activating video
after voice-only inputs.

The replay uses the checked-in fixture, sends one input per second, and receives
PCM concurrently. It measures the native `/backend` protocol, not microphone,
browser playback or network latency. Reports and audio stay in ignored
`verification/`. The benchmark explicitly leaves human speech quality acceptance
false. Compare matching seeds, inputs, context and precision, and include startup
and graph-capture costs separately from steady state.

The 12 GB device has limited headroom with every module resident. Do not infer
that more workers or a larger context will fit from the short replay. See the
measurement report for peak VRAM and the remaining acceptance work.

The current profile uses 4096 context to reserve more VRAM for speech and the
desktop. The earlier 8192 configuration remains documented in the measurement
report. Increasing context consumes more VRAM; a smaller context retains less
conversation history. Browser chat uses the native structured turn-based
protocol, including generation limits, history, speech toggles and cancellation.
LLM temperature and TTS temperature are independent. The replay explicitly sets
both to zero for deterministic comparisons; everyday speech keeps TTS at 0.8.

The NVIDIA profile now uses `OMNI_VISION_LAZY=1` and
`OMNI_TTS_CPU_HEAD=avx2_ilp`. Voice-only replay saved about 1.14 GiB of sampled
whole-card peak VRAM; the acoustic operator is 1.65–1.74× faster than the earlier
AVX2 path. Whole-pipeline latency gains were inconsistent. The first image pays
a model load (751 ms in one cached-file trial). Vision remains resident during
an active session; `OMNI_VISION_IDLE_SECONDS=60` releases it after a closed
session is idle for 60 seconds, recovering about 1.11 GiB in four local probes.
Reopening images then pays another load. Set the timeout to `0` to retain vision
across sessions, `OMNI_VISION_LAZY=0` for eager camera loading and
`OMNI_TTS_CPU_HEAD=avx2` for the earlier CPU path. See the
[voice-path report](voice-optimization-rtx3060.md) for repeated-session evidence
and the remaining optimization priorities.

## 中文

Windows 日常使用可双击 `scripts/windows/start.bat`、`status.bat`、`stop.bat`，
或运行同名 PowerShell 脚本。启动会等待模型就绪并显示配置中的 HTTPS 地址，
重复启动复用现有服务；停止会中断会话并退出本项目进程、释放引擎显存。
日志在 `logs/windows-*.log`，后台运行不需要保留启动窗口。

已在本机 i7-10700F、32 GB 内存、RTX 3060 12 GB 上完成原生 Windows CUDA
编译与实测，不需要 WSL2。按上述 PowerShell 流程配置、编译，分别启动 Worker
和 Gateway；模型仍为 Q4_K_M 主模型与原精度音频、视觉、TTS 模块。

配置针对延迟与显存预算选择线程数、Flash Attention、GPU 放置及 AVX2 声学输出层，
保留 Apple M5 路径。显存接近容量上限，默认单 Worker。局部算子加速倍数不代表
全链路提升；详细数值、音频一致性和长会话限制见[实测报告](performance-rtx3060.md)。
新的[优化研究](inference-optimization.md)包含词表与工作区削减、等值预处理、
CUDA 实验，以及量化、剪枝与后续优化的优先级。新 CUDA 声学输出层和音频
Flash Attention 默认关闭，选定配置保留 AVX2 输出层。

最新配置启用视觉按需加载和 AVX2 指令并行：纯语音回放的整卡峰值采样约减少
1.14 GiB，声学投影算子比此前 AVX2 快 1.65–1.74 倍；全链路延迟未显示稳定收益。
首次图片需要加载模型，活动会话内保持驻留；会话关闭后空闲 60 秒释放视觉，
四次本机验证均回收约 1.11 GiB，再次打开图片需重新加载。
`OMNI_VISION_IDLE_SECONDS=0` 可恢复跨会话常驻。完整对照、回退开关和后续方向见
[语音路径实测](voice-optimization-rtx3060.md)。

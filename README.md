# duplex.cpp

**Hardware-tuned, native full-duplex voice inference.**

[English](README.md) · [简体中文](README.zh-CN.md)

Run a local voice engine that listens while it speaks. Built on MiniCPM-o 4.5 and
llama.cpp-omni, duplex.cpp combines C++ inference, streaming speech generation,
interruption handling and a WebSocket API, with an optional browser interface.
The goal is fluid, high-quality conversation on a single consumer machine.

- **Native duplex:** model-driven listen/speak decisions with continuous audio input.
- **Streaming output:** incremental text and speech, with interruption and stale-audio suppression.
- **Hardware tuning:** preserve the Apple Silicon path while developing a CUDA path for NVIDIA GPUs.
- **Local serving:** a Python gateway and worker expose HTTP and WebSocket endpoints.

## Status

| Target | Status |
| --- | --- |
| Apple Silicon / M5 | Existing Metal, CoreML/ANE and Accelerate optimizations preserved |
| Windows / NVIDIA RTX 3060 12 GB | Native CUDA build, GPU operators and fixture replay verified; human speech quality acceptance pending |

Long-session audio continuity and speech quality remain active work. See the
[Apple baseline](docs/apple-silicon.md) and [NVIDIA development guide](docs/nvidia.md).

## Quick start — Apple Silicon

Requires macOS on Apple Silicon, Python 3.11, CMake, Ninja and Xcode Command Line Tools.

```bash
git clone https://github.com/EitanWong/duplex.cpp.git
cd duplex.cpp
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r runtime/requirements-native.txt
```

Prepare the [model files](docs/models.md), then configure and build:

```bash
python scripts/configure.py --model-dir /absolute/path/to/MiniCPM-o-4_5-gguf
./scripts/macos/build.sh
./scripts/macos/start.sh
```

Open **http://localhost:8040/omni** and enable the microphone. Headphones are
recommended. Use `./scripts/macos/status.sh` and `./scripts/macos/stop.sh`
to inspect or stop the service. Model weights are downloaded separately.

## Quick start — Windows / RTX 3060

With a Python environment, the [model files](docs/models.md), VS 2022 C++ tools,
CUDA 12.8+, CMake and Ninja ready, use PowerShell:

```powershell
.venv/Scripts/python scripts/configure.py --profile nvidia-rtx3060
./scripts/windows/build.ps1 -CudaRoot $env:CUDA_PATH
./scripts/windows/start.ps1
```

Open the address printed by the launcher (default **http://localhost:8040/omni**).
Use `./scripts/windows/status.ps1` to check readiness and `./scripts/windows/stop.ps1`
to stop all service processes. Double-clickable `.bat` entries are in the same folder.
The profile uses one worker, a 4096-token
context, CUDA Flash Attention and an AVX2 acoustic head. See the
[NVIDIA guide](docs/nvidia.md) for setup, measured results and limits.
The [optimization report](docs/inference-optimization.md) covers pipeline
bottlenecks, VRAM reductions, CUDA kernels and quantization/pruning priorities;
[voice-path measurements](docs/voice-optimization-rtx3060.md) cover lazy vision and AVX2 tuning.
For other devices on your LAN, follow the [HTTPS setup](docs/nvidia.md#private-lan-access--局域网访问).

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Gateway health; check `/workers` for model readiness |
| `GET /workers` | Worker availability and model state |
| `WS /ws/duplex/{session_id}` | Full-duplex voice session; use an `adx_` prefix for audio-only |

Connect → wait for `queue_done` → send `prepare` → wait for `prepared` → stream
16 kHz mono audio while receiving text and 24 kHz speech. See the
[protocol quick reference](docs/api.md) for message formats.
For remote browser access, put the gateway behind an authenticated HTTPS/WSS proxy;
the local launcher binds to loopback.

## Development

`main` holds stable versions; daily development uses `dev`. Promote validated
changes with a `dev` → `main` PR and **Squash and merge**. Keep experimental
branches local. See [CONTRIBUTING.md](CONTRIBUTING.md).

`engine/` contains the native engine, `runtime/` the service and client,
and `profiles/` the hardware profiles, including the preserved Apple M5 profile.
Platform entry points are grouped under `scripts/windows/` and `scripts/macos/`.
See the [script index](scripts/README.md), [project layout](docs/project-layout.md)
and [documentation index](docs/README.md).

## Credits and licensing

Based on [llama.cpp-omni](https://github.com/OpenBMB/llama.cpp-omni),
[llama.cpp](https://github.com/ggml-org/llama.cpp),
[MiniCPM-o](https://github.com/OpenBMB/MiniCPM-o) and
[MiniCPM-o-Demo](https://github.com/OpenBMB/MiniCPM-o-Demo).
See [LICENSE](LICENSE) and [THIRD_PARTY.md](THIRD_PARTY.md) for scope and upstream terms.

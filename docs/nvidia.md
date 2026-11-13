# NVIDIA development — RTX 3060 12 GB

Windows/NVIDIA is the next target. These commands are a starting point for bring-up,
not a measured RTX 3060 performance profile. The Apple launcher uses POSIX process
controls and macOS utilities; it is not a native Windows service manager.

## WSL2 starting point

Use a current Windows NVIDIA driver with WSL2 GPU support, an Ubuntu WSL2 environment,
Python 3.11, a C++17 compiler, CMake, Ninja and a compatible CUDA toolkit. Confirm
`nvidia-smi` works inside WSL2. Keep the checkout in the Linux filesystem.

```bash
git clone https://github.com/EitanWong/duplex.cpp.git
cd duplex.cpp
git switch dev
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r runtime/requirements-native.txt
python scripts/configure.py --model-dir /absolute/path/to/MiniCPM-o-4_5-gguf
cmake -S engine -B engine/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_METAL=OFF -DENABLE_COREML=OFF \
  -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build engine/build --target llama-omni-server -j 4
```

To begin testing, run the worker and gateway in separate activated terminals from
`runtime/`. Keep ports local; do not run the macOS supervisor.

```bash
# Terminal 1: generic GPU vision path (anything other than "coreml" selects it).
cd runtime
OMNI_VISION_BACKEND=metal CUDA_VISIBLE_DEVICES=0 \
  ../.venv/bin/python worker.py --host 127.0.0.1 --port 22440 --gpu-id 0 --worker-index 0
```

```bash
# Terminal 2
cd runtime
../.venv/bin/python gateway.py --host 127.0.0.1 --port 8040 \
  --workers 127.0.0.1:22440 --http
```

The legacy `metal` selector here means the non-CoreML ggml vision implementation;
the compiled backend determines GPU placement. It does not enable Metal on NVIDIA.
Backend naming and device placement need cleanup during CUDA bring-up.

## Native Windows build

With Visual Studio 2022 C++ tools, CMake and CUDA installed, run from a developer
PowerShell terminal:

```powershell
cmake -S engine -B engine/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DENABLE_COREML=OFF -DLLAMA_OPENSSL=OFF
cmake --build engine/build --config Release --target llama-omni-server -j 4
```

Native Windows worker lifecycle and `.exe` discovery still need adaptation and
on-device validation. WSL2 is the initial service bring-up route.

## Optimization gates

Start with Q4_K_M LLM, original F16 audio/TTS modules and one active worker. Measure
peak VRAM, including KV caches, workspace and Windows display use; 12 GB is a budget,
not a guarantee of fit at every context size.

Establish matching audio-only and audio/video baselines before tuning CUDA graphs,
attention, device placement, memory reuse and streaming overlap. Accept a change
only when repeatable latency or memory gains preserve speech quality, interruption
behavior and long-session playback continuity.

## 中文

RTX 3060 12 GB 是下一阶段目标，目前没有该显卡的实测性能结论。
建议先在 WSL2 中建立 CUDA 基线；原生 Windows 的服务进程管理和 `.exe` 查找仍需适配。
先测单 Worker、Q4_K_M 主模型与原精度音频/TTS，在计入 KV、临时缓冲和桌面占用后
确认显存余量，再逐项优化，保留 Apple 路径并同步验收音质、连续性与打断。

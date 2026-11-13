# duplex.cpp

**针对硬件深度优化的原生全双工语音推理引擎。**

[English](README.md) · [简体中文](README.zh-CN.md)

在本地运行能够一边听、一边说的语音引擎。duplex.cpp 基于 MiniCPM-o 4.5 和
llama.cpp-omni，整合 C++ 推理、流式语音生成、打断处理与 WebSocket API，
提供可选的浏览器界面，目标是在单台消费级主机上实现流畅、高质量的语音对话。

- **原生双工**：模型自主决定倾听与发声，持续接收音频输入。
- **流式输出**：增量生成文本与语音，支持打断与过期音频抑制。
- **硬件优化**：保留 Apple Silicon 优化路径，继续面向 NVIDIA GPU 深度优化。
- **本地服务**：通过 Python Gateway 与 Worker 提供 HTTP / WebSocket 接口。

## 当前状态

| 平台 | 状态 |
| --- | --- |
| Apple Silicon / M5 | 已保留 Metal、CoreML/ANE 与 Accelerate 优化 |
| Windows / NVIDIA RTX 3060 12 GB | 下一阶段优化目标，性能与兼容性待实机验收 |

长会话语音连续性与音质仍在持续改进。详见
[Apple 优化基线](docs/apple-silicon.md)和 [NVIDIA 开发指南](docs/nvidia.md)。

## 快速开始 — Apple Silicon

需要 Apple Silicon Mac、Python 3.11、CMake、Ninja 与 Xcode Command Line Tools。

```bash
git clone https://github.com/EitanWong/duplex.cpp.git
cd duplex.cpp
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r runtime/requirements-native.txt
```

准备[模型文件](docs/models.md)，然后配置、编译并启动：

```bash
python scripts/configure.py --model-dir /absolute/path/to/MiniCPM-o-4_5-gguf
./scripts/build-release.sh
./project-control/start.sh
```

访问 **http://localhost:8040/omni** 并开启麦克风，建议使用耳机。
通过 `./project-control/status.sh` 查看状态，`./project-control/stop.sh` 停止服务。
模型权重需单独下载。

## API

| 接口 | 用途 |
| --- | --- |
| `GET /health` | Gateway 健康状态；模型就绪状态请查看 `/workers` |
| `GET /workers` | Worker 可用性与模型状态 |
| `WS /ws/duplex/{session_id}` | 全双工会话；纯音频会话使用 `adx_` 前缀 |

连接后等待 `queue_done`，发送 `prepare`，收到 `prepared` 后持续发送
16 kHz 单声道音频，同时接收文本与 24 kHz 语音。
消息格式见 [API 速查](docs/api.md)。远程浏览器访问需通过带认证的 HTTPS/WSS
反向代理；本地启动器默认绑定回环地址。

## 开发

`main` 保存稳定版本，日常开发统一在 `dev`。验证后提交 `dev` → `main` PR，
统一 **Squash and merge**；实验分支只在本地保留。
详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

`engine/` 为原生引擎，`runtime/` 为服务与客户端，
`profiles/apple-m5.json` 保存 Apple 优化配置。

## 致谢与许可

基于 [llama.cpp-omni](https://github.com/OpenBMB/llama.cpp-omni)、
[llama.cpp](https://github.com/ggml-org/llama.cpp)、
[MiniCPM-o](https://github.com/OpenBMB/MiniCPM-o) 与
[MiniCPM-o-Demo](https://github.com/OpenBMB/MiniCPM-o-Demo)。
许可范围与上游条款见 [LICENSE](LICENSE) 和 [THIRD_PARTY.md](THIRD_PARTY.md)。

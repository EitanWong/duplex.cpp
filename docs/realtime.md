# OpenAI Realtime audio API / 实时语音接口

This page describes OpenAI events selected with `model=gpt-realtime`. For the
ModelBest official audio/video format and `MiniCPM-o-4.5-Realtime`, see
[ModelBest Realtime](modelbest-realtime.md). The two protocols share port 8006
but use different event fields and audio formats. Specify `model` explicitly.

An isolated, authenticated WebSocket adapter for the existing duplex engine.
It implements the OpenAI Realtime **audio event subset** in GA and Beta dialects;
the model remains MiniCPM-o, with the native hardware profile and speech decisions.
It does not implement the complete OpenAI service, WebRTC, SIP, Agents tool calling,
or OpenAI-hosted model/voice quality. Existing `/ws/duplex` clients keep working.

这是现有双工引擎的独立 OpenAI Realtime **音频协议兼容层**，支持 GA 与 Beta
事件名称。自动模式复用双工会话，手动提交模式复用 turn-based 音频会话；
两者沿用已有 Gateway 的排队、Worker 分配及会话清理。
引擎、Apple M5 配置和 CUDA 推理参数不变。

## Start / 启动

Start the existing Gateway and Worker first. The API starts separately and reads
the upstream address and local TLS paths from ignored `runtime/config.json`.
It defaults to port **8006**, independent of the demo Gateway's usual **8040**.

Windows（已有局域网 HTTPS 配置）：

```powershell
./scripts/windows/start.ps1
./scripts/windows/realtime.ps1 -Action start
./scripts/windows/realtime.ps1 -Action status
# Stop only the compatibility API / 仅停止兼容入口：
./scripts/windows/realtime.ps1 -Action stop
```

`realtime.ps1` defaults to `0.0.0.0:8006`, requires TLS for a LAN listener, and
starts a hidden background process with logs in `logs/realtime.*.log`.
Repeated starts reuse the same listener; stop before changing its options.
The adapter is not automatically started/stopped by the existing service commands.

尚未配置局域网证书时，先按 [Windows LAN guide](nvidia.md#private-lan-access--局域网访问)
运行 `scripts/configure-lan.py --ip <本机局域网IP>`，并按该指南应用配置。
客户端应信任 `runtime/certs/lan-ca.crt`，并使用证书包含的局域网 IP 或 `localhost`。
Windows 防火墙启用时，允许私有网络的 TCP 8006；不必开放 Worker 或引擎端口。

前台运行（Windows 与 macOS 使用相应 Python 虚拟环境）：

```bash
python runtime/realtime.py --host 127.0.0.1 --port 8006 --http
```

This is the local `ws://127.0.0.1:8006/v1/realtime?mode=audio&model=gpt-realtime` endpoint.
For LAN use, specify `--host <LAN-IP>` and certificate/key files, or let the adapter
reuse HTTPS paths in `runtime/config.json`. For a separately hosted upstream,
use `--gateway-url wss://host:8040 --gateway-ca /path/to/ca.crt`.
Upstream certificate verification is always enabled.

## Authentication / 认证

The first start creates a random key in **`.local/realtime/api-key`**. Set
`MODELBEST_API_KEY` or `DUPLEX_REALTIME_API_KEY` before starting to supply a key instead
(the former takes precedence); keys need at
least 32 ASCII characters without whitespace. Keys, logs and service state are
ignored by Git. The key is never printed by the launcher or placed in URLs.

服务器客户端通过请求头传入密钥：

```text
Authorization: Bearer <API_KEY>
```

| Endpoint | Purpose / 用途 |
| --- | --- |
| `WS /v1/realtime?model=gpt-realtime` | GA audio events; adding `mode=audio` is also accepted |
| `GET /v1/models` | OpenAI aliases: `duplex`, `gpt-realtime`, `gpt-4o-realtime-preview`; also lists the ModelBest protocol alias; all run the same local model |
| `POST /v1/realtime/client_secrets` | Authenticated backend creates a browser token, 60 seconds by default |
| `POST /v1/realtime/sessions` | Beta form of browser token creation |
| `GET /health` | Adapter liveness only; does not imply model readiness |

Add `OpenAI-Beta: realtime=v1` for Beta events (`response.audio.delta`,
`response.audio_transcript.delta`). Without it the adapter emits GA events
(`response.output_audio.delta`, `response.output_audio_transcript.delta`).

浏览器不要保存主密钥。你的业务后端用主密钥调用 `client_secrets`，只把短期
`value` 交给浏览器。可传入 `session` 预设和
`expires_after: {"anchor":"created_at","seconds":60}`（范围 10–600 秒）。
短期密钥只用于 WebSocket，不能签发新密钥；过期后不能建立新连接。
会话最长 30 分钟。短期密钥存于单进程内存，服务重启后失效。

```javascript
// Obtain clientSecret from your own authenticated backend first.
const ws = new WebSocket("wss://voice.example.com/v1/realtime?model=gpt-realtime", [
  "realtime", "openai-insecure-api-key." + clientSecret
]);
```

Use `--allowed-origins https://your-app.example` (or Windows launcher
`-AllowedOrigins`) to restrict browser origins. Non-browser clients still use
Bearer authentication. Only the compatibility API is published: no demo admin,
recording download, upload, static files or arbitrary proxy routes.

## Client / 调用示例

Input/output audio is **24 kHz, mono, signed PCM16 little-endian**, Base64-encoded,
without a WAV header. The adapter resamples input to native 16 kHz float32,
preserving filter state across packets, and converts 24 kHz float32 output to PCM16.

```python
import asyncio, base64, json, ssl
from pathlib import Path
from websockets.asyncio.client import connect

async def main():
    key = Path(".local/realtime/api-key").read_text().strip()
    context = ssl.create_default_context(cafile="runtime/certs/lan-ca.crt")
    async with connect(
        "wss://YOUR_LAN_IP:8006/v1/realtime?model=gpt-realtime",
        additional_headers={"Authorization": "Bearer " + key},
        ssl=context, proxy=None,
    ) as ws:
        print(json.loads(await ws.recv())["type"])  # session.created
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "type": "realtime", "instructions": "请用简短中文回答。",
                "audio": {"input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "turn_detection": {"type": "server_vad"}
                }}
            }
        }))
        async def send_microphone():
            # Your async capture source, e.g. 100 ms per packet.
            # Keep sending silence between utterances and during playback.
            async for pcm16_bytes in microphone_pcm16_chunks():
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm16_bytes).decode()
                }))
        async def receive():
            async for raw in ws:
                event = json.loads(raw)
                if event["type"] == "response.output_audio.delta":
                    play_pcm16(base64.b64decode(event["delta"]), sample_rate=24000)
                elif event["type"] == "response.output_audio_transcript.delta":
                    print(event["delta"], end="", flush=True)
                elif event["type"] == "response.done":
                    if event["response"]["status"] == "cancelled":
                        flush_playback_queue()
                elif event["type"] == "error":
                    print(event["error"])
        await asyncio.gather(send_microphone(), receive())

asyncio.run(main())
```

`microphone_pcm16_chunks`, `play_pcm16` and `flush_playback_queue` are
application-provided capture/playback callbacks. A complete file replay client
is included, requiring no new dependency:

```powershell
.venv/Scripts/python scripts/benchmarks/realtime.py `
  --url "wss://YOUR_LAN_IP:8006/v1/realtime?mode=audio&model=gpt-realtime" `
  --ca runtime/certs/lan-ca.crt --output verification/realtime
```

For the OpenAI Python SDK, use `base_url="https://YOUR_LAN_IP:8006/v1"`, the local
key and a WebSocket transport that trusts your CA. Do not use OpenAI's default
endpoint. Event payloads follow the [official SDK types](https://github.com/openai/openai-python/tree/main/src/openai/types/realtime).
The automated smoke client uses raw WebSocket events; an installed OpenAI SDK is
not required or covered by that test.

## Supported scope / 兼容范围

| Client event | Behavior / 行为 |
| --- | --- |
| `session.update` | Instructions, PCM16 formats, audio/text output, server VAD or manual turns; configure before first audio |
| `input_audio_buffer.append` | 24 kHz PCM16, preferably 20–100 ms packets at real-time speed |
| `input_audio_buffer.commit` | Create a user audio item; does not itself start a manual response |
| `input_audio_buffer.clear` | Clear unsent/uncommitted input; already delivered native context cannot be rewound |
| `response.create` | After commit in manual mode; optional `metadata` only |
| `response.cancel` | Cancel, mute native output, suppress obsolete audio epochs; client must flush its playback |
| `conversation.item.create` | User `input_audio` only, with turn detection disabled |

Server events include `session.created/updated`, `conversation.created`,
`conversation.item.created`, `input_audio_buffer.speech_started/stopped/committed/cleared`,
`response.created`, `response.output_item.added/done`, `response.content_part.added/done`,
audio/transcript (or text) `delta/done`, `response.done`, and structured `error` events.
`response.done` waits for native `speech_end`, so late vocoder audio is not cut off
just because the model has returned to listening.

自动模式默认使用简单能量 VAD（20 ms 分析帧），不是 OpenAI 的 VAD 模型。
VAD 产生语音起止和提交事件，具体回复时机仍由双工引擎决定。
`threshold` 的 0–1 范围映射到本地 RMS 阈值；`prefix_padding_ms` 用于事件时间标记，
连续音频已完整送入引擎。默认静音 500 ms 触发提交。
客户端必须并行收发：用户暂停时继续发送静音，引擎说话时也继续采集。
兼容层每秒发一个原生输入块，不会等待回复结束再收音。

手动模式设置 `audio.input.turn_detection: null`，发送音频后依次执行
`input_audio_buffer.commit`、`response.create`。也支持
`server_vad.create_response: false` 后显式 `response.create`。
手动模式复用已有原生 turn-based 音频路径，先完整送入已提交音频再开始生成，
逐轮携带此前的音频和助手文本。为限制内存，最多保留 90 秒输入及 32 条历史消息，
超限返回 `conversation_context_full`，需新建会话。回放脚本加 `--manual` 可验证。
输入最多缓冲 30 秒、最多 2 个推理块在途，防止慢引擎无限积压；超限会关闭连接。
默认最多接入 8 条连接；实际并发推理由现有 Worker 数量决定，通常为 1。
排队/准备超时 120 秒，排队期间也受缓冲限制。

Unsupported options return errors: G.711, semantic VAD, input transcription,
noise reduction, text/image prompts, tools/functions, arbitrary model selection,
OpenAI voice switching, per-response inference settings, item delete/truncate/retrieve,
WebRTC `/v1/realtime/calls`, and SIP. `alloy` aliases the existing reference voice.
Usage is `null`; token billing is not fabricated. Session changes that would
require rebuilding native context are rejected after audio starts.
Cancellation does not erase native conversation history.
Automatic full-duplex mode cannot disable the native speech interruption gate;
`interrupt_response: false` is rejected there. Use manual turns for controlled responses.

原 Gateway 的录制策略仍然生效，输出保存在本机忽略目录，不通过此 API 对外发布。

## Internet deployment / 公网部署

Use a public domain with DNS pointing to your reverse proxy. On the same host,
bind the isolated API to loopback with HTTP:

```powershell
./scripts/windows/realtime.ps1 -Action stop
./scripts/windows/realtime.ps1 -Action start -ListenHost 127.0.0.1 -Http
```

Copy [realtime.Caddyfile](../deploy/realtime.Caddyfile), replace
`voice.example.com` with your domain, and run Caddy with that file. Caddy handles
WebSocket upgrades and provisions public TLS. Forward TCP 80/443 to Caddy; keep
8006, the demo Gateway, Worker and native engine ports off the public router.
If using an existing tunnel, point it at this loopback API instead of the demo.

公网最终地址为 `wss://你的域名/v1/realtime?model=gpt-realtime`，仍需 Bearer
或短期密钥。配置文件不自动修改 DNS、路由器或创建外网隧道。只有配置真实域名、
证书和外部路由并从外部网络验证后，才能确认公网可达。

## Verification / 验证

```powershell
cd runtime
../.venv/Scripts/python -m pytest tests/test_realtime.py -q
```

Tests cover GA/Beta lifecycles, PCM encoding and streaming resampler boundaries,
manual commits, VAD, cancellation/stale epochs, auth/expired browser tokens,
origin/capacity limits, malformed events, upstream failure and teardown.
The replay client stores event traces and `response.wav` in ignored `verification/`.
Listen to the WAV on target hardware to assess speech quality; waveform checks
alone do not establish human listening quality.

2026-11-16，Windows / RTX 3060 的真实模型回放记录：

| Case | Observed result |
| --- | --- |
| GA automatic, 25-second replay | PCM16 audio and transcript; native interruption produced `cancelled`, a subsequent response produced `completed` |
| Beta manual, 11.34-second source WAV | `completed`, 5.04 seconds of output audio; response finished 14.17 seconds after starting the real-time replay |
| LAN TLS/auth | Trusted CA verified; unauthenticated WebSocket rejected with HTTP 401; authenticated models endpoint works; admin/worker routes return 404 |

手动回放的 14.17 秒包含输入音频发送时间，不能当成提交后推理延迟。
输出 WAV 已检查采样率、有限样本和有效音量，人工音质验收尚未完成。
协议/音频/边界与已有原生会话回归检查共 46 项通过。此验证不代表已完成
公网连通、macOS 运行或所有 OpenAI SDK 客户端的兼容验证。

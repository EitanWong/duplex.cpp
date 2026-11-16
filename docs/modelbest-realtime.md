# ModelBest Realtime / 面壁实时音视频接口

独立入口 `WS /v1/realtime` 参考面壁官方平台的音频/视频事件格式，复用已有
Gateway 排队、Worker 分配和原生 MiniCPM-o 双工推理。接口默认端口 **8006**。
这是本地协议适配，支持范围见下表；不是面壁托管服务的完整实现。

协议来源：[官方 Realtime 文档](https://platform.modelbest.cn/console/docs/api/realtime)、
[全双工模型文档](https://platform.modelbest.cn/console/docs/models/full-duplex)，
并对照官方平台实际使用的 WebSocket 客户端；核对日期为 2026-11-16。

接入 qwen-audio-agent 2.1.0 时需加 `protocol=qwen-audio-agent`，输出其内置
MiniCPM-o Provider 读取的统一事件格式；配置与启动见[接入指南](qwen-audio-agent.md)。

## 启动与连接

```powershell
./scripts/windows/start.ps1
./scripts/windows/realtime.ps1 -Action start
./scripts/windows/realtime.ps1 -Action status
```

启动器默认监听 `0.0.0.0:8006`，沿用忽略文件 `runtime/config.json` 的 Gateway
地址和 TLS 证书。日志为 `logs/realtime.stdout.log`、`logs/realtime.stderr.log`。
此入口独立启停；`realtime.ps1 -Action stop` 只停止 8006 服务。
其他平台可前台运行 `python runtime/realtime.py --host <LAN-IP> --port 8006`，
或显式指定 `--ssl-certfile` 和 `--ssl-keyfile`。局域网监听要求 TLS。

```text
wss://YOUR_LAN_IP:8006/v1/realtime?mode=audio&model=MiniCPM-o-4.5-Realtime
```

`model` 必填，当前支持 `MiniCPM-o-4.5-Realtime`。`mode=audio` 为纯音频，
`mode=video` 为音频加视频帧；按照官方约定，省略 `mode` 时默认 `video`。
同一端口保留 [OpenAI GA/Beta 音频协议](realtime.md)，需显式使用
`model=gpt-realtime` 等 OpenAI 别名，其事件和 PCM16 音频格式与本页不同。

局域网客户端需信任 `runtime/certs/lan-ca.crt`，并使用证书包含的地址。
证书配置见 [Windows 局域网指南](nvidia.md#private-lan-access--局域网访问)。
防火墙启用时允许私有网络 TCP 8006。Worker 和原生引擎继续使用回环地址。

## 认证

首次启动生成随机主密钥，位于 `.local/realtime/api-key`。也可在服务启动前设置
`MODELBEST_API_KEY`，其次是 `DUPLEX_REALTIME_API_KEY`；两者共用同一认证入口。
密钥需至少 32 个不含空白的 ASCII 字符。不要把真实密钥放入 URL 或仓库。

服务端客户端使用请求头：

```text
Authorization: Bearer <API_KEY>
```

浏览器子协议与官方格式一致：

```javascript
const ws = new WebSocket(
  "wss://YOUR_LAN_IP:8006/v1/realtime?mode=audio&model=MiniCPM-o-4.5-Realtime",
  ["map.realtime", "map-api-key." + apiKey]
);
```

浏览器直连只适合可信的本地客户端；公开前端应由自己的服务端保管密钥并代理连接。
可通过 `-AllowedOrigins https://your-app.example` 限制浏览器来源。
OpenAI 路径签发的短期 client secret 不适用于面壁会话。
`GET /v1/models` 同样需要 Bearer 认证，`GET /health` 只反映入口进程存活。

## 会话时序与格式

```text
连接建立
  <- session.queued / session.queue_update（繁忙时）
  <- session.queue_done
  -> session.init
  <- session.created（顶层 session_id）
  -> input.append（持续并行收发）
  <- response.output_text.delta / response.output_audio.delta / response.listen
  -> session.close
  <- session.closed（顶层 reason）
```

也接受连接后立即发送一次 `session.init`，内部等到 Worker 分配后再准备模型。
收到 `session.created` 前不要发送输入；发送 `session.close` 后停止输入并等待确认。
所有消息为 JSON 文本帧；客户端应忽略未知的输出事件和附加字段。

```json
{
  "type": "session.init",
  "payload": {
    "system_prompt": "请用自然、简洁的口语回答。",
    "config": {"temperature": 0.7, "tts_enabled": true}
  }
}
```

可选的 `payload.voice` 与官方同名，支持分别指定 LLM 与 TTS 参考音频：

```json
{
  "ref_audio_base64": "BASE64_LLM_REFERENCE_AUDIO",
  "tts_ref_audio_base64": "BASE64_TTS_REFERENCE_AUDIO"
}
```

参考音频同样为裸 16 kHz 单声道 Float32 PCM，各不超过 30 秒。仅提供 LLM 参考时
TTS 复用它；未提供的参考使用现有语言配置。它们不是 WAV 文件的 Base64。

```json
{
  "type": "input.append",
  "input": {
    "audio": "BASE64_FLOAT32_PCM_16K_MONO",
    "video_frames": ["BASE64_JPEG_FRAME"],
    "force_listen": false
  }
}
```

纯音频省略 `video_frames`。输入为 **16 kHz、单声道、Float32 小端 PCM**，样本需
有限且在 `[-1, 1]`，没有 WAV 头。建议每次发送 1 秒（16000 样本），按真实时间推送。
支持较小包，内部累积至整秒再送入模型，不额外补静音；关闭时不足一秒的尾包不推理。
用户停顿或助手播放时继续发送麦克风输入，包括真实静音，才能维持全双工会话。

每个输入最多一张 JPEG（裸 Base64、压缩后不超过约 1.5 MiB，像素不超过
4096 × 4096）。小音频包聚合时使用该秒最新一帧。
`force_listen: true` 请求打断并让下一原生输入块进入聆听；客户端同时清空待播放音频。

```json
{"type":"response.output_text.delta","text":"你好，"}
```

```json
{"type":"response.output_audio.delta","audio":"BASE64_FLOAT32_PCM_24K_MONO","sample_rate":24000,"audio_epoch":0,"speech_end":false}
```

输出音频是 **24 kHz 单声道 Float32 小端 PCM**，解码 `audio` 字段后连续播放或拼接。
文本读取顶层 `text`；不是 OpenAI 的 `delta` 字段。`speech_end` 与 `audio_epoch`
是本地附加元数据，用于音频结尾和打断后丢弃旧队列。
`response.listen` 表示模型聆听状态，不把助手文本冒充输入语音转写。
音视频模式不伪造 token usage，也不依赖 `response.done` 判断原生语音结束。

## 支持范围

| 字段或事件 | 本地行为 |
| --- | --- |
| `session.init` / `input.append` / `session.close` | 官方命名和字段；含排队、初始化与关闭确认 |
| `config.temperature` | 0–2，传递到原生采样配置 |
| `config.tts_enabled` | 控制原生语音生成；false 时只返回文本与状态 |
| `voice.ref_audio_base64` / `tts_ref_audio_base64` | 两路参考独立传递 |
| `config.length_penalty` | 0.1–5，传递到原生采样配置 |
| `config.vision_hd` / `payload.max_slice_nums` | 高清图像切片，最多 3；输入也可带 `max_slice_nums` |
| `input.force_listen` | 立即请求原生打断；下一整秒输入携带 force_listen |
| `session.pause` / `session.resume` / `session.interrupt` | 本地扩展；官方 SDK 的 pause/resume 通常只是暂停/恢复采集 |
| `config.max_tokens` / `config.max_kv_tokens` | **暂不支持**；返回 `unsupported_feature`，请从官方示例中移除这两项 |
| `mode=chat`、工具调用、输入转写 | 暂不支持；chat 业务事件未在官方通用文档中定义 |

原生完整双工路径没有官方“每次回复最大 token 数”的控制，不能把它当成“每秒生成
token 上限”；KV 预算继续由硬件配置管理。上表未实现的配置会明确报错并关闭连接。
因此官方包含 `max_tokens` 的初始化示例需要按本页调整。

输入最多缓冲 30 秒、2 个推理块在途，超限返回错误。默认最多 8 条连接，真正的并发
由 Worker 数量决定。排队和准备共限 120 秒，会话最长 30 分钟。
断连、错误、超时和关闭都会释放上游连接与 Worker。
原有录制设置继续生效；此 API 不暴露录音、上传、管理或任意代理路径。

## 回放与部署

包含可运行的文件回放客户端，直接发送本页协议：

```powershell
.venv/Scripts/python scripts/benchmarks/modelbest-realtime.py `
  --url "wss://YOUR_LAN_IP:8006/v1/realtime?mode=audio&model=MiniCPM-o-4.5-Realtime" `
  --ca runtime/certs/lan-ca.crt --browser-auth --output verification/modelbest-audio
```

加 `--no-tts` 测试纯文本；`--packet-samples 1600` 测试 100 ms 小包；
`--reference <16k-wav>` 指定参考，`--tts-reference <16k-wav>` 可独立覆盖 TTS 参考；
`--interrupt-at 9` 标记第 9 秒输入打断。
视频测试使用 `mode=video` 并加 `--frame <jpeg>`。
脚本只在客户端读取 WAV 并编码；线上传输始终为裸 PCM。
输出事件、摘要及 `response.wav` 写入 Git 忽略的 `verification/`。
验证采用原始 WebSocket；没有声称已验证官方 JavaScript/Python SDK 的全部行为。

公网配置保留在 [realtime.Caddyfile](../deploy/realtime.Caddyfile)。先使用
`realtime.ps1 -Action start -ListenHost 127.0.0.1 -Http` 启动代理上游（已有监听时先停止），
替换 Caddy 中的域名并配置 DNS、TCP 80/443 路由与证书后，外部地址为：

```text
wss://voice.example.com/v1/realtime?mode=audio&model=MiniCPM-o-4.5-Realtime
```

Caddy 支持 WebSocket Upgrade，认证仍由 API 执行。8006、Gateway、Worker、引擎端口
无需公网映射。当前工作仅启动局域网入口，不创建公网 DNS、隧道或端口映射。

## 验证

```powershell
cd runtime
../.venv/Scripts/python -m pytest tests/test_gateway_duplex_cleanup.py tests/test_queue.py tests/test_modelbest_realtime.py tests/test_realtime.py tests/test_native_duplex.py tests/test_native_chat.py tests/test_cpp_launch.py -q
```

协议测试覆盖官方字段、两种认证、排队时序、音频原样传递与小包聚合、视频映射、
独立参考、关闭 TTS、打断旧音频、输入校验、缓冲上限与资源清理。
实际 GPU 回放验证与人工音质评价分别记录；波形和文本正常不等于完成人工听感验收。

2026-11-16，Windows / RTX 3060、Q4_K_M、4096 上下文配置的实机结果：

| 用例 | 结果 |
| --- | --- |
| 面壁浏览器子协议认证，1 秒音频包 | 11.34 秒源 WAV，返回文本与 3.68 秒 Float32 语音，正常关闭 |
| 100 ms 音频包、参考音频、`tts_enabled: false` | 返回文本，音频输出为 0；原生输入没有丢包 |
| 每秒一张 448 × 448 JPEG，含 `force_listen` | 返回文本与 5.36 秒语音；原生日志确认每块 1 张图、64 个视觉 token |
| 仅独立 TTS 参考、纯音频打断 | LLM 保留语言默认参考，TTS 使用上传参考；返回 3.84 秒语音 |
| 最终服务重启后的 20 秒音频回放 | 返回文本与 1.76 秒语音；20 块输入全部处理，正常关闭 |
| 排队时提前初始化后关闭、分配后未初始化即关闭 | 连续 5 轮均正常确认；每轮 Worker 恢复 idle，原生引擎 PID 保持不变 |
| 局域网 IP 的 TLS / 认证 / 路由 | CA 验证通过；无认证 WebSocket 为 401；管理与 Worker 路由为 404 |

首个音频事件分别在测试启动后约 9.55 秒（纯音频）与 14.22 秒（视频）到达，
包含排队、模型准备及源音频按真实时间发送，不能当作说完后的响应延迟。
除最终 20 秒复验外，输入持续发送 25 秒（包含开头 1 秒静音及源 WAV 后的静音），
各会话收到关闭确认。完整双工中输入可打断输出，回放之间的文本和语音长度不完全相同。
事件记录验证打断之后没有继续传出旧 epoch 音频。输出 WAV 为 24 kHz，样本有限且
有有效音量，保存在 `verification/modelbest-*`；人工听感验收尚未完成。
上述命令的 109 项回归检查与仓库结构检查通过。此记录只验证本机通过局域网 IP 访问，
未验证另一台设备、公网或 macOS，也未完成官方 SDK 的集成测试。

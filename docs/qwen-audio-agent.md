# qwen-audio-agent 接入

使用官方 [QwenAudio/qwen-audio-agent](https://github.com/QwenAudio/qwen-audio-agent)
内置的 **`minicpm-o` Provider**，将本项目作为本地语音模型。已使用官方 npm 包
**2.1.0** 的未修改 Provider 和桌面配置模块验证语音链路。

部署关系是：**另一台电脑的 qwen-audio-agent 应用 → 本机 8006 模型 API →
MiniCPM-o 推理**。qwen 应用使用它自己的 Gateway；模型主机无需运行 qwen
网页或 qwen Gateway。`scripts/windows/qwen-audio-agent.ps1` 只是可选联调工具。

## 模型主机与另一台电脑

模型主机执行：

```powershell
./scripts/windows/start.ps1
./scripts/windows/realtime.ps1 -Action start
./scripts/windows/export-qwen-client.ps1
```

将生成的 **`.local/qwen-client.zip`** 复制到客户端电脑并解压。包内包含完整服务地址、
API Key、CA 公钥、配置说明和无依赖连接检查器，不包含服务器私钥或模型权重。
此包包含调用凭据，不要提交到 Git。通用说明见 [客户端配置](../deploy/qwen-client-setup.md)。

在那台电脑的 qwen 应用“语音前台 → 面壁智能 / MiniCPM-o”填写包中的
URL 和 API Key；**8006 是模型服务地址，不是 qwen 的 Gateway 地址，也不是
后台 Agent 地址**。不要将它填入“远程连接 / Gateway”或“后台 Agent”。

先在客户端电脑配置信任包中的 `lan-ca.crt`，再完全退出并重启应用。
可先在解压目录检查模型服务：

```powershell
$env:NODE_EXTRA_CA_CERTS = (Resolve-Path ./lan-ca.crt).Path
node ./check-connection.mjs
```

检查器从运行它的电脑验证 TLS、正确/错误密钥、模型列表、初始化、正常关闭和重新连接。
它不启动网页或访问麦克风。模型主机可通过 LAN IP 自测；另一台电脑运行成功才是
该电脑路由、防火墙和证书信任的验证。实际桌面版的麦克风/扬声器仍需在客户端试用。

## 连接配置

在桌面版“语音前台 → 面壁智能”设置服务地址和 API Key，或将以下配置放到
`qwenaudio config` 显示的 `config.env` 中：

```dotenv
QWEN_AUDIO_REALTIME_PROVIDER=minicpm-o
QWEN_AUDIO_REALTIME_URL=wss://YOUR_LAN_IP:8006/v1/realtime?mode=audio&model=MiniCPM-o-4.5-Realtime&protocol=qwen-audio-agent
QWEN_AUDIO_REALTIME_API_KEY=YOUR_LOCAL_API_KEY
QWEN_AUDIO_REALTIME_MODEL=
QWEN_AUDIO_REALTIME_VOICE=
AGENT_PROTOCOL=none
```

将 `YOUR_LAN_IP` 替换为模型主机的局域网 IP，API Key 取自该主机的
`.local/realtime/api-key`。模型和音色字段**留空**；模型名通过 URL 的 `model`
参数传给本服务。不要选择 `gpt-live` 或 `dashscope` Provider。
模板见 [qwen-audio-agent.env.example](../deploy/qwen-audio-agent.env.example)。

本地 CA 需在 **qwen-audio-agent 所在主机**受信任。Node 进程启动前设置：

```powershell
$env:NODE_EXTRA_CA_CERTS = 'E:/Projects/duplex.cpp/runtime/certs/lan-ca.crt'
qwenaudio
```

其他主机需复制 CA 公钥并改为自己的路径；无需复制 TLS 私钥。
桌面版也需在进程启动前设置此变量或安装受信任 CA。不要禁用 TLS 验证。
`NODE_EXTRA_CA_CERTS` 只写进运行中应用的设置文件并不会更新 Node 的信任链。

## 可选的官方 WebUI 联调

本节用于开发联调，**不属于服务端部署或另一台电脑接入的前置条件**。

需要 Node.js 22.22.2+（22.x）、24.15.0+（24.x）或 26+。官方推荐全局安装：

```powershell
npm install -g qwen-audio-agent@2.1.0
```

本项目也可隔离安装，不修改全局 qwen 设置：

```powershell
npm install --prefix .local/qwen-audio-agent --ignore-scripts qwen-audio-agent@2.1.0
```

隔离安装中的 WebUI 和语音 Provider 可以运行；`--ignore-scripts` 不安装 TUI 原生
音频等可选功能。已有源码或 npm 安装时，启动器可传 `-QwenRoot <包根目录>`。

```powershell
./scripts/windows/start.ps1
./scripts/windows/realtime.ps1 -Action start
./scripts/windows/qwen-audio-agent.ps1 -Action start
./scripts/windows/qwen-audio-agent.ps1 -Action status
```

打开 **`http://127.0.0.1:3101`**，连接并允许麦克风，开启语音输入即可对话。
该 WebUI 由 qwen 自己的 Gateway 提供，Gateway 通过受认证的 WSS 连接模型。
localhost 可使用浏览器麦克风。当前启动器只让 WebUI 监听回环地址，模型 API
仍可从局域网接入。其他机器上的 qwen 可以使用相同的模型服务 URL。

启动器将配置、身份、聊天记录保存在 `.local/qwen-audio-agent/` 的隔离目录，
不覆盖用户已有的 `~/.config/qwaudio` 或桌面版配置。
启动器为 2.1.0 预加载一个兼容修复：其官方 MiniCPM-o Provider 有模型档案但漏了
`modelCatalog()`，导致 WebUI 显示“Model capability information unavailable”。
修复让目录包含 Provider 当前的模型档案，保留音频/视频模式各自的传输能力；
不修改已安装的 npm 包，不对其他版本应用此修复。
日志位于 `logs/qwen-audio-agent.stdout.log` 和 `logs/qwen-audio-agent.stderr.log`。
重复启动复用现有进程；改变选项需先停止。

```powershell
# 只停止 qwen；模型 API、Gateway 和 Worker 继续运行
./scripts/windows/qwen-audio-agent.ps1 -Action stop
# 视频输入：先停止 qwen 后再启动；WebUI 可协商摄像头输入
./scripts/windows/qwen-audio-agent.ps1 -Action start -Video
```

默认读取 `runtime/config.json` 中的 LAN 地址，并使用 `.local/realtime/api-key`
或服务启动时同样设置的 `MODELBEST_API_KEY` / `DUPLEX_REALTIME_API_KEY`。
非默认模型服务可传 `-ApiUrl <完整WSS地址>`。

## 协议差异

qwen-audio-agent 2.1.0 的内置适配器读取早期 MiniCPM-o 事件：

```json
{"type":"response.output.delta","kind":"text","text":"你好","response_id":"resp_..."}
```

因此 URL 必须加 **`protocol=qwen-audio-agent`**。该选项输出统一的
`response.output.delta` 及 `kind=text/audio/listen`，同一回复使用稳定的
`response_id`。音频保持 16 kHz Float32 输入、24 kHz Float32 输出；qwen 的
Provider 完成它与客户端 PCM16 的双向转换。
回复结束要等原生最终音频抵达，避免 qwen 提前结束后把迟到语音当成新一轮。

省略此选项仍为[面壁文档的拆分事件格式](modelbest-realtime.md)，现有面壁客户端
不变；OpenAI 兼容路径也继续使用自己的事件。
`model` 参数仍必填。qwen 文档中的无 model 示例需改成上面的完整 URL。

## 应用侧能力提示与限制

未修改的 qwen-audio-agent 2.1.0 在 MiniCPM-o Provider 中缺少 `modelCatalog()`，
所以 UI 可能显示“Model capability information unavailable”。这是 qwen 应用
自身模型目录的缺陷；服务端没有被它调用来获取这段目录，增加 HTTP 接口或
WebSocket 字段不能修复另一台电脑的此项显示。它不阻止 Provider 初始化和语音收发。
上一节启动器中的兼容修复只作用于那个联调进程，独立桌面版需升级到包含修复的
qwen 版本，或在应用侧补丁中让 `modelCatalog()` 返回当前 `modelProfile()`。

| 能力 | 当前接入 |
| --- | --- |
| 实时语音输入、文本和语音回复 | 支持 |
| 视频帧输入 | `mode=video`，qwen 每秒携带最新 JPEG |
| 用户讲话打断助手 | 沿用原生全双工语音打断；qwen 内置 Provider 不支持客户端 response.cancel |
| 选择 qwen 内置其他音色 | 不支持；参考音频在模型服务侧配置 |
| 文字输入、结构化函数调用、后台 Agent 编排 | 此 Provider 暂不支持 |
| 后台任务主动播报、历史恢复、输入转写、记忆写入 | 此 Provider 暂不支持 |

这些限制与 qwen 官方 [MiniCPM-o 接入说明](https://github.com/QwenAudio/qwen-audio-agent/blob/main/docs/voice-frontends/minicpm-o.zh.md)
一致。`AGENT_PROTOCOL=none` 用于当前可用的语音/视觉聊天；仅添加后台 Agent 配置
不会让 MiniCPM-o 获得工具调用能力。若要用语音派发后台任务，还需要额外的
文本或工具调用适配，不在本次语音 Provider 联通验证之内。

## 验证

可运行的回放客户端直接导入已安装包的 `RealtimeFrontend`、`miniCpmOProvider`
和可选的官方 `GatewayClient`，不复制协议实现：

```powershell
$env:NODE_EXTRA_CA_CERTS = Join-Path (Get-Location) 'runtime/certs/lan-ca.crt'
# Provider 直接连接本地模型
node scripts/benchmarks/qwen-audio-agent.mjs
# 不运行 qwen 网页：验证官方桌面设置的保存/重载、运行时配置、Provider 语音链路
node scripts/benchmarks/qwen-audio-agent.mjs --desktop-settings `
  --output verification/qwen-independent-app-audio
# 经 qwen Gateway 的客户端链路
node scripts/benchmarks/qwen-audio-agent.mjs `
  --gateway ws://127.0.0.1:3101/api/realtime `
  --output verification/qwen-audio-agent-gateway
# Provider 视频帧链路（默认地址自动使用 mode=video）
node scripts/benchmarks/qwen-audio-agent.mjs --frame path/to/frame.jpg `
  --output verification/qwen-audio-agent-video
```

脚本支持 `--qwen-root`、`--url`、`--key-file`、`--audio`、`--seconds`、`--frame`、`--desktop-settings`；
音频文件使用 16 kHz 单声道 PCM16 WAV，模拟 100 ms 实时采集。
摘要、事件和 24 kHz PCM16 `response.wav` 写入 Git 忽略的 `verification/`。

2026-11-16，Windows / RTX 3060，官方 npm 2.1.0，回放 11.34 秒源音频：

| 路径 | 文本/语音 | 输出语音时长 | 完成回复 | 首音频距启动 |
| --- | --- | --- | --- | --- |
| 官方 Provider → 模型 API | 成功 | 3.24 秒 | 1 | 17.25 秒 |
| 官方 GatewayClient → qwen Gateway → Provider → 模型 API | 成功 | 4.44 秒 | 1 | 16.61 秒 |
| 官方 Provider → 模型 API，携带 JPEG 帧 | 成功 | 3.96 秒 | 4 | 10.16 秒 |
| 官方桌面设置保存/重载 → 运行时配置 → 未修改 Provider → LAN 模型 API | 成功 | 4.40 秒 | 1 | 16.31 秒 |
| 同上，`mode=video` 并携带 JPEG 帧 | 成功 | 3.40 秒 | 1 | 17.24 秒 |

Gateway 链路验证了 `voice.ready`、`response.started`、`transcript.delta`、
`audio.delta` 和 `audio.done`，文本和全部音频使用同一回复 ID，结束后无残留活动回复。
音频两条路径的摘要分别保存在 `verification/qwen-audio-agent/summary.json` 和
`verification/qwen-audio-agent-gateway/summary.json`，视频帧传输摘要保存在
`verification/qwen-audio-agent-video/summary.json`。相关自动化回归共 112 项通过。
独立配置路径的结果在 `verification/qwen-independent-app-audio/summary.json` 和
`verification/qwen-independent-app-video/summary.json`；
验证期间 3101 联调进程已停止，没有预加载 UI 修复。
客户端连接检查器在模型主机通过 LAN IP 自测：TLS、认证、错误密钥拒绝、
初始化、关闭和重连均成功；尚未在另一台电脑运行检查器或验证其桌面 GUI。
上述首音频时间包含模型准备和实时输入发送，不是说完后的推理延迟。
此项验证不代表人工麦克风听感、其他操作系统或 qwen 后续版本均已验证。

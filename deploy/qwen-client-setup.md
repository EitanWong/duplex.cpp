# 在另一台电脑的 qwen-audio-agent 中连接模型

这个目录是模型客户端配置包。模型运行在 duplex.cpp 主机；qwen-audio-agent
应用在你的客户端电脑运行。不需要在模型主机启动 qwen 网页或 qwen Gateway。

## 桌面应用设置

1. 进入 qwen-audio-agent 设置的“语音前台 → 面壁智能 / MiniCPM-o”。
2. 服务地址填写本目录 `config.env` 的 `QWEN_AUDIO_REALTIME_URL` 完整值。
3. API Key 填写 `QWEN_AUDIO_REALTIME_API_KEY` 的值；模型和音色留空。
4. 先配置下面的 CA 信任，再完全退出并重新启动应用。开启麦克风后使用语音对话。

这是**语音模型服务地址**。不要把 8006 填到 qwen 的“Gateway 地址 / 远程连接”
或“后台 Agent 地址”。这些入口使用不同的协议；桌面应用仍使用它自己的 Gateway。
`AGENT_PROTOCOL=none` 适用于只做语音聊天；不会为 MiniCPM-o 添加工具调用能力。

## CA 信任和检查

本目录 `lan-ca.crt` 是模型服务的 CA 公钥。Node/Electron 需要在进程启动前
设置 `NODE_EXTRA_CA_CERTS`。Windows PowerShell 示例：

```powershell
$env:NODE_EXTRA_CA_CERTS = (Resolve-Path ./lan-ca.crt).Path
# 把路径替换成实际已安装的 qwen-audio-agent 程序；确保旧进程已经完全退出
& 'C:/path/to/qwen-audio-agent.exe'
```

也可以在系统中配置应用进程的环境变量，然后重启应用。只导入浏览器的证书，
或只在应用运行后修改 `config.env`，不能保证其 Node 语音进程信任该证书。
macOS/Linux 也应在启动应用的环境里设置指向本地 `lan-ca.crt` 的绝对路径。

如果客户端装有 Node.js 22.22.2+（22.x）、24.15.0+（24.x）或 26+，可在本目录运行：

```powershell
$env:NODE_EXTRA_CA_CERTS = (Resolve-Path ./lan-ca.crt).Path
node ./check-connection.mjs
```

检查器不安装依赖、不启动网页、不占用麦克风。它从这台电脑检查 TLS、认证、
模型列表、会话初始化、正常关闭和重新连接。成功时所有布尔字段均为 `true`。
它不会输出密钥，也不替代实际麦克风和扬声器测试。

连接超时：确认与模型主机网络互通，并允许模型主机 TCP 8006 入站。
证书错误：检查启动环境中的 CA 路径，以及地址是否仍为证书覆盖的模型主机 IP。
401：检查是否使用本配置包中的 API Key。模型主机 IP 或密钥改变后需要重新导出。

## qwen 2.1.0 的能力提示

未修改的官方 2.1.0 MiniCPM-o Provider 缺少 `modelCatalog()`，其 UI 可能显示
“Model capability information unavailable”。这是客户端内置模型目录的缺陷，
不是模型服务器的查询失败；它不会阻止 Provider 初始化和语音收发。
服务器不能通过 WebSocket 修改客户端内置代码。本项目联调启动器的修复只作用于
那个联调进程；独立桌面版需要包含此修复的 qwen 版本或应用侧补丁。

本配置包包含调用凭据，仅供你自己的 qwen 客户端使用。包内没有模型权重、
服务器 TLS 私钥或 CA 私钥。

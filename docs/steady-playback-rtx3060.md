# RTX 3060：稳态归因、浏览器播放与持续会话

本轮在 2026-11-16 的 RTX 3060 12GB、i7-10700F、32GB RAM、Windows/WDDM、
CUDA 12.8.1、610.88 驱动上继续[前轮实现](long-session-rtx3060.md)。
原始证据在本机忽略目录 `verification/steady-20261116/`。本报告不把前轮结果
计为本轮新验证，也不把主机等待相加当成可删除的 GPU 时间。

当前默认仍为单 Worker、4096 LLM 上下文、F16 K/V、CUDA Flash Attention 和现有
CUDA Graphs。TTS 独立保持 4096；Apple M5 profile、Metal/CoreML/Accelerate 路径
未更改。没有提交、推送或创建 PR。初始工作区快照与文件 hash 已保存；原服务状态依据本轮开始时的工具观测，
`service-before.txt` 重定向为空，另有明确标注来源的 reconstructed JSON；
已有改动和独立的 Strata 文档继续保留。

## 本轮实现与归因

`OMNI_KV_TRACE=1` 现在记录同步调用方、context、模型层数和 queued token 数；
CUDA Graph 记录 key/uid、节点数、首个发生变化的节点及形状/指针/stride/source/
参数/flags 差异。计时包括 capture、instantiate 和 launch，不能称为纯 capture
内核耗时。日志中最近的 LLM phase 只代表时间邻近，不能证明异步 TTS 的归属。

120 段固定音视频、seed 42、LLM greedy、TTS 温度 0.8、第 11 段强制 LISTEN 的
新诊断基线（`trace-control`）：输入响应 p50/p95 **208.94/305.61 ms**；文本、
听说决策及 23.48 秒 PCM 与前轮默认基线逐字节相同，PCM SHA-256 为
`336aea439ceea6929a0a7504d37c3df9a2b6d706714541df4357b952feb00035`。
这是固定样本等价证据，不代替真人语音质量验收。

以引擎日志启动后 15 秒作为 startup/steady 分界，稳态累计主机等待：

| 调用 | 次数 | 累计 ms | p50 ms | 含义 |
| --- | ---: | ---: | ---: | --- |
| 36 层 logits getter，有 pending | 112 | 4971.70 | 45.95 | 读 GPU 结果前的等待 |
| 36 层 embeddings getter，有 pending | 193 | 2965.48 | 14.72 | 读 GPU 隐状态前的等待 |
| 20 层 TTS embeddings getter，有 pending | 463 | 758.87 | 1.77 | 声学 decoder 依赖 |
| 36 层显式 synchronize，queued=0 | 209 | 0.43 | 0.001 | 非主要延迟来源 |

不能按 queued=0 跳过同步：startup 中有一次此类 embeddings 等待达 12.40 ms；
queued token 不是所有后台设备工作的一致状态机。未实施这种不安全的删同步。

较大的两类 Graph（11740/11746 节点，Token2Mel flow）在 startup 后各捕获 4 次，
累计分别 288.95/289.26 ms；LLM 1265 节点的 22 次共 36.85 ms；TTS decoder
624 节点的 22 次共 15.16 ms。频繁 property 变化不等于每次都 capture：当前
backend 会先回到 eager/warmup，稳定后才 capture。

曾尝试 CUDA prompt-cache session 复用：保留图和分配，但重新加载 speaker、四组
缓存与 warmup。**该实现已撤回，不提供此开关。** 实测文本/决策一致，但 PCM hash
改变且普通响应更慢，未满足输出等价目标；单靠显式缓存重载不足以证明构图、噪声与
执行状态等价。被拒补丁仅留在忽略目录 `rejected-t2m-reuse.patch` 作负面实验记录。
代码回到本轮之前的 Token2Mel 生命周期，不碰原有 Apple 路径。

另外独立测试 `GGML_CUDA_GRAPH_IDLE_MS=60000`：只改变 CUDA backend 图的空闲
过期时间（默认仍是 10000 ms，每 5 秒 sweep；合法范围 1..120000 ms）。日志加入
`KV_GRAPH_EVICT`，可区分过期清理与 property 变化。保留图增加 metadata/executable
驻留，有限 TTL 不等于图数量硬上限，必须看资源。该实验与已否决的 session 复用分别对照，
不叠加收益。对照结果见后文。

## 浏览器测量定义

`playback-server.py` 是仅监听 127.0.0.1 的诊断中继。浏览器加载正式
`AudioPlayer`，经透明 AudioWorklet 后连接真实 AudioContext destination；不使用
麦克风。可选 `onSchedule` 记录实际 source.start 时间、AudioContext 时钟、
getOutputTimestamp、baseLatency/outputLatency。普通播放器无需安装探针。

原生响应从完整输入发送计时；浏览器区间从收到中继 input notice 计时，分别报告
到 PCM 到达、到调度回调、到设备呈现估计。两套时钟不能直接相减；浏览器结果
不含 native send 到 input notice 的中继延迟。
设备呈现用 getOutputTimestamp 映射，不重复加 outputLatency；这不是扬声器声学
延迟，也不包含一秒输入采集、麦克风/AEC 或公网 Gateway 路径。渲染心跳与非零
渲染块证明浏览器实际执行了音频渲染，不证明音色、可懂度或人耳听感合格。
欠载按播放队列调度缺口计算，不能把语音中的正常静音当成欠载。

小时测试开始时发现旧观测页将 fixture index 加 1 关联 native trace.input，实际上
参考音频临时文件占了一个编号，当前固定 fixture 应加 **2**。已修复中继显式传入
native_input，旧小时日志必须用 `analyze-playback.py --legacy-input-offset 2`
离线关联；不采用当时页面写出的错误 input_to_pcm_ms 值。原始事件保留。

## 持续会话与记忆验收边界

测试采用 3601 个完整一秒音视频输入、单一持续 session、每 181 段一次强制
LISTEN，打开 turn planner、有来源文本记忆及诊断。真实持续至少一小时，不能用
加速回放替代。fixture 每 36 段循环；记忆输入为明确标记的合成 user_text 事实，
不冒充 fixture 用户语音转录。completed_turns 是引擎记录的结束边界计数，不能
等同于真人语义问答轮数。正常 1Hz 回放与过载 burst 测量分开解释。

`native.py` 增加逐事件 flush 的 journal，失败也保留已完成输入；`--interrupt-every`
支持反复打断；`--burst-size` 连续发送完整输入后按 FIFO 收齐响应，额外覆盖 socket
积压。原生 input_age 从本机服务处理阶段计时，不能把它当成包含 socket 排队的
端到端年龄；burst 报告中的发送到响应延迟包含这部分积压。burst 禁止同时提交
记忆更新，以保持来源轮次更新在明确安全边界。

`long-history.cpp` 增加 `crowded-memory` 模式：119 条有来源文本、纠错链、已完成
任务和 112 条干扰事实，以固定 512-token 注入预算检查中英文、否定、联合事实、
旧值排除及缺失视觉/声音信息。报告实际检索文本与 token 数，检索丢失和模型
答错分别记录。模型无关测试覆盖 128 条容量上限、纠错的纠错、拒绝后的事务完整性。文本 probe
使用独立 context，接在小时测试结束之后顺序运行；不能声称它验证了小时音视频
会话中早期用户语音的事实召回。小时测试验证记忆协议、注入及多次淘汰后记账。

长期支持语义仍是：有限近期多模态 KV 全量 attention + 窗口外调用方提供的文本
事实。没有自动 ASR、没有并发线程操作同一 llama_context、没有窗口外原始 KV
检索，也不是无损 KV 压缩。文本记忆不能恢复原始音色、语气、画面及任意逐字细节。

## 复现

先记录并暂停本项目托管服务，顺序执行 GPU 测试，结束恢复原配置。不要停止无关
进程，不要在 hour run 使用 DLL 时覆盖构建产物。完整构建方式见 Windows 指南。

```powershell
$env:PATH = "$PWD/optimization/cuda-12.8/bin;$env:PATH"
# 终端一：本地诊断中继。浏览器打开下方 URL，点击开始播放观测。
.venv/Scripts/python scripts/benchmarks/playback-server.py --output verification/playback
# http://127.0.0.1:19280/?run=hour

# 终端二：实际 >=1 小时，原生服务和资源采样由工具持有并清理。
.venv/Scripts/python scripts/benchmarks/pipeline.py `
  --model-dir models/MiniCPM-o-4_5-gguf --output verification/hour `
  --count 3601 --rounds 1 --interrupt-at 0 --interrupt-every 181 `
  --tts-temperature 0.8 --resources --env OMNI_DUPLEX_WINDOW=turn `
  --env OMNI_TEXT_MEMORY=1 --env OMNI_KV_TRACE=1 `
  --memory-file verification/your-user-facts.json --memory-query 'room task' `
  --browser-relay 'ws://127.0.0.1:19280/stream?run=hour'
.venv/Scripts/python scripts/benchmarks/summarize-session.py verification/hour
.venv/Scripts/python scripts/benchmarks/analyze-steady.py verification/hour
.venv/Scripts/python scripts/benchmarks/analyze-playback.py verification/playback/hour/browser-events.jsonl
```

Graph 空闲保留对照使用 4096/F16、120 段、`--interrupt-at 11 --tts-temperature 0.8
--env OMNI_KV_TRACE=1`，分别设置 `--env GGML_CUDA_GRAPH_IDLE_MS=10000` 和
`=60000`，其他配置完全相同，按 A/B/A 顺序运行。保留 PCM/text/decision 等价
及首 PCM，不能仅比主响应 p50。回退只需移除开关或设为 10000。

积压用 `--count 40 --recovery-count 12 --interval-ms 0 --burst-size 8 --interrupt-at 11`，
前段 burst 后同一 session 恢复 1Hz；不启用记忆
更新。长历史 probe 编译见前轮文档，`4096 3072 f16 crowded-memory`；可附题号。
工具输出、录音、模型、生成二进制始终留在 Git 忽略目录。

## 浏览器打断控制探针

不占用 GPU，在小时测试期间把诊断基线已生成的 PCM 回放至独立浏览器探针。
先排程两段 1 秒 PCM，0.6 秒时强制停止，有 1 次 was_draining=true；再主动发送
旧 epoch 包，确认其被拒绝，随后播放新 epoch。AudioWorklet 的停止窗口为
**2.142 秒、738 块，非零块 0**。这是渲染端边界证据，不是声学回录。
工具中的 stale_pcm_packets=1 是主动注入且被丢弃的包，不能算服务泄漏。
探针最后故意延后 speech_end，播放器确实报告约 811 ms 空队列等待；不能把
scheduleGaps=0 等同于所有形式的欠载为零。

小时初版观测页记录 schedule gap 与渲染心跳，未保存逐回调 player_metrics。
后续页已补记录 queuedAudioMs/underflowMs，每个 turn 取最大值，不能据初版
日志追认这一项通过。原生与浏览器探针均不会把样本静音判为欠载。

## 结果与待验收

小时测试完成，最后输入在 session 计时 **3602.081 秒**返回；此计时含初始化，
首尾发送间隔约 3600 秒，完成 3601 段输入。含尾部等待的 report 另有 elapsed_ms。

| 项目 | 本轮小时实测 |
| --- | --- |
| 输入响应 p50/p95/max | 215.42 / 343.34 / 1644.29 ms |
| 初次 speak 输入到首 PCM | 554.94 ms |
| 完整淘汰事件 / unit fallback | 508 / 0 |
| accounted_tokens 与真实 KV 长度不一致 | 0 |
| 强制 LISTEN 输入 / epoch 转换 | 20 / 20 |
| 打断响应确认后收到旧 epoch PCM | 0 |
| LLM 最大 used / physical / attention span | 4049 / 4049 / 4096 |
| 最后 used / physical / attention span | 3459 / 4021 / 4096 |
| LLM KV 分配 | 576 MiB，全程容量固定 |
| 来源事实注入次数 / 累计 token | 82 / 16147（累计量，不是单次预算） |
| 累计记忆 retrieval / prefill | 2910.49 / 8294.44 ms |
| 累计 KV 删除元数据主机时间 | 1037.62 ms |
| RoPE update + reserve | 508 次，累计 2293.91 ms，p50/p95 4.49/4.62 ms |
| Graph capture + instantiate + launch | 694 次，累计 5612.87 ms，p95 69.58 ms |
| PCM | 330.12 秒，有限、无削波；未人工听验 |

首个和最后完整 300 段窗口的响应 p50/p95 为 216.21/345.09 → 215.84/329.73 ms。
正常 1Hz 输入的主机 schedule lag p50/p95/max 为 11.24/20.07/663.19 ms；引擎
input_queue p50/p95 0.040/0.066 ms，input_age 124.10/130.64 ms。这些数据支持
此循环负载下稳定运行一小时，不是任意真人多小时会话或过载吞吐承诺。

稳态首个完整 5 分钟后与最后完整 5 分钟，进程 private memory 中位数
13564.1 → 13563.9 MiB，working set 6486.2 → 6486.3 MiB，专用/共享显存
9589/96 → 9589/96 MiB；无单调持续增长证据。冷期进程峰值 private 16883.55 MiB、
working set 11007.20 MiB；专用峰值 9589 MiB，共享峰值 96 MiB，整卡峰值
10429 MiB（含桌面/浏览器，不能全归因于引擎）。采样不覆盖瞬时未命中峰值。workspace 日志分别报告 LLM CUDA/host 76.10/5 MiB、
TTS CUDA/host 6.63/1.75 MiB、audio CUDA/CPU 59.30/18.30 MiB；这些不是全模型
workspace 峰值或可直接相加的进程显存，端到端峰值以上述 WDDM 采样为准。

浏览器实收 366 个有 PCM 的包、160 个 native speech turn；实际渲染 1260976 块，
其中非零 113804 块。input notice 到各 turn 首 PCM p50/p95 **600.20/729.92 ms**，
到该首包设备呈现估计 **762.20/955.07 ms**；outputLatency 为 64 ms。
未观测 schedule gap，render 心跳 p50/p95/max 1000.10/1010.20/1030.20 ms。
上述首包按 native trace turn 分组；不是用户语义轮次或语音端点时延。
小时页缺完整 player_metrics，不能从此结论推出所有空队列等待为零。

## 拥挤文本记忆：保留负面结果

在本轮 trace 构建上运行独立模型 probe，实际 prefill 3072 token、4096/F16，119 条
record、每题实际检索 512 token。7 题中 6 题回答符合预期：青鹭/7319、二次纠正的
26 楼、Amber Cabin/B42、茶且不喝咖啡、三事实联合，以及缺失画面/音色回答 UNKNOWN。
任务题要求区分已完成与未完成，模型回答“未知”，判为失败。检索文本第一条就是
`task-done` 的“任务已完成”并保留 status=done/supersedes，旧 pending 记录未入选，
所以这是模型利用/回答失败，不能算检索正确就验收通过。当前记忆仍为实验选项。

这些是显式合成文本，不是中文/英文语音 ASR。该 probe 在 CUDA 公共头重新构建
期间顺序使用 GPU，编译占用 CPU；它可作上述确定输出的质量记录，retrieval
57.64–123.34 ms 与 prefill ~1.77–1.80 s 不作为无干扰性能基线。构建后以最终默认二进制复测，7 题输出与上述完全相同；实际检索仍每题 512 token，
retrieval 49.36–55.87 ms，prefill 1755.76–1785.49 ms（不含检索），TTFT 另见 JSON。
单条待办的 UNKNOWN 未改善。与小时测试只有两条事实的亚毫秒级平均检索成本
不同，119 条存储的逐候选完整 tokenizer 预算检查已有明显开销；当前不扩大上限。

## 积压与恢复

在公共头 CPU 构建期间，用保留的 trace 二进制顺序执行 GPU 压力：

| 场景 | 实际输入 | 最大在途 | 响应 p50/p95 ms | 打断确认后旧 PCM |
| --- | ---: | ---: | ---: | ---: |
| 8 个完整输入/批，连续 72 段 | 72/72 | 8 | 1402.17/2305.00 | 0（4 次强制） |
| turn planner，同 session 前 40 段 burst | 40/40 | 8 | 1452.87/2812.78 | 0 |
| 接着恢复 1Hz | 12/12 | 1 | 299.28/418.34 | 0 |

后一 session 合计 3 次强制打断，KV 账本不一致 0。以上证明短负载下能排空并恢复
输入处理，不能称为 8 倍实时吞吐；延迟含 socket FIFO 排队，编译还施加了 CPU
压力，不能与无干扰 215 ms 基线当作算法回归。恢复段只有 12 秒，需与小时稳态
结果合读，不能据此证明任何持续过载都能自愈。未丢弃或截断完整多模态输入。

## Graph 无干扰对照

`ab-control-a` 与公共头 CPU 重编译重叠，p50/p95 246.58/371.40 ms；PCM 仍与
默认 hash 相同。此项明确排除出速度比较。构建完成后重新以 `clean-control-a`、
`clean-reuse`、`clean-idle60`、`clean-control-b` 顺序测量，采用同一最终构建、输入、
模型、4/8 CPU threads、采样与 GPU，两个实验分别开启。session 复用结果：p50/p95 217.88/309.89 ms、23.48 秒 PCM hash
`9f0d0a52974a6592fa0432c16f495356d1d4fae80d0b32aedc700eefc6343e86`，与默认
`336aea...0035` 不同。文本与听说决策一致仍不足以通过，因此撤回。初始化 warmup
574.2 ms 不能抵消输出差异。控制 A p50/p95 210.57/305.28 ms，首 PCM 540.75 ms。
60 秒空闲保留组已完成：文本、决策与 23.48 秒 PCM 逐字节一致，p50/p95
210.93/305.05 ms，初次首 PCM 546.34 ms，未改善普通响应或冷首包。按 native
turn 分组首包 p50/p95 从控制 A 的 540.75/593.62 降至 500.80/585.12 ms。
重 flow 图捕获从 10 次/777.38 ms 减至 5 次/422.96 ms；进程专用显存采样峰值
9628.98 → 9748.99 MiB，整卡 10443 → 10567 MiB。保留图以额外驻留换取减少发声间歇
后的重捕获，不能称为 decoder 加速。最终控制 B p50/p95 210.16/307.71 ms，冷首 PCM 535.63 ms，各 native turn 首包
p50/p95 535.63/598.41 ms，flow 捕获 10 次/795.48 ms，PCM 再次等于默认 hash。
因此 idle60 的 turn 首包 p50 比两侧控制减少 34.83–39.95 ms（6.5–7.4%），
p95 少 8.49–13.29 ms；冷首包慢 5.58–10.70 ms。主响应与文本 TTFT 没有稳定改善。
这是单组 A/B/A、固定视频音频的局部首包收益，未证明真实浏览器听感收益或长会话
下同样收益，不能宣称全链路加速。默认 profile 保持 10 秒，60 秒仅供显存足够时实验。




## 本轮验收与回退状态

79 项相关 Python 测试、43 项前端测试、C++ 完整输入预算/128 条记忆事务测试、
真实 GPU KV 超大删除/陈旧账本/不可能输入拒绝路径均通过。PowerShell 进程归属
回归、页面 JS 语法和仓库卫生检查通过。编译沿用已有 build_modulate 递归警告，
本轮未更改其代码，不把构建成功视为该未触发路径已修复。

未引入删同步、忽略任意 tensor metadata、物理压紧、RAM 冷 KV 或新量化格式。
当前物理 scan 仍常达分配容量，删除只回收槽位。60 秒 Graph TTL 是有限时间驻留，
不是图数量硬上限，也不是长期 KV。拒绝的 session 复用不保留产品开关。

最终默认构建已再跑 36 段 AV：响应 p50/p95 210.08/333.94 ms，文本/听说决策
等于控制 A 前 36 段，9.60 秒 PCM 与控制 A 对应前缀逐字节一致。实际浏览器
4 个 native turn 的首 PCM p50/p95 564.10/627.37 ms、设备呈现估计
786.25/921.63 ms，37 条 player_metrics、每 turn 欠载计时均为 0，调度 gap 0，
旧 PCM 0。该小样本补验了新版探针，不能追认小时页缺失的 underflow 时间序列。

服务已恢复至 Worker 22440、native 19080 和 HTTPS Gateway 8040，均就绪。
保持单 Worker、4096/F16/FA、默认 10 秒 Graph 过期；turn memory、idle60 和已撤回
复用实验未启用。原进程已停止，恢复的是同配置的新进程；测试 observer 已停止。

仍待真人验收：中英文自然长谈、早期语音名称/数字与否定纠错、多说话者归属、
跨轮任务、真实视频细节、音色/语气连续性，以及麦克风/AEC/扬声器实声插话恢复。
本轮浏览器测试为 localhost PCM 中继+正式播放器，未覆盖完整 HTTPS Gateway
传输和麦克风到扬声器声学回录；没有宣布这些通过。全文报告保留 6/7 记忆失败、
复用 PCM 变化、idle60 的显存代价及不改善冷首包/主响应的负面结果。

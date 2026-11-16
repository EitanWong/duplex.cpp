# RTX 3060 本机优化实测（2026-11-14）

本次完成原生 Windows CUDA 构建、设备配置与两轮推理优化。局部算子有明确收益，
声学输出层和设备传递的对照 PCM 一致；视觉 Flash Attention 会改变数值和生成输出。
长时间真人音质和浏览器连续播放仍待验收。
下面区分独立组件测速、原生服务回放和公开 Gateway 通路，避免混用吞吐与延迟。

## 硬件、模型与构建

| 项目 | 实测环境 |
| --- | --- |
| CPU | Intel i7-10700F，8 核 16 线程，AVX2 |
| 内存 | 32 GB |
| GPU | NVIDIA GeForce RTX 3060，12,288 MiB，SM 8.6 |
| 系统 / 驱动 | Windows 11 / NVIDIA 610.88，WDDM |
| 编译 | MSVC 14.38、Windows SDK 10.0.22621.0、CUDA 12.8.1、Ninja、Release |
| 存储 | 项目与权重位于 E 盘机械硬盘；本次加载时间有系统文件缓存 |
| 主模型 | 官方 MiniCPM-o 4.5 Q4_K_M；8,189,195,264 参数 |
| 其他模块 | 官方原精度 audio、vision、TTS、Token2Wav 权重 |

模型来源为 `openbmb/MiniCPM-o-4_5-gguf`，固定 revision
`47eeee5b877167846601d816a192f9a2ee62fc25`。所下载文件均按官方 LFS SHA256
校验；完整清单位于本机 `models/MiniCPM-o-4_5-gguf/source-manifest.json`。
主 GGUF 文件 SHA256 为
`1237a97ee081b8abebc47aa7dad565701e8f5f904cdc92f6723ac4281bbc0932`。
权重、生成音频、构建产物和本机配置均不纳入版本控制。

编译针对 SM 8.6，启用 CUDA Graphs、Flash Attention 和 AVX2。
Windows 上游 `iovec` / `ssize_t` 构建问题已修复；Worker 支持 `.exe` 发现、
DLL 搜索路径与配置传递。[构建及启动方法](nvidia.md)可复现此环境。

## 选定的运行配置

首轮测速使用 LLM 全层 GPU，8192 上下文，F16 KV，
Flash Attention；4 个 decode 线程、8 个 batch 线程，batch 512 / ubatch 128。
TTS 保留原精度 GPU decoder、F16 KV、4096 上下文及现有 hidden-only 路径。
声学输出层使用 CPU AVX2，Token2Wav 和 vocoder 使用 CUDA 0，并启用现有全路径预热。
默认单 Worker。

后续浏览器修复后，`profiles/nvidia-rtx3060.json` 与本机改用 4096 上下文，
为桌面和语音模块保留更多显存；对照数据和历史长度限制见文末。

线程数按延迟选择。单会话自回归 decode 受权重读取、内存带宽和串行依赖限制，
增加线程或观察到较低 Tensor Core 利用率，并不意味着能按算力比例增加 token/s。
视频帧、声学采样与 flow steps 保持原有语义。

## 声学输出层：相同计算顺序的 AVX2

原 Windows 路径逐个计算 6562 × 768 的标量点积。新实现把八个输出的权重
交错打包，同时计算八个输出；每个输出仍按原顺序逐项乘、加。
使用 `/fp:strict`，不做水平归约或 FMA 合并，额外常驻 CPU 内存 20,176,896 bytes。
Apple 继续使用 Accelerate。`OMNI_TTS_CPU_HEAD=scalar` 可回退。

真实 GGUF 声学权重、100 组固定随机 hidden 输入、交替测量顺序，空闲时三次独立复测：

| 次数 | 标量 ms / 次 | AVX2 ms / 次 | 局部加速 |
| --- | ---: | ---: | ---: |
| 1 | 4.5081 | 0.9288 | 4.854× |
| 2 | 4.5469 | 0.9484 | 4.794× |
| 3 | 4.4933 | 0.8968 | 5.011× |

三次均为 100 × 6562 个 logits 逐字节一致，包含不足八个输出的尾部。
此前带其他负载的一次测量为 4.6616 → 1.2999 ms（3.586×），说明耗时也受系统负载影响。
合成权重也通过同样校验。可运行 `test-tts-cpu-head.exe`；附加 row-major F32
权重文件参数即可检查真实权重。原始结果：`logs/tts-head-final.log`。

## 独立 LLM：Flash Attention 与线程扫描

`llama-bench`，Q4_K_M，全层 GPU、F16 KV，128-token prefill / 64-token generation，
每项三次重复。扫描 4/8 线程、ubatch 128/256、Flash Attention 开/关。

| 4 线程，ubatch 128 | Prefill token/s | Decode token/s |
| --- | ---: | ---: |
| Flash Attention 关闭 | 1725.09 | 59.41 |
| Flash Attention 开启 | 1840.27 | 62.79 |

这组测量中分别增加约 6.7% / 5.7%。8 线程和 ubatch 256 在短输入下没有稳定优势，
因此使用较小 decode 线程数与 workspace。该结果不是全双工吞吐，也不代表长上下文表现。
原始结果：`verification/llm-sweep.json`，监控：`logs/llm-gpu.csv`。

## 独立 Token2Wav：现有 CUDA Graph 开关

固定声学 token，重复五次，生成 18.8 秒音频；六次独立进程按关、开、开、关、关、开运行。
RTF 为计算时间 / 音频时长，越小越快。

| 路径 | 三次 RTF |
| --- | --- |
| Graph 关闭 | 0.1279 / 0.1279 / 0.1284 |
| Graph 开启 | 0.1708 / 0.1124 / 0.1111 |

Graph 首次 capture 有额外成本；后续两次的稳态测量约快 12%–13%，不能把首次成本排除后
称为所有请求都加速。内部 token2mel compute p50 为 78.5 → 57.3 ms。
六个 WAV 文件 SHA256 完全相同：
`6d5d53550d761456bb37ac710bc8e975ccb8773d78e9e70048e9129cc0069836`。
这里评测的是已有 CUDA Graph 实现。原始结果：`verification/t2w-ab/results.json`。

## 完整原生音视频：标量与 AVX2 对照（预热关闭）

使用仓库 36 秒 fixture，单帧视觉、16 kHz 输入，按一秒间隔发送并独立接收 24 kHz PCM。
两个进程均 `OMNI_SAMPLER_SEED=42`，LLM 为 temperature 0 / top_k 1，TTS 保留原采样设置。
除声学输出层开关外，配置一致。每个进程连续两轮；轮次 1 包含更多首次执行成本。
延迟从发出完整一秒输入开始计算，不包含采集这一秒所需的时间。

| 轮次 / 路径 | 所有输入 p50 ms | p95 ms | 最大 ms | 首个 SPEAK 输入至首包 PCM ms |
| --- | ---: | ---: | ---: | ---: |
| 1 标量 | 284.10 | 426.52 | 492.28 | 753.04 |
| 1 AVX2 | 284.86 | 431.98 | 462.82 | 676.08 |
| 2 标量 | 284.37 | 420.17 | 474.99 | 680.98 |
| 2 AVX2 | 272.40 | 374.24 | 402.46 | 563.35 |

首包分别减少约 10.2% / 17.3%。整体中位延迟一轮持平，另一轮降低约 4.2%；
两个顺序运行的进程与两轮样本不足以证明所有负载的固定加速比例。
视频编码通常占约 210–220 ms，仍是此 fixture 的主要耗时。

两轮分别生成 245760 / 249600 个样本（10.24 / 10.4 秒），对应轮次的文本与 PCM
逐字节一致。PCM SHA256 分别为：

- `e507a86acee7d11d5a06574eafe4ae620c3c117d4bda8fa3535f83f068b095ce`
- `c06059e6a20e1aa5986f7324d5c9f2eee217116b9b4734fd0e44234c2f994fac`

全部为有限值，无削波。参考音频为上游 `default_ref_audio.wav`。
原始报告及 PCM：`verification/duplex-scalar-seed42/`、`verification/duplex-avx2-seed42/`。

## 全路径预热：选入最终配置

另外启动一个相同 seed 的 AVX2 进程，仅设置 `OMNI_T2W_WARMUP=1`。
沿用现有两次 25+3 window 预热，丢弃其 PCM，并恢复 prompt、stream 和随机噪声状态。
本机预热耗时 655.6 ms，转移到 Worker 就绪之前。

| 轮次 | 所有输入 p50 ms | p95 ms | 最大 ms | 首个 SPEAK 至首包 PCM ms |
| --- | ---: | ---: | ---: | ---: |
| 1，AVX2 + 预热 | 276.24 | 396.64 | 406.90 | 603.23 |
| 2，AVX2 + 预热 | 273.57 | 392.80 | 434.83 | 571.71 |

相对未预热的 AVX2，首轮首包减少 72.84 ms；第二轮略慢 8.36 ms。
两轮文本和 PCM 均与对应的标量、未预热 AVX2 完全一致，哈希同上。
最终配置选择预热以降低首次交互成本，不把它解释为所有请求的稳态加速。
原始报告：`verification/duplex-warmup-seed42/`，预热日志：`logs/warmup-worker.log`。

## 持续输入、打断与服务验证

AVX2、预热关闭的路径连续 120 秒音视频输入，第 6 个输入主动 force-listen：

- 输入响应 p50 274.53 ms、p95 444.36 ms、最大 671.85 ms。
- 强制倾听返回成功，耗时 286.99 ms；确认返回后未再收到 epoch 0 的旧 PCM。
- KV 最大观测值 7612，触发 7613 → 2006 的滑动回收，日志确认 TTS KV 保留；
  最后一帧 KV 4086，滑动后继续响应。
- 输出 31.48 秒 PCM，全部有限，无削波，峰值 0.64943。

另测 36 秒纯音频：p50 155.52 ms、p95 193.38 ms、最大 240.79 ms，
生成 23.72 秒有限、无削波 PCM。其文本与发声时长不同，不与音视频模式直接比较。

公开 Gateway 通路完成 `queue_done → prepared → 18 个 result → stopped`，
收到 14 个音频包、322560 个样本，均有限值。`/health`、`/workers`、`/omni` 返回 200。
本测试涵盖 Python Gateway、Worker 和原生引擎，不涵盖实际浏览器收音与播放。

原始数据：`verification/duplex-long-interrupt/`、`verification/duplex-audio/`、
`verification/gateway-smoke.json`。CUDA ADD / IM2COL / CONV_TRANSPOSE_1D 的 303 项、
F16 Flash Attention 的 84 项正确性测试通过；52 项 Python 配置、启动与 schema 测试通过。

## 显存、启动与剩余限制

以 250 ms 间隔采样，整卡显存峰值 11,648 MiB，距离 12,288 MiB 容量约 640 MiB。
此数包括桌面和其他进程。测试期间 GPU 利用率最高 100%，功率最高 114.84 W，温度最高 61°C。
另一次 WDDM 进程快照为约 10.37 GiB dedicated、98 MiB shared；shared 分配不等同于
已经证明发生显存换页，也不能据此证明完全没有换页。原始监控：`logs/seeded-gpu.csv`。

热文件缓存下模型初始化约 6–7 秒；session 初始化约 2–3 秒。
本次没有测量清空操作系统缓存后的机械盘冷启动，不能把上述数据当成磁盘冷启动时间。

首包之后的供音也需要继续优化。用每个 epoch/turn 独立、100 ms 首包缓冲的简化模拟，
未预热 AVX2 两轮的最大供音缺口分别约 37.2 / 0 ms，预热后约 38.1 / 0 ms，
120 秒回放约 71.8 ms。预热没有消除这里观测到的间隙。
它不是浏览器自适应播放缓冲的实测结果，但说明当前证据不足以宣称绝无停顿。
真人长会话的自然度、说话人相似度、回声、AEC 和多小时稳定性均未验收。

当前结果支持此硬件上的可回退实验配置。更大上下文、更多视觉切片、多个 Worker、
其他量化精度或修改视觉 attention 数学路径，需要重新测量显存、延迟和音质。

## 浏览器回复、语音延迟及显存余量修复

后续排查发现普通聊天旧 HTTP 桥接把第一条用户消息放在 `index=0`，该索引实际只做
system/reference 初始化；SSE provider 还引用了已经退出的请求栈变量，可能导致原生
进程崩溃。浏览器聊天现使用原生结构化 `turn_based` 协议，传递消息角色、历史、
生成上限与语音开关，错误返回页面。SSE provider 另修复了请求参数生命周期。
参考音频加载统一使用已有的 soundfile / scipy，支持 Windows、双声道下混及
22.05 kHz → 16 kHz 重采样，不再依赖未安装的 librosa 或 macOS 固定 ffmpeg 路径。

语音原先在 PCM 已输出后还等待至少 3 秒，现依据最后一个 PCM 回调确认完成。
同时修复 TTS 错误读取 LLM 温度的行为：文本 `temperature=0` 不再强制声学采样
argmax。对照测试显式指定 `tts_temperature=0`；日常语音仍使用自身的默认 0.8。
这项修复改变了此前“LLM 温度为零、TTS 温度省略”请求的声学采样行为，不能把
该情形的 PCM 改变当作算子误差或宣称与旧行为音质完全相同。

本机通过局域网 HTTPS/WSS，单客户端、短句“请只回答：测试成功”，实测如下：

| 项目 | 修复后观测 |
| --- | ---: |
| 同模式连续文本回复（流式 / 非流式） | 约 0.22–0.25 秒 |
| 流式语音首个 PCM | 约 0.76 秒 |
| 流式语音整个请求完成 | 约 0.89 秒 |
| 非流式语音整个请求完成 | 约 0.73 秒 |
| 两次输出音频时长 | 1.52 / 1.68 秒，均为有限 PCM |
| 停止长流式 / 非流式请求后恢复 Worker idle | 约 330 / 112 ms |

这不是长回答、真实麦克风或其他客户端的延迟保证。从全双工切换到普通聊天仍会
重建模式相关模块，本次首个请求约 8 秒。多轮测试能正确回答先前给定的“蓝鲸”名字，
8-token 上限生效，关闭语音未收到 PCM，空消息错误能返回并释放会话。
记录：`verification/browser-repair-4096-https.json`、`logs/chat-control.log`。

为 12 GB 显卡保留更多余量，当前 NVIDIA profile 改用 **4096 上下文、F16 KV**。
未改模型权重、视觉切片或声学步数。8192 与 4096 的同一 36 秒音视频回放中，
文本与 PCM 逐字节一致，PCM SHA256 均为
`e507a86acee7d11d5a06574eafe4ae620c3c117d4bda8fa3535f83f068b095ce`。

| 上下文 | 整卡显存峰值 MiB | 输入响应 p50 / p95 ms |
| --- | ---: | ---: |
| 8192 | 11334 | 276.32 / 405.88 |
| 4096 | 10825 | 302.20 / 460.82 |

本轮释放约 **509 MiB**，没有测得吞吐提升。4096 会更早回收较旧历史，是显存余量
与历史长度的取舍。两轮测量有桌面负载差异，显存数包括整卡其他进程。
记录：`verification/browser-repair-duplex-8192/`、`verification/browser-repair-duplex-4096/`，
监控：`logs/repair-8192-gpu.csv`、`logs/repair-4096-gpu.csv`。

4096 上下文另完成 120 秒连续音视频输入和第 6 帧 force-listen：响应 p50 304.08 ms、
p95 425.64 ms、最大 544.76 ms；打断返回倾听耗时 330.12 ms。观测 KV 最高 3534，
输出 26.76 秒有限、无削波 PCM，整卡显存峰值 11098 MiB。HTTPS/WSS Gateway
另完成 `queue_done → prepared → 18 个 result → stopped`，收到 13 包音频、
301440 个有限样本。记录：`verification/browser-repair-long-4096/`、
`verification/gateway-https-final.json`。
另验证系统提示词生效、图片与音频联合输入返回文字，以及使用原生 22.05 kHz
英语参考音频完成 `prepared → stopped`。记录：`verification/browser-extra.json`、
`logs/browser-extra.log`。

HTTPS/WSS 已完成证书链验证，Windows 当前用户的系统证书验证请求返回 200。
其他局域网设备需各自信任公共 `lan-ca.crt`。浏览器 HTTP 页面曾实际显示正确回复；
HTTPS 的浏览器自动化验收被工具的 URL / 证书环境检查阻止，未绕过安全警告。
63 项 Python 回归测试通过；真人收音、扬声器回放与主观音质验收仍未完成。

## 第二轮：调试输出、视觉 attention、KV、会话与声码器

本轮按上述顺序实现与对照；选入 NVIDIA 配置的是关闭高频诊断及中间文件、
视觉 CUDA Flash Attention、跨模式复用已加载模块，以及纯文本 token 前缀复用。
主模型仍为 Q4_K_M、4096 上下文和 F16 KV；Q8 KV 与声码器设备传递可单独试用。
Apple M5 配置未修改，CPU / Metal / ANE 保持原有 attention 与声码器数据通路。

### 调试输出与 Windows 增量构建

`OMNI_VERBOSE=0` 在格式化时间戳之前跳过诊断；`OMNI_KEEP_ARTIFACTS=0` 跳过
hidden states、logits、token 文本和中间 WAV，原生 PCM 回调仍正常工作。
关闭中间文件后也跳过 WAV 合并，避免无效文件扫描和“无文件可合并”的告警。
保留默认兼容行为：未设置这些环境变量的旧文件消费者继续保存输出。
首轮对照未测得调试清理带来的稳定吞吐提升，不将它计入加速比例。

本机 CMake 将 MSVC 中文 `/showIncludes` 前缀误解码，Ninja 给旧对象记录了
零个头文件依赖。修改类字段后，这会把不同结构布局链接到一起，导致堆损坏。
构建脚本修正前缀并重建缺失依赖的原生对象，覆盖 Token2Wav 包装层和独立工具；
修复后的两者分别记录 116 / 137 个依赖。修复完成后使用构建目录中的标记，
避免每次重建确实没有头文件的源文件。记录：`logs/optimization-v2-build-deps-all.log`、
`logs/optimization-v2-build-final.log`。

### 视觉 CUDA attention

SigLIP 的 head dimension 为 72。CUDA 路径将 Q/K/V 补零至 80，保留原始 72 维
缩放系数，使用 F16 K/V、F32 attention 精度，再裁回 72 维。
后端不支持时使用原实现；仅 CUDA 显式开启此分支。

真实图像，六次编码，排除首次执行后的中位数：

| 输入布局 | 原 attention ms | CUDA Flash ms | 相对 RMSE | Cosine |
| --- | ---: | ---: | ---: | ---: |
| 单帧 | 167.917 | 123.252 | 0.01131 | 0.999932 |
| 概览加切片，合计三张 | 501.355 | 381.204 | 0.01417 | 0.999892 |
| 两张图的 batch | 290.405 | 202.769 | 0.01602 | 0.999867 |

单帧耗时减少约 26.6%。这条路径包含 F16 转换，不能宣称 embedding 或语音与原
attention 逐字节一致。单帧、切片和 batch 输出均为有限值；固定 36 秒回放的
输入响应 p50 / p95 从 284.09 / 384.86 ms 变为 246.62 / 339.36 ms。
语音输出从 10.24 秒变为 11.28 秒，均有限且无削波；文本也有变化，需要真人审核
视频理解与语音质量。高 cosine 和无削波只作为数值检查，不代替主观音质验收。
原始 embedding 与报告：`verification/optimization-v2/vision-*.f32`、
`vision-comparison.json`、`debug-clean/`、`flash-f16/`。

### 主模型 Q8 KV 对照：保持可选

视觉 Flash 开启，其余设置相同的两个进程，各回放一次 36 秒 fixture：

| LLM KV | 输入 p50 / p95 ms | 整卡显存峰值 MiB | 输出音频秒数 |
| --- | ---: | ---: | ---: |
| F16 | 246.62 / 339.36 | 10769 | 11.28 |
| Q8_0 | 252.02 / 351.05 | 10628 | 12.40 |

采样区间内最低整卡占用分别为 10615 / 10346 MiB。峰值减少 141 MiB、最低观测
减少 269 MiB，均含桌面与其他进程，不能当作精确进程分配差。Q8 的响应中位数
约慢 2.2%，输出变化，未做真人音质验收，因此默认仍为 F16。
`OMNI_LLM_KV_TYPE=q8_0` 可启用此取舍；TTS 的 KV 不随该变量量化。
记录：`verification/optimization-v2/flash-q8/`、`logs/optimization-v2-{fa,q8}-gpu.csv`。

### 模式与历史复用

跨模式保留模型权重和上下文对象，停止并 join 原线程，清空队列、滑窗状态和
会话状态，再应用下一会话的配置。纯文本复用必须验证 token 前缀；不复用生成
token 的可信性假设，最后一个输入 token 始终重算，以重新产生正确 logits。
更换系统消息、音频、图像、TTS 模板或滑窗时按实际 token / 模态失效。
当前聊天仍只构建最后最多八条非 system 消息，缓存不会扩大历史长度。

同一最终二进制、F16 KV、视觉 Flash 开启，对比关闭模式/文本复用与开启复用的
独立进程，直接测原生协议：

| 请求 | 对照 ms | 复用 ms |
| --- | ---: | ---: |
| 相同短文本，再次请求 | 204.64 | 128.61 |
| 相同短文本，第三次请求 | 202.96 | 128.90 |
| 相同历史，再次请求 | 335.50 | 129.79 |
| 继续追加历史 | 330.69 | 123.63 |

九组返回文本全部一致，覆盖更换 system、再次返回原 system、历史和历史追加。
日志确认复用 24 / 302 / 303 个 token；未把前缀命中次数当作速度测量。
原始报告：`verification/optimization-v2/history-{control,candidate}.json`。

同一二进制分别交替切换模式三轮，已加载 Worker 下测得：

| 模式初始化 | 关闭复用 ms | 开启复用 ms |
| --- | ---: | ---: |
| 全双工切入普通聊天，三轮 | 7150.96 / 6910.25 / 6832.94 | 3.20 / 2.35 / 2.04 |
| 普通聊天切回全双工，后两轮 | 8303.44 / 8193.93 | 1577.48 / 1564.46 |

前者是原生 `session.created` 的初始化耗时，不是整段回复耗时；开启复用时，
该短文本请求连同初始化约 118–125 ms。全双工仍需要参考音频预填充。
原始报告：`verification/optimization-v2/modes-{control,selected}.json`。

补测回退路径时发现旧全局 `prefill_done=true` 可让解码先于预填充完成，两个线程
并发使用同一 LLM，偶发 RoPE 断言。完成信号改为上下文独立、初始 false，
入队、等待及发布均使用同一队列锁；关闭诊断输出后，两种配置均完成三轮切换。

同一 36 秒音视频输入，对照与候选分别为 p50 / p95 243.89 / 337.68 ms 与
237.35 / 334.62 ms。这一小差异不构成稳定加速结论。两个进程的文本与 PCM
一致，均为 270720 个样本、11.28 秒，PCM SHA256 为
`1e0814ad007f6efe168c33b4bf9e283ee4520a41d463914b8981741772641b7c`。
候选同时启用了下述声码器设备传递，未改变视觉数学路径。
记录：`verification/optimization-v2/{control,candidate}-final-build/`。

### 声码器数据通路：实现并保留实验开关

`OMNI_T2W_DEVICE_MEL=1` 在 flow 与 vocoder 都位于同一 CUDA 设备时，
直接传递 CTB mel 并在显存中保留八帧 overlap，跳过 mel 下载、CPU 转置和重新上传。
vocoder 使用独立输入 buffer，避免流模型图被遍历或释放；对需要跨窗口读取的
输入明确保留生命周期，防止分配器在计算后覆盖 overlap。跨设备、CPU 和 ANE
使用原通路。声学 token、flow steps、随机噪声、PCM 回调和 crossfade 不变。

六个独立进程，按关、开、开、关、关、开顺序，各生成 18.8 秒音频：

| 设备传递 | 三次计算秒数 | 三次 RTF |
| --- | --- | --- |
| 关闭 | 2.110 / 2.070 / 2.088 | 0.1122 / 0.1101 / 0.1110 |
| 开启 | 2.080 / 2.076 / 2.085 | 0.1107 / 0.1104 / 0.1109 |

六个 WAV 哈希全部相同，为
`6d5d53550d761456bb37ac710bc8e975ccb8773d78e9e70048e9129cc0069836`。
未观察到足够大的稳定收益，默认关闭；不将设备传递称为显著提速。
原始日志、WAV 与结果：`verification/optimization-v2/t2w-device/`。

### 选定配置的持续回放与 HTTPS 回归

选定配置为 F16 KV、视觉 Flash 开启、模式与文本前缀复用开启，设备 mel 传递关闭。
从聊天切回全双工后，完成 120 秒音视频连续输入及第六帧 force-listen：

- 输入响应 p50 / p95 为 237.12 / 334.35 ms，最大 342.84 ms。
- force-listen 用时 224.57 ms；KV 最大观测值 3522。
- 输出 26.68 秒 PCM，全部有限、无削波，峰值 0.47687。
- 250 ms 整卡采样峰值 11298 / 12288 MiB，剩余约 990 MiB；本轮没有大幅降低显存。

原始报告：`verification/optimization-v2/selected-long-after-chat/round-0/report.json`。
HTTPS 音视频通路另完成 18 帧输入，收到 13 包音频、303360 个有限样本；
系统提示、图像与音频联合输入、22.05 kHz 英文参考音频通过回归。
记录：`verification/optimization-v2/gateway-https-selected.json`、
`verification/optimization-v2/browser-media-selected.json`。

预填充同步修复后的最终二进制再次通过 HTTPS 流式、非流式、历史、语音和生成上限测试。
短文本后续请求约 157–167 ms；语音首包约 645 ms，整个请求约 767 ms，
非流式语音约 739 ms。语音输出均为有限 PCM；这些数值来自本机脚本通路。
流式 / 非流式取消后恢复 idle 分别约 146 / 129 ms，空消息错误能正常返回。
记录：`verification/browser-repair-optimization-v2-sync.json`、
`logs/optimization-v2-sync-chat-control-bench.log`。

相同前置条件的新进程回放中，最终同步修复版本的 36 秒输入响应 p50 / p95 为
243.84 / 334.82 ms，文本与此前选定配置完全一致；PCM 仍为 270720 个样本、
11.28 秒，SHA256 仍为上述 `1e0814ad…2641b7c`。全部有限且无削波。
记录：`verification/optimization-v2/fresh-prefill-sync/round-0/report.json`。
另一次在语音和取消测试后执行的回放也完成了 36 帧，文本一致、PCM 有限且无削波，
但音频哈希不同。Token2Wav 的噪声流按进程持续推进，不能只固定 LLM seed 后就把
不同语音前置历史的 PCM 当作逐字节对照。

视觉 Flash 和 Q8 KV 的数学结果不同，不能据短回放宣称语音完全无损。
真人麦克风、扬声器连续播放、主观自然度和说话人相似度仍待人工验收。

## 2026-11-14：显存与预处理优化补测

详细瓶颈、原始来源、复现命令及量化、剪枝、稀疏计算等方案的取舍，见
[全链路优化研究](inference-optimization.md)。以下相对上述已优化工作树增量测量，
不应归因到未经修改的上游版本。

选定配置将未使用的 TTS 词表留在 CPU、只上传条件嵌入需要的行，按一秒查询预留
音频工作区，并以四线程执行等值 bicubic 预处理；Q4_K_M 主模型、F16 KV、声学
权重、采样参数与 flow steps 均保留。Apple M5 配置与默认路径不变。

两个新进程分别连续回放两轮 36 秒音视频输入，TTS 温度 0：

| 输入响应 | 对照 ms | 选定配置 ms |
| --- | ---: | ---: |
| 第一轮 p50 / p95 | 236.77 / 339.26 | 207.92 / 308.27 |
| 第二轮 p50 / p95 | 233.54 / 332.27 | 207.97 / 309.52 |

对应文本与 PCM 逐字节一致；整卡每 200 ms 采样峰值为 10921 → 10408 MiB，
包含桌面与其他进程。纯音频 TTS 温度 0.8 的两轮文本与 PCM 也一致，但延迟
波动重叠，尚无稳定加速结论。普通聊天避免额外 445.50 MiB 条件词表；实测整卡
峰值为 11471 → 10600 MiB，不能把此差值全归因于单张词表。

普通聊天一次对照 PCM 与选定配置不同，另一次原二进制对照的三段 PCM 全部与
选定配置相同，表明原路径也存在进程间变化。真实权重的条件图独立对照覆盖
六种 token 数，122112 个 F32 输出逐字节一致，不能只凭 seed 宣称所有语音确定性。

CUDA 声学投影算子提速约 6.4–6.7 倍，独立 logits 一致；但第二轮随机语音
回放出现差异，故保留实验开关、默认关闭。音频 Flash Attention 也因输出变化且
全链路无明显收益而关闭。默认仍使用 AVX2 声学输出层。

三次语音聊天后切回全双工，TTS 温度 0.8，连续 120 秒音视频输入并在第六帧打断：
p50 / p95 为 206.86 / 300.90 ms，最大 320.93 ms；force-listen 用时 201.59 ms，
确认后无旧 epoch PCM。输出 22.20 秒，有限且无削波，KV 最大 3522；整卡峰值
10634 MiB。HTTPS 文本、历史、流式与非流式语音、生成上限及参考音频接口通过，
短流式语音首 PCM 为 643.68 ms。

这些计时从完整一秒输入发送后开始，不含采集、播放与浏览器 AEC，不能称为
低于 100 ms 的端到端对话。真人音质与麦克风、扬声器验收仍待完成。

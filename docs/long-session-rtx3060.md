# RTX 3060：长会话实现与验证（2026-11-15）

后续同步/Graph 归因、真实浏览器播放与小时压力测试见
[稳态推理与播放验收](steady-playback-rtx3060.md)。

本轮落实了共享 KV 删除事务、完整输入预算、真实物理跨度观测，以及默认关闭的
有来源文本记忆。默认仍为单 Worker、4096、F16 K/V、CUDA Flash Attention 和
现有 CUDA Graphs。8K F16 可作为增加近期历史的实验选项；Q8 节省容量但更慢。
16K 分配和实际历史深度均达到，但事实回答及延迟不合格，未继续 32K。
本轮没有证明默认链路稳定加速，也没有完成人工语音质量验收。

## 基线、工作区与测量范围

硬件为 RTX 3060 12 GB、i7-10700F、32 GB RAM、Windows 11/WDDM、CUDA 12.8.1、
驱动 610.88，负载下 PCIe Gen 3 ×16。模型和校验值沿用
[本机性能记录](performance-rtx3060.md)，LLM 为官方 Q4_K_M，其他模块保持原精度。
Qwen3 为 36 层、32 query heads、8 KV heads、head dimension 128，原生位置范围
40960。没有展开 GQA、改 RoPE 配置或扩大 TTS 上下文。

开始时保存了 dirty 文件清单、SHA256、staged/unstaged patch、原文件及基线二进制。
用户随后授权整理原有成果，已在本地 `dev` 保存为 `de96c81`，含 signoff 和 AI
协助披露；未推送、未创建 PR。本轮新增优化留在工作区。其他任务的 Strata 文档保留。
Apple M5 profile、selected profile 以及 Metal/CoreML/Accelerate 专用路径未修改，
本轮没有 M5 硬件测量。

原始结果全部在忽略目录 `verification/long-session-20261115/`。每次 pipeline
启动独立进程，固定同一组 36 段音视频循环、1 秒输入、参考音频、LLM greedy/top-k 1、
每 chunk 10 个 speak token、TTS temperature 0.8、seed 42、4/8 CPU 线程、
batch 512/ubatch 128。GPU 测试顺序执行；TTS 始终为独立的 4096/F16。
基线使用保存的旧二进制。模型初始化与首个图像的冷激活、后续稳态分别记录。
文件缓存已热，不能把启动数据解释为磁盘冷启动。

延迟从准备好的完整输入发送开始，到原生响应结束，不含麦克风收集、端点判断、
浏览器 AEC、网络和播放。`first_pcm_ms` 从首次 speak 输入发送到首 PCM。
`input_queue_ms` / `llm_queue_ms` 是原生编码前/LLM 前排队；`input_age_ms` 是
LLM 消费时距提交的年龄，包含编码。脚本另记发送节奏落后量。阶段重叠，不能相加。

## 实现和回退

`kv_remove_range()` 是两套历史路径共用的删除/逻辑位置移动事务。成功删除后才更新
`n_past`、轮次边界、unit 长度、pending input 起点、累计位置与淘汰计数。执行前核对
真实 KV 最大位置，陈旧计数直接失败；删除失败不推进账本。原有 basic/context 策略
保留，但位置变化通过共享事务协调；native duplex 的 turn 模式只运行一个预算规划器。

`OMNI_DUPLEX_WINDOW=turn` 启用实验规划器：保护 system/参考音频前缀，按完整音视频
输入预留容量，再留 `max(32, speak_budget + 16)` 的生成空间。超预算优先一次删除
足够的已结束轮次，并预留 `min(1024, capacity/8)` 的水位间隔。只有一个活动轮次
本身过大且不能容纳输入时，才退到删除其最老的完整 unit；记录 `capacity-unit`，
不会切当前输入的任意 token。整个输入即使分段 prefill，也先计算总预算。
fused prefill 部分改变 KV 后失败不会重试第二遍输入；decode 失败清除失败 batch 并
以 `decode_failed` 关闭会话，重新建会话后再用，避免继续使用半成品状态。

省略 `OMNI_DUPLEX_WINDOW` 即回到已有默认滑窗选择。默认输出等价性单独验证，
实验 turn 策略会保留更多历史，可能改变回答和语音。LLM 裁剪不操作 TTS KV。

新增只读元数据 API `llama_memory_get_kv_stats()`，当前支持 plain KV cache。
由唯一 LLM 线程发布快照；WS/TTS 不遍历正在变化的 KV。`metrics.kv` 包含：

| 字段 | 含义 |
| --- | --- |
| `used_cells` | 实际有效槽位数 |
| `physical_span` | 最高占用物理槽位 + 1 |
| `attention_span` | 按现有后端规则对齐后的扫描跨度；本模型单序列 |
| `capacity_cells`, `allocated_bytes` | 已分配容量与 buffer bytes |
| `accounted_tokens`, `completed_turns` | 应用 unit 账本、已结束轮次数量 |
| `evictions`, `evicted_tokens`, `unit_fallbacks` | 淘汰次数、token 与活动轮次兜底次数 |
| `prune_metadata_ms` | rm/shift 元数据 host 耗时累计，不含后续 GPU RoPE |
| `input_queue_ms`, `llm_queue_ms`, `input_age_ms` | 队列与输入年龄 |
| `memory_injections`, `memory_tokens`, `memory_prefill_ms`, `memory_retrieval_ms` | 记忆累计注入、预填充、检索费用 |

`OMNI_KV_TRACE=1` 额外打印淘汰原因、区间、位置变化、保留轮次/unit，以及
synchronize、graph build、RoPE update/reserve、CUDA capture/instantiate/launch
的 host 耗时，并启用核心 trace 日志。它包括 LLM/TTS 与启动阶段，不是逐 kernel GPU
profiler；CUDA capture 项只在需要捕获/更新时记录。各项包含重叠，不能求和当总开销。
日常关闭 trace，避免把日志费用加进默认链路。

## 默认配置对照：没有可归因的延迟改善

120 输入、index 11 打断，按旧 → 新 → 新 → 旧顺序运行：

| 运行目录 | 输入 p50 / p95 ms | 除首输入 p50 / p95 ms | 首 PCM ms | 首输入 ms |
| --- | ---: | ---: | ---: | ---: |
| baseline-av | 211.80 / 330.98 | 211.80 / 320.31 | 721.10 | 1299.13 |
| candidate-av | 207.94 / 306.89 | 207.87 / 304.54 | 554.27 | 1036.22 |
| final-av | 208.63 / 304.17 | 208.36 / 302.55 | 538.51 | 1045.13 |
| baseline-reverse | 206.96 / 302.86 | 206.95 / 302.45 | 558.92 | 1110.02 |

四次全部 120 个文本/听说决策和 23.48 秒 PCM 一致；SHA256：
`336aea439ceea6929a0a7504d37c3df9a2b6d706714541df4357b952feb00035`。
新版默认在输入 43/61/80/98/116 淘汰，共五次，账本与 KV 长度逐输入一致。
前后优势随顺序消失，因此不宣布加速。此处是本轮新测量，未复用旧报告的 25 项通过数。
默认路径未额外取得整套 logits dump；PCM/text/decision 等价只覆盖固定 fixture。

匹配资源采样的 final/reverse：进程 dedicated 均 9628.98 MiB、shared 均 96 MiB；
working set 约 10.07 GiB、private 约 16.49 GiB；整卡峰值 10214 / 10221 MiB。
进程资源从模型载入后开始以约 1 秒加查询时间采样，可能漏掉瞬时峰值，不能作为加载
阶段 RAM 上界。private 是提交量，不能直接等同于物理驻留 RAM。

## 真正达到的长历史与资源

独立 `llama-bench`，每项 3 次、生成 128 token，固定 FA、线程和 batch：

| 实际历史 token | F16/F16 token/s | Q8_0/Q8_0 token/s |
| --- | ---: | ---: |
| 3072 | 57.7429 | 54.2364 |
| 7168 | 52.3323 | 47.3876 |
| 14336 | 未测 | 38.6121 |

Q8 保持容量选项。先前 4K 的 F16/Q8 测量 54.84 token/s 不是本轮重测结果。

完整 AV 流启用 turn 规划器，index 11 打断；不启用文本记忆：

| 配置 / 运行目录 | 输入数 | 最大有效 token | ≥7168 / ≥14336 的输入数 | 首次淘汰输入（0 起） | LLM KV MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8K F16 / av-8k-f16 | 140 | 8125 | 44 / 0 | 100 | 1152 |
| 8K Q8 / av-8k-q8 | 140 | 8122 | 44 / 0 | 100 | 612 |
| 16K Q8 / av-16k-q8 | 240 | 16348 | 152 / 63 | 202 | 1224 |

三组均发生 3 次完整轮次淘汰，无活动 unit 兜底，逐输入账本一致。它们确实计算过
长历史 attention，不只是分配成功；不能因此推断模型正确利用了每段历史。
最长流式回放为 240 个一秒输入，约四分钟，尚不构成数小时持续运行压力验收。
删除只回收槽位，不缩显存。8K F16 最后有效 7795，物理跨度 8125、扫描 8192；
逻辑长度下降未使物理扫描同步下降。

| 配置 | 输入 p50 / p95 ms | 首 PCM ms | 打断响应 ms | 发送落后 p95 / max ms | 播放代理最大缺口 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8K F16 | 217.85 / 359.45 | 538.72 | 201.57 | 14.08 / 146.46 | 10.16 |
| 8K Q8 | 231.71 / 350.74 | 562.20 | 206.91 | 17.23 / 123.10 | 109.99 |
| 16K Q8 | 242.19 / 484.21 | 563.68 | 200.90 | 18.46 / 414.28 | 1140.86 |

三组 input age p95 为 134.53 / 134.90 / 134.25 ms，编码前排队 p95 ≤0.063 ms，
LLM 前排队 p95 ≤0.042 ms。打断确认后未收到旧 epoch 的非空 PCM，但这是
index 11 一次程序打断，不覆盖真实说话中插话及扬声器残音。播放代理按 epoch/turn
模拟 100 ms 首缓冲，依据原生到包时间；不是浏览器实际欠载次数，亦不测麦克风。

| 配置 | 进程 dedicated / shared MiB | working set / private GiB | 整卡峰值 MiB |
| --- | ---: | ---: | ---: |
| 8K F16 | 10204.99 / 98 | 10.37 / 17.05 | 11036 |
| 8K Q8 | 9684.98 / 96 | 10.11 / 16.53 | 10532 |
| 16K Q8 | 10382.99 / 98 | 10.08 / 17.13 | 11129 |

整卡包含桌面与其他应用。8K F16 和 Q8 的同条件 140 输入有 **29 个**文本/听说
结果不同，PCM 不同；输出改变也改变后续负载，p95 差异不能作为等价加速。
全部生成 PCM 有限、无削波，但未以此判定音色、情绪和语音质量通过。

## 事实质量：8K 有限成功，16K 明确失败

`long-history.cpp` 把真实 prompt 填到指定 token 深度，问题覆盖中文早期项目名/数字、
英文名称/钥匙、否定与纠正楼层、未完成跨轮任务，以及楼层+门禁码+饮料三事实联合。
每题清 KV 并完整 prefill，greedy 生成最多 80 token。填充为合成重复天气文字，
不是自然会话或音视频质量基准。`full` 使用早期完整文本，`memory` 只把选中引文放在
近期尾部，`recent` 丢掉早期事实。后两种都以 4K 配置测试。

| 配置 / 实际 prompt 深度 | 本套回答 | prefill / TTFT 中位数 ms |
| --- | --- | ---: |
| 8K F16 / 7168 | 六类正确，联合回答“23楼，7319，茶” | 4504.06 / 4504.39 |
| 8K Q8 / 7168 | 六类正确，联合回答“23楼，7319，茶” | 4801.36 / 4801.69 |
| 16K Q8 / 14336 | 楼层、否定、任务及联合回答失败；有 think 标记或空输出 | 11196.79 / 11197.12 |
| 16K F16 文本控制 / 14336 | 同类失败，不能全部归因于 Q8 | 10420.53 / 10420.86 |
| 4K F16 + 文本记忆 / 3072 | 六类正确；联合包含楼层、码与茶 | 1756.78 / 1757.11 |
| 4K F16 recent-only / 3072 | 多数 UNKNOWN，楼层幻觉为 1，联合 UNKNOWN | 1755.38 / 1755.72 |

最初联合题实际只问饮料，不充分；原结果保留，修正后三事实题另存
`recall-joint-*.jsonl`。表中 prefill/TTFT 是初始整套题的中位数，联合题采用修正后的
结果。有限六类成功不能外推长历史普遍质量。16K F16 仅有隔离文本控制，没有完整
语音链路测试。4K 记忆工作量较小，不能把 TTFT 差解释为相同计算的内核加速。

## 可用的实验文本记忆

当前 duplex 的输出文本是助手回复，不能当作用户转录。本轮不增加 ASR 或让同一
`llama_context` 被多个线程并发提取。只接受调用方提供、审核过的用户原文引述，
`source` 为 `user_text` 或 `external_transcript`；这校验来源形状，不能认证外部转录
内容真实。浏览器/Gateway 尚未自动产生这些事实，原生 `/backend` 和 Python SDK
可调用。没有可靠用户文本时保留空记忆，不编造麦克风事实。

同时设置 `OMNI_DUPLEX_WINDOW=turn` 和 `OMNI_TEXT_MEMORY=1`。已完成输入步骤后，
发送原生消息；`turn` 为零起来源轮次，必须小于 `metrics.kv.completed_turns`。
这里的轮次是 native LISTEN 边界计数，不等同于 ASR 端点或人的完整语义发言：

```json
{"type":"memory.update","facts":[{"id":"room-v1","source":"user_text","speaker":"user","time":"2026-11-15T22:00:00+08:00","turn":0,"text":"会议室在17楼。","negated":false,"status":"fact"}]}
```

返回 `memory.updated`（revision、facts 数）或 `memory.error`，不会混入 listen/speak
结果队列。纠错用新 id、`supersedes: "room-v1"`，明确引用“不是17楼，是23楼”；
旧事实保留来源但不再检索。状态 `fact` / `pending` / `done`，任务完成也用新记录
supersede 原 pending。id 不可重用，引用目标须存在。错误 batch 全部拒绝，不部分写入。
每次 1–16 条，每条 text ≤1024 UTF-8 bytes，最多 128 条/128 KiB；满时显式拒绝，
不静默丢失事实。`speaker` 和 `time` 必填；时间是来源元数据，不是服务器测得时间。

Python 原生 Worker 用法，在已完成的 `duplex_generate` 之间调用：

```python
ack = worker.duplex_update_memory(facts, query="会议室 楼层")
```

也可调用 `NativeDuplex.update_memory(facts, query=...)`。query 最多 2048 UTF-8 bytes，
只用于确定性词面排序，再兼顾 pending 和新近记录，不执行向量搜索。引文连同出处、
否定、时间、纠错关系一起选入，不能容纳的完整条目跳过；含两个 unit 控制 token
总预算不超过 512。引文 tokenizer 禁止解析 special-token 字符串为控制 token。
模型仍可能受恶意引文影响，这不是提示注入防护保证。

检索在 WS 串行步骤上只读 tokenizer，实际 prefill 只在唯一 LLM 线程、前一轮
LISTEN 且 KV 正好处于记录边界时执行，活动回复中延后。生成的 `memory` unit 参与
同一账本，已注入且仍存活的相同引文不重复写；被淘汰或检索改变后再注入。
记忆只驻留当前会话，关闭/重建会话清空，不写磁盘；禁用开关并新建会话可回退。

集成验证使用明确标记的合成用户事实，**不是 AV fixture 的转录**：先写楼层与 pending
任务，再 supersede 楼层。最初 `av-memory-4k` 虽 ACK 成功却零次注入，原因是 WS
预先置 `text_streaming`；保留失败记录，改查真实边界。随后
`av-memory-injected-4k` 证明注入，但每次任何淘汰后都重注入，存在冗余；最终改为检查
该 memory unit 是否仍存活。最终 `av-memory-final-4k` 的 120 输入中：

| 项目 | 每次淘汰后重注入的原型 | 最终按 unit 存活去重 |
| --- | ---: | ---: |
| 注入次数 / 累计 token | 15 / 2948 | 4 / 781 |
| 累计记忆 prefill ms | 1491.77 | 379.79 |
| 淘汰次数 | 13 | 10 |
| 输入 p50 / p95 ms | 218.27 / 366.00 | 216.50 / 334.37 |

两次使用相同 fixture、seed 和事实 schedule；保留引文的次数变化也改变上下文和输出，
PCM 为 22.72 / 22.92 秒且不同。因此这里只证明避免了冗余注入和相应工作量，不宣称
等价语音链路加速。最终每次注入 190/197 token、75.98–106.24 ms；词面检索 120 次
累计 72.86 ms。两次更新 ACK 成功、纠正记录取代旧引文参与检索，最终无账本偏差、
无 unit 兜底。最终最大有效 4040，最后有效 3466/物理 4032/扫描 4096，仍没有压紧。
首 PCM 599.13 ms，发送落后 p95/max 21.02/135.06 ms，播放代理缺口为 0。
最终 PCM 有限、无削波；人工质量未验收。

最终 trace 的 host 时长如下，包含启动及 LLM/TTS，不能相加：

| 阶段 | 样本数 | p50 / p95 / max ms |
| --- | ---: | ---: |
| graph build | 406 | 0.318 / 0.352 / 0.564 |
| synchronize | 2252 | 0.001 / 33.627 / 49.268 |
| RoPE update + reserve | 10 | 4.474 / 7.009 / 7.076 |
| CUDA capture + instantiate + launch | 63 | 1.636 / 74.581 / 108.848 |

rm/shift 元数据累计 19.89 ms。日志声明的 LLM CUDA/host compute buffer 为
76.10/5.00 MiB、TTS 为 6.63/1.75 MiB，后续 graph reserve 还报告 CUDA 59.30 / CPU
18.30 MiB；这是可见的工作区分配记录，不是整个进程所有 allocator 的动态峰值。
进程/整卡采样覆盖 graph、模型和其他缓存的合计；没有把声明值相加伪造 workspace 峰值。
最终进程 dedicated/shared 约 9582.98/94 MiB、working set/private 10.07/16.49 GiB，
整卡峰值 10208 MiB。

文本记忆属于有损事实检索，无法完整保留语气、音色、原始画面、时序和任意逐字细节，
不是无损 KV 压缩。AV 集成验证能证明 provenance/correction、预算和注入执行；
独立文本 probe 能证明本套事实召回，二者不能代替真实语音事实抽取、听觉及视觉验收。

## 本轮收敛的方案

32K F16/Q8 的 LLM KV 约 4.5/2.39 GiB，尚未包括语音、视觉、graph/workspace。
16K Q8 已出现长尾延迟、播放代理缺口和回答失败，因此没有为了“分配成功”继续 32K。
Q8、8K、turn planner 和记忆均不进入默认 profile。

没有引入 GPU 物理压紧：旧探针 CPU save/load 约 302 MB 状态需 108–120 ms，
生成约快 2%，下 token logits 最大差 0.4334，不能直接用于服务。本轮确认剪裁后
物理扫描仍高，但没有证明一次 GPU gather/位置修正/同步/重捕获能带来净收益。
没有实现 RAM 冷 KV、逐 head 预算或新低比特格式：完整 32K F16 每 token 经 PCIe
回传的理论下界已约 0.31 秒/token；原始历史检索需求与本机稀疏选择质量尚无支持。

InfLLM/InfiniGen 的块检索与预取、Quest 的页界估计、ShadowKV 的低秩重建、MagicPIG
的 CPU LSH、DuoAttention 的 head 分组、KIVI/KVQuant 的不同精度布局解决不同问题，
本轮不移植整套框架或采用论文收益。机制与来源见
[低显存长上下文研究](long-context-low-vram.md)。当前只有 8 个 KV heads，任何未来
query head 选页都须计入集合并集成本，并保持位置、RoPE、mask、物理块与完整 unit
一致。分页、FA、chunked prefill 和前缀缓存不是通用长上下文加速开关。

当前支持语义：持续运行靠有限窗口淘汰；窗口内保留近期多模态 KV 并全量 attention；
窗口外只可保留调用方提交的有来源文本事实。没有窗口外原始 KV 稀疏检索，也没有
跨全部已丢弃历史的全量 attention，更不承诺无限会话记忆。

## 复现与验收

记录托管服务状态后暂停，仅顺序执行本项目 GPU 测试，结束恢复原配置：

```powershell
./scripts/windows/status.ps1
./scripts/windows/stop.ps1
./scripts/windows/build.ps1 -CudaRoot $env:CUDA_PATH -AllKvQuants
$env:PATH = "$PWD/engine/build-cuda/bin;$env:CUDA_PATH/bin;$env:PATH"
./engine/build-cuda/bin/test-kv-budget.exe
./engine/build-cuda/bin/test-kv-ledger.exe models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf
.venv/Scripts/python scripts/benchmarks/kv-cache.py `
  --model models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf `
  --depths 3072 7168 --repetitions 3 --formats f16:f16 q8_0:q8_0 `
  --output verification/repeat-decoder
.venv/Scripts/python scripts/benchmarks/pipeline.py `
  --model-dir models/MiniCPM-o-4_5-gguf --count 140 --rounds 1 `
  --interrupt-at 11 --tts-temperature 0.8 --ctx-size 8192 --resources `
  --env OMNI_DUPLEX_WINDOW=turn --output verification/repeat-8k-f16
.venv/Scripts/python scripts/benchmarks/summarize-session.py verification/repeat-8k-f16
./scripts/windows/start.ps1
```

Q8 附加 `--env OMNI_LLM_KV_TYPE=q8_0`；16K 用 `--ctx-size 16384 --count 240`，
仅作失败复现。记忆用 4096/120 输入，附加 `--env OMNI_TEXT_MEMORY=1
--env OMNI_KV_TRACE=1 --memory-file verification/your-user-facts.json
--memory-query '事实关键词'`；schedule 格式为 `[{"after_input":0,"facts":[...]}]`。
测试中的 index 0 强制 LISTEN 是为了可复现来源轮次，不代表说话中打断质量。
`pipeline.py` 自动记录配置、PCM、时间、WDDM/RAM/整卡样本并结束子进程。

文本 probe 在 VS developer PowerShell 编译，所有产物放忽略目录：

```powershell
cl /nologo /O2 /utf-8 /std:c++17 /EHsc /Iengine/include /Iengine/ggml/include /Iengine/vendor `
  scripts/benchmarks/long-history.cpp /Feverification/long-history.exe /Foverification/long-history.obj `
  /link engine/build-cuda/src/llama.lib engine/build-cuda/ggml/src/ggml.lib engine/build-cuda/ggml/src/ggml-base.lib
verification/long-history.exe models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf 8192 7168 f16 full
```

模式 `memory/recent` 用 `4096 3072 f16`；末尾可附 test index（0–5），5 为修正后的联合
问题。不要同时运行 probe、decoder、pipeline 或托管服务。

已通过 74 项相关 Python 测试、model-free C++ 预算/记忆事务测试、真实 GPU KV 拒绝
路径测试。服务控制器另修复 Windows parent PID 复用导致误认子进程的问题，并有纯
inventory 回归测试；曾误停止较旧 Edge 进程，已向用户说明，后续测试使用修复控制器。
仓库卫生、文档链接、PowerShell 语法与 diff 检查记录在同一 verification 目录。
相关 Python 命令为 `.venv/Scripts/python -m pytest runtime/tests/test_cpp_launch.py
runtime/tests/test_native_chat.py runtime/tests/test_native_duplex.py runtime/tests/test_schemas.py -q`。
测试结束已恢复原本健康的 Worker（22440）、native（19080）和 HTTPS Gateway（8040）；
Worker idle/model_loaded，Gateway 注册就绪，仍用 4096/F16/FA，实验环境变量未启用。

人工仍须检查中英文自然长谈、早期语音名称/数字、否定纠错和跨轮任务、多人归属、
多事实联合、画面细节、音色/语气连续性、真实插话后恢复与残音，以及浏览器实际欠载。
没有把 finite/无削波 PCM、六题召回或 token/s 视为这些验收已通过。

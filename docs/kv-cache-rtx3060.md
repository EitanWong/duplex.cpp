# RTX 3060：KV cache 与滑窗优化

后续已完成的预算/账本实现、真实 8K/16K 深度实验和可选文本记忆，见
[长会话实施与验证](long-session-rtx3060.md)。本文保留前一轮的原始结论与测试计数。

调研、构建与本机测试日期：2026-11-15。硬件为 RTX 3060 12 GB / SM 8.6、
i7-10700F（8 核 16 线程）、32 GB RAM，Windows WDDM、驱动 610.88、CUDA 12.8。
实测 PCIe 链路为 **Gen 3 ×16**。本轮沿用现有 Q4_K_M LLM、F16 声学模型和
已优化的 CUDA/AVX2 路径；此前结果见[语音优化记录](voice-optimization-rtx3060.md)。

## 推荐的平衡点

当前 4096 上下文优先保留 **F16 K + F16 V、单个常驻 Worker、现有 Flash Attention
与 CUDA Graphs**。本机测试中，Q8 KV 节省容量，但四种组合都没有超过 F16 的生成速度。
不能把论文中的长上下文吞吐收益直接用作这台机器的语音响应收益。

该模型的流式听说决策约为 **1 Hz**。缩短内部推理并不能直接消除输入收集、轮次
判断和声学窗口的等待；随意改成 100 ms 输入或减少 flow 步数会改变模型条件。
后续优化还应同时记录播放欠载、输入队列年龄和打断后残留音频，不能只看 token/s。

新增 `OMNI_LLM_K_TYPE`、`OMNI_LLM_V_TYPE`，可独立设置 `f16` / `q8_0`；未设置时
继承原来的 `OMNI_LLM_KV_TYPE`，再回退到 `f16`。TTS cache 仍由声学路径独立使用 F16。
混合精度是容量实验选项，NVIDIA 默认 profile 未开启。

## 本机 decoder 测量

同一模型与 CUDA 构建，`llama-bench`，Flash Attention 开启，`-t 4`、batch
512、ubatch 128、每次生成 128 token，每个历史深度重复 5 次：

| K / V | 历史 0 token，token/s | 历史 1024 token | 历史 3072 token |
| --- | ---: | ---: | ---: |
| F16 / F16 | 63.04 | 60.86 | 57.83 |
| Q8_0 / Q8_0 | 61.44 | 58.79 | 54.29 |
| Q8_0 / F16 | 61.86 | 59.10 | 54.36 |
| F16 / Q8_0 | 62.01 | 59.23 | 54.84 |

3072 历史 token 下，只压缩 V 比 F16 慢 **5.2%**，双 Q8 慢 **6.1%**。同组重复的
标准差约 0.03–0.13 token/s；固定顺序测试结束后再跑 F16，3072 深度为 **57.81
token/s**，未发现能解释该差距的持续降速。重复样本不等于跨日置信区间。
这些测试没有采样、音频编码或语音合成，不能作为完整对话延迟。

原始结果：`verification/kv-research/llm-formats/`、`f16-repeat-console.log`。

## 默认配置的完整语音回归

原始 binary 与修正后的 binary 各启动一个新进程，按 1 秒间隔回放同一组 **120 段
音视频输入**，LLM greedy、TTS temperature 0.8、seed 42、input 11 强制 LISTEN。
两次运行均发生 **5 次 KV 长度下降**，最高长度均为 3542；覆盖实际滑窗而非仅测短历史。

| 版本 | 输入 p50 ms | p95 ms | 最大 ms | 语音输出 |
| --- | ---: | ---: | ---: | --- |
| 本轮前的工作区 binary | 210.66 | 303.99 | 1291.65 | 23.48 s |
| 边界修正版、F16 KV | 208.56 | 303.57 | 1049.52 | 23.48 s，PCM 完全相同 |

**全部 120 次文本/听说决策及 PCM 字节一致**，音频 finite、无削波。PCM SHA256：
`336aea439ceea6929a0a7504d37c3df9a2b6d706714541df4357b952feb00035`。
约 1% 的 p50 差距不构成稳定加速证据；最大延迟包含首次视觉加载，不能视为稳态收益。
全卡 200 ms 采样峰值 10527 / 10380 MiB，包含桌面与其它应用，不能把其差值记为
裁剪释放的显存。它验证本组输入的输出一致性，不代表真人麦克风与所有长对话质量验收。

证据：`verification/kv-research/{av-baseline,av-selected}/` 和
`av-selected-comparison.json`。

同样的 120 段音视频、采样和打断设置下，只把 V 改为 Q8_0：

| K / V | p50 ms | p95 ms | 首次 SPEAK 输入至 PCM ms | 音频时长 |
| --- | ---: | ---: | ---: | ---: |
| F16 / F16 | 208.56 | 303.57 | 544.10 | 23.48 s |
| F16 / Q8_0 | 216.65 | 340.46 | 536.86 | 26.84 s |

有 **18/120** 个输入的文本/听说结果不同，从 index 9 开始；PCM 也不同。首包约 7 ms
的差异与明显变差的整体延迟并存，不能选择性解释为加速。输出变化本身也改变了合成
工作量，因此这组回放不是严格相同工作的 kernel 比较。量化后的音频 finite、无削波，
但这不证明自然度、识别能力或事实准确度相同。结合同步的 decoder 测量，未选用该配置。
全卡峰值 10380 / 10181 MiB 仅为容量观察，精确 KV 张量减少量仍是上表的 135 MiB。

证据：`verification/kv-research/av-vq8/`、`av-vq8-comparison.json`。

## 为什么“删了历史”不等于“显存下降、推理更快”

本模型为 36 层、8 个 KV heads、head dimension 128。4096 token 的 LLM KV payload：

| K / V | MiB | 比 F16 少 |
| --- | ---: | ---: |
| F16 / F16 | 576 | — |
| Q8_0 / F16 或 F16 / Q8_0 | 441 | 135 MiB，23.4% |
| Q8_0 / Q8_0 | 306 | 270 MiB，46.9% |

Q8_0 每 32 个数另有一个 F16 scale，因此不能按“正好减半”计算。这些数字是
LLM K+V 张量容量，不含权重、TTS/audio cache、workspace、CUDA Graphs 或桌面显存。
TTS 4096-token F16 cache 另约 240 MiB；不能把 LLM 压缩比例套到整个服务。

`llama_memory_seq_rm` 释放的是预分配缓存内的槽位。当前实现不会据此缩小 CUDA
buffer，`seq_add` 也主要修改位置和 shift 元数据。下一次 decode 才执行 K 的 RoPE
修正。量化 K 还需经历 F32 解量化、旋转、重新量化，增加计算并可能积累误差。

此外，`llama_kv_cache::get_n_kv` 根据 **最高占用的物理槽位**计算注意力范围，再对齐
到至少 256 的倍数；它并不只看有效 token 数。删除中间历史后，尾部槽位仍被占用时，
逻辑长度下降并不保证 attention 扫描范围同步下降。该问题与权重量化是两个瓶颈。

独立 CUDA 探针实际验证：3072 个物理位置中删除 1024 个后，有效 cell 为 **2048**，
物理跨度仍为 **3072**；通过公开 sequence state API 保存并装载后，下一步探测跨度为
**2049**（包含已追加的一个 token）。六次交替顺序测量表明，后续 128 个匹配 token
的生成耗时从约 **2205 ms 降到 2162 ms，约 2%**；但保存、装载约 302 MB 状态需
**108–120 ms**，且下一 token logits 最大绝对差为 **0.4334**。缓存重排不能自动
宣称数值无损；该合成输入也不是语音质量测试。

因此未把 CPU save/load 用作服务中的压紧策略。它可作为进一步设计 GPU 内低频压紧
的参考基线，但新实现还需要评估是否足以抵偿搬移与图重建，以及真实语音输出变化。
原始证据为 `verification/kv-research/fragmentation-debug.log`。早期未成功启用 DLL
诊断环境的 screen 不用于物理跨度结论；可复现命令见下文。

## 当前滑窗的缺点与本轮修正

原生 duplex 实际主要使用 `kv_cache_slide_window`。另有 `SlidingWindowConfig` 的
unit/turn 实现，但默认 `mode=off`；不能把另一条服务器路径的 turn 模式视为原生协议
已经开启的重要性裁剪。当前策略更接近“系统前缀 + 最近历史”，不是 SnapKV 或 H2O。

本轮只修正现有策略的容量与失败处理：

- 2048/3072 等小上下文改用可容纳的保留长度和不同的高低水位，避免固定保留 2048
  token 导致无法回收空间。
- 把当前输入 batch 和系统前缀计入预算；完整轮次边界不足以容纳当前 batch 时使用
  现有的尾部回退策略。
- 双工裁剪确认 cache 支持 shift 且删除成功后才修改 `n_past` 和轮次边界，避免删除失败后计数
  与真实 KV 分离。

普通 4096+ 上下文、可容纳的输入继续使用原有保留长度和触发策略。回退裁剪依然
可能切开一个很长的活动轮次；本轮修正不代表已经解决长期记忆或语义重要性选择。

2048 上下文另做了修正前后各 **72 段音视频**回放，均完成且音频 finite、无削波：

| 版本 | p50 / p95 ms | KV 长度下降次数 | 最大 KV 长度 | 音频时长 |
| --- | ---: | ---: | ---: | ---: |
| 原版 2048 | 210.49 / 321.12 | 9 | 1535 | 17.20 s |
| 修正版 2048 | 209.66 / 323.72 | 8 | 1532 | 18.64 s |

小上下文的新水位改变了保留历史，文本与 PCM 因而不同，没有稳定的速度收益，未设为
默认。这组 fixture 没有复现原版溢出，不能据此宣称已证明某次崩溃得到修复；容量修正
针对的是固定保留长度与实际剩余空间冲突的边界。证据在
`verification/kv-research/{small-baseline,small-selected}/`、`small-comparison.json`。

## 公开方案的适用性

以下结论来自项目 README 与公开实现，检索时也检查了 llama.cpp 的相关 issue/PR。
所列项目都不能直接作为本仓库原生 MiniCPM-o duplex 的已验证替换品。

| 方案 | 主要机制 | 对本项目的价值与限制 |
| --- | --- | --- |
| [StreamingLLM](https://github.com/mit-han-lab/streaming-llm) | 保留 attention sinks 和最近窗口 | 适合流式稳定性，当前保留前缀已具有类似目的；被丢弃的早期事实仍会遗忘，不能宣称无限上下文记忆。 |
| [SnapKV](https://github.com/FasterDecoding/SnapKV) | 在 prompt 尾部 observation window 统计逐 head 注意力，池化并保留重要位置 | 可用于完整轮次结束后的历史筛选；原实现主要在 prefill 压缩，不能保证下一轮未知问题所需历史仍被保留。每秒重新选取也有成本。 |
| [PyramidKV / KVCache-Factory](https://github.com/Zefan-Cai/PyramidKV) | 按层分配不同预算 | 比所有层相同窗口更有研究价值；需要适配 Qwen3 的逐层容量、mask 和 kernel，不能直接借用 Llama 的预算或收益。 |
| [H2O](https://github.com/FMInference/H2O) | 历史累计 attention heavy hitters + 最近 token | 比纯按时间删除更能保留重要位置；当前 Flash Attention 不输出完整概率矩阵，直接物化它会抵消优化，需要内核内累计或低频采样。 |
| [DuoAttention](https://github.com/mit-han-lab/duo-attention) | retrieval heads 保留长历史，streaming heads 仅保留 sinks 和短窗口 | 有望比逐 token top-k 更适合持续对话；需要用 MiniCPM-o 的双语音频/视频与控制 token 校准 head 类型，并实现不同 head 的 cache 布局。公开模式主要是 Llama/Mistral。 |
| [KIVI](https://github.com/jy-yuan/KIVI) | K 按 channel、V 按 token 低比特量化，保留近期高精度 residual | 适合“旧历史压缩 + 近期高精度”的设计；不是把 ggml 的 `q4_0` 或 `q8_0` 开关改名。需新存储、packing、attention 和 RoPE 路径。 |
| [KVQuant](https://github.com/SqueezeAILab/KVQuant) | pre-RoPE K 量化、非均匀码本、离群值处理与 sink 保护 | 有助于避免反复 shift 时对已量化 K 再量化；需要校准、元数据和融合内核，4K 小 cache 上额外成本可能大于收益。 |
| [InfLLM](https://github.com/thunlp/InfLLM) | 将远端历史存为 memory blocks，按需检索参与注意力 | 比单纯 sliding window 更能保存远期依赖；检索与 PCIe 数据传输需要计入每轮延迟，CPU offload 不能视为免费扩展。 |

注意 GQA：本模型只有 8 个 KV heads。若照搬按 query head 展开 KV 的实现，可能
放大缓存，抵消压缩收益。PyramidKV 当前项目也区分 `query_head` 和 `kv_head` 布局。
修改逐 head/逐层预算还需要调整 ggml 的统一 cell 元数据和 Flash Attention 接口。

## 下一版裁剪方案

推荐按三步推进，先控制工程成本，再考虑新的注意力布局：

1. **先统一预算和记账。** 用一个 ledger 管理 system、输入音频/图像完整 unit、生成
   回复与当前活动轮次。入队前按实际输入 token、decode 预留与上下文容量计算硬边界；
   高低水位留出间隔，一次计算整段淘汰区间，再执行一次 rm/shift。两套裁剪器不应
   各自修改同一份位置而不知道对方的记账。
2. **给长期事实独立的容量。** 保留系统指令、sinks 和近期完整 unit；已结束轮次中的
   名称、约束、数字与未完成任务可做有来源的文本记忆，在适当轮次边界重新 prefill。
   必须保留否定、时间和说话者归属，支持纠正旧事实。它属于有损摘要/检索，不能称为
   无损 KV 压缩，也不能把旧文本 KV 无条件复制到新的多模态位置。
3. **只有长历史确实成为瓶颈，再做 head 与精度分层。** 优先研究 DuoAttention 式
   head 分组与 KIVI/KVQuant 式旧历史压缩的组合；先测近期 F16、旧 V Q8，随后才评估
   K 的低比特化。对 retrieval heads 保留更长历史，对 streaming heads 使用短窗口。
   本机 4K 的实测并不支持直接启用这一复杂方案。

物理压紧可另做低频 CUDA gather/copy 实验，但必须计入搬移开销、图重建和数值变化。
逐 token 回传 CPU 排序、每帧完整 cache save/load、每秒重算全部注意力分数，都不适合
本机的低延迟目标。新的分页布局也不会自动带来单会话加速；当前并非多租户服务。

对于当前 4K 单会话，LLM 单 token 生成还要读取约 4.67 GB 的 decoder 权重，KV 不是
唯一主因。更大的速度空间仍在用原始 BF16/F16 权重、双语多模态校准数据生成更合适的
低比特权重，以及 profiler 确认的 CUDA 小算子融合与同步优化。当前本地只有 Q4_K_M
LLM 文件，继续从 Q4 重量化不能当作经过原始权重校准的新方案。3060 无原生 FP8/FP4
Tensor Cores；较新 GPU 或另一种 INT4 packing 的宣传数据不能直接套用。

## 复现

先停止当前托管服务，所有 GPU 测试顺序执行，完成后恢复服务。模型、日志、音频和
报告都保留在被 Git 忽略的 `verification/` 中。

```powershell
./scripts/windows/stop.ps1
# VS developer PowerShell，或按 Windows 指南提供 -VsDevCmd / -WindowsSdkRoot：
./scripts/windows/build.ps1 -CudaRoot $env:CUDA_PATH -AllKvQuants
$env:PATH = "$env:CUDA_PATH/bin;$env:PATH"
.venv/Scripts/python scripts/benchmarks/kv-cache.py `
  --model models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf `
  --mixed --output verification/kv-formats
.venv/Scripts/python scripts/benchmarks/pipeline.py `
  --model-dir models/MiniCPM-o-4_5-gguf --count 120 --rounds 1 `
  --tts-temperature 0.8 --output verification/kv-control
.venv/Scripts/python scripts/benchmarks/pipeline.py `
  --model-dir models/MiniCPM-o-4_5-gguf --count 120 --rounds 1 `
  --tts-temperature 0.8 --env OMNI_LLM_V_TYPE=q8_0 --output verification/kv-vq8
.venv/Scripts/python scripts/benchmarks/compare.py `
  verification/kv-control verification/kv-vq8 --output verification/kv-comparison.json
./scripts/windows/start.ps1
```

`-AllKvQuants` 对应已有的 `GGML_CUDA_FA_ALL_QUANTS=ON`，使 CUDA 支持混合 K/V
类型。普通构建不带此参数时为 OFF，不能假定混合组合仍走 Flash Attention。
原生 benchmark 是 decoder 算子测试，不能代替语音测试；pipeline 的计时始于完整
输入发出，不含麦克风收集、真人停顿判断、浏览器播放缓冲和网络抖动。

`scripts/benchmarks/kv-fragmentation.cpp` 是基于公开 llama API 的物理跨度探针。在 VS
developer PowerShell 中可编译为本地产物，运行前需从父进程设置诊断环境：

```powershell
cl /nologo /O2 /std:c++17 /EHsc /Iengine/include /Iengine/ggml/include `
  scripts/benchmarks/kv-fragmentation.cpp `
  /Feverification/kv-fragmentation.exe /Foverification/kv-fragmentation.obj `
  /link engine/build-cuda/src/llama.lib engine/build-cuda/ggml/src/ggml.lib `
  engine/build-cuda/ggml/src/ggml-base.lib
$env:PATH = "$PWD/engine/build-cuda/bin;$env:CUDA_PATH/bin;$env:PATH"
$env:LLAMA_KV_CACHE_DEBUG = '1'
verification/kv-fragmentation.exe models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf
Remove-Item Env:LLAMA_KV_CACHE_DEBUG
```

探针对比中间裁剪与公开 state API 重新装载后的缓存，按交替顺序重复六次，并报告
物理跨度、save/load 费用和下一 token 的 logits 差异。这个 save/load 参考路径不用于
实际服务，也不构成语音质量验收。

本轮 CUDA 构建、上述硬件测试、25 项 native launch/chat/duplex Python 测试、仓库
卫生检查和 `git diff --check` 通过。汇总为
`verification/kv-research/results-summary.json`。NVIDIA 默认配置保持 4096 + F16 KV；
服务已在测试结束后恢复。Apple M5 profile 与 Metal/ANE 专用路径未修改，未做新的 M5
硬件测量。未提交权重、音频、构建产物或本地配置。

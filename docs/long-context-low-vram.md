# 低显存设备的高性能长上下文调研

日期：2026-11-15。范围：当前 duplex.cpp / MiniCPM-o 4.5，重点为 RTX 3060
12 GB、i7-10700F、32 GB RAM、Windows/WDDM；保留 Apple M5 路径。
本文核对本地代码、GGUF 元数据、既有本机记录和公开论文/实现，未执行新的 GPU
性能或语音质量测试。容量计算、论文结果与本机实测分别标明。

## 结论

可以支持，但应按需求选择方案：

- **8K–16K 的完整近期历史**：先评估现有 Flash Attention + GPU 常驻 KV，必要时
  使用 Q8。这是当前改动最少的路径；16K 的完整语音链路表现尚未验证。
- **长时间持续对话、记住较早事实**：近期多模态 KV 常驻 GPU，已结束轮次形成可
  检索的文本记忆，在需要时重新 prefill。适合作为第一版产品方案，属于有损长期
  记忆，不保证保留原始声音细节、全部视觉信息或任意逐字回忆。
- **要求早期原始 token 仍可参与注意力**：研究 GPU 热缓存 + RAM 冷 KV + 查询相关
  块检索/预取。冷历史不立即删除，每步只访问部分块；这能同时降低显存和访存量，
  但属于近似稀疏注意力，需要改造原生引擎并验证召回和语音质量。
- **要求每步都精确关注全部原始历史**：CPU/GPU 分区 attention 或逐层 KV offload
  能扩容量，但需要扫描全部 KV；在这台 PCIe 3.0 主机上不能承诺维持短上下文延迟。
  更大显存、更高主存带宽或多 GPU 是更直接的硬件路径。

推荐顺序是 **先测 8K/16K 常驻配置，再做近期窗口 + 文本长期记忆；确有原始历史
检索需求时，才投入分层 KV 系统**。32K 原始历史可作为后续实验目标，不能预先
承诺其速度、完整召回或音质。

## 当前项目的真实边界

本地 GGUF 与上游 [config.json][model-config] 一致：Qwen3，36 层、32 个 query
heads、8 个 KV heads、head dimension 128，原生位置长度 **40,960 token**，
`rope_scaling=null`。所以 128K/1M 的容量估算不等于模型已经具备相应上下文能力；
扩展位置编码和验证多模态质量是另外的工作。

当前 NVIDIA profile 为单 Worker、4096 上下文、F16 KV、CUDA Flash Attention。
`runtime/core/processors/cpp_backend.py` 只接受 `f16` / `q8_0`，并分别传递 K/V
类型。底层 CLI 和 CUDA attention 有 `q4_0` 等路径，但不能直接把环境变量改成
Q4 就视为现有产品配置已支持，更不等于实现了 KIVI/KVQuant。

`engine/src/llama-kv-cache.cpp` 按配置容量预分配张量，GPU/CPU 放置按现有层设备
和 offload 设置选择。`seq_rm` 回收槽位不会缩小已分配的显存；当前 Qwen3 attention
路径没有“冷页在 RAM、查询时按需取回”的管理器。

`engine/tools/omni/omni.cpp` 的双工滑窗保留系统前缀和近期历史，删除较旧部分。
它使会话能持续运行，但删除后的原始 KV 不再可检索。逻辑历史长度、实际扫描的
物理槽位跨度、预分配显存容量也不是同一个量，详见[已有 KV 研究](kv-cache-rtx3060.md)。

## 显存预算：先算整个语音服务

单会话 LLM KV 张量容量为：

`2 × 层数 × token 数 × KV heads × head dimension × 每元素字节数`

按本模型尺寸计算，F16 为 **144 KiB/token**。Q8_0 每 32 个数有 32 字节数据和
2 字节 scale；Q4_0 则为 16 + 2 字节，不能只按理想位宽计算。

| 历史容量 | F16 K+V | Q8_0 K+V | Q4_0 K+V，仅容量计算 |
| --- | ---: | ---: | ---: |
| 4,096 | 576 MiB | 306 MiB | 162 MiB |
| 8,192 | 1,152 MiB | 612 MiB | 324 MiB |
| 16,384 | 2,304 MiB | 1,224 MiB | 648 MiB |
| 32,768 | 4,608 MiB | 2,448 MiB | 1,296 MiB |
| 40,960 | 5,760 MiB | 3,060 MiB | 1,620 MiB |
| 131,072，超出当前原生位置范围 | 18,432 MiB | 9,792 MiB | 5,184 MiB |

表中不含权重、TTS/音频 cache、索引、临时 workspace、图缓存和桌面占用。
TTS 的 4096-token F16 KV 另约 **240 MiB**，无需随 LLM 历史一同扩大。
文件张量清单也不能直接当作 GPU 常驻占用相加。

既有 4K 音视频回放曾测得整卡峰值 **10,380 MiB**；单纯用这次峰值替换 LLM KV
做粗算，16K F16 约为 12,108 MiB，32K Q8 约为 12,252 MiB，均缺少可靠余量。
这些是预算提示，**不是对应配置的实测峰值**：workspace、视觉首次加载、图缓存和
桌面负载会变化。纯语音已有 9,291 MiB（约 9.07 GiB）整卡峰值记录，按需视觉加载已经在现有
profile 中，不能把它重复计作新优化收益。

量化也不保证加速。既有相同模型 decoder 测试，在历史深度 3072 token 时：
F16/F16 为 **57.83 token/s**，Q8/Q8 为 **54.29 token/s**，后者约慢 **6.1%**。
只量化 V 的完整音视频回放还出现了文本/听说决策变化。因此 Q8 是容量选项，
它在 16K/32K 能否抵消更大的访存成本，需要重新测量。

## 为什么“放到内存里”仍可能很慢

若 GPU 每生成一个 token 都从 RAM 重新读入完整 32K F16 KV，需要传输约
**4.5 GiB/token**。这台主机此前负载测试为 PCIe Gen 3 ×16；按该链路单向编码后
理论上限约 15.75 GB/s 计算，仅传输就至少约 **0.31 秒/token**，尚未计协议损耗、
层间同步和计算。分层搬运降低峰值显存，却不会消除这些总字节数。

缩成 2K F16 的被选中 KV，所有层合计也有 **288 MiB**；若每步全部未命中，
按同一理论带宽仍约需 **19 ms/token**。所以分层系统必须实测传输带宽，并提高
热块命中率、减少真正缺页量、合并拷贝、尝试异步预取；只写“支持 CPU offload”
不足以说明低延迟。上述数字是传输下界估算，不是完整模型 token/s。

另一条路是在 CPU 上计算冷历史 attention，只传 query 和分区结果。各分区若
保留 softmax 的归一化统计量，再正确合并，可以保持全量 attention 的数学语义
（仍可能有浮点舍入差异）；它避免搬运全部 KV，却仍须扫描主存中的历史，并受
CPU 带宽、计算和逐层同步限制。它与丢弃/采样历史是不同的取舍。

SSD 适合保存会话或更冷的资料；随机取 KV 进入逐 token 关键路径会引入 I/O 延迟，
不宜作为本机实时双工的首选热数据层。Windows shared GPU memory、UVM 或 mmap
也不会消除物理带宽差距，不能把自动换页视为性能保证。

## 公开方案比较

以下来源均为论文或作者公开实现。论文速度来自其各自模型、硬件和工作负载，
不能直接作为 RTX 3060 / Qwen3 / 原生双工的收益。公开适配列表未证明支持本项目。

| 方案 | 对低显存长历史的作用 | 对当前项目的判断 |
| --- | --- | --- |
| GPU 常驻 Q8 / 低比特 KV | 保留所有位置，减少容量和读取字节；数值有损 | Q8 已有入口，先测更长历史。全 cache Q4 与“近期 F16、旧历史低比特”应分别评估，后者需要新布局。 |
| [KIVI][kivi] / [KVQuant][kvquant] | K/V 使用不同量化策略；处理 channel 离群值、近期 residual 或 pre-RoPE K | 比普通 Q4 开关更复杂；需要校准/格式、融合 attention，以及 RoPE/shift 路径适配。适合作为第二阶段压缩研究。 |
| [Quest][quest] | 用 query 和每页 K 的 min/max 估计重要页，只读 top-k 页 | 适合借鉴页选择内核。原始“少读 GPU KV”不等于“完整 KV 已移出 GPU”，解决容量还需冷页存储与预取。 |
| [InfLLM][infllm] | 远端上下文放入 memory blocks，检索相关块；公开实现含 CPU 存储和 GPU 块缓存 | 与持续流输入比较接近。原实现有 CPU 索引/同步路径，需量化检索和传输开销，不能直接移植 Python 层。 |
| [InfiniGen][infinigen] | 用下一层的小规模 rehearsal 预测重要 KV，提前从主存取回 | 重点解决 offload 的取数开销；值得借鉴预取机制，但要适配 Qwen3 的 Q/K norm、GQA 和当前调度器。 |
| [ShadowKV][shadowkv] | 保存低秩 K、将 V offload，再重建/取回少量 KV | 容量和吞吐方向有价值；公开效率实验为 A100，支持列表含 Qwen2，不能推定 Qwen3 双工可用。持续追加历史时还要考虑低秩表示更新成本。 |
| [MagicPIG][magicpig] | CPU 保存 LSH 表并计算采样 attention，降低 GPU 容量与取数要求 | 展示了消费 GPU + CPU 的可行性，但示例设置 96 个 OMP 线程；本机 8 核 CPU 不能套用其结果。采样仍是近似。 |
| [DuoAttention][duoattention] | retrieval heads 保留长历史，streaming heads 只保留短窗口 | 同时减少容量和访存，但须重新识别本模型的 head 类型；只有 8 个 KV heads，不能照搬 MHA 的压缩比例或展开 GQA。 |
| [FlexGen / FlexLLMGen][flexgen] | GPU/RAM/SSD 分层与压缩，大 batch 摊薄 offload 开销 | 作者明确以吞吐为目标，小 batch 可明显变慢。适合离线批处理，优先级低于单会话实时路径。 |
| [LMCache][lmcache] | 分层持久化、跨请求复用 KV，减少重复 prefill 和 TTFT | 有助于重复前缀与会话恢复；仅接入存储层不代表活动会话每步可以高效关注 RAM 中全部 KV。 |

MagicPIG 论文给出的一个例子是 RTX 4090、Llama-3.1-8B、96K context 下
**54 ms/token**，说明该方向有实际系统证据；这不是当前 3060 的预测值，也不是
语音端到端延迟。ShadowKV 的最高约 3.04× 是 A100 吞吐结果，同样不能换算为
本项目单会话首包收益。

Flash Attention 能避免显式保存巨大的 attention 中间矩阵，但不会自动压缩
持久 KV，完整 prefill 的 attention 计算量也仍随长度平方增长。Chunked prefill
控制工作区和调度，不减少最终完整 KV。PagedAttention 主要解决布局、碎片、共享
和批处理，也不会自动把唯一历史的有效 KV 字节数压小。

## 适合 duplex.cpp 的接入路线

第一步测现成配置：8K F16 作为完整链路对照，再测 8K/16K Q8；单独固定
`OMNI_TTS_CTX_SIZE=4096`。先确认真实历史深度、Flash Attention 实际启用和显存余量，
不要根据已分配长度推断模型确实访问过等长历史。当前双工滑窗会提前触发，测试应
记录触发前后的 KV 长度；32K 深度实验需要另行安排窗口预算。

第二步做对话记忆：系统指令/音色参考 + 近期完整音频/图像 unit 留在 GPU，已结束
轮次保存带说话者、时间、来源和纠错关系的文本记忆。检索后在合适的轮次边界加入
有限 token 的记忆并重新 prefill。完整转录需要另行确认来源：现有双工回复文本
不能当作用户语音的完整转录；如果需要补充 ASR，应计入延迟和资源预算。

第三步才实现分层原始 KV。建议先在原生 LLM decoder 上建立可测的最小方案：

1. 分离逻辑位置和物理槽位；GPU 热池放 system/sinks、近期完整 unit 和命中冷块，
   RAM 保存仍在预算内的旧 KV。暂以 2K–4K 热 token 做预算实验，不固定为最终参数。
2. 以块为单位检索和缓存；比较 GPU 页统计量选择与 CPU 索引，选取真正适合本机
   的路径。增加命中率、缺页字节数和逐层等待计时，再考虑跨层预测与异步预取。
3. 保持 GQA 共享；选择预算要考虑各 query head 选中集合的并集，避免把 8 个
   KV heads 展开成 32 个导致缓存反而变大。kernel 必须真正跳过未选块，仅加 mask
   不代表访存量下降。
4. 缓存热/冷块的 RoPE 位置、mask 和 unit 边界必须一致；不能沿用删除旧历史后的
   位移计数，又把原位置的冷 KV 无条件加回。块迁移、重排和 CUDA Graph 重捕获的
   成本都应计入。纯语音与音视频分别验收，保持现有 M5 配置与路径。

若需要完全保留旧 token 的全量 attention，则另做 CPU/GPU 分区合并实验，与稀疏
方案分开报告。它适合质量对照和容量兜底，性能是否足够由实测决定。

## 验收条件

应覆盖 4K / 8K / 16K / 32K 的真实历史深度，并把静态长 prompt 的 prefill/TTFT
和持续双工的 decode、首 PCM、播放欠载分开测量。设备资源还应记录专用/共享显存、
RAM/锁页量、主机传输字节数、热缓存命中率和输入队列年龄。

质量至少覆盖中文/英文、较早名称与数字、多处事实联合推理、否定与纠错、跨轮任务、
音色/情绪依赖、画面信息和打断后恢复。单次 Needle/passkey 不能代表完整长上下文
能力；token/s、有限且无削波的 PCM 也不能代替人工语音质量验收。

本轮材料保存在被 Git 忽略的 `verification/long-context-research/sources/`。
既有硬件测量及其复现方法见 [KV cache 研究](kv-cache-rtx3060.md)、
[语音优化](voice-optimization-rtx3060.md)和[全链路优化](inference-optimization.md)。

[model-config]: https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/main/config.json
[kivi]: https://github.com/jy-yuan/KIVI
[kvquant]: https://github.com/SqueezeAILab/KVQuant
[quest]: https://github.com/mit-han-lab/Quest
[infllm]: https://github.com/thunlp/InfLLM
[infinigen]: https://github.com/snu-comparch/InfiniGen
[shadowkv]: https://arxiv.org/abs/2410.21465
[magicpig]: https://arxiv.org/abs/2410.16179
[duoattention]: https://github.com/mit-han-lab/duo-attention
[flexgen]: https://github.com/FMInference/FlexGen
[lmcache]: https://github.com/LMCache/LMCache

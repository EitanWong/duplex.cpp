# Strata 核心技术与 duplex.cpp 的借鉴路线

调研日期：2026-11-15。Strata 源码固定在
[`13eedf552417a6a791a7d13c6597e24300ee3a09`][strata-tree]，提交日期为
2026-11-12；其文档标注 app/Terminal 0.2.13、engine 0.1.59。
当前项目对照包含 `dev` 工作区尚未提交的优化。本文依据源码、作者文档和本项目已有
测量；本轮未编译 Strata、下载模型或执行新的硬件基准，也未修改推理路径与默认配置。
下文的 Strata 性能数字均为作者报告，本机数字引用既有报告。

**结论：值得借鉴，近期优先研究 SM 8.6 的访存与融合方法、设备端状态管理和可验证的
内存预算。GPU/RAM 分层 KV 有长期价值，但需要先为 MiniCPM-o 设计自己的选择器。
直接更换推理引擎、搬运 MoE 专家缓存、套用其速度数字，适用性很低。**

## 1. 先区分模型能力与运行时优化

Strata 当前公开模型列表有 DeepSeek V4.1-Flash、Qwen3.8-Flash-Next 和
Qwen3.6-35B-A3B。以下机制主要分析其 Flash-Next 路径；DeepSeek 有独立引擎，
不能把这一条路径的结构与数字推广到所有受支持模型，详见[支持列表][strata-reference]。

| 项目 | Strata Flash-Next 路径 | 当前 duplex.cpp |
| --- | --- | --- |
| 主干 | 48 层 MoE；每层 512 个路由专家、每 token 选 10 个，另有共享专家 | MiniCPM-o 4.5 的 Qwen3 8B 稠密主干，36 层 |
| 注意力 | 36 层 Gated DeltaNet，12 层带索引器的 QSA | 各层常规 GQA、自注意力和 RoPE |
| KV 几何 | QSA 为 2 KV heads × 256；索引器预算 2048 cells 加未满块尾部 | 8 KV heads × 128；默认 LLM 4096 上下文、F16 KV |
| 主要资源问题 | 大量专家无法全部常驻 VRAM；CPU、PCIe 和 GPU 协作 | 已量化稠密主干常驻 GPU；还要运行音频编码、TTS、flow、vocoder 和可选视觉 |
| 输出约束 | 文本 token、推测验证和会话恢复 | 文本、听说控制 token、对应 hidden states、声学状态与打断 epoch |

结构依据为 Strata 的 [`geometry.hpp`][strata-geometry]、[`qsa.hpp`][strata-qsa]，
以及本项目 [`qwen3.cpp`](../engine/src/models/qwen3.cpp)、
[`omni.h`](../engine/tools/omni/omni.h) 和[现有容量分析](inference-optimization.md)。
Strata 的较小 KV、稀疏注意力和 MTP 能力都部分来自目标模型本身；运行时无法给现有
MiniCPM-o 权重直接增加相同结构。

## 2. 核心技术及可迁移程度

### 2.1 专家缓存与 CPU/GPU/PCIe 重叠：理解价值高，直接迁移价值低

Strata 把常用专家放入 GPU 缓存，其余专家保存在主存。GPU 计算命中部分，CPU
计算部分 miss，也可把部分 miss 权重经 PCIe 取到 GPU。初始专家排序来自路由
profile，运行中继续调整；拆分比例参考实测 CPU 专家速度和链路带宽。
[`expert_cache.hpp`][strata-experts] 定义槽位与 residency table，
[`pcie_share_rule.hpp`][strata-pcie] 明确写出了分配规则和硬件条件。

它能受益于“每步只需要少数专家”。当前稠密 Qwen3 每 token 都需要使用大部分
decoder 矩阵，没有同样的路由稀疏性；把这些矩阵搬到 RAM 会引入持续 PCIe 读取。
当前配置也已把 LLM/TTS decoder 尽量放在 GPU。

可借鉴的是用实测成本决定任务放置，并把计算、传输与同步一起计时。CPU 还承担
AVX2 声学 head、预处理和服务调度，不能用一个独立 MoE 测试的 CPU 分配比例决定
整条语音链路的线程数。Apple 统一内存路径也不应套用离散显卡的 PCIe 规则。

### 2.2 小矩阵访存与融合：近期最值得检查的性能方向

源码中有三个具体可参考的实现：

- [`native_mmvq.cu`][strata-mmvq] 的 B6 路径让激活片段跨多个输出行复用，保持
  每个输出的累加和归约顺序；覆盖范围包含 Q4_K、Q5_K、Q6_K 等指定格式。
- [`iq_kernels.cu`][strata-gu] 把 gate/up、SwiGLU 和下一阶段需要的 q8_1 打包合并，
  减少中间张量读写及 launch。该实现针对其专家布局，需重新适配稠密 FFN。
- [`dequant_bf16.cu`][strata-dequant] 的 wide dequant 调整相邻线程处理的数据和
  成组读写方式；保持单个数值的解码表达式。它作用于所选 prefill 路径的反量化，
  不能推断逐 token MMVQ 都有同样成本。

Strata 的[历史记录][strata-details]报告：RTX 3060、IQ3_XXS、engine 0.1.45.3，
整套架构默认优化使 decode 从 60.36 到 66.52 token/s，约 +10.2%。该组合包括
专家调度、QSA/GDN 等本项目没有的路径，**不能作为本项目的收益预测**。同一文档
中，单独 B6 MMVQ 在其 3060 上只有约 0.2–1% 的收益；组合收益不能简单相加。

本项目已有以下基础，研究应从实际命中率开始：

- [`ggml-cuda.cu`](../engine/ggml/src/ggml-cuda/ggml-cuda.cu) 已识别
  `MUL_MAT + MUL_MAT + GLU` 融合，也已有 RMSNorm、RoPE 写 cache 等融合入口及
  CUDA Graph 捕获/重放。
- [`mmvq.cu`](../engine/ggml/src/ggml-cuda/mmvq.cu) 已有多行/多列计算和
  gate/up epilogue；存在类似实现不代表当前形状与张量类型一定进入该分支。
- [`tts-cpu-head.cpp`](../engine/tools/omni/tts-cpu-head.cpp) 已复用 hidden
  broadcast，并交错四组 AVX2 累加器；[现有实测](voice-optimization-rtx3060.md)
  已记录约 1.65–1.74× 算子加速，不能再次计为 Strata 带来的新收益。

建议先记录单 token、短 prefill、TTS decode 各自选择的 kernel、融合失败原因、
反量化时间、寄存器压力与访存效率，再选一个缺口实验。当前 Q4_K_M 稠密 decoder
每 token 约读取 4.67 GB 权重，隔离测量约 63 token/s；理想带宽估算约 77 token/s，
说明保持相同权重字节数的小改动有明确空间限制。该估算不计并行语音任务和实际
缓存行为，不能视为完整系统的硬上限。

### 2.3 GPU 驱动的 KV 分页：长期最有参考价值的系统实现

[`kv_stream.hpp`][strata-kv-stream] 和 [`kv_stream.cu`][strata-kv-stream-cu]
给出了较完整的原生实现：

1. 完整、权威的 KV 保存在 pinned、device-mapped 主存；GPU 保留有限槽位。
2. 模型索引器先选择本次 query 需要的 cells；page table 映射逻辑块到 GPU 槽位。
3. GPU resolve kernel 合并 miss，使用 CLOCK/second-chance 选择槽位，保护本次
   调用正在使用的块；copy kernel 经 PCIe 读取主存中的缺页。
4. append 始终更新主存副本，命中时同时更新 GPU，避免两份副本内容失配。
5. page map、miss list 和计数器保持固定地址；解析及复制可进入 CUDA Graph，
   无需逐 token 把选择列表回传 CPU 后再发起每个请求。

这一机制保存完整 KV，但每步只为模型本来选择的部分建立 residency。MiniCPM-o
当前没有其 `indexer.q_proj/k_proj` 权重，也没有同样的稀疏 attention 语义。
如果保持现有全量 attention，每步需要的唯一页集合仍接近全部历史；CLOCK 本身
不会减少这些必要字节。引入 Quest/InfLLM 式选择会成为新的近似推理方案。

容量差距也很大：当前 36 层 LLM 的 F16 KV 为 144 KiB/token，4K 约 576 MiB，
16K 约 2.25 GiB，仅计算 K/V payload。若每 token 把 16K 的完整 KV 都经实测
Gen 3 ×16 链路搬一遍，按约 15.75 GB/s 的理论单向带宽也需约 153 ms，尚未包含
协议开销、计算和逐层同步。这是带宽估算，并非本轮实测。

接入应先建立与模型匹配的页选择实验，再借鉴其 GPU resolve、固定地址和一致性
机制，并保留本项目的 GQA 共享、多模态 unit 边界、逻辑 RoPE 位置、滑窗和打断
状态。该路线补充[既有长上下文方案](long-context-low-vram.md)，优先级低于当前
4K 单会话中的实测瓶颈。

### 2.4 Hadamard 低比特 KV：值得独立实验，不能等同现有 Q4 开关

[`kv_q4.hpp`][strata-kv-q4] 对 QSA 的 K/V 做正交 Hadamard 旋转，再保存 q4_0；
query 同样旋转，attention 输出对 V 的旋转做逆变换。旋转在精确算术中保留点积，
有助于摊平离群值；4-bit 舍入仍有损。代码还支持 K8V4 混合布局。

当前模型每 head 为 128 维，Strata 这条实现专门处理 256 维；需要新的变换、
packing 与融合 attention。尤其不能忽略 RoPE/滑窗：Hadamard 与 RoPE 一般不交换，
不能对旋转域中的 K 直接套用现有原坐标域的 shift。需明确变换位置和位置更新方法。

本机[已有 KV 对照](kv-cache-rtx3060.md)显示，在 3072 历史 token 下，F16/F16
为 57.83 token/s，双 Q8 为 54.29；仅 V Q8 为 54.84。短上下文的反量化成本可能
超过节省带宽的收益。故该方向应以 8K/16K 容量与质量为目标，保持 TTS F16、
近期高精度，并与现有 Q8 基线比较；没有依据直接替换默认 F16。

### 2.5 推测解码：借鉴选择策略，接入需要完整语音状态契约

Strata 同时使用模型 MTP、prompt/suffix lookup，并在
[`draft_policy.hpp`][strata-draft] 中比较“预计提交 token 数 / 实测窗口耗时”。
它记录接受率和不同窗口成本；`draft_policy.cpp` 还避免用首次 graph capture 的
成本代表稳态验证成本。这种策略比只追求更长 draft 更适合作为参考。

当前 MiniCPM-o 不能直接使用另一模型的 MTP。接入独立 drafter 需额外权重、KV
和验证工作区，必须返回每个接受 token 对齐的目标模型 hidden state，才能继续
TTS。还要处理 LISTEN/SPEAK、拒绝后的 KV 回滚、采样随机状态、音频 epoch 和
短语音片段的收益。草稿验证期间也不能延迟处理新的输入和打断。

Strata 最新提交修复的 [`serve_stop_bound.hpp`][strata-stop] 很值得借鉴：
commit 数必须受剩余输出预算与 EOS 约束，不能把调用者没有看到的额外验证行
留在后续会话状态。对本项目还需把同样边界扩展到 hidden-state 与声学队列。

作者文档明确区分 greedy 和 sampled drafting，后者没有稳定提速结论；部分数值
路径也会让 greedy 结果随窗口或缓存发生变化。不能用“目标模型验证过”替代
采样分布、状态恢复和整条语音链路的一致性验证。

### 2.6 内存生命周期、固定地址和运行记录：近期工程收益明确

Strata 的 [`run_manifest.hpp`][strata-manifest] 记录实际引擎、预算、加载后格式、
fast path 是否选中及拒绝原因。它比只打印请求的配置更能解释“开关打开但没有
生效”，也便于比较不同硬件。建议本项目建立相同用途的可选 JSON 记录：LLM/TTS
KV 实际类型、Flash Attention/Graph 命中、融合分支、CPU head、视觉 residency、
工作区实际尺寸和回退原因。

[`qsa.hpp`][strata-qsa] 还把 position、KV 长度、选择宽度集中在设备端 step buffer，
由一个入口更新，避免新增自定义图捕获时把每步变化的 host scalar 固定在首步。
本项目已有 ggml 图更新机制；这一点适用于后续自定义 kernel 的设计与验证，不能
据此断言现有图存在同样错误。

其 [`ExpertCache`][strata-experts] 还通过 CUDA VMM 保持虚拟地址不变，按段缩放
物理存储，保护捕获图中的指针；[弹性显存文档][strata-elastic] 说明了约束和
部分失败状态。这是可参考的地址管理方法，当前稠密主干不必引入整个专家缓存体系。

本项目已实现视觉按需加载、60 秒会话空闲释放、host embedding、缩小音频 reserve
及 flow/vocoder metadata arena。新的收益应寻找尚未重复利用的互斥工作区，
但 LLM、TTS、flow 和 vocoder 的异步阶段可能同时使用缓冲区，不能只依据单个
函数返回便进行 alias。首次视觉加载、graph 预热和声学输出并发也须计入峰值。

另一个具体检查点是 [`eval_tokens_with_hidden`](../engine/tools/omni/omni.cpp)
读取 host hidden states，再交给 projector。`llama_get_embeddings` 在
[`llama-context.cpp`](../engine/src/llama-context.cpp) 中显式同步。可以测量保持
hidden/projector 数据在设备上的收益，同时保留 CPU fallback；现有采样读取也
可能已要求同步，不能按 getter 调用次数推算可消除的等待时间。

## 3. 哪些“提速”不能当作保持结果的优化

Strata 的[默认设置说明][strata-details]包含 `STRATA_ROUTE_TAIL_SKIP`：满足条件时
跳过低排名的未缓存专家；可选 `STRATA_ROUTE_PRIOR` 让路由偏向已缓存专家。
这些操作改变参与计算的模型部分。文档还记录 route prior 在长期 agent 任务上的
质量退化，并撤回过自动启用策略。

因此应分别报告：保持同样运算/归约顺序的访存调整、只在数学上等价但改变舍入的
融合、低比特量化、稀疏选择/路由裁剪。文本 KL 或 top-1 一致率也不能单独证明
听说决策、中文发音、音色与打断恢复的质量。相关功能在当前稠密主干也缺少直接
对应结构。

NVFP4 的模型格式支持同样不代表 RTX 3060 拥有原生 FP4 Tensor Cores。
新硬件与不同模型的吞吐不能转换成此机的首 PCM 收益。

## 4. 建议的执行顺序与验收

| 顺序 | 建议工作 | 决策依据 |
| --- | --- | --- |
| 1 | 在现有回放中补实际路径记录与 Nsight 时间线 | 分离 kernel 时间、host gap、同步、输入队列和 cold vision；先确认现有融合/Graph 是否生效 |
| 2 | 从 MMVQ 激活复用、FFN epilogue、wide dequant 中挑一个实际热点 | 采用本模型真实形状、Q4_K/Q6_K 等实际类型；每次一个 opt-in 改动，保留原实现 |
| 3 | 测 hidden → projector 的同步及工作区生命周期 | 同时核查异步使用者、峰值专用/共享显存、RAM 和图重捕获；无热点则终止该方向 |
| 4 | 长历史场景评估旋转 Q4/K8V4 | 先做 8K/16K 完整历史对照，包含 shift、跨页、中文数字与纠错；现有 F16 为质量基线 |
| 5 | 有明确长期记忆需求后再做 GPU/RAM KV 页实验 | 先选页，再 resolve/copy；记录 miss 字节、命中率、逐层等待与近似质量损失 |
| 6 | 最后考虑推测解码 | 先定义目标 hidden-state、控制 token、回滚、EOS/预算和打断契约，再测额外显存及首包 |

保持现有 M5 profile 与 Metal/CoreML/Accelerate 路径；CUDA 改动只在实测更好的
硬件/形状上选择。性能实验采用交错 A/B 和相同输入历史，分别记录纯音频与音视频、
冷启动与稳态。至少覆盖实际滑窗、多会话复用、打断、恢复及持续运行。

预期保持输出的优化比较 logits、hidden states、听说决策、文本与 PCM；允许数值
变化的方案另做双语理解、早期事实/纠错、发音、音色、自然度和真人试听。共同指标
包括首 PCM、输入 p50/p95、队列年龄、播放欠载、打断后残留音频与资源峰值；
不能只用 token/s 或音频 finite/无削波作为通过条件。运行记录、模型和音频继续放
在忽略目录，不提交到仓库。

Strata 主体为 MIT，移植实际代码需保留其作者、版权和许可证；ggml 及模型衍生
数据另有来源声明。本文只提出借鉴路线，未引入其代码或模型资产。

[strata-tree]: https://github.com/niko1221/strata/tree/13eedf552417a6a791a7d13c6597e24300ee3a09
[strata-reference]: https://github.com/niko1221/strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/docs/AI_REFERENCE.md
[strata-details]: https://github.com/niko1221/strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/docs/DETAILS.md
[strata-geometry]: https://github.com/niko1221/strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/include/strata/core/geometry.hpp
[strata-qsa]: https://github.com/niko1221/strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/include/strata/kernels/qsa.hpp
[strata-experts]: https://github.com/niko1221/strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/include/strata/core/expert_cache.hpp
[strata-pcie]: https://github.com/niko1221/strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/include/strata/core/pcie_share_rule.hpp
[strata-mmvq]: https://github.com/niko1221/strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/src/kernels/cuda/native_mmvq.cu#L1383
[strata-gu]: https://github.com/niko1221/strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/src/kernels/cuda/iq_kernels.cu#L2872
[strata-dequant]: https://github.com/niko1221/strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/src/kernels/cuda/dequant_bf16.cu#L196
[strata-kv-stream]: https://github.com/niko1221/strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/include/strata/kernels/kv_stream.hpp
[strata-kv-stream-cu]: https://github.com/niko1221/strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/src/kernels/cuda/kv_stream.cu#L89
[strata-kv-q4]: https://github.com/niko1221/strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/include/strata/kernels/kv_q4.hpp
[strata-draft]: https://github.com/niko1221/strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/include/strata/spec/draft_policy.hpp
[strata-stop]: https://github.com/niko1221/strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/include/strata/core/serve_stop_bound.hpp
[strata-manifest]: https://github.com/niko1221/strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/include/strata/core/run_manifest.hpp
[strata-elastic]: https://github.com/niko1221/strata/blob/13eedf552417a6a791a7d13c6597e24300ee3a09/docs/VRAM_ELASTIC.md

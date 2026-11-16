# Documentation / 文档索引

| Document | 内容 |
| --- | --- |
| [Project layout](project-layout.md) | 目录规范、平台脚本与本机生成产物 |
| [Script index](../scripts/README.md) | Windows / macOS 的日常启动、停止、构建与测试入口 |
| [Model files](models.md) | 模型准备、来源及校验 |
| [Windows / NVIDIA](nvidia.md) | Windows CUDA 构建、HTTPS 与运行配置 |
| [Apple Silicon](apple-silicon.md) | 保留的 Apple M5 路径及验收边界 |
| [RTX 3060 measurements](performance-rtx3060.md) | 本机性能、显存和音频对照证据 |
| [Inference optimization research](inference-optimization.md) | 全链路瓶颈、低层算子、量化与剪枝优先级及新测量 |
| [Voice-path optimization](voice-optimization-rtx3060.md) | 视觉按需加载、AVX2 指令并行、语音一致性与冷启动代价 |
| [KV cache and sliding windows](kv-cache-rtx3060.md) | 本机混合精度 KV 测试、滑窗边界修正与长期记忆方案 |
| [Long context with limited VRAM](long-context-low-vram.md) | 长上下文容量预算、GPU/RAM 分层 KV、稀疏检索与接入路线 |
| [Long-session implementation](long-session-rtx3060.md) | 统一 KV 预算与观测、真实 8K/16K 历史、文本记忆实现和质量限制 |
| [Steady inference and browser playback](steady-playback-rtx3060.md) | 同步/Graph 归因、真实 WebAudio、小时压力与记忆质量 |
| [Strata technical review](strata-technical-review.md) | Strata 核心实现、模型结构差异及对本项目的借鉴优先级 |
| [API reference](api.md) | 原生全双工与 Gateway 协议 |
| [ModelBest Realtime](modelbest-realtime.md) | 面壁官方音视频事件格式、Float32 音频、认证与部署 |
| [qwen-audio-agent](qwen-audio-agent.md) | MiniCPM-o Provider 配置、WebUI 启动与实际联调 |
| [OpenAI Realtime audio](realtime.md) | GA/Beta 音频事件兼容、密钥认证、局域网与公网部署 |

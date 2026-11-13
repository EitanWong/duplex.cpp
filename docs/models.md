# Model files / 模型文件

Download the [official MiniCPM-o 4.5 GGUF bundle](https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf).
Choose the **Q4_K_M LLM** and retain the original F16 audio, vision and TTS modules.
Model weights are not included in Git.

```text
MiniCPM-o-4_5-gguf/
├── MiniCPM-o-4_5-Q4_K_M.gguf
├── audio/MiniCPM-o-4_5-audio-F16.gguf
├── vision/MiniCPM-o-4_5-vision-F16.gguf
├── tts/
│   ├── MiniCPM-o-4_5-tts-F16.gguf
│   └── MiniCPM-o-4_5-projector-F16.gguf
└── token2wav-gguf/
    ├── encoder.gguf
    ├── flow_matching.gguf
    ├── flow_extra.gguf
    ├── hifigan2.gguf
    └── prompt_cache.gguf
```

To download only the reference bundle, install the Hugging Face CLI and run:

```bash
python -m pip install huggingface_hub
hf download openbmb/MiniCPM-o-4_5-gguf --local-dir models/MiniCPM-o-4_5-gguf \
  --include 'MiniCPM-o-4_5-Q4_K_M.gguf' 'audio/*' 'vision/*' 'tts/*' 'token2wav-gguf/*'
```

The Apple profile additionally requires
`vision/coreml_minicpmo45_vit_all_f16.mlmodelc/` from the upstream CoreML assets or
conversion workflow. Keep the complete asset bundle and any accompanying metadata.

Configure absolute paths for your machine:

```bash
python scripts/configure.py --model-dir /absolute/path/to/MiniCPM-o-4_5-gguf
```

This generates the ignored `runtime/config.json`; it does not download models.
Use `--force` to deliberately replace an existing local configuration.

使用上游的完整 GGUF 模型包，主模型选择 Q4_K_M，其余模块保留原精度。
Apple 的 CoreML 路径还需要上述 `.mlmodelc` 目录。
配置脚本会生成本机绝对路径，不会自动下载模型或覆盖已有配置。

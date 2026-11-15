# 项目目录规范

正式脚本统一放在 `scripts/`，按平台与用途分类。原来的 `project-control/`
已合并到此目录，根目录不再放平台启停脚本。新入口见[脚本索引](../scripts/README.md)。

```text
duplex.cpp/
├── engine/                 C++ 推理引擎及其上游构建、测试与许可证
├── runtime/                Python Gateway、Worker、前端、运行时资源与测试
├── scripts/
│   ├── windows/            Windows 启停、状态、CUDA 构建及平台辅助工具
│   ├── macos/              macOS 启停、状态、Apple 构建及平台辅助工具
│   ├── benchmarks/         原生回放、模式切换与历史复用测速
│   ├── configure.py        共享的本机配置生成器
│   ├── configure-lan.py    局域网 HTTPS 配置
│   └── check-repository.py 仓库结构、语法、链接和发布检查
├── profiles/               可版本化的硬件配置；Apple 与 NVIDIA 分开维护
├── docs/                   安装、协议、性能证据及维护说明
├── .github/                CI 与协作模板
└── selected-profile.json   保留的 macOS 当前配置，与 Apple M5 基线一致
```

根目录只保留项目说明、协作规则、许可证、上游来源和仓库配置。
`engine/`、`runtime/` 的模块边界、资源路径及上游声明保持现状；其内部工具服务于
各自组件，项目级启停与构建以 `scripts/` 的入口为准。

## 本机数据与生成产物

| 位置 | 内容 |
| --- | --- |
| `.local/state/` | 平台服务状态、锁与停止标记，不存放源码 |
| `runtime/config.json` | 本机模型路径、端口、HTTPS 与推理开关 |
| `runtime/certs/` | 本机 TLS 证书及私钥 |
| `models/` | 下载的模型与来源校验清单 |
| `engine/build*/`、`releases/` | 编译缓存与本机可执行产物 |
| `logs/` | 下载、启动、运行与构建日志 |
| `verification/` | 本机回归报告、音频与实验辅助脚本 |
| `optimization/` | 本机 CUDA 工具链与其他优化依赖 |
| `.venv/`、运行时数据目录 | Python 环境、会话数据及缓存 |

以上本机产物均由忽略规则保护，不应加入版本控制。历史报告中的旧命令记录保留为
当时的实验依据；需要重跑时使用当前脚本索引。整理不会删除权重、录音或历史证据。

## 维护约定

- Windows 脚本放在 `scripts/windows/`，macOS 脚本放在 `scripts/macos/`。
- 共享配置与仓库检查工具放在 `scripts/` 顶层，性能测量放在 `scripts/benchmarks/`。
- 文件名使用明确用途，如 `start`、`stop`、`status`、`build`、`service`；入口以所在
  平台目录区分，避免再增加含糊的顶层控制目录。
- 移动脚本时同步调整项目根路径计算、子进程路径、文档和 CI；保留 shell 可执行权限。
- Python / shell 文件使用 LF，PowerShell / batch 文件由 `.gitattributes` 统一 CRLF。
- 修改结构后运行 `python scripts/check-repository.py` 及对应平台的启停验证。

旧部署应先停止服务再升级；控制器首次启停时将旧根目录状态文件归入 `.local/state/`。
配置、证书、模型路径和推理默认值不随目录迁移重置。

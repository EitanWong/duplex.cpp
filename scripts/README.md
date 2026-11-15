# Scripts / 脚本入口

Run these commands from the repository root. Platform launchers also work from
other working directories. Configure the model paths before the first start.
所有日常入口按平台分类；首次使用先配置模型路径，构建对应平台引擎。

| Location / 目录 | Purpose / 用途 |
| --- | --- |
| [windows/](windows/) | Windows CUDA build, background start/stop/status, foreground debugging |
| [macos/](macos/) | Apple Silicon build, start/stop/status/restart, supervision and monitoring |
| [benchmarks/](benchmarks/) | Isolated pipeline, native replay and mode/history reuse measurements |
| `configure.py` | Generate ignored `runtime/config.json` from a hardware profile |
| `configure-lan.py` | Configure LAN HTTPS and local certificates |
| `check-repository.py` | Check layout, syntax, documentation links and publication hygiene |

## Windows

Double-click `windows/start.bat`, `windows/status.bat` or `windows/stop.bat`.
可双击这三个文件；关闭启动窗口不会停止后台服务。

```powershell
./scripts/windows/start.ps1
./scripts/windows/status.ps1
./scripts/windows/stop.ps1
# Optional / 可选：
./scripts/windows/start.ps1 -OpenBrowser -TimeoutSeconds 300
```

`build.ps1` builds CUDA binaries; `run.ps1 -Role worker|gateway` runs a service
in the foreground. `service.ps1` and `detach.py` are implementation helpers.
Start waits for the model and Gateway worker registration. Repeated starts reuse
services; stop interrupts active sessions and releases the native engine's VRAM.
The HTTP/HTTPS config is retained. See the [Windows guide](../docs/nvidia.md).

## macOS

```bash
./scripts/macos/build.sh
./scripts/macos/start.sh
./scripts/macos/status.sh
./scripts/macos/stop.sh
./scripts/macos/service.sh restart
```

Finder entries `Start.command`, `Status.command` and `Stop.command` live in
`macos/`. The controller uses `.venv/bin/python`, `selected-profile.json` and the
existing M5 release/profile; GPU inference settings are preserved.
`manage.py`, `service.py` and `monitor.py` are platform helpers.
详见 [Apple Silicon 基线](../docs/apple-silicon.md)。

## Benchmarks / 性能测试

```bash
python scripts/benchmarks/native.py --help
python scripts/benchmarks/pipeline.py --help
python scripts/benchmarks/session-reuse.py --help
python scripts/benchmarks/kv-cache.py --help
python scripts/benchmarks/compare.py --help
python scripts/check-repository.py
```

`pipeline.py` owns a fresh native process, records inference settings and whole-GPU
samples, and cleans up on exit. Run A/B variants sequentially with matching input
and speech history. See [optimization evidence](../docs/inference-optimization.md)
for measured results, experimental switches and commands.
For KV formats, pruning limits and the physical-span diagnostic, see
[KV cache research](../docs/kv-cache-rtx3060.md).

There is currently no supported Linux production launcher. Linux CI validates
source and schemas; it does not establish Linux inference support.
目前未提供正式 Linux 推理启动器，不能把 Linux CI 通过等同于推理已验证。

Service state and locks are local to `.local/state/`; logs remain in `logs/`, and
benchmark evidence stays in `verification/`. These directories are ignored by Git.
Stop old services before upgrading an existing deployment. Legacy root state files
are migrated by the platform controller on its first mutating command.
项目目录职责与本机产物约定见 [目录规范](../docs/project-layout.md)。

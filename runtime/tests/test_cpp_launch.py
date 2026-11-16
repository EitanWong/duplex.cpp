"""Native executable discovery and launch settings, without model loading."""
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.processors.cpp_backend import CppBackendWorker


def worker_at(tmp_path):
    worker = CppBackendWorker.__new__(CppBackendWorker)
    worker.llamacpp_root = str(tmp_path)
    worker.native_binary_dir = ""
    worker.runtime_env = {}
    worker.model_dir = str(tmp_path / "models")
    worker.llm_model = "test.gguf"
    worker.gpu_id = 0
    worker.ctx_size = 8192
    worker.n_gpu_layers = 99
    worker._cpp_server_port = 19080
    worker._cpp_server_url = "http://127.0.0.1:19080"
    return worker


@pytest.mark.parametrize("directory", ["build/bin", "build/bin/Release", "build-cuda/bin"])
def test_windows_finds_omni_executable(tmp_path, monkeypatch, directory):
    monkeypatch.delenv("OMNI_NATIVE_RELEASE_DIR", raising=False)
    monkeypatch.setattr("platform.system", lambda: "Windows")
    executable = tmp_path / directory / "llama-omni-server.exe"
    executable.parent.mkdir(parents=True)
    executable.touch()
    assert Path(worker_at(tmp_path)._find_server_binary()) == executable


def test_explicit_binary_directory_does_not_fall_back(tmp_path, monkeypatch):
    monkeypatch.delenv("OMNI_NATIVE_RELEASE_DIR", raising=False)
    worker = worker_at(tmp_path)
    worker.native_binary_dir = str(tmp_path / "missing")
    with pytest.raises(RuntimeError, match="Selected native release"):
        worker._find_server_binary()


@pytest.mark.parametrize("kv_settings,expected", [
    ({"OMNI_LLM_KV_TYPE": "f16"}, ("f16", "f16")),
    ({"OMNI_LLM_KV_TYPE": "q8_0", "OMNI_LLM_V_TYPE": "f16"}, ("q8_0", "f16")),
    ({"OMNI_LLM_K_TYPE": "f16", "OMNI_LLM_V_TYPE": "q8_0"}, ("f16", "q8_0")),
])
def test_cuda_settings_reach_child_and_command(tmp_path, monkeypatch, kv_settings, expected):
    for key in list(__import__('os').environ):
        if key.startswith('OMNI_'):
            monkeypatch.delenv(key)
    monkeypatch.setattr("platform.system", lambda: "Windows")
    worker = worker_at(tmp_path)
    worker.runtime_env = {"OMNI_FLASH_ATTN": "on", "OMNI_CPU_THREADS": "4",
                          "OMNI_CPU_THREADS_BATCH": "8",
                          "OMNI_UBATCH_SIZE": "128", "OMNI_VOC_DEVICE": "gpu:0",
                          "OMNI_SAMPLER_SEED": "42"}
    worker.runtime_env.update(kv_settings)
    executable = tmp_path / "build-cuda/bin/llama-omni-server.exe"
    executable.parent.mkdir(parents=True)
    executable.touch()
    model = Path(worker.model_dir) / worker.llm_model
    model.parent.mkdir()
    model.touch()
    calls = []

    def spawn(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=io.StringIO(""), poll=lambda: None)

    monkeypatch.setattr("subprocess.Popen", spawn)
    monkeypatch.setattr("requests.get", lambda *a, **k: SimpleNamespace(status_code=200))
    monkeypatch.setenv("OMNI_CPU_THREADS", "6")
    worker._start_cpp_server()
    command, options = calls[0]
    assert command[command.index('--threads') + 1] == '6'
    assert command[command.index('--threads-batch') + 1] == '8'
    assert command[command.index('--flash-attn') + 1] == 'on'
    assert command[command.index('--cache-type-k') + 1] == expected[0]
    assert command[command.index('--cache-type-v') + 1] == expected[1]
    assert command[command.index('--ubatch-size') + 1] == '128'
    assert command[command.index('--seed') + 1] == '42'
    assert options['env']['OMNI_VOC_DEVICE'] == 'gpu:0'
    assert options['env']['PATH'].startswith(str(executable.parent))

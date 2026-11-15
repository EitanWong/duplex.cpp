#!/usr/bin/env python3
"""Run an isolated native pipeline replay, recording configuration and GPU usage.

Each invocation owns a fresh native process. Run A/B variants sequentially with
the same inputs and seed; never benchmark competing variants on the same GPU.
Reports, logs and recordings must stay in ignored verification/.
"""
import argparse
import asyncio
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--binary-dir', type=Path, default=ROOT / 'engine/build-cuda/bin')
    parser.add_argument('--profile', type=Path, default=ROOT / 'profiles/nvidia-rtx3060.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=19180)
    parser.add_argument('--ctx-size', type=int)
    parser.add_argument('--env', action='append', default=[], metavar='NAME=VALUE')
    parser.add_argument('--count', type=int, default=36)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--interval-ms', type=int, default=1000)
    parser.add_argument('--tail-seconds', type=float, default=5)
    parser.add_argument('--audio-only', action='store_true')
    parser.add_argument('--video-start-at', type=int, default=0,
                        help='First frame with video (zero based); measures cold vision activation')
    parser.add_argument('--interrupt-at', type=int, default=-1)
    parser.add_argument('--tts-temperature', type=float, default=0.0)
    args = parser.parse_args()
    if (args.count < 1 or args.rounds < 1 or args.interval_ms < 0 or args.video_start_at < 0 or
            args.tail_seconds < 0 or not 1 <= args.port <= 65535 or
            not math.isfinite(args.tts_temperature) or args.tts_temperature < 0):
        parser.error('Invalid count, interval, duration or port')
    profile = json.loads(args.profile.read_text(encoding='utf-8'))
    # Resolve before changing cwd for the backend's existing relative paths.
    args.output = args.output.resolve()
    args.model_dir = args.model_dir.resolve()
    args.binary_dir = args.binary_dir.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    settings = dict(profile['environment'])
    settings['OMNI_SAMPLER_SEED'] = '42'
    for override in args.env:
        key, sep, value = override.partition('=')
        if not sep or not key.startswith(('OMNI_', 'GGML_CUDA_', 'CUDA_', 'OMP_', 'OPENBLAS_', 'MKL_')):
            parser.error('--env requires an inference variable NAME=VALUE')
        settings[key] = value
    # Explicitly record inherited inference overrides too: the native launcher
    # gives process environment precedence over runtime_env.
    inherited = {k: v for k, v in os.environ.items() if
                 k.startswith(('OMNI_', 'GGML_CUDA_', 'CUDA_', 'OMP_', 'OPENBLAS_', 'MKL_'))}
    settings.update(inherited)
    for override in args.env:
        key, _, value = override.partition('=')
        settings[key] = value
    os.environ.update(settings)
    logging.basicConfig(filename=args.output / 'backend.log', level=logging.INFO,
                        format='%(asctime)s %(name)s %(message)s', encoding='utf-8')
    sys.path.insert(0, str(ROOT / 'runtime'))
    from core.processors.cpp_backend import CppBackendWorker
    spec = importlib.util.spec_from_file_location('native_replay', Path(__file__).with_name('native.py'))
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    cfg = profile['cpp_backend']
    ctx_size = args.ctx_size or cfg['ctx_size']
    worker = CppBackendWorker(
        llamacpp_root=str(ROOT / 'engine'), model_dir=str(args.model_dir),
        cpp_server_port=args.port, ctx_size=ctx_size,
        n_gpu_layers=cfg['n_gpu_layers'], native_binary_dir=str(args.binary_dir),
        runtime_env=settings)
    manifest = {'platform': platform.platform(), 'profile': str(args.profile.resolve()),
                'model_dir': str(args.model_dir), 'binary_dir': str(args.binary_dir),
                'ctx_size': ctx_size, 'environment': settings,
                'audio_only': args.audio_only, 'video_start_at': args.video_start_at,
                'count': args.count, 'rounds': args.rounds, 'interval_ms': args.interval_ms,
                'tail_seconds': args.tail_seconds, 'interrupt_at': args.interrupt_at,
                'tts_temperature': args.tts_temperature,
                'gpu_memory_scope': 'whole GPU including desktop and other processes',
                'speech_quality_accepted': False}
    monitor = None
    gpu_file = None
    try:
        if shutil.which('nvidia-smi'):
            manifest['gpu'] = subprocess.check_output([
                'nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                '--format=csv'], text=True).strip()
            gpu_file = (args.output / 'gpu.csv').open('w', encoding='utf-8')
            monitor = subprocess.Popen([
                'nvidia-smi', '--query-gpu=timestamp,memory.used,utilization.gpu,power.draw',
                '--format=csv', '--loop-ms=200'], stdout=gpu_file, stderr=subprocess.DEVNULL)
        os.chdir(ROOT / 'runtime')
        started = time.perf_counter()
        worker.load_model()
        manifest['load_ms'] = (time.perf_counter() - started) * 1000
        args.url = f'ws://127.0.0.1:{args.port}/backend'
        for iteration in range(args.rounds):
            asyncio.run(native.replay(args, iteration))
    finally:
        worker._stop_cpp_server()
        if worker._http_client is not None:
            worker._http_client.close()
        if monitor is not None:
            monitor.terminate()
            monitor.wait(timeout=10)
        if gpu_file is not None:
            gpu_file.close()
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()

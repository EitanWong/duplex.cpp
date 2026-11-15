#!/usr/bin/env python3
"""Measure native LLM KV formats at several history depths on one idle GPU.

This measures decoder performance, not duplex latency or speech quality. Use
pipeline.py separately with the same K/V settings and compare.py for output
checks. Mixed formats require GGML_CUDA_FA_ALL_QUANTS=ON on CUDA.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--binary-dir', type=Path, default=ROOT / 'engine/build-cuda/bin')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--depths', type=int, nargs='+', default=[0, 1024, 3072])
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--mixed', action='store_true', help='Also test q8_0/f16 and f16/q8_0')
    args = parser.parse_args()
    if min(args.depths) < 0 or args.tokens < 1 or args.repetitions < 1:
        parser.error('Depths must be nonnegative; tokens and repetitions must be positive')
    if not args.model.is_file():
        parser.error('Model file does not exist')
    suffix = '.exe' if platform.system() == 'Windows' else ''
    binary_dir = args.binary_dir.resolve()
    binary = binary_dir / f'llama-bench{suffix}'
    if not binary.is_file():
        parser.error(f'Benchmark executable not found: {binary}')
    # A second resident pipeline would distort timings and can exhaust VRAM.
    # Under WDDM, --query-compute-apps also lists desktop C+G processes.
    inventory = ET.fromstring(subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True))
    gpu_processes = [{key: process.findtext(key) for key in ('pid', 'type', 'process_name')}
                     for process in inventory.findall('.//process_info')]
    compute_processes = [process for process in gpu_processes if process['type'] == 'C']
    if compute_processes:
        parser.error('Stop GPU compute processes before benchmarking: ' + str(compute_processes))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        'platform': platform.platform(), 'model': str(args.model.resolve()),
        'binary': str(binary), 'depths': args.depths, 'tokens': args.tokens,
        'repetitions': args.repetitions, 'mixed': args.mixed,
        'gpu_processes_at_start': gpu_processes,
        'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                                        '--format=csv'], text=True).strip(),
        'environment': {k: v for k, v in os.environ.items() if
                        k.startswith(('OMNI_', 'GGML_', 'CUDA_', 'OMP_', 'OPENBLAS_'))},
        'speech_quality_accepted': False,
    }
    cuda_library = binary_dir / 'ggml-cuda.dll'
    if cuda_library.is_file():
        with cuda_library.open('rb') as source:
            manifest['ggml_cuda_sha256'] = hashlib.file_digest(source, 'sha256').hexdigest()
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    pairs = [('f16', 'f16'), ('q8_0', 'q8_0')]
    if args.mixed:
        pairs += [('q8_0', 'f16'), ('f16', 'q8_0')]
    results = []
    for key_type, value_type in pairs:
        command = [str(binary), '-m', str(args.model.resolve()), '-p', '0', '-n', str(args.tokens),
                   '-d', ','.join(map(str, args.depths)), '-r', str(args.repetitions),
                   '-b', '512', '-ub', '128', '-t', '4', '-ngl', '99', '-fa', '1',
                   '-ctk', key_type, '-ctv', value_type, '-o', 'json']
        name = f'{key_type}-{value_type}'
        started = time.perf_counter()
        with (args.output / f'{name}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=log, text=True,
                                    encoding='utf-8', errors='replace', timeout=1800)
        (args.output / f'{name}.json').write_text(result.stdout, encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'{name} failed ({result.returncode}); see {args.output / (name + ".log")}')
        rows = json.loads(result.stdout)
        for row in rows:
            results.append({'k': key_type, 'v': value_type, 'depth': row['n_depth'],
                            'tokens_per_second': row['avg_ts'], 'stddev': row['stddev_ts'],
                            'flash_attn_requested': row['flash_attn']})
        (args.output / 'summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(f'{name}: {time.perf_counter() - started:.1f}s', flush=True)
        for row in results[-len(rows):]:
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()

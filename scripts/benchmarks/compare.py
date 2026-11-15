#!/usr/bin/env python3
"""Compare matched pipeline replays, including output and pruning observations."""
import argparse
import json
from pathlib import Path

import numpy as np


def distribution(values):
    return {key: float(np.percentile(values, p)) if values else None
            for key, p in (('p50', 50), ('p95', 95), ('max', 100))}


def describe(report):
    frames = report['frames']
    lengths = [frame['metrics'].get('kv_cache_length') for frame in frames]
    drops = [i for i in range(1, len(lengths)) if lengths[i] is not None and
             lengths[i - 1] is not None and lengths[i] < lengths[i - 1]]
    return {
        'inputs': len(frames), 'latency_ms': report['latency_ms'],
        'listen_latency_ms': distribution([f['latency_ms'] for f in frames if f['listen']]),
        'speak_latency_ms': distribution([f['latency_ms'] for f in frames if not f['listen']]),
        'kv_length_max': max((v for v in lengths if v is not None), default=None),
        'kv_length_drop_indices': drops,
        'kv_drop_latency_ms': distribution([frames[i]['latency_ms'] for i in drops]),
        'first_speak_to_pcm_ms': report['first_speak_to_pcm_ms'], 'audio': report['audio'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    baseline_manifest = json.loads((args.baseline / 'manifest.json').read_text(encoding='utf-8'))
    candidate_manifest = json.loads((args.candidate / 'manifest.json').read_text(encoding='utf-8'))
    keys = ('model_dir', 'ctx_size', 'audio_only', 'video_start_at', 'count', 'rounds', 'interval_ms',
            'tail_seconds', 'interrupt_at', 'tts_temperature')
    manifest_differences = {key: [baseline_manifest.get(key), candidate_manifest.get(key)] for key in keys
                            if baseline_manifest.get(key) != candidate_manifest.get(key)}
    baseline_paths = {path.parent.name: path for path in args.baseline.glob('round-*/report.json')}
    candidate_paths = {path.parent.name: path for path in args.candidate.glob('round-*/report.json')}
    if not baseline_paths or baseline_paths.keys() != candidate_paths.keys():
        parser.error('Both runs must contain the same nonempty set of rounds')
    results = {'manifest_differences': manifest_differences, 'rounds': [],
               'speech_quality_accepted': False,
               'note': 'Output changes can change the workload; latency ratios alone do not prove speedup.'}
    for name in sorted(baseline_paths):
        baseline = json.loads(baseline_paths[name].read_text(encoding='utf-8'))
        candidate = json.loads(candidate_paths[name].read_text(encoding='utf-8'))
        before, after = baseline['frames'], candidate['frames']
        changed = [i for i, (left, right) in enumerate(zip(before, after)) if
                   (left['listen'], left['text']) != (right['listen'], right['text'])]
        same_lengths = len(before) == len(after)
        results['rounds'].append({
            'round': name, 'baseline': describe(baseline), 'candidate': describe(candidate),
            'same_frame_count': same_lengths, 'decision_text_changed_indices': changed,
            'same_decisions_text': same_lengths and not changed,
            'same_sampling': baseline['sampling'] == candidate['sampling'],
            'same_pcm_bytes': baseline['audio']['sha256'] == candidate['audio']['sha256'],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Summarize real KV depth, accounting, resources and delivery from a pipeline run."""
import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np


def distribution(values):
    return {name: float(np.percentile(values, p)) if values else None
            for name, p in (('p50', 50), ('p95', 95), ('max', 100))}


def summarize(root):
    rounds = []
    for path in sorted(root.glob('round-*/report.json')):
        report = json.loads(path.read_text(encoding='utf-8'))
        frames = report['frames']
        kv = [f['metrics'].get('kv', {}) for f in frames]
        evictions = [f['index'] for i, f in enumerate(frames) if i and
                     kv[i].get('evictions', 0) > kv[i-1].get('evictions', 0)]
        groups = {}
        for packet in report['audio_packets']:
            if packet['samples']:
                groups.setdefault((packet['epoch'], packet.get('trace', {}).get('turn')), []).append(packet)
        gaps = []
        for packets in groups.values():
            end = packets[0]['received_ms'] + 100
            for packet in packets:
                gaps.append(max(0, packet['received_ms'] - end))
                end = max(end, packet['received_ms']) + packet['samples'] / 24
        forced = next((f for f in frames if f['forced_listen']), None)
        epochs = []
        epoch, forced_active = 0, False
        for frame in frames:
            if frame['forced_listen'] and not forced_active:
                epoch += 1
                epochs.append((frame['received_ms'], epoch))
            forced_active = frame['forced_listen']
        stale = []
        for packet in report['audio_packets']:
            prior = [value for received, value in epochs if received < packet['received_ms']]
            if packet['samples'] and prior and packet['epoch'] < prior[-1]:
                stale.append(packet)
        rounds.append({
            'round': path.parent.name, 'inputs': len(frames), 'latency_ms': report['latency_ms'],
            'cold_first_input_ms': frames[0]['latency_ms'],
            'steady_latency_ms': distribution([f['latency_ms'] for f in frames[1:]]),
            'first_pcm_ms': report['first_speak_to_pcm_ms'], 'audio': report['audio'],
            'max_used': max((k.get('used_cells', 0) for k in kv), default=0),
            'max_physical_span': max((k.get('physical_span', 0) for k in kv), default=0),
            'max_attention_span': max((k.get('attention_span', 0) for k in kv), default=0),
            'depth_inputs': {str(n): sum(k.get('used_cells', 0) >= n for k in kv) for n in (3072, 7168, 14336)},
            'eviction_inputs': evictions, 'last_kv': kv[-1],
            'accounting_mismatch_inputs': [f['index'] for f, k in zip(frames, kv)
                                           if k.get('accounted_tokens', -1) >= 0 and k['accounted_tokens'] != f['metrics'].get('kv_cache_length')],
            'queue_ms': {key: distribution([k[key] for k in kv if key in k])
                         for key in ('input_queue_ms', 'llm_queue_ms', 'input_age_ms')},
            'schedule_lag_ms': report.get('schedule_lag_ms'),
            'interrupt_ms': forced['latency_ms'] if forced else None,
            'stale_pcm_packets_after_interrupt_ack': len(stale),
            'playback_proxy_100ms': {'max_supply_gap_ms': max(gaps, default=0),
                                     'total_supply_gap_ms': sum(gaps),
                                     'scope': 'per epoch/turn native arrival simulation, not browser or microphone acceptance'},
        })
    result = {'rounds': rounds, 'speech_quality_accepted': False}
    resources = root / 'resources.csv'
    if resources.exists():
        with resources.open(encoding='utf-8-sig', newline='') as source:
            rows = list(csv.DictReader(source))
        result['process_peak_bytes'] = {key: max((float(row[key]) for row in rows if row.get(key)), default=None)
                                        for key in ('working_set_bytes', 'private_bytes', 'dedicated_bytes', 'shared_bytes', 'committed_bytes')}
        result['resource_samples'] = len(rows)
    gpu = root / 'gpu.csv'
    if gpu.exists():
        with gpu.open(encoding='utf-8-sig', newline='') as source:
            rows = list(csv.DictReader(source, skipinitialspace=True))
        key = next((k for k in rows[0] if 'memory.used' in k), None) if rows else None
        if key:
            result['whole_gpu_peak_mib'] = max(float(re.search(r'[0-9.]+', row[key])[0]) for row in rows)
    log = (root / 'backend.log').read_text(encoding='utf-8', errors='replace')
    timings = {}
    for stage, ms in re.findall(r'KV_TIMING stage=(\S+) ms=([0-9.]+)', log):
        timings.setdefault(stage, []).append(float(ms))
    result['timing_ms'] = {key: {'samples': len(v), 'sum': sum(v), **distribution(v)} for key, v in timings.items()}
    result['eviction_events'] = re.findall(r'KV_EVENT[^\r\n]+', log)
    result['reported_workspace_allocations'] = re.findall(r'[^\r\n]*compute buffer size[^\r\n]*', log)
    result['timing_note'] = 'Host durations overlap and must not be added as independent GPU kernel costs. Trace logging is opt-in.'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = summarize(args.run)
    (args.run / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

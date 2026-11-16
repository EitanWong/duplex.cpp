#!/usr/bin/env python3
"""Attribute trace cost and summarize completed or interrupted stress replays.

Host waits overlap GPU work. This report does not treat their sum as lost time.
Reads the durable journal when a run has not produced its final report yet.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import json
from pathlib import Path
import re

import numpy as np


def stats(values):
    return {'n': len(values), 'sum': float(sum(values)), **{
        key: float(np.percentile(values, p)) if values else None
        for key, p in [('p50', 50), ('p95', 95), ('max', 100)]}}


def analyze(root, cold_seconds=15):
    sync, capture = defaultdict(list), defaultdict(list)
    changes = Counter()
    capture_windows = defaultdict(list)
    property_nodes = Counter()
    graph_evictions = defaultdict(list)
    graph_phase = Counter()
    phase = 'initialization'
    started = None
    # Group by model layers/caller, then separate initialization from steady.
    with (root / 'backend.log').open(encoding='utf-8', errors='replace') as source:
        for line in source:
            clock = re.search(r'^\S+ (\d+):(\d+):(\d+),(\d+)', line)
            if clock:
                seconds = sum(int(v) * scale for v, scale in zip(clock.groups(), [3600, 60, 1, .001]))
                if started is None:
                    started = seconds
                elapsed = (seconds - started) % 86400
            else:
                elapsed = 0
            period = 'startup' if elapsed < cold_seconds else 'steady'
            if 'KV_PHASE ' in line:
                phase = re.search(r'phase=(\w+)', line)[1]
            match = re.search(r'stage=synchronize ms=([0-9.]+).*layers=(\d+) queued=(\d+) reason=(\w+)', line)
            if match:
                sync[f'{period}/layers-{match[2]}/{match[4]}/' + ('empty' if match[3] == '0' else 'pending')].append(float(match[1]))
            match = re.search(r'stage=cuda_capture_instantiate_launch ms=([0-9.]+).*nodes=(\d+).*last=(.*)', line)
            if match:
                capture[f'{period}/nodes-{match[2]}/{match[3]}'].append(float(match[1]))
                graph_phase[f'{period}/{phase}'] += 1
                capture_windows[f'{int(elapsed//300)*300}/nodes-{match[2]}'].append(float(match[1]))
            match = re.search(r'KV_GRAPH_EVICT .*nodes=(\d+) idle_ms=([0-9.]+)', line)
            if match:
                graph_evictions[match[1]].append(float(match[2]))
            match = re.search(r'KV_GRAPH .*reason=(\w+)', line)
            if match:
                changes[match[1]] += 1
            match = re.search(r'reason=properties node=(\d+) name=(.*?) op=(\S+) (.*)', line)
            if match:
                # Retain the diagnostic difference fields without graph addresses.
                detail = re.sub(r'old_ne1=\d+ ne1=\d+', 'shape-lengths', match[4])
                property_nodes[f'{match[2]}/{match[3]}/{detail.strip()}'] += 1
    events = []
    journal = root / 'round-0/events.jsonl'
    if journal.exists():
        lines = journal.read_text(encoding='utf-8').splitlines(keepends=True)
        for i, line in enumerate(lines):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                if i == len(lines)-1 and not line.endswith('\n'):
                    continue  # Writer may still be appending this last line.
                raise
    report_file = root / 'round-0/report.json'
    if not events and report_file.exists():
        report = json.loads(report_file.read_text(encoding='utf-8'))
        events = [{'kind': 'input', **r} for r in report['frames']] + [
            {'kind': 'audio', **r} for r in report['audio_packets']]
    frames = [row for row in events if row['kind'] == 'input']
    sent_events = [row for row in events if row['kind'] == 'sent']
    observed_inflight_max = None
    if sent_events:
        timeline = [(r['sent_ms'], 1) for r in sent_events] + [(r['received_ms'], -1) for r in frames]
        outstanding = 0
        observed_inflight_max = 0
        for _, delta in sorted(timeline):
            outstanding += delta
            observed_inflight_max = max(observed_inflight_max, outstanding)
    interruptions = [row for row in frames if row['forced_listen']]
    audio = [row for row in events if row['kind'] == 'audio']
    windows = []
    for low in range(0, len(frames), 300):
        rows = frames[low:low+300]
        kv = [row['metrics'].get('kv', {}) for row in rows]
        windows.append({'start_input': low, 'inputs': len(rows),
            'latency_ms': stats([r['latency_ms'] for r in rows]),
            'schedule_lag_ms': stats([r['schedule_lag_ms'] for r in rows]),
            'input_age_ms': stats([k['input_age_ms'] for k in kv if 'input_age_ms' in k]),
            'accounting_mismatches': [r['index'] for r, k in zip(rows, kv) if
                k.get('accounted_tokens', -1) >= 0 and k['accounted_tokens'] != r['metrics'].get('kv_cache_length')],
            'last_kv': kv[-1]})
    # A packet from any older epoch after a forced response is stale. Input
    # interruption count itself is not proof of a speaking-time interruption.
    stale, acknowledgements = [], []
    epoch, forced_active = 0, False
    for row in frames:
        if row['forced_listen'] and not forced_active:
            epoch += 1
            acknowledgements.append((row['received_ms'], epoch))
        forced_active = row['forced_listen']
    for packet in audio:
        prior = [value for received, value in acknowledgements if received < packet['received_ms']]
        if packet['samples'] and prior and packet['epoch'] < prior[-1]:
            stale.append(packet)
    resource_windows = []
    resource_five_minutes = []
    resource_file = root / 'resources.csv'
    if resource_file.exists():
        with resource_file.open(encoding='utf-8-sig') as source:
            resources = list(csv.DictReader(source))
        if resources:
            beginning = datetime.fromisoformat(resources[0]['timestamp'].replace('Z', '+00:00'))
            groups = defaultdict(list)
            for row in resources:
                elapsed = (datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00'))-beginning).total_seconds()
                groups[int(elapsed//300)*300].append(row)
            for low, rows in groups.items():
                resource_five_minutes.append({'start_seconds_after_monitor': low, 'samples': len(rows),
                    **{key: stats([float(r[key]) for r in rows if r.get(key)]) for key in
                        ['working_set_bytes', 'private_bytes', 'dedicated_bytes', 'shared_bytes']}})
        for label, rows in [('first_60_samples', resources[:60]), ('last_60_samples', resources[-60:])]:
            resource_windows.append({'window': label, **{key: stats([float(r[key]) for r in rows if r.get(key)])
                for key in ['working_set_bytes', 'private_bytes', 'dedicated_bytes', 'shared_bytes']}})
    return {'scope': 'host durations; sums overlap; startup cutoff from engine launch',
        'startup_cutoff_seconds': cold_seconds,
        'sync': {k: stats(v) for k, v in sync.items()},
        'capture': {k: stats(v) for k, v in capture.items()},
        'graph_evictions_idle_ms': {k: stats(v) for k, v in graph_evictions.items()},
        'graph_change_reasons': dict(changes), 'property_nodes': dict(property_nodes),
        'capture_5min': {k: stats(v) for k, v in capture_windows.items()}, 'capture_nearest_llm_phase_not_owner': dict(graph_phase),
        'inputs': len(frames), 'max_observed_inflight_inputs': observed_inflight_max, 'elapsed_ms': frames[-1].get('elapsed_ms', frames[-1]['received_ms']) if frames else None,
        'interruptions': len(interruptions), 'interruption_epoch_transitions': len(acknowledgements), 'stale_pcm_after_ack': len(stale),
        'phases': {phase: {'inputs': len(selected), 'latency_ms': stats([r['latency_ms'] for r in selected])}
            for phase in ('primary', 'recovery') if (selected := [r for r in frames if r.get('phase', 'primary') == phase])},
        'windows': windows, 'resource_windows': resource_windows,
        'resource_5min': resource_five_minutes,
        'manual_speech_quality_accepted': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--cold-seconds', type=float, default=15)
    args = parser.parse_args()
    result = analyze(args.run, args.cold_seconds)
    (args.run / 'steady-summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['inputs', 'elapsed_ms', 'interruptions', 'stale_pcm_after_ack']}, indent=2))

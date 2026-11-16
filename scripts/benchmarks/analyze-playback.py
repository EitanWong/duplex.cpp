#!/usr/bin/env python3
"""Summarize browser PCM arrival, production scheduling and real render probes.

getOutputTimestamp estimates device presentation, not acoustic speaker latency.
No waveform-silence-as-underrun inference. Old observer logs require an explicit
fixture input offset; never guess it from timings.
"""
import argparse
from collections import deque
import json
from pathlib import Path


def distribution(values):
    import numpy as np
    return {'n': len(values), **{key: float(np.percentile(values, p)) if values else None
        for key, p in [('p50', 50), ('p95', 95), ('max', 100)]}}


def analyze(rows, legacy_input_offset=None):
    inputs, pending, turns = {}, deque(), set()
    arrival, scheduled, device = [], [], []
    first_arrival, first_device, delays, gaps, heartbeats = [], [], [], [], []
    last_heartbeat, last_epoch = None, -1
    scheduled_turns = set()
    rendered = nonzero = 0
    interrupted_pending = stale = 0
    starts, complete, context_changes = [], [], []
    for row in rows:
        kind = row['type']
        if kind == 'start':
            starts.append(row)
            # Separate clocks/turn IDs across page starts.
            inputs.clear(); pending.clear(); turns.clear(); scheduled_turns.clear()
            last_heartbeat = None; last_epoch = -1
        elif kind == 'input':
            native = row.get('native_input')
            if native is None and legacy_input_offset is not None:
                native = row['index'] + legacy_input_offset
            if native is not None:
                inputs[native] = row['browser_ms']
        elif kind == 'interrupt':
            interrupted_pending += len(pending); pending.clear()
        elif kind == 'pcm_arrival':
            if row['epoch'] < last_epoch:
                stale += 1
                continue
            if row['epoch'] > last_epoch:
                interrupted_pending += len(pending); pending.clear(); last_epoch = row['epoch']
            pending.append(row)
            if row.get('input') in inputs:
                delta = row['browser_ms'] - inputs[row['input']]
                arrival.append(delta)
                turn = (row['epoch'], row.get('turn'))
                if turn not in turns:
                    first_arrival.append(delta); turns.add(turn)
        elif kind == 'schedule':
            # FIFO matches production pending-chunk order, clearing on epoch/interrupt.
            pcm = pending.popleft() if pending else None
            if pcm:
                delays.append(row['browser_ms'] - pcm['browser_ms'])
                if pcm.get('input') in inputs:
                    scheduled.append(row['browser_ms'] - inputs[pcm['input']])
                ct, pt = row.get('outputContextTime'), row.get('outputPerformanceTime')
                if ct is not None and pt is not None and pt > 0:
                    presentation = pt + (row['scheduledAudioTime'] - ct)*1000
                    if pcm.get('input') in inputs:
                        delta = presentation - inputs[pcm['input']]
                        device.append(delta)
                        turn = (pcm['epoch'], pcm.get('turn'))
                        if turn not in scheduled_turns:
                            first_device.append(delta); scheduled_turns.add(turn)
        elif kind == 'gap':
            gaps.append(row['gap_ms'])
        elif kind == 'stale_packet':
            stale += 1
        elif kind == 'render':
            if last_heartbeat is not None:
                heartbeats.append(row['browser_ms']-last_heartbeat)
            last_heartbeat = row['browser_ms']
            rendered = max(rendered, row['blocks']); nonzero = max(nonzero, row['nonzero'])
        elif kind == 'context_state':
            context_changes.append(row)
        elif kind == 'complete':
            complete.append(row)
    schedules = [r for r in rows if r['type'] == 'schedule']
    player_metrics = [r for r in rows if r['type'] == 'player_metrics']
    underflow_by_turn = {}
    for row in player_metrics:
        underflow_by_turn[row['turn']] = max(underflow_by_turn.get(row['turn'], 0), row['underflowMs'])
    return {
        'scope': 'loopback relay + production AudioPlayer + actual AudioWorklet rendering; no Gateway/microphone/acoustic or subjective acceptance',
        'browser_input_clock': 'input notice receipt in browser; native send clock is reported separately',
        'legacy_input_offset': legacy_input_offset, 'starts': starts, 'complete': complete,
        'input_to_pcm_arrival_ms': distribution(arrival),
        'first_pcm_per_native_turn_ms': distribution(first_arrival),
        'input_to_schedule_callback_ms': distribution(scheduled),
        'input_to_estimated_device_presentation_ms': distribution(device),
        'first_device_presentation_per_native_turn_ms': distribution(first_device),
        'pcm_arrival_to_schedule_callback_ms': distribution(delays),
        'output_latency_ms': distribution([r['outputLatencyMs'] for r in schedules]),
        'scheduled_chunks': len(schedules), 'rendered_blocks': rendered, 'nonzero_rendered_blocks': nonzero,
        'schedule_underflow_ms': distribution(gaps), 'schedule_underflow_total_ms': sum(gaps),
        'player_underflow_per_turn_ms': distribution(list(underflow_by_turn.values())),
        'player_metrics_samples': len(player_metrics),
        'render_heartbeat_wall_ms': distribution(heartbeats), 'context_changes': context_changes,
        'stale_pcm_packets': stale,
        'interruptions_while_draining': (sum(r['type']=='interrupt' and r.get('was_draining',False) for r in rows)
            if any(r['type']=='interrupt' and 'was_draining' in r for r in rows) else None),
        'interrupt_render_windows': [r for r in rows if r['type']=='interrupt_render_window'],
        'muted_pcm_packets': sum(r['type']=='muted_packet' for r in rows), 'canceled_unscheduled_packets': interrupted_pending,
        'unmatched_schedule_callbacks': max(0, len(schedules)-len(delays)),
        'manual_speech_quality_accepted': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    parser.add_argument('--legacy-input-offset', type=int)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.journal.read_text(encoding='utf-8').splitlines() if line]
    result = analyze(rows, args.legacy_input_offset)
    args.journal.with_name('playback-summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('starts', 'complete', 'context_changes')}, indent=2))


if __name__ == '__main__':
    main()

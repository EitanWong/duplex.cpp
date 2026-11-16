#!/usr/bin/env python3
"""Replay the checked-in duplex fixture against a running native /backend server.

Reports observed request latency and PCM delivery, without claiming human speech
quality acceptance. Generated audio and reports belong in ignored verification/.
"""
import argparse
import asyncio
import base64
import hashlib
import json
from pathlib import Path
import time
import sys

import numpy as np
import soundfile as sf
from websockets.asyncio.client import connect

ROOT = Path(__file__).resolve().parents[2]


def pcm(path):
    samples, rate = sf.read(path, dtype='float32')
    if rate != 16000 or samples.ndim != 1:
        raise ValueError(f'Fixture must be mono 16 kHz: {path}')
    return base64.b64encode(samples.astype('<f4').tobytes()).decode()


def distribution(values):
    return {key: float(np.percentile(values, p)) if values else None
            for key, p in (('p50', 50), ('p95', 95), ('max', 100))}


async def replay(args, iteration):
    started = time.perf_counter()
    output = args.output / f'round-{iteration}'
    output.mkdir(parents=True, exist_ok=True)
    # Durable progress survives a failed hour-long run. Never store PCM/base64
    # here; packets remain in the separate ignored recording.
    def record(kind, **data):
        journal.write(json.dumps({'kind': kind, 'elapsed_ms': now(), **data}, ensure_ascii=False) + '\n')
    now = lambda: (time.perf_counter() - started) * 1000
    results = asyncio.Queue()
    memory_results = asyncio.Queue()
    memory_events = []
    first_text = {}
    memory_file = getattr(args, 'memory_file', None)
    memory_updates = json.loads(memory_file.read_text(encoding='utf-8')) if memory_file else []
    browser_relay = None
    frames, audio, buffers = [], [], []
    tts_temperature = getattr(args, 'tts_temperature', 0.0)
    ref = pcm(ROOT / 'engine/tools/omni/assets/default_ref_audio/default_ref_audio.wav')
    async with connect(args.url, proxy=None, max_size=32 * 1024 * 1024,
                       ping_interval=None, open_timeout=30, close_timeout=10) as ws:
        await ws.send(json.dumps({'type': 'session.init', 'payload': {
            'mode': 'full_duplex', 'use_tts': True,
            'system_prompt': 'Streaming Duplex Conversation.',
            'voice': {'ref_audio': ref, 'tts_ref_audio': ref},
            'config': {'temperature': 0.0, 'top_k': 1, 'tts_temperature': tts_temperature,
                       'max_new_speak_tokens_per_chunk': 10}}}))
        created = json.loads(await asyncio.wait_for(ws.recv(), 180))
        if created.get('type') != 'session.created':
            raise RuntimeError(created)
        init_ms = now()
        burst_size = getattr(args, 'burst_size', 1)
        recovery_count = getattr(args, 'recovery_count', 0)
        total_count = args.count + recovery_count
        if burst_size > 1 and memory_updates:
            raise ValueError('Memory updates require serial safe-boundary replay')
        journal = (output / 'events.jsonl').open('w', encoding='utf-8', buffering=1)

        async def receive():
            try:
                async for raw in ws:
                    event = json.loads(raw)
                    if event.get('kind') == 'audio':
                        raw_pcm = base64.b64decode(event.get('audio') or '')
                        buffers.append(raw_pcm)
                        audio.append({'received_ms': now(), 'samples': len(raw_pcm) // 4,
                                      'epoch': event.get('audio_epoch'), 'final': event.get('speech_end'),
                                      'trace': event.get('trace', {})})
                        record('audio', **audio[-1])
                        if browser_relay:
                            await browser_relay.send(json.dumps({'type': 'audio', 'event': event,
                                'native_received_ms': audio[-1]['received_ms']}))
                    elif event.get('kind') == 'text' and event.get('text') and event.get('response_id'):
                        first_text.setdefault(event['response_id'], now())
                    elif event.get('type') == 'response.done' or event.get('kind') == 'listen':
                        await results.put((now(), event))
                    elif event.get('type') in ('error', 'session.closed'):
                        raise RuntimeError(event)
                    elif event.get('type') in ('memory.updated', 'memory.error'):
                        await memory_results.put(event)
            except Exception as error:
                await results.put(error)

        reader = asyncio.create_task(receive())
        try:
            if getattr(args, 'browser_relay', None):
                browser_relay = await connect(args.browser_relay, proxy=None, max_size=32 * 1024 * 1024)
            pending_frames = []
            expected_epoch, forced_active = 0, False
            recovery_start = None
            for index in range(total_count):
                recovery = index >= args.count
                if recovery and recovery_start is None:
                    recovery_start = now()
                target = (recovery_start + (index-args.count)*1000 if recovery else
                          init_ms + index*args.interval_ms)
                await asyncio.sleep(max(0, (target - now()) / 1000))
                prefix = ROOT / 'engine/tools/omni/assets/test_case/duplex_omni_test_case'
                wav = prefix / f'duplex_omni_test_case_{index % 36:04}.wav'
                jpg = wav.with_suffix('.jpg')
                prepare_start = now()
                image = ([] if args.audio_only or index < getattr(args, 'video_start_at', 0)
                         else [base64.b64encode(jpg.read_bytes()).decode()])
                interrupt_every = getattr(args, 'interrupt_every', 0)
                forced = index == args.interrupt_at or (interrupt_every > 0 and index > 0 and index % interrupt_every == 0)
                payload = json.dumps({'type': 'input.append', 'input': {
                    'audio': pcm(wav), 'video_frames': image, 'max_slice_nums': 1,
                    'memory_query': getattr(args, 'memory_query', ''),
                    'force_listen': forced}})
                sent = now()
                prepare_ms = sent - prepare_start
                await ws.send(payload)
                send_ms = now() - sent
                if browser_relay:
                    await browser_relay.send(json.dumps({'type': 'input', 'index': index, 'forced': forced,
                        # One reference-audio temporary file consumes msg_counter=1.
                        'sent_ms': sent, 'native_input': index + 2}))
                record('sent', index=index, sent_ms=sent, forced_listen=forced, phase='recovery' if recovery else 'primary')
                # Native /backend handles complete inputs in FIFO order. Burst
                # mode intentionally measures transport backlog as well as compute.
                pending_frames.append((index, image, sent, prepare_ms, send_ms, forced, target, recovery))
                if len(pending_frames) < (1 if recovery else burst_size) and index < args.count-1:
                    continue
                for index, image, sent, prepare_ms, send_ms, forced, target, recovery in pending_frames:
                    result = await asyncio.wait_for(results.get(), 180)
                    if isinstance(result, Exception):
                        raise result
                    received, event = result
                    text_time = first_text.pop(event.get('response_id'), None)
                    frames.append({'index': index, 'input_mode': 'audio_video' if image else 'audio',
                                   'schedule_lag_ms': max(0, sent - target),
                                   'phase': 'recovery' if recovery else 'primary',
                                   'input_prepare_ms': prepare_ms, 'send_ms': send_ms,
                                   'sent_ms': sent, 'received_ms': received,
                                   'latency_ms': received - sent,
                                   'first_text_ms': text_time - sent if text_time is not None else None,
                                   'forced_listen': forced,
                                   'listen': event.get('kind') == 'listen', 'text': event.get('text', ''),
                                   'metrics': event.get('metrics', {})})
                    record('input', **frames[-1])
                    if forced and not forced_active:
                        expected_epoch += 1
                    forced_active = forced
                    if browser_relay:
                        await browser_relay.send(json.dumps({'type': 'input_done', 'index': index,
                            'forced': forced, 'epoch': expected_epoch}))
                    print(f"round={iteration} frame={index} latency={received-sent:.1f}ms "
                          f"listen={frames[-1]['listen']} text={frames[-1]['text'][:50]}", flush=True)
                    for update in memory_updates:
                        if update['after_input'] == index:
                            await ws.send(json.dumps({'type': 'memory.update', 'facts': update['facts']}))
                            acknowledgement = await asyncio.wait_for(memory_results.get(), 20)
                            memory_events.append({'after_input': index, **acknowledgement})
                            record('memory', **memory_events[-1])
                            if acknowledgement['type'] != 'memory.updated':
                                raise RuntimeError(acknowledgement)
                pending_frames.clear()
            await asyncio.sleep(args.tail_seconds)
        finally:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
            journal.close()
            if browser_relay:
                await browser_relay.send(json.dumps({'type': 'complete', 'inputs': len(frames), 'success': len(frames) == total_count}))
                await browser_relay.close()
    samples = np.frombuffer(b''.join(buffers), dtype='<f4')
    output = args.output / f'round-{iteration}'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'audio.f32').write_bytes(samples.tobytes())
    if len(samples):
        sf.write(output / 'audio.wav', samples, 24000, subtype='FLOAT')
    first_speak = next((f for f in frames if not f['listen']), None)
    first_pcm = next((packet for packet in audio if packet['samples'] > 0), None)
    first_pcm_ms = (first_pcm['received_ms'] - first_speak['sent_ms']) if first_pcm and first_speak else None
    native_inputs = {frame['index']+2: frame for frame in frames}
    turn_first, seen_turns = [], set()
    for packet in audio:
        trace = packet.get('trace', {})
        turn = (packet['epoch'], trace.get('turn'))
        source = native_inputs.get(trace.get('input'))
        if packet['samples'] and turn not in seen_turns and source:
            seen_turns.add(turn)
            turn_first.append({'epoch': turn[0], 'turn': turn[1], 'source_input': source['index'],
                               'ms': packet['received_ms']-source['sent_ms']})
    video_start_at = getattr(args, 'video_start_at', 0)
    report = {'mode': 'audio' if args.audio_only else ('mixed' if video_start_at else 'audio_video'),
              'video_start_at': video_start_at, 'init_ms': init_ms,
              'elapsed_ms': now(),
              'latency_clock': 'prepared input send start to response; fixture I/O and encoding excluded',
              'burst_size': burst_size, 'recovery_count': recovery_count,
              'sampling': {'temperature': 0.0, 'top_k': 1, 'tts_temperature': tts_temperature, 'speak_budget': 10},
              'frames': frames, 'audio_packets': audio,
              'memory_events': memory_events,
              'first_speak_to_pcm_ms': first_pcm_ms,
              'source_input_to_first_pcm_per_native_turn': turn_first,
              'source_input_to_first_pcm_per_native_turn_ms': distribution([row['ms'] for row in turn_first]),
              'schedule_lag_ms': distribution([frame['schedule_lag_ms'] for frame in frames]),
              'latency_ms': distribution([frame['latency_ms'] for frame in frames]),
              'first_text_ms': distribution([frame['first_text_ms'] for frame in frames if frame['first_text_ms'] is not None]),
              'audio': {'samples': len(samples), 'seconds': len(samples) / 24000,
                        'finite': bool(np.isfinite(samples).all()),
                        'peak': float(np.max(np.abs(samples))) if len(samples) else None,
                        'clipping_fraction': float(np.mean(np.abs(samples) >= 1)) if len(samples) else None,
                        'sha256': hashlib.sha256(samples.tobytes()).hexdigest()},
              'speech_quality_accepted': False}
    (output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('latency_ms', 'audio')}, ensure_ascii=False), flush=True)
    return report


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='ws://127.0.0.1:19080/backend')
    parser.add_argument('--count', type=int, default=36)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--interval-ms', type=int, default=1000)
    parser.add_argument('--tail-seconds', type=float, default=10)
    parser.add_argument('--audio-only', action='store_true')
    parser.add_argument('--video-start-at', type=int, default=0,
                        help='First frame with video (zero based); measures cold vision activation')
    parser.add_argument('--interrupt-at', type=int, default=-1)
    parser.add_argument('--recovery-count', type=int, default=0, help='Additional serial 1Hz inputs after primary stress')
    parser.add_argument('--burst-size', type=int, default=1, help='Intact inputs sent before draining FIFO responses; backlog stress only')
    parser.add_argument('--interrupt-every', type=int, default=0, help='Periodic intact-input interruption; 0 disables')
    parser.add_argument('--tts-temperature', type=float, default=0.0)
    parser.add_argument('--browser-relay', help='Local observer producer WebSocket')
    parser.add_argument('--memory-file', type=Path, help='Explicit user fact schedule, never inferred from audio')
    parser.add_argument('--memory-query', default='')
    parser.add_argument('--output', type=Path, default=ROOT / 'verification/native-duplex')
    args = parser.parse_args()
    if (args.count < 1 or args.rounds < 1 or args.burst_size < 1 or args.recovery_count < 0 or args.interrupt_every < 0 or args.interval_ms < 0 or args.tail_seconds < 0 or args.video_start_at < 0 or
            not np.isfinite(args.tts_temperature) or args.tts_temperature < 0):
        parser.error('Counts must be positive and intervals nonnegative')
    for iteration in range(args.rounds):
        await replay(args, iteration)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    asyncio.run(main())

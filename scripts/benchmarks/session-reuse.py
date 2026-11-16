#!/usr/bin/env python3
"""Measure native mode switches and repeated text with matching sampling.

Run separately with reuse switches disabled/enabled; keep reports outside Git.
This does not validate microphone playback or subjective speech quality.
"""
import argparse
import asyncio
import base64
import json
from pathlib import Path
import sys
import time

import soundfile as sf
from websockets.asyncio.client import connect

ROOT = Path(__file__).resolve().parents[2]


async def session(args, mode, name, messages, ref):
    started = time.perf_counter()
    payload = {'mode': mode, 'use_tts': True,
               'config': {'temperature': 0, 'top_k': 1, 'tts_temperature': 0}}
    if mode == 'full_duplex':
        payload['system_prompt'] = 'Streaming Duplex Conversation.'
        payload['voice'] = {'ref_audio': ref, 'tts_ref_audio': ref}
    async with connect(args.url, proxy=None, ping_interval=None,
                       max_size=32*1024*1024, close_timeout=10) as ws:
        await ws.send(json.dumps({'type': 'session.init', 'payload': payload}))
        event = json.loads(await asyncio.wait_for(ws.recv(), 180))
        if event.get('type') != 'session.created':
            raise RuntimeError(event)
        init_ms = (time.perf_counter()-started)*1000
        if mode == 'turn_based':
            await ws.send(json.dumps({'type': 'input.append', 'input': {
                'messages': messages, 'streaming': True, 'use_tts_template': False,
                'tts': {'enabled': False}, 'generation': {'max_new_tokens': 48}}}))
            while True:
                event = json.loads(await asyncio.wait_for(ws.recv(), 180))
                if event.get('type') in ('error', 'session.closed'):
                    raise RuntimeError(event)
                if event.get('type') == 'response.done':
                    break
            if not event.get('text'):
                raise RuntimeError('Empty text response')
        result = {'name': name, 'mode': mode, 'init_ms': init_ms,
                  'wall_ms': (time.perf_counter()-started)*1000,
                  'text': event.get('text'), 'metrics': event.get('metrics', {})}
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='ws://127.0.0.1:19080/backend')
    parser.add_argument('--cycles', type=int, default=2)
    parser.add_argument('--output', type=Path, default=ROOT/'verification/session-reuse.json')
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error('--cycles must be positive')
    pcm, rate = sf.read(ROOT/'engine/tools/omni/assets/default_ref_audio/default_ref_audio.wav', dtype='float32')
    if rate != 16000 or pcm.ndim != 1:
        raise ValueError('Reference must be mono 16 kHz')
    ref = base64.b64encode(pcm.astype('<f4').tobytes()).decode()
    messages = [{'role': 'system', 'content': '你是简洁的中文助手。'},
                {'role': 'user', 'content': '请只回答：测试成功'}]
    results = []
    for cycle in range(args.cycles):
        results.append(await session(args, 'full_duplex', f'duplex-{cycle}', [], ref))
        results.append(await session(args, 'turn_based', f'chat-{cycle}', messages, ref))
        results.append(await session(args, 'turn_based', f'exact-{cycle}', messages, ref))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'sampling': {'temperature': 0, 'top_k': 1},
                                      'results': results, 'speech_quality_accepted': False},
                                     indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    asyncio.run(main())

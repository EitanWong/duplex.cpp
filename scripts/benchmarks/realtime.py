#!/usr/bin/env python3
"""Replay a WAV through the authenticated Realtime API and save local evidence."""
import argparse
import asyncio
import base64
import json
from pathlib import Path
import ssl
import sys
import time

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
from websockets.asyncio.client import connect

ROOT = Path(__file__).resolve().parents[2]


async def run(args):
    key = args.key_file.read_text(encoding="utf-8").strip()
    kwargs = {"additional_headers": {"Authorization": "Bearer " + key}, "proxy": None,
              "max_size": 4 * 1024 * 1024}
    if args.beta:
        kwargs["additional_headers"]["OpenAI-Beta"] = "realtime=v1"
    if args.url.startswith("wss:"):
        kwargs["ssl"] = ssl.create_default_context(cafile=str(args.ca) if args.ca else None)
    wave, rate = sf.read(args.audio, dtype="float32", always_2d=True)
    wave = np.mean(wave, axis=1)
    if rate != 24000:
        from math import gcd
        divisor = gcd(rate, 24000)
        wave = resample_poly(wave, 24000 // divisor, rate // divisor)
    pcm = np.clip(np.rint(wave * 32768), -32768, 32767).astype("<i2")
    args.output.mkdir(parents=True, exist_ok=True)
    events, output, text = [], bytearray(), []
    started = time.monotonic()
    first_audio = None
    response_done = asyncio.Event()
    async with connect(args.url, **kwargs) as ws:
        created = json.loads(await ws.recv())
        assert created["type"] == "session.created", created
        await ws.send(json.dumps({"type": "session.update", "session": {
            "instructions": "你是一个友善的中文语音助手。请用简短的一两句话回答用户。",
            **({"turn_detection": None} if args.manual else {})}}))

        async def sender():
            # One second of initial silence preserves the native startup listen guard.
            samples = np.concatenate((np.zeros(24000, dtype="<i2"), pcm))
            deadline = time.monotonic()
            for offset in range(0, args.seconds * 24000, 2400):
                if args.manual and offset >= len(samples):
                    await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                    await ws.send(json.dumps({"type": "response.create"}))
                    return
                chunk = samples[offset:offset + 2400].tobytes()
                chunk += b"\0" * (4800 - len(chunk))
                await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(chunk).decode()}))
                deadline += .1
                await asyncio.sleep(max(0, deadline - time.monotonic()))

        async def receiver():
            nonlocal first_audio
            async for raw in ws:
                event = json.loads(raw)
                kind = event["type"]
                if kind in ("response.output_audio.delta", "response.audio.delta"):
                    if first_audio is None:
                        first_audio = time.monotonic() - started
                    output.extend(base64.b64decode(event["delta"]))
                    event["delta"] = f"<{len(event['delta'])} Base64 characters>"
                elif kind in ("response.output_audio_transcript.delta", "response.audio_transcript.delta"):
                    text.append(event["delta"])
                events.append({"elapsed_s": round(time.monotonic() - started, 3), **event})
                if kind == "error":
                    raise RuntimeError(event["error"])
                if kind == "response.created":
                    response_done.clear()
                elif kind == "response.done":
                    response_done.set()

        tasks = [asyncio.create_task(sender()), asyncio.create_task(receiver())]
        try:
            async with asyncio.timeout(args.seconds + 30):
                # Surface either task's exception immediately.
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
                # The model may answer during a pause inside the WAV. Complete
                # the whole real-time input replay before judging the response.
                if tasks[0] not in done:
                    raise RuntimeError("The server closed before replay finished")
                if not response_done.is_set():
                    completion = asyncio.create_task(response_done.wait())
                    try:
                        done, _ = await asyncio.wait((completion, tasks[1]), return_when=asyncio.FIRST_COMPLETED)
                        for task in done:
                            task.result()
                    finally:
                        completion.cancel()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    summary = {"input": str(args.audio), "input_duration_s": round(len(pcm) / 24000, 3),
               "elapsed_s": round(time.monotonic() - started, 3), "first_audio_s": first_audio,
               "output_duration_s": round(len(output) / 48000, 3), "transcript": "".join(text),
               "events": len(events), "protocol": "beta" if args.beta else "GA", "manual": args.manual,
               "responses": [e["response"]["status"] for e in events if e["type"] == "response.done"]}
    (args.output / "events.json").write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if output:
        sf.write(args.output / "response.wav", np.frombuffer(output, dtype="<i2"), 24000, subtype="PCM_16")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not output or not text or "completed" not in summary["responses"]:
        raise RuntimeError("No complete speech and transcript response received")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8006/v1/realtime?mode=audio&model=gpt-realtime")
    parser.add_argument("--key-file", type=Path, default=ROOT / ".local/realtime/api-key")
    parser.add_argument("--ca", type=Path)
    parser.add_argument("--audio", type=Path, default=ROOT / "runtime/tests/cases/common/user_audio/000_user_audio0.wav")
    parser.add_argument("--output", type=Path, default=ROOT / "verification/realtime")
    parser.add_argument("--seconds", type=int, default=40)
    parser.add_argument("--beta", action="store_true")
    parser.add_argument("--manual", action="store_true", help="Commit the WAV and explicitly create a response")
    args = parser.parse_args()
    if not 5 <= args.seconds <= 120:
        parser.error("seconds must be between 5 and 120")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

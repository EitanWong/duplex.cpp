#!/usr/bin/env python3
"""Replay official ModelBest session.init/input.append messages on real hardware."""
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
MODEL = "MiniCPM-o-4.5-Realtime"


def reference_audio(path):
    ref, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != 16000:
        raise ValueError("reference must be 16000 Hz")
    return base64.b64encode(ref.mean(axis=1).astype("<f4").tobytes()).decode()


async def run(args):
    key = args.key_file.read_text(encoding="utf-8").strip()
    kwargs = {"proxy": None, "max_size": 8 * 1024 * 1024}
    if args.browser_auth:
        kwargs["subprotocols"] = ["map.realtime", "map-api-key." + key]
    else:
        kwargs["additional_headers"] = {"Authorization": "Bearer " + key}
    if args.url.startswith("wss:"):
        kwargs["ssl"] = ssl.create_default_context(cafile=str(args.ca) if args.ca else None)
    samples, rate = sf.read(args.audio, dtype="float32", always_2d=True)
    samples = samples.mean(axis=1)
    if rate != 16000:
        from math import gcd
        factor = gcd(rate, 16000)
        samples = resample_poly(samples, 16000 // factor, rate // factor)
    samples = np.concatenate((np.zeros(16000, dtype="<f4"), samples)).astype("<f4")
    frame = base64.b64encode(args.frame.read_bytes()).decode() if args.frame else None
    args.output.mkdir(parents=True, exist_ok=True)
    events, audio, text = [], bytearray(), []
    first_audio = None
    started = time.monotonic()
    session_closed = asyncio.Event()
    init_sent = False

    async with connect(args.url, **kwargs) as ws:
        async def receiver():
            nonlocal first_audio, init_sent
            async for raw in ws:
                event = json.loads(raw)
                kind = event["type"]
                if kind == "response.output_audio.delta" and event.get("audio"):
                    first_audio = first_audio if first_audio is not None else time.monotonic() - started
                    audio.extend(base64.b64decode(event["audio"]))
                    event["audio"] = f"<{len(event['audio'])} Base64 characters>"
                elif kind == "response.output_text.delta":
                    text.append(event["text"])
                events.append({"elapsed_s": round(time.monotonic() - started, 3), **event})
                if kind == "session.queue_done" and not init_sent:
                    payload = {"system_prompt": "你是一个友善的中文语音助手，请简短回答。",
                               "config": {"temperature": .7, "tts_enabled": not args.no_tts}}
                    if args.reference:
                        encoded = reference_audio(args.reference)
                        payload["voice"] = {"ref_audio_base64": encoded, "tts_ref_audio_base64": encoded}
                    if args.tts_reference:
                        payload.setdefault("voice", {})["tts_ref_audio_base64"] = reference_audio(args.tts_reference)
                    await ws.send(json.dumps({"type": "session.init", "payload": payload}))
                    init_sent = True
                elif kind == "session.created":
                    created.set()
                elif kind == "error":
                    raise RuntimeError(event["error"])
                elif kind == "session.closed":
                    session_closed.set()
                    return

        created = asyncio.Event()
        receiver_task = asyncio.create_task(receiver())
        async def replay():
            await created.wait()
            deadline = time.monotonic()
            for offset in range(0, args.seconds * 16000, args.packet_samples):
                size = min(args.packet_samples, args.seconds * 16000 - offset)
                chunk = samples[offset:offset + size].tobytes()
                chunk += b"\0" * (size * 4 - len(chunk))
                payload = {"audio": base64.b64encode(chunk).decode()}
                if frame:
                    payload["video_frames"] = [frame]
                if args.interrupt_at is not None and offset <= args.interrupt_at * 16000 < offset + size:
                    payload["force_listen"] = True
                await ws.send(json.dumps({"type": "input.append", "input": payload}))
                deadline += size / 16000
                await asyncio.sleep(max(0, deadline - time.monotonic()))
            await ws.send(json.dumps({"type": "session.close", "reason": "smoke_test_complete"}))
            await session_closed.wait()

        sender_task = asyncio.create_task(replay())
        try:
            async with asyncio.timeout(args.seconds + 130):
                done, _ = await asyncio.wait((sender_task, receiver_task), return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
                if not session_closed.is_set():
                    raise RuntimeError("Session ended without close acknowledgement")
        finally:
            for task in (sender_task, receiver_task):
                task.cancel()
            await asyncio.gather(sender_task, receiver_task, return_exceptions=True)
    summary = {"protocol": "ModelBest", "input": str(args.audio), "input_duration_s": round(len(samples) / 16000 - 1, 3),
               "elapsed_s": round(time.monotonic() - started, 3), "first_audio_s": first_audio,
               "output_duration_s": round(len(audio) / 96000, 3), "transcript": "".join(text),
               "events": len(events), "no_tts": args.no_tts, "frame": str(args.frame) if args.frame else None,
               "closed": session_closed.is_set()}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "events.json").write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    if audio:
        sf.write(args.output / "response.wav", np.frombuffer(audio, dtype="<f4"), 24000, subtype="FLOAT")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not text or (not args.no_tts and not audio) or (args.no_tts and audio):
        raise RuntimeError("Missing text/audio output or tts_enabled was not honored")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=f"ws://127.0.0.1:8006/v1/realtime?mode=audio&model={MODEL}")
    parser.add_argument("--key-file", type=Path, default=ROOT / ".local/realtime/api-key")
    parser.add_argument("--ca", type=Path)
    parser.add_argument("--audio", type=Path, default=ROOT / "runtime/tests/cases/common/user_audio/000_user_audio0.wav")
    parser.add_argument("--output", type=Path, default=ROOT / "verification/modelbest-realtime")
    parser.add_argument("--seconds", type=int, default=25)
    parser.add_argument("--packet-samples", type=int, choices=(1600, 16000), default=16000)
    parser.add_argument("--frame", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--tts-reference", type=Path, help="Independent TTS reference, also works without --reference")
    parser.add_argument("--browser-auth", action="store_true")
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument("--interrupt-at", type=float)
    args = parser.parse_args()
    if not 10 <= args.seconds <= 120:
        parser.error("seconds must be 10-120")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

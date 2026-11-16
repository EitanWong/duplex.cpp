"""ModelBest Realtime audio/video wire protocol over the existing Gateway.

Contract: https://platform.modelbest.cn/console/docs/api/realtime
Also checked against the official platform's DuplexExperience client (2026-11-16).
Audio stays float32 throughout; no OpenAI PCM16 conversion or extra VAD is used.
"""
import asyncio
import base64
import binascii
from collections import deque
from io import BytesIO
import json
import logging
import math

import numpy as np
from PIL import Image, UnidentifiedImageError
from websockets.asyncio.client import connect

from .realtime import ProtocolError, new_id

logger = logging.getLogger("realtime.modelbest")
MODEL = "MiniCPM-o-4.5-Realtime"
MODEL_ALIASES = (MODEL,)


def decode_audio(value, param="input.audio", max_seconds=1):
    if not isinstance(value, str) or len(value) > max_seconds * 16000 * 4 * 4 // 3 + 8:
        raise ProtocolError(f"{param} must be Base64 float32 PCM, at most {max_seconds} seconds", param=param)
    try:
        data = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error):
        raise ProtocolError("Invalid Base64 audio", param=param) from None
    if not data or len(data) % 4 or len(data) > max_seconds * 64000:
        raise ProtocolError("Use whole float32 PCM samples at 16000 Hz, mono", param=param)
    samples = np.frombuffer(data, dtype="<f4")
    if not np.isfinite(samples).all() or np.max(np.abs(samples)) > 1:
        raise ProtocolError("Float32 audio samples must be finite and in [-1, 1]", param=param)
    return data


def validate_frame(value):
    if not isinstance(value, str) or len(value) > 2 * 1024 * 1024:
        raise ProtocolError("Use a Base64 JPEG frame under 1.5 MiB", param="input.video_frames")
    try:
        data = base64.b64decode(value, validate=True)
        with Image.open(BytesIO(data)) as frame:
            if frame.format != "JPEG" or frame.width * frame.height > 4096 * 4096:
                raise ValueError("invalid frame")
            frame.verify()
    except (ValueError, binascii.Error, OSError, UnidentifiedImageError, Image.DecompressionBombError):
        raise ProtocolError("Invalid JPEG frame or excessive image dimensions", param="input.video_frames") from None
    return value


class ModelBestSession:
    def __init__(self, websocket, settings, model=MODEL, mode="video", protocol="modelbest"):
        self.ws, self.settings, self.model, self.mode = websocket, settings, model, mode
        self.protocol = protocol
        self.id = new_id("omni_map_" if mode == "video" else "adx_map_")
        self.upstream = None
        self.queue_ready = False
        self.prepare = None
        self.prepare_sent = False
        self.created = asyncio.Event()
        self.finished = asyncio.Event()
        self.closing = False
        self.closed = False
        self.paused = False
        self.send_lock = asyncio.Lock()
        self.close_lock = asyncio.Lock()
        self.pending = bytearray()
        self.pending_frames = []
        self.pending_force = False
        self.inputs = deque()
        self.input_ready = asyncio.Event()
        self.slots = asyncio.Semaphore(2)
        self.in_flight = 0
        self.buffered_bytes = 0
        self.epoch = 0
        self.min_epoch = 0
        self.tts_enabled = True
        self.native_speaking = False
        self.response_id = None
        self.response_audio_finished = False
        self.response_listening = False

    async def output(self, kind, **fields):
        if self.protocol != "qwen-audio-agent":
            names = {"text": "response.output_text.delta", "audio": "response.output_audio.delta", "listen": "response.listen"}
            await self.emit(names[kind], **fields)
            return
        # qwen-audio-agent 2.1.0's built-in minicpm-o Provider consumes the
        # earlier unified event form. Keep its response ID until the native
        # final audio arrives so late vocoder packets don't create a new turn.
        if kind == "audio" and not fields.get("audio") and self.response_id is None:
            return
        if kind == "listen":
            self.response_listening = True
            if self.response_id and self.tts_enabled and not self.response_audio_finished:
                return
        elif self.response_id is None:
            self.response_id = new_id("resp_")
            self.response_audio_finished = False
            self.response_listening = False
        # An empty speech_end packet closes the current response but must not
        # start a phantom response in qwen's normalizer.
        if kind != "audio" or fields.get("audio"):
            await self.emit("response.output.delta", kind=kind, response_id=self.response_id or "", **fields)
        if kind == "listen":
            self.response_id = None
        elif kind == "audio" and fields.get("speech_end"):
            self.response_audio_finished = True
            if self.response_listening:
                await self.output("listen")

    async def emit(self, event_type, **fields):
        if self.closed:
            return
        async with self.send_lock:
            await self.ws.send_json({"type": event_type, "session_id": self.id, **fields})

    async def close(self, reason="client_closed", code=1000):
        async with self.close_lock:
            if self.closed:
                return
            self.closing = True
            try:
                await self.emit("session.closed", reason=reason)
            finally:
                self.closed = True
                self.finished.set()
            await self.ws.close(code=code)

    async def fail(self, exc, server=False, event_id=None):
        await self.emit("error", error={"type": "server_error" if server else "invalid_request_error",
                                        "code": exc.code, "message": str(exc), "param": exc.param,
                                        **({"event_id": event_id} if event_id else {})})
        await self.close("server_error" if server else "invalid_request", 1011 if server else 1008)

    def init_payload(self, payload):
        if not isinstance(payload, dict) or payload.keys() - {"system_prompt", "config", "voice", "max_slice_nums"}:
            raise ProtocolError("Invalid session.init payload", param="payload")
        prompt = payload.get("system_prompt", "You are a helpful voice assistant.")
        if not isinstance(prompt, str) or len(prompt) > 16000:
            raise ProtocolError("system_prompt must be a string of at most 16000 characters", param="payload.system_prompt")
        config = payload.get("config", {})
        if not isinstance(config, dict):
            raise ProtocolError("config must be an object", param="payload.config")
        allowed = {"temperature", "tts_enabled", "length_penalty", "vision_hd", "max_tokens", "max_kv_tokens"}
        if config.keys() - allowed:
            raise ProtocolError("Unsupported session config option", "unsupported_feature", "payload.config")
        # These two options require native context/token-budget controls that
        # this transport adapter cannot enforce. Never silently ignore them.
        for key in ("max_tokens", "max_kv_tokens"):
            if key in config:
                raise ProtocolError(f"{key} is controlled by the native hardware profile in this deployment; omit it", "unsupported_feature", f"payload.config.{key}")
        native = {"chunk_ms": 1000}
        for field, default, low, high in (("temperature", .7, 0, 2), ("length_penalty", 1.1, .1, 5)):
            value = config.get(field, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ProtocolError(f"Invalid {field}", param=f"payload.config.{field}")
            native[field] = value
        for field in ("tts_enabled", "vision_hd"):
            if field in config and not isinstance(config[field], bool):
                raise ProtocolError(f"{field} must be boolean", param=f"payload.config.{field}")
        self.tts_enabled = config.get("tts_enabled", True)
        native["generate_audio"] = self.tts_enabled
        slices = payload.get("max_slice_nums", 3 if config.get("vision_hd") else 1)
        if isinstance(slices, bool) or not isinstance(slices, int) or not 1 <= slices <= 3:
            raise ProtocolError("max_slice_nums must be 1-3", param="payload.max_slice_nums")
        request = {"type": "prepare", "system_prompt": prompt, "config": native, "max_slice_nums": slices}
        voice = payload.get("voice", {})
        if not isinstance(voice, dict) or voice.keys() - {"ref_audio_base64", "tts_ref_audio_base64"}:
            raise ProtocolError("Invalid voice reference configuration", param="payload.voice")
        for key, value in voice.items():
            decode_audio(value, f"payload.voice.{key}", max_seconds=30)
            request[key] = value
        return request

    async def maybe_prepare(self):
        if self.queue_ready and self.prepare is not None and not self.prepare_sent:
            self.prepare_sent = True
            await self.upstream.send(json.dumps(self.prepare))

    async def append(self, payload):
        if not self.created.is_set() or self.closing or self.paused:
            raise ProtocolError("Wait for session.created and resume before sending input", "invalid_state")
        if not isinstance(payload, dict) or payload.keys() - {"audio", "video_frames", "force_listen", "max_slice_nums"}:
            raise ProtocolError("Invalid input.append input", param="input")
        data = decode_audio(payload.get("audio"))
        frames = payload.get("video_frames", [])
        if not isinstance(frames, list) or len(frames) > 1:
            raise ProtocolError("This native backend supports at most one video frame per input", "unsupported_feature", "input.video_frames")
        if self.mode == "audio" and frames:
            raise ProtocolError("video_frames requires mode=video", param="input.video_frames")
        frames = [validate_frame(frame) for frame in frames]
        force = payload.get("force_listen", False)
        if not isinstance(force, bool):
            raise ProtocolError("force_listen must be boolean", param="input.force_listen")
        slices = payload.get("max_slice_nums", self.prepare["max_slice_nums"])
        if isinstance(slices, bool) or not isinstance(slices, int) or not 1 <= slices <= 3:
            raise ProtocolError("max_slice_nums must be 1-3", param="input.max_slice_nums")
        if self.buffered_bytes + len(data) > self.settings.max_buffer_seconds * 64000:
            raise ProtocolError("Audio buffer full; send at real-time speed", "audio_buffer_full", "input.audio")
        if force and not self.pending_force:
            self.min_epoch = max(self.min_epoch, self.epoch + 1)
            await self.upstream.send(json.dumps({"type": "interrupt"}))
        self.pending_force |= force
        if frames:
            self.pending_frames = frames
        self.pending.extend(data)
        self.buffered_bytes += len(data)
        if len(self.pending) >= 64000:
            chunk = bytes(self.pending[:64000])
            del self.pending[:64000]
            self.inputs.append({"type": "audio_chunk", "audio_base64": base64.b64encode(chunk).decode(),
                                "frame_base64_list": self.pending_frames, "force_listen": self.pending_force,
                                "max_slice_nums": slices})
            self.pending_frames = []
            self.pending_force = False
            self.input_ready.set()

    async def pump(self):
        await self.created.wait()
        while True:
            await self.input_ready.wait()
            await self.slots.acquire()
            if self.paused or not self.inputs:
                self.slots.release()
                self.input_ready.clear()
                continue
            item = self.inputs.popleft()
            if not self.inputs:
                self.input_ready.clear()
            self.in_flight += 1
            await self.upstream.send(json.dumps(item))

    async def native_event(self, event):
        kind = event.get("type")
        if kind == "queue_done":
            self.queue_ready = True
            await self.emit("session.queue_done")
            await self.maybe_prepare()
        elif kind in ("queued", "queue_update"):
            await self.emit("session.queued" if kind == "queued" else "session.queue_update",
                            **{key: event[key] for key in ("position", "estimated_wait_s", "queue_length", "ticket_id") if key in event})
        elif kind == "prepared":
            self.created.set()
            await self.emit("session.created", model=self.model, mode=self.mode)
        elif kind == "interrupted":
            self.epoch = max(self.epoch, event.get("audio_epoch", self.epoch))
            self.min_epoch = max(self.min_epoch, self.epoch)
            self.native_speaking = False
            # The old response ends on interruption even if its final audio
            # was discarded by the native epoch gate.
            self.response_audio_finished = True
            if self.protocol == "qwen-audio-agent" and self.response_id:
                await self.output("listen")
            await self.emit("response.interrupted", audio_epoch=self.epoch)
        elif kind in ("result", "audio_only"):
            if kind == "result" and self.in_flight:
                self.in_flight -= 1
                self.buffered_bytes = max(0, self.buffered_bytes - 64000)
                self.slots.release()
            epoch = event.get("audio_epoch", self.epoch)
            self.epoch = max(self.epoch, epoch)
            if epoch < self.min_epoch:
                return
            if kind == "result":
                text = event.get("text") or ""
                if text:
                    self.native_speaking = True
                    await self.output("text", text=text, end_of_turn=bool(event.get("end_of_turn")))
                if event.get("is_listen"):
                    self.native_speaking = False
                    # Native result.text is generated assistant text, never a
                    # user transcript, including on a LISTEN boundary.
                    # Defer a qwen response boundary until after audio below;
                    # text and final audio may share this native result.
                    if self.protocol != "qwen-audio-agent":
                        await self.output("listen")
            encoded = event.get("audio_data") or ""
            if self.tts_enabled and (encoded or event.get("speech_end")):
                await self.output("audio", audio=encoded, sample_rate=24000,
                                  audio_epoch=epoch, speech_end=bool(event.get("speech_end")))
            if self.protocol == "qwen-audio-agent" and kind == "result" and event.get("is_listen"):
                await self.output("listen")
        elif kind in ("paused", "resumed"):
            await self.emit("session.paused" if kind == "paused" else "session.resumed")
        elif kind == "stopped":
            await self.close("client_closed")
        elif kind in ("error", "timeout"):
            raise ProtocolError("The Gateway could not complete the realtime session", "upstream_unavailable")

    async def receive(self):
        from fastapi import WebSocketDisconnect
        while not self.closed:
            incoming = await self.ws.receive()
            if incoming["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(incoming.get("code", 1000))
            event = None
            try:
                if incoming.get("text") is None:
                    raise ProtocolError("Send JSON text frames", "invalid_event")
                event = json.loads(incoming["text"])
                if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                    raise ProtocolError("An event object with a type is required", param="type")
                kind = event["type"]
                if kind == "session.init":
                    if self.prepare is not None:
                        raise ProtocolError("session.init was already received", "invalid_state")
                    self.prepare = self.init_payload(event.get("payload", {}))
                    await self.maybe_prepare()
                elif kind == "input.append":
                    await self.append(event.get("input"))
                elif kind == "session.close":
                    self.closing = True
                    self.inputs.clear()
                    if self.created.is_set():
                        await self.upstream.send(json.dumps({"type": "stop"}))
                        async def deadline():
                            await asyncio.sleep(5)
                            await self.close("client_closed")
                        # run() owns and cancels this task on teardown.
                        self.close_task = asyncio.create_task(deadline())
                    else:
                        await self.close("client_closed")
                    return
                elif kind in ("session.pause", "session.resume", "session.interrupt"):
                    if not self.created.is_set() or self.closing:
                        raise ProtocolError("Wait for session.created", "invalid_state")
                    if kind == "session.pause":
                        self.paused = True
                        self.inputs.clear()
                        self.pending.clear()
                        self.pending_frames = []
                        self.pending_force = False
                        self.buffered_bytes = self.in_flight * 64000
                        await self.upstream.send(json.dumps({"type": "pause"}))
                    elif kind == "session.resume":
                        self.paused = False
                        await self.upstream.send(json.dumps({"type": "resume"}))
                    else:
                        self.pending_force = True
                        self.min_epoch = max(self.min_epoch, self.epoch + 1)
                        await self.upstream.send(json.dumps({"type": "interrupt"}))
                else:
                    raise ProtocolError(f"Unsupported ModelBest event: {kind}", "unsupported_event", "type")
            except (json.JSONDecodeError, ProtocolError) as exc:
                if not isinstance(exc, ProtocolError):
                    exc = ProtocolError("Invalid JSON", "invalid_json")
                await self.fail(exc, event_id=event.get("event_id") if isinstance(event, dict) else None)
                return

    async def run(self):
        from fastapi import WebSocketDisconnect
        tasks = []
        self.close_task = None
        try:
            kwargs = {"max_size": 8 * 1024 * 1024, "proxy": None, "open_timeout": 10}
            if self.settings.gateway_url.startswith("wss:"):
                kwargs["ssl"] = self.settings.upstream_ssl
            async with connect(self.settings.gateway_url.rstrip("/") + "/ws/duplex/" + self.id, **kwargs) as upstream:
                self.upstream = upstream
                async def reader():
                    async for raw in upstream:
                        await self.native_event(json.loads(raw))
                        if self.closed:
                            return
                    if not self.closed:
                        raise ProtocolError("The Gateway closed the realtime session", "upstream_unavailable")
                async def guard():
                    await asyncio.wait_for(self.created.wait(), self.settings.prepare_timeout)
                    await asyncio.Event().wait()
                async def receive_and_close():
                    await self.receive()
                    if self.closing and not self.closed:
                        # Keep the reader alive for the server's stopped/close ack.
                        await asyncio.Event().wait()
                tasks = [asyncio.create_task(reader()), asyncio.create_task(receive_and_close()),
                         asyncio.create_task(self.pump()), asyncio.create_task(guard()),
                         asyncio.create_task(self.finished.wait())]
                async with asyncio.timeout(self.settings.max_session_seconds):
                    done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        task.result()
        except WebSocketDisconnect:
            pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("ModelBest session ended: %s", type(exc).__name__)
            if not self.closed:
                try:
                    await self.fail(ProtocolError("The realtime service is unavailable or the session timed out", "upstream_unavailable"), server=True)
                except (RuntimeError, WebSocketDisconnect):
                    pass
        finally:
            if self.close_task:
                tasks.append(self.close_task)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

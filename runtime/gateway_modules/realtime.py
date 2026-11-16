"""OpenAI Realtime audio events over the existing duplex Gateway protocol.

This module deliberately rejects features the native model cannot implement.
It doesn't change inference, own a Worker pool, or expose the Gateway admin API.
"""
import asyncio
import base64
import binascii
import copy
import json
import logging
import time
import uuid

import numpy as np
from scipy.signal import firwin, lfilter
from websockets.asyncio.client import connect

logger = logging.getLogger("realtime")


def new_id(prefix):
    return prefix + uuid.uuid4().hex[:24]


class ProtocolError(ValueError):
    def __init__(self, message, code="invalid_value", param=None):
        super().__init__(message)
        self.code, self.param = code, param


class AudioResampler:
    """Stateful, band-limited 24 kHz PCM16 -> 16 kHz float32 conversion.

    Preserve FIR state and decimation phase across arbitrarily sized appends.
    The filter adds 0.625 ms latency; packet boundaries add no discontinuities.
    """
    def __init__(self):
        self.kernel = firwin(61, 1 / 3).astype(np.float64) * 2
        self.state = np.zeros(len(self.kernel) - 1)
        self.phase = 0

    def convert(self, data):
        samples = np.frombuffer(data, dtype="<i2").astype(np.float64) / 32768
        upsampled = np.zeros(len(samples) * 2)
        upsampled[::2] = samples
        filtered, self.state = lfilter(self.kernel, [1], upsampled, zi=self.state)
        output = filtered[self.phase::3].astype("<f4")
        self.phase = (self.phase - len(upsampled)) % 3
        return output.tobytes()


def output_pcm16(encoded):
    samples = np.frombuffer(base64.b64decode(encoded, validate=True), dtype="<f4")
    samples = np.nan_to_num(samples, nan=0, posinf=1, neginf=-1)
    pcm = np.clip(np.rint(samples.astype(np.float64) * 32768), -32768, 32767).astype("<i2")
    return base64.b64encode(pcm.tobytes()).decode("ascii")


class RealtimeSession:
    def __init__(self, websocket, settings, model="gpt-realtime", beta=False, initial=None):
        self.ws, self.settings, self.beta = websocket, settings, beta
        self.id, self.conversation_id = new_id("sess_"), new_id("conv_")
        self.config = {
            "instructions": "You are a helpful voice assistant.",
            "output_modalities": ["audio"],
            "turn_detection": {"type": "server_vad", "threshold": 0.5,
                               "prefix_padding_ms": 300, "silence_duration_ms": 500,
                               "create_response": True, "interrupt_response": True},
        }
        self.model = model
        if initial:
            self.update(initial)
        self.ready = asyncio.Event()
        self.lock = asyncio.Lock()
        self.send_lock = asyncio.Lock()
        self.backend_task = None
        self.backend = None
        self.pcm = bytearray()
        self.manual = bytearray()
        self.committed_pcm = bytearray()
        self.vad_pending = bytearray()
        self.resampler = AudioResampler()
        self.input_samples = 0
        self.uncommitted_samples = 0
        self.speech_id = None
        self.silence_ms = 0
        self.previous_item = None
        self.committed = False
        self.response_allowed = False
        self.response = None
        self.transcript = ""
        self.metadata = None
        self.force_listen = False
        self.min_epoch = 0
        self.last_epoch = 0
        self.discard_until_listen = False
        self.in_flight = 0
        self.history = []
        self.history_bytes = 0
        self.input_started = False

    def update(self, patch):
        if not isinstance(patch, dict):
            raise ProtocolError("session must be an object", param="session")
        candidate = copy.deepcopy(self.config)
        allowed = {"type", "model", "instructions", "audio", "output_modalities", "modalities",
                   "input_audio_format", "output_audio_format", "voice", "turn_detection",
                   "input_audio_transcription", "tools", "tool_choice", "temperature",
                   "max_response_output_tokens", "max_output_tokens"}
        for key in patch.keys() - allowed:
            raise ProtocolError(f"Unsupported session field: {key}", "unsupported_feature", key)
        if patch.get("type", "realtime") != "realtime":
            raise ProtocolError("Only realtime audio sessions are supported", "unsupported_feature", "type")
        if "model" in patch and patch["model"] != self.model:
            raise ProtocolError("The session model cannot be changed", param="model")
        if "instructions" in patch:
            if not isinstance(patch["instructions"], str) or len(patch["instructions"]) > 16000:
                raise ProtocolError("instructions must be a string of at most 16000 characters", param="instructions")
            candidate["instructions"] = patch["instructions"]
        if "modalities" in patch:
            modalities = patch["modalities"]
            if modalities not in (["text"], ["audio"], ["text", "audio"], ["audio", "text"]):
                raise ProtocolError("Unsupported modalities", param="modalities")
            candidate["output_modalities"] = ["audio"] if "audio" in modalities else ["text"]
        if "output_modalities" in patch:
            if patch["output_modalities"] not in (["audio"], ["text"]):
                raise ProtocolError("Use output_modalities ['audio'] or ['text']", param="output_modalities")
            candidate["output_modalities"] = patch["output_modalities"]
        for field in ("input_audio_format", "output_audio_format"):
            if field in patch and patch[field] != "pcm16":
                raise ProtocolError("Only 24 kHz mono PCM16 is supported", "unsupported_feature", field)
        if patch.get("voice", "alloy") != "alloy":
            raise ProtocolError("Only alloy (an alias for the native reference voice) is supported", "unsupported_feature", "voice")
        if patch.get("input_audio_transcription") is not None:
            raise ProtocolError("Input transcription is unavailable", "unsupported_feature", "input_audio_transcription")
        if patch.get("tools", []) != [] or patch.get("tool_choice", "auto") not in ("auto", "none"):
            raise ProtocolError("Tool calling is unavailable", "unsupported_feature", "tools")
        # These controls don't have equivalent per-response semantics in the engine.
        for field in ("temperature", "max_response_output_tokens", "max_output_tokens"):
            if field in patch:
                raise ProtocolError(f"{field} is controlled by the native profile", "unsupported_feature", field)
        td_present = "turn_detection" in patch
        td = patch.get("turn_detection")
        if "audio" in patch:
            audio = patch["audio"]
            if not isinstance(audio, dict) or audio.keys() - {"input", "output"}:
                raise ProtocolError("Invalid audio configuration", param="audio")
            for direction, fields in (("input", {"format", "turn_detection", "transcription", "noise_reduction"}),
                                      ("output", {"format", "voice", "speed"})):
                section = audio.get(direction, {})
                if not isinstance(section, dict) or section.keys() - fields:
                    raise ProtocolError(f"Unsupported audio.{direction} configuration", "unsupported_feature", f"audio.{direction}")
                if "format" in section and section["format"] != {"type": "audio/pcm", "rate": 24000}:
                    raise ProtocolError("Only audio/pcm at 24000 Hz is supported", "unsupported_feature", f"audio.{direction}.format")
                if direction == "input":
                    for field in ("transcription", "noise_reduction"):
                        if section.get(field) is not None:
                            raise ProtocolError(f"{field} is unavailable", "unsupported_feature", f"audio.input.{field}")
                    if "turn_detection" in section:
                        td_present, td = True, section["turn_detection"]
                elif section.get("voice", "alloy") != "alloy" or section.get("speed", 1) != 1:
                    raise ProtocolError("Only the native voice at speed 1 is supported", "unsupported_feature", "audio.output")
        if td_present:
            if td is None:
                candidate["turn_detection"] = None
            else:
                defaults = {"type": "server_vad", "threshold": 0.5, "prefix_padding_ms": 300,
                            "silence_duration_ms": 500, "create_response": True, "interrupt_response": True}
                if not isinstance(td, dict) or td.keys() - defaults.keys() or td.get("type") != "server_vad":
                    raise ProtocolError("Only server_vad or null is supported", "unsupported_feature", "turn_detection")
                defaults.update(td)
                for field, lo, hi in (("threshold", 0, 1), ("prefix_padding_ms", 0, 2000), ("silence_duration_ms", 100, 3000)):
                    value = defaults[field]
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not lo <= value <= hi:
                        raise ProtocolError(f"Invalid {field}", param=f"turn_detection.{field}")
                    if field != "threshold" and not isinstance(value, int):
                        raise ProtocolError(f"{field} must be an integer", param=f"turn_detection.{field}")
                for field in ("create_response", "interrupt_response"):
                    if not isinstance(defaults[field], bool):
                        raise ProtocolError(f"{field} must be boolean", param=f"turn_detection.{field}")
                if defaults["create_response"] and not defaults["interrupt_response"]:
                    raise ProtocolError("Native full-duplex speech interruption cannot be disabled; use manual turns", "unsupported_feature", "turn_detection.interrupt_response")
                candidate["turn_detection"] = defaults
        if (getattr(self, "backend_task", None) or getattr(self, "input_started", False)) and candidate != self.config:
            raise ProtocolError("Configure the session before sending audio; start a new session to reconfigure", "session_configuration_locked", "session")
        self.config = candidate

    def description(self):
        common = {"id": self.id, "object": "realtime.session", "type": "realtime",
                  "model": self.model, "instructions": self.config["instructions"],
                  "tools": [], "tool_choice": "none"}
        if self.beta:
            return {**common, "modalities": ["text", "audio"] if self.audio_mode else ["text"],
                    "input_audio_format": "pcm16", "output_audio_format": "pcm16", "voice": "alloy",
                    "input_audio_transcription": None, "turn_detection": self.config["turn_detection"]}
        return {**common, "output_modalities": self.config["output_modalities"],
                "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000},
                                     "turn_detection": self.config["turn_detection"],
                                     "transcription": None, "noise_reduction": None},
                          "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "alloy", "speed": 1}}}

    @property
    def audio_mode(self):
        return self.config["output_modalities"] == ["audio"]

    async def emit(self, event_type, **fields):
        async with self.send_lock:
            await self.ws.send_json({"type": event_type, "event_id": new_id("event_"), **fields})

    async def error(self, error, event_id=None):
        await self.emit("error", error={"type": "invalid_request_error", "code": error.code,
                                        "message": str(error), "param": error.param, "event_id": event_id})

    def start_backend(self):
        if self.backend_task is None:
            self.backend_task = asyncio.create_task(self.backend_loop())

    async def backend_loop(self):
        try:
            kwargs = {"max_size": 8 * 1024 * 1024, "proxy": None, "open_timeout": 10}
            if self.settings.gateway_url.startswith("wss:"):
                kwargs["ssl"] = self.settings.upstream_ssl
            async with connect(self.settings.gateway_url.rstrip("/") + "/ws/duplex/" + new_id("adx_rt_"), **kwargs) as upstream:
                self.backend = upstream
                async with asyncio.timeout(self.settings.prepare_timeout):
                    async for raw in upstream:
                        event = json.loads(raw)
                        kind = event.get("type")
                        if kind == "queue_done":
                            await upstream.send(json.dumps({"type": "prepare", "system_prompt": self.config["instructions"],
                                                            "config": {"generate_audio": self.audio_mode, "chunk_ms": 1000}}))
                        elif kind == "prepared":
                            self.ready.set()
                            break
                        elif kind in ("error", "timeout"):
                            raise RuntimeError("Gateway could not prepare the session")
                if not self.ready.is_set():
                    raise RuntimeError("Gateway closed before preparation")
                async for raw in upstream:
                    event = json.loads(raw)
                    async with self.lock:
                        await self.native_event(event)
                raise RuntimeError("Gateway connection closed")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Realtime upstream failure: %s", type(exc).__name__)
            # Never reflect upstream exception strings (addresses, paths, prompts) to public clients.
            async with self.lock:
                await self.finish("failed")
                await self.emit("error", error={"type": "server_error", "code": "upstream_unavailable",
                                                "message": "The duplex Gateway is unavailable or the session timed out."})
                await self.ws.close(code=1011)

    @property
    def manual_mode(self):
        td = self.config["turn_detection"]
        return td is None or not td["create_response"]

    async def manual_loop(self):
        """An explicitly committed item uses the existing turn-based path.

        Feeding a recording into forced full-duplex LISTEN can leave that model
        silent indefinitely. The native chat path consumes the whole item before
        generation and streams the same speech encoding, with explicit history.
        """
        try:
            kwargs = {"max_size": 8 * 1024 * 1024, "proxy": None, "open_timeout": 10}
            if self.settings.gateway_url.startswith("wss:"):
                kwargs["ssl"] = self.settings.upstream_ssl
            async with connect(self.settings.gateway_url.rstrip("/") + "/ws/chat", **kwargs) as upstream:
                self.backend = upstream
                await upstream.send(json.dumps({
                    "messages": [{"role": "system", "content": self.config["instructions"]}, *self.history],
                    "streaming": True, "tts": {"enabled": self.audio_mode},
                    "use_tts_template": self.audio_mode, "generation": {"max_new_tokens": 256},
                }))
                async with asyncio.timeout(self.settings.prepare_timeout):
                    async for raw in upstream:
                        event = json.loads(raw)
                        async with self.lock:
                            if event.get("type") == "chunk":
                                await self.native_event({"type": "result", "text": event.get("text_delta", ""),
                                                         "audio_data": event.get("audio_data", ""), "native_stream": True})
                            elif event.get("type") == "done":
                                if self.transcript:
                                    self.history.append({"role": "assistant", "content": self.transcript})
                                await self.finish()
                                return
                            elif event.get("type") == "error":
                                raise RuntimeError("Native chat failed")
                raise RuntimeError("Gateway closed before response completion")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Realtime manual response failure: %s", type(exc).__name__)
            async with self.lock:
                await self.finish("failed")
                await self.emit("error", error={"type": "server_error", "code": "upstream_unavailable",
                                                "message": "The Gateway could not complete the audio response."})

    async def pump(self):
        await self.ready.wait()
        deadline = time.monotonic()
        while True:
            # Bound in-flight inference as well as buffered input. Never queue an
            # unbounded amount of audio on the existing Worker WebSocket.
            if self.in_flight < 2:
                chunk = bytes(self.pcm[:48000])
                del self.pcm[:48000]
                chunk += b"\0" * (48000 - len(chunk))
                native = self.resampler.convert(chunk)
                force = self.force_listen
                self.force_listen = False
                self.in_flight += 1
                await self.backend.send(json.dumps({"type": "audio_chunk", "audio_base64": base64.b64encode(native).decode(),
                                                    "force_listen": force}))
            deadline += 1
            await asyncio.sleep(max(0, deadline - time.monotonic()))
            deadline = max(deadline, time.monotonic() - 1)

    async def commit(self, item_id=None):
        if not self.uncommitted_samples:
            raise ProtocolError("The input audio buffer is empty", "input_audio_buffer_commit_empty")
        item_id = item_id or new_id("item_")
        await self.emit("input_audio_buffer.committed", item_id=item_id, previous_item_id=self.previous_item)
        await self.emit("conversation.item.created", previous_item_id=self.previous_item,
                        item={"id": item_id, "object": "realtime.item", "type": "message", "role": "user",
                              "status": "completed", "content": [{"type": "input_audio"}]})
        self.previous_item = item_id
        self.uncommitted_samples = 0
        self.committed = True
        if self.manual:
            self.committed_pcm.extend(self.manual)
            self.manual.clear()

    async def vad(self, data):
        td = self.config["turn_detection"]
        if td is None:
            self.input_samples += len(data) // 2
            return
        self.vad_pending.extend(data)
        while len(self.vad_pending) >= 960:  # 20 ms frames, independent of packet boundaries
            frame = bytes(self.vad_pending[:960])
            del self.vad_pending[:960]
            samples = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768
            active = float(np.sqrt(np.mean(samples * samples))) > max(0.001, td["threshold"] * 0.04)
            position_ms = self.input_samples // 24
            self.input_samples += 480
            if active:
                self.silence_ms = 0
                if self.speech_id is None:
                    self.speech_id = new_id("item_")
                    await self.emit("input_audio_buffer.speech_started", item_id=self.speech_id,
                                    audio_start_ms=max(0, position_ms - td["prefix_padding_ms"]))
                    if td["interrupt_response"] and self.response:
                        await self.cancel(reason="turn_detected")
            elif self.speech_id:
                self.silence_ms += 20
                if self.silence_ms >= td["silence_duration_ms"]:
                    item_id = self.speech_id
                    self.speech_id = None
                    await self.emit("input_audio_buffer.speech_stopped", item_id=item_id,
                                    audio_end_ms=self.input_samples // 24)
                    # Count is tracked by append; multiple turns in one packet
                    # must still each create a committed user item.
                    self.uncommitted_samples = max(1, self.uncommitted_samples)
                    await self.commit(item_id)
                    if td["create_response"]:
                        self.response_allowed = True

    async def append(self, encoded):
        if not isinstance(encoded, str):
            raise ProtocolError("audio must be Base64 PCM16", param="audio")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            raise ProtocolError("Invalid Base64 audio", param="audio") from None
        if not data or len(data) % 2:
            raise ProtocolError("audio must contain whole PCM16 samples", param="audio")
        if len(self.pcm) + len(self.manual) + len(self.committed_pcm) + len(data) > 48000 * self.settings.max_buffer_seconds:
            raise ProtocolError("Input audio buffer limit exceeded; send audio at real-time speed", "audio_buffer_full", "audio")
        self.uncommitted_samples += len(data) // 2
        self.input_started = True
        td = self.config["turn_detection"]
        if td is not None and td["create_response"]:
            self.pcm.extend(data)
            self.start_backend()
        else:
            self.manual.extend(data)
        await self.vad(data)

    async def handle(self, event):
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            raise ProtocolError("An event object with a type is required", param="type")
        kind = event["type"]
        if kind == "session.update":
            self.update(event.get("session"))
            await self.emit("session.updated", session=self.description())
        elif kind == "input_audio_buffer.append":
            await self.append(event.get("audio"))
        elif kind == "input_audio_buffer.commit":
            await self.commit()
        elif kind == "input_audio_buffer.clear":
            # Audio already delivered to the model is committed context and
            # cannot be rewound, just as clearing an OpenAI buffer doesn't undo items.
            td = self.config["turn_detection"]
            if td is not None and td["create_response"]:
                self.pcm.clear()
            self.manual.clear()
            self.vad_pending.clear()
            self.speech_id = None
            self.uncommitted_samples = 0
            await self.emit("input_audio_buffer.cleared")
        elif kind == "response.create":
            params = event.get("response", {})
            if not isinstance(params, dict) or params.keys() - {"metadata"}:
                raise ProtocolError("Only response.metadata is supported; configure output in session.update", "unsupported_feature", "response")
            if self.response or self.response_allowed:
                raise ProtocolError("A response is already active", "conversation_already_has_active_response")
            if not self.committed:
                raise ProtocolError("Commit input audio before response.create", "input_audio_buffer_commit_required")
            if params.get("metadata") is not None and not isinstance(params["metadata"], dict):
                raise ProtocolError("metadata must be an object", param="response.metadata")
            if self.manual_mode and (self.history_bytes + len(self.committed_pcm) > 48000 * 90 or len(self.history) >= 32):
                raise ProtocolError("Manual conversation history is full; start a new session", "conversation_context_full")
            self.metadata = params.get("metadata")
            self.response_allowed = True
            self.committed = False
            await self.begin()
            if self.manual_mode:
                self.history_bytes += len(self.committed_pcm)
                native = AudioResampler().convert(bytes(self.committed_pcm))
                self.committed_pcm.clear()
                self.history.append({"role": "user", "content": [{"type": "audio", "data": base64.b64encode(native).decode()}]})
                self.backend_task = asyncio.create_task(self.manual_loop())
            else:
                self.start_backend()
        elif kind == "response.cancel":
            if not self.response:
                raise ProtocolError("No active response to cancel", "response_cancel_not_active")
            if event.get("response_id", self.response["id"]) != self.response["id"]:
                raise ProtocolError("Unknown response_id", param="response_id")
            await self.cancel()
        elif kind == "conversation.item.create":
            item = event.get("item", {})
            content = item.get("content", []) if isinstance(item, dict) else []
            if (not isinstance(item, dict) or item.get("type", "message") != "message" or item.get("role") != "user"
                    or not isinstance(content, list) or len(content) != 1 or not isinstance(content[0], dict)
                    or content[0].get("type") != "input_audio" or event.get("previous_item_id") is not None):
                raise ProtocolError("Only appended user input_audio items are supported", "unsupported_feature", "item")
            if self.config["turn_detection"] is not None:
                raise ProtocolError("Use turn_detection null for explicit audio items", "unsupported_feature", "item")
            if "id" in item and (not isinstance(item["id"], str) or not 1 <= len(item["id"]) <= 128):
                raise ProtocolError("item.id must be a string of 1-128 characters", param="item.id")
            await self.append(content[0].get("audio"))
            await self.commit(item.get("id"))
        else:
            raise ProtocolError(f"Unsupported client event: {kind}", "unsupported_event", "type")

    def part(self, done=False):
        return ({"type": "audio", "transcript": self.transcript if done else ""} if self.audio_mode
                else {"type": "text", "text": self.transcript if done else ""})

    def response_object(self, status):
        return {"id": self.response["id"], "object": "realtime.response", "status": status,
                "status_details": {"type": status, "reason": self.response.get("reason", "client_cancelled")} if status == "cancelled" else None,
                "output": [self.response["item"]] if status != "in_progress" else [],
                "conversation_id": self.conversation_id, "metadata": self.metadata,
                **({"modalities": ["text", "audio"] if self.audio_mode else ["text"], "output_audio_format": "pcm16", "voice": "alloy"}
                   if self.beta else {"output_modalities": self.config["output_modalities"],
                                     "audio": {"output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "alloy"}}}),
                "usage": None}

    async def begin(self):
        if self.response:
            return
        self.transcript = ""
        item = {"id": new_id("item_"), "object": "realtime.item", "type": "message", "role": "assistant",
                "status": "in_progress", "content": []}
        self.response = {"id": new_id("resp_"), "item": item}
        self.committed = False
        await self.emit("response.created", response=self.response_object("in_progress"))
        await self.emit("response.output_item.added", response_id=self.response["id"], output_index=0, item=copy.deepcopy(item))
        await self.emit("conversation.item.created", previous_item_id=self.previous_item, item=copy.deepcopy(item))
        self.previous_item = item["id"]
        await self.emit("response.content_part.added", **self.indices(), part=self.part())

    def indices(self):
        return {"response_id": self.response["id"], "item_id": self.response["item"]["id"],
                "output_index": 0, "content_index": 0}

    def audio_event(self, suffix):
        return ("response.audio." if self.beta else "response.output_audio.") + suffix

    def text_event(self, suffix):
        if self.audio_mode:
            return ("response.audio_transcript." if self.beta else "response.output_audio_transcript.") + suffix
        return ("response.text." if self.beta else "response.output_text.") + suffix

    async def finish(self, status="completed", reason=None):
        if not self.response:
            return
        if reason:
            self.response["reason"] = reason
        if self.audio_mode:
            await self.emit(self.audio_event("done"), **self.indices())
        await self.emit(self.text_event("done"), **self.indices(), **({"transcript": self.transcript} if self.audio_mode else {"text": self.transcript}))
        await self.emit("response.content_part.done", **self.indices(), part=self.part(True))
        item = self.response["item"]
        item["content"] = [self.part(True)]
        item["status"] = "completed" if status == "completed" else "incomplete"
        await self.emit("response.output_item.done", response_id=self.response["id"], output_index=0, item=item)
        await self.emit("response.done", response=self.response_object(status))
        self.response = None
        self.response_allowed = False
        self.metadata = None

    async def cancel(self, reason="client_cancelled"):
        if self.manual_mode:
            if self.backend_task and not self.backend_task.done():
                self.backend_task.cancel()
            if self.transcript:
                self.history.append({"role": "assistant", "content": self.transcript})
            await self.finish("cancelled", reason=reason)
            return
        await self.finish("cancelled", reason=reason)
        self.force_listen = True
        self.min_epoch = self.last_epoch + 1
        self.discard_until_listen = True
        if self.backend and self.ready.is_set():
            # The native Worker supports immediate muting, before the next full
            # input second arrives. force_listen also commits LISTEN on that unit.
            await self.backend.send(json.dumps({"type": "interrupt"}))

    async def native_event(self, event):
        kind = event.get("type")
        if kind == "result":
            self.in_flight = max(0, self.in_flight - 1)
        if kind in ("error", "timeout", "stopped"):
            raise RuntimeError("Native session failed")
        epoch = event.get("audio_epoch")
        if epoch is not None:
            self.last_epoch = max(self.last_epoch, epoch)
        if kind == "interrupted":
            await self.finish("cancelled", reason="turn_detected" if event.get("reason") == "foreground_speech" else "client_cancelled")
            self.min_epoch = max(self.min_epoch, epoch or 0)
            self.discard_until_listen = False
            return
        if epoch is not None and epoch < self.min_epoch:
            return
        if self.discard_until_listen:
            if kind == "result" and event.get("is_listen"):
                self.discard_until_listen = False
            return
        text = event.get("text", "") if kind == "result" else ""
        audio = event.get("audio_data", "") if kind in ("result", "audio_only") else ""
        if self.response_allowed and (text or audio):
            await self.begin()
            if text:
                self.transcript += text
                await self.emit(self.text_event("delta"), **self.indices(), delta=text)
            if audio and self.audio_mode:
                await self.emit(self.audio_event("delta"), **self.indices(), delta=output_pcm16(audio))
        if self.response and (event.get("speech_end") or
                              (kind == "result" and event.get("end_of_turn") and (not event.get("native_stream") or not self.audio_mode))):
            await self.finish()

    async def run(self):
        from fastapi import WebSocketDisconnect
        await self.emit("session.created", session=self.description())
        await self.emit("conversation.created", conversation={"id": self.conversation_id, "object": "realtime.conversation"})
        async def receive():
            while True:
                message = await self.ws.receive()
                if message["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect(message.get("code", 1000))
                raw = message.get("text")
                if raw is None:
                    await self.ws.close(code=1003, reason="Send JSON text events")
                    return
                if len(raw) > 2 * 1024 * 1024:
                    await self.ws.close(code=1009)
                    return
                event = None
                async with self.lock:
                    try:
                        event = json.loads(raw)
                        await self.handle(event)
                    except (json.JSONDecodeError, ProtocolError) as exc:
                        error = exc if isinstance(exc, ProtocolError) else ProtocolError("Invalid JSON", "invalid_json")
                        await self.error(error, event.get("event_id") if isinstance(event, dict) else None)
                        if error.code == "audio_buffer_full":
                            await self.ws.close(code=1009)
                            return
        pump_task = asyncio.create_task(self.pump())
        receive_task = asyncio.create_task(receive())
        try:
            async with asyncio.timeout(self.settings.max_session_seconds):
                done, _ = await asyncio.wait((pump_task, receive_task), return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
        except WebSocketDisconnect:
            pass
        except TimeoutError:
            await self.ws.close(code=1000, reason="Session time limit reached")
        except Exception as exc:
            logger.warning("Realtime transport failure: %s", type(exc).__name__)
            try:
                await self.ws.close(code=1011, reason="Realtime transport closed")
            except RuntimeError:
                pass
        finally:
            tasks = [pump_task, receive_task] + ([self.backend_task] if self.backend_task else [])
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

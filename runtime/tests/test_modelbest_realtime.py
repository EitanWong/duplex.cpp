"""ModelBest official wire examples, lifecycle, audio and public boundary checks."""
import asyncio
import base64
from io import BytesIO
import json
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from gateway_modules.modelbest_realtime import MODEL, ModelBestSession, decode_audio
from realtime import Settings, create_app

KEY = "test-only-modelbest-key-00000000000000000000"
URL = f"/v1/realtime?mode=audio&model={MODEL}"


def encoded(samples):
    return base64.b64encode(np.asarray(samples, dtype="<f4").tobytes()).decode()


def until(ws, kind):
    events = []
    for _ in range(30):
        event = ws.receive_json()
        events.append(event)
        if event["type"] == kind:
            return events
    raise AssertionError(f"Missing {kind}: {events}")


class Gateway:
    instances = []
    queued = False

    def __init__(self, *args, **kwargs):
        self.url = args[0]
        self.sent = []
        self.events = asyncio.Queue()
        self.closed = False
        self.instances.append(self)

    async def __aenter__(self):
        if self.queued:
            await self.events.put(json.dumps({"type": "queued", "position": 1, "estimated_wait_s": 12, "queue_length": 1}))
        await self.events.put(json.dumps({"type": "queue_done"}))
        return self

    async def __aexit__(self, *args):
        self.closed = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.events.get()

    async def send(self, raw):
        msg = json.loads(raw)
        self.sent.append(msg)
        event = msg["type"]
        if event == "prepare":
            await self.events.put(json.dumps({"type": "prepared"}))
        elif event == "audio_chunk":
            await self.events.put(json.dumps({"type": "result", "text": "你好", "is_listen": True,
                                               "end_of_turn": True, "native_stream": True, "audio_epoch": 1}))
            await self.events.put(json.dumps({"type": "audio_only", "audio_data": encoded([.25, -.25]),
                                               "speech_end": True, "audio_epoch": 1}))
        elif event == "interrupt":
            await self.events.put(json.dumps({"type": "interrupted", "audio_epoch": 1}))
        elif event in ("pause", "resume", "stop"):
            await self.events.put(json.dumps({"type": {"pause": "paused", "resume": "resumed", "stop": "stopped"}[event]}))


@pytest.fixture
def client(monkeypatch):
    Gateway.instances = []
    Gateway.queued = False
    monkeypatch.setattr("gateway_modules.modelbest_realtime.connect", Gateway)
    with TestClient(create_app(Settings(api_key=KEY))) as client:
        yield client


def connect(client, url=URL):
    return client.websocket_connect(url, headers={"Authorization": "Bearer " + KEY})


def initialize(ws, payload=None):
    ws.send_json({"type": "session.init", "payload": payload or {
        "system_prompt": "请用自然、简洁的口语回答。", "config": {"temperature": .7, "tts_enabled": True}}})
    return until(ws, "session.created")[-1]


def test_official_browser_auth_queue_init_and_float32_output(client):
    Gateway.queued = True
    with client.websocket_connect(URL, subprotocols=["map.realtime", "map-api-key." + KEY]) as ws:
        assert ws.accepted_subprotocol == "map.realtime"
        queued = ws.receive_json()
        assert queued["type"] == "session.queued" and queued["position"] == 1
        assert ws.receive_json()["type"] == "session.queue_done"
        created = initialize(ws)
        assert created["model"] == MODEL and created["mode"] == "audio"
        assert created["session_id"].startswith("adx_map_")
        pcm = np.linspace(-.5, .5, 16000, dtype="<f4")
        ws.send_json({"type": "input.append", "input": {"audio": encoded(pcm)}})
        events = until(ws, "response.output_audio.delta")
        assert [event["type"] for event in events] == ["response.output_text.delta", "response.listen", "response.output_audio.delta"]
        assert events[0]["text"] == "你好" and "delta" not in events[0]
        assert "text" not in events[1]  # assistant text must not become input transcription
        assert np.array_equal(np.frombuffer(base64.b64decode(events[-1]["audio"]), dtype="<f4"), [.25, -.25])
        assert events[-1]["sample_rate"] == 24000 and events[-1]["speech_end"]
        native = next(event for event in Gateway.instances[0].sent if event["type"] == "audio_chunk")
        assert base64.b64decode(native["audio_base64"]) == pcm.tobytes()  # no format conversion
        ws.send_json({"type": "session.close", "reason": "user_stop"})
        assert until(ws, "session.closed")[-1]["reason"] == "client_closed"
        assert ws.receive()["code"] == 1000
    assert Gateway.instances[0].closed
    assert client.app.state.active_connections == 0


def test_early_init_is_buffered_until_worker_assignment(client):
    with connect(client) as ws:
        created = initialize(ws)
        assert created["type"] == "session.created"
        assert len([e for e in Gateway.instances[0].sent if e["type"] == "prepare"]) == 1


def test_qwen_event_format_groups_text_and_late_audio_in_one_response(client):
    with connect(client, URL + "&protocol=qwen-audio-agent") as ws:
        initialize(ws)
        ws.send_json({"type": "input.append", "input": {"audio": encoded([0] * 16000)}})
        events = [ws.receive_json() for _ in range(3)]
        assert all(event["type"] == "response.output.delta" for event in events)
        assert [event["kind"] for event in events] == ["text", "audio", "listen"]
        assert len({event["response_id"] for event in events}) == 1
        assert events[0]["response_id"].startswith("resp_")
        assert events[0]["text"] == "你好" and events[1]["audio"] == encoded([.25, -.25])
        ws.send_json({"type": "session.close"})
        until(ws, "session.closed")


def test_unknown_or_mismatched_protocol_is_rejected(client):
    for url in (URL + "&protocol=unknown", "/v1/realtime?model=gpt-realtime&protocol=qwen-audio-agent"):
        with pytest.raises(Exception) as error:
            with connect(client, url):
                pass
        assert error.value.status_code == 400


@pytest.mark.asyncio
async def test_qwen_response_boundary_handles_empty_final_audio_and_interrupt():
    class Socket:
        def __init__(self):
            self.events = []
        async def send_json(self, event):
            self.events.append(event)
    ws = Socket()
    session = ModelBestSession(ws, Settings(api_key=KEY), protocol="qwen-audio-agent")
    await session.native_event({"type": "audio_only", "speech_end": True})
    assert not ws.events and session.response_id is None
    await session.native_event({"type": "result", "text": "hello", "is_listen": True, "native_stream": True})
    assert [event["kind"] for event in ws.events] == ["text"]
    await session.native_event({"type": "audio_only", "audio_data": encoded([.1]), "speech_end": False})
    await session.native_event({"type": "audio_only", "speech_end": True})
    assert [event["kind"] for event in ws.events] == ["text", "audio", "listen"]
    assert len({event["response_id"] for event in ws.events}) == 1
    await session.native_event({"type": "result", "text": "next", "audio_epoch": 1})
    await session.native_event({"type": "interrupted", "audio_epoch": 2})
    assert ws.events[-2]["kind"] == "listen"
    assert ws.events[-1]["type"] == "response.interrupted"
    assert ws.events[-3]["response_id"] != ws.events[0]["response_id"]


def test_queued_close_releases_gateway_without_preparing(client, monkeypatch):
    class QueuedGateway(Gateway):
        async def __aenter__(self):
            await self.events.put(json.dumps({"type": "queued", "position": 1}))
            return self
    monkeypatch.setattr("gateway_modules.modelbest_realtime.connect", QueuedGateway)
    with connect(client) as ws:
        assert ws.receive_json()["type"] == "session.queued"
        ws.send_json({"type": "session.init", "payload": {}})
        ws.send_json({"type": "session.close"})
        assert ws.receive_json()["type"] == "session.closed"
        assert ws.receive()["code"] == 1000
    assert Gateway.instances[0].sent == [] and Gateway.instances[0].closed
    assert client.app.state.active_connections == 0


@pytest.mark.parametrize("raw", [b"binary-pcm", "{invalid", "[]"])
def test_binary_and_invalid_json_close_cleanly(client, raw):
    with connect(client) as ws:
        until(ws, "session.queue_done")
        ws.send_bytes(raw) if isinstance(raw, bytes) else ws.send_text(raw)
        assert until(ws, "error")[-1]["error"]["type"] == "invalid_request_error"
        assert ws.receive_json()["type"] == "session.closed"
        assert ws.receive()["code"] == 1008


def test_short_packets_are_joined_without_synthetic_silence(client):
    with connect(client) as ws:
        initialize(ws)
        packet = np.full(1600, .2, dtype="<f4")
        for _ in range(10):
            ws.send_json({"type": "input.append", "input": {"audio": encoded(packet)}})
        until(ws, "response.output_audio.delta")
        chunks = [e for e in Gateway.instances[0].sent if e["type"] == "audio_chunk"]
        assert len(chunks) == 1
        assert base64.b64decode(chunks[0]["audio_base64"]) == packet.tobytes() * 10


def test_video_default_mode_and_jpeg_mapping(client):
    image = BytesIO()
    Image.new("RGB", (16, 16), "red").save(image, format="JPEG")
    frame = base64.b64encode(image.getvalue()).decode()
    with connect(client, f"/v1/realtime?model={MODEL}") as ws:
        assert initialize(ws, {"config": {"vision_hd": True}})["mode"] == "video"
        assert Gateway.instances[0].url.split("/")[-1].startswith("omni_map_")
        ws.send_json({"type": "input.append", "input": {"audio": encoded([0] * 16000), "video_frames": [frame]}})
        until(ws, "response.output_audio.delta")
        native = Gateway.instances[0].sent[-1]
        assert native["frame_base64_list"] == [frame] and native["max_slice_nums"] == 3


def test_pause_resume_interrupt_and_close(client):
    with connect(client) as ws:
        initialize(ws)
        ws.send_json({"type": "session.pause"})
        assert ws.receive_json()["type"] == "session.paused"
        ws.send_json({"type": "session.resume"})
        assert ws.receive_json()["type"] == "session.resumed"
        ws.send_json({"type": "session.interrupt"})
        assert ws.receive_json()["type"] == "response.interrupted"
        ws.send_json({"type": "input.append", "input": {"audio": encoded([0] * 16000)}})
        until(ws, "response.output_audio.delta")
        assert Gateway.instances[0].sent[-1]["force_listen"]
        ws.send_json({"type": "session.close"})
        until(ws, "session.closed")


def test_voice_refs_and_temperature_are_forwarded(client):
    ref1, ref2 = encoded([.1] * 100), encoded([.2] * 100)
    with connect(client) as ws:
        initialize(ws, {"system_prompt": "test", "config": {"temperature": .4, "length_penalty": 1.2},
                        "voice": {"ref_audio_base64": ref1, "tts_ref_audio_base64": ref2}})
        msg = Gateway.instances[0].sent[0]
        assert msg["ref_audio_base64"] == ref1 and msg["tts_ref_audio_base64"] == ref2
        assert msg["config"]["temperature"] == .4 and msg["config"]["length_penalty"] == 1.2


@pytest.mark.parametrize("input", [
    {"audio": "not-base64"}, {"audio": base64.b64encode(b"xx").decode()},
    {"audio": encoded([np.nan])}, {"audio": encoded([2.0])},
    {"audio": encoded([0]), "video_frames": ["invalid-jpeg"]},
    {"audio": encoded([0]), "force_listen": "yes"},
])
def test_invalid_input_is_rejected_and_closed(client, input):
    with connect(client) as ws:
        initialize(ws)
        ws.send_json({"type": "input.append", "input": input})
        assert ws.receive_json()["type"] == "error"
        assert ws.receive_json()["type"] == "session.closed"
        assert ws.receive()["code"] == 1008


def test_input_before_created_and_duplicate_init_are_errors(client):
    for event in ({"type": "input.append", "input": {"audio": encoded([0])}},
                  {"type": "session.init", "payload": {}}):
        with connect(client) as ws:
            until(ws, "session.queue_done")
            if event["type"] == "session.init":
                initialize(ws)
            ws.send_json(event)
            assert until(ws, "error")[-1]["error"]["code"] == "invalid_state"
            until(ws, "session.closed")


def test_model_required_and_unsupported_mode_and_auth(client):
    for url, expected in (("/v1/realtime?mode=audio", 400),
                          (f"/v1/realtime?mode=chat&model={MODEL}", 400)):
        with pytest.raises(Exception) as error:
            with connect(client, url):
                pass
        assert error.value.status_code == expected
    with pytest.raises(Exception) as error:
        with client.websocket_connect(URL, subprotocols=["map.realtime", "map-api-key.wrong"]):
            pass
    assert error.value.status_code == 401


@pytest.mark.asyncio
async def test_backpressure_epochs_and_close_before_init():
    class Socket:
        def __init__(self):
            self.events = []
        async def send_json(self, event):
            await asyncio.sleep(0)  # exercise concurrent stopped/deadline close
            self.events.append(event)
        async def close(self, **kwargs):
            pass
    settings = Settings(api_key=KEY, max_buffer_seconds=1)
    ws = Socket()
    session = ModelBestSession(ws, settings, mode="audio")
    session.created.set()
    session.prepare = {"max_slice_nums": 1}
    await session.append({"audio": encoded([0] * 16000)})
    from gateway_modules.realtime import ProtocolError
    with pytest.raises(ProtocolError, match="buffer full"):
        await session.append({"audio": encoded([0])})
    session.min_epoch = 2
    await session.native_event({"type": "audio_only", "audio_data": "old", "audio_epoch": 1})
    assert not ws.events
    await session.native_event({"type": "audio_only", "audio_data": "new", "audio_epoch": 2})
    assert ws.events[-1]["audio"] == "new"
    await asyncio.gather(session.close(), session.close())
    assert len([e for e in ws.events if e["type"] == "session.closed"]) == 1


def test_unsupported_native_budget_controls_are_explicit(client):
    with connect(client) as ws:
        ws.send_json({"type": "session.init", "payload": {"config": {"max_tokens": 1024}}})
        error = until(ws, "error")[-1]["error"]
        assert error["code"] == "unsupported_feature" and error["param"] == "payload.config.max_tokens"


def test_native_init_keeps_independent_refs_and_tts_flag(monkeypatch):
    from core.processors.m5_duplex import NativeDuplex
    socket = Mock()
    socket.recv.return_value = json.dumps({"type": "session.created", "session_id": "native"})
    monkeypatch.setattr("core.processors.m5_duplex.connect", lambda *args, **kwargs: socket)
    monkeypatch.setattr("core.processors.m5_duplex.threading.Thread", Mock())
    monkeypatch.setattr("core.audio.load_mono_audio", lambda path: np.array([.1 if path == "llm.wav" else .2], dtype="<f4"))
    owner = SimpleNamespace(ref_audio_path="llm.wav", _cpp_server_url="http://localhost:19080",
                            _maybe_update_kv_cache_length=Mock())
    NativeDuplex(owner, "test", "llm.wav", {"use_tts": False}, 1.1, tts_ref_audio_path="tts.wav")
    payload = json.loads(socket.send.call_args.args[0])["payload"]
    assert payload["use_tts"] is False
    assert payload["voice"]["ref_audio"] == encoded([.1])
    assert payload["voice"]["tts_ref_audio"] == encoded([.2])

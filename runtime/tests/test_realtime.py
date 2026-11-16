"""Protocol, audio and public-boundary checks without loading model weights."""
import asyncio
import base64
import json

from fastapi.testclient import TestClient
import numpy as np
import pytest

from gateway_modules.realtime import AudioResampler, ProtocolError, RealtimeSession, output_pcm16
from realtime import Settings, create_app, load_key


KEY = "test-only-realtime-key-00000000000000000000"


@pytest.fixture
def settings():
    return Settings(api_key=KEY)


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as client:
        yield client


def headers():
    return {"Authorization": f"Bearer {KEY}"}


def connect(client, **kwargs):
    return client.websocket_connect("/v1/realtime?mode=audio&model=gpt-realtime", headers=headers(), **kwargs)


def receive_until(ws, kind):
    events = []
    for _ in range(40):
        event = ws.receive_json()
        events.append(event)
        if event["type"] == kind:
            return events
    raise AssertionError(f"Missing {kind}: {events}")


def audio(samples):
    return base64.b64encode(np.asarray(samples, dtype="<i2").tobytes()).decode()


class FakeGateway:
    instances = []

    def __init__(self, *args, **kwargs):
        self.events = asyncio.Queue()
        self.sent = []
        self.closed = False
        self.chat = args[0].endswith("/ws/chat")
        self.instances.append(self)

    async def __aenter__(self):
        if not self.chat:
            await self.events.put(json.dumps({"type": "queue_done"}))
        return self

    async def __aexit__(self, *args):
        self.closed = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.events.get()

    async def send(self, raw):
        event = json.loads(raw)
        self.sent.append(event)
        if self.chat:
            await self.events.put(json.dumps({"type": "chunk", "text_delta": "你好", "audio_data": base64.b64encode(
                np.array([0, .5, -.5], dtype="<f4").tobytes()).decode()}))
            await self.events.put(json.dumps({"type": "done", "text": "你好"}))
        elif event["type"] == "prepare":
            await self.events.put(json.dumps({"type": "prepared"}))
        elif event["type"] == "audio_chunk":
            await self.events.put(json.dumps({"type": "result", "text": "你好", "is_listen": False,
                                               "native_stream": True, "audio_epoch": 0}))
            await self.events.put(json.dumps({"type": "audio_only", "audio_data": base64.b64encode(
                np.array([0, .5, -.5], dtype="<f4").tobytes()).decode(), "speech_end": True, "audio_epoch": 0}))


@pytest.fixture
def gateway(monkeypatch):
    FakeGateway.instances = []
    monkeypatch.setattr("gateway_modules.realtime.connect", FakeGateway)
    return FakeGateway


def test_only_public_api_and_auth(client):
    assert client.get("/health").json()["api"] == "realtime"
    for path in ("/admin", "/workers", "/api/sessions", "/docs", "/openapi.json", "/static/lan-ca.crt"):
        assert client.get(path).status_code == 404
    assert client.get("/v1/models").status_code == 401
    assert "duplex" in [m["id"] for m in client.get("/v1/models", headers=headers()).json()["data"]]
    with pytest.raises(Exception) as rejected:
        with client.websocket_connect("/v1/realtime?api_key=" + KEY):
            pass
    assert rejected.value.status_code == 401


def test_session_created_update_and_atomic_rejection(client):
    with connect(client) as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"
        assert created["session"]["audio"]["input"]["format"]["rate"] == 24000
        assert ws.receive_json()["type"] == "conversation.created"
        ws.send_json({"type": "session.update", "session": {"type": "realtime", "instructions": "说中文"}})
        assert ws.receive_json()["session"]["instructions"] == "说中文"
        ws.send_json({"type": "session.update", "event_id": "bad-update", "session": {
            "instructions": "must not apply", "audio": {"input": {"format": {"type": "audio/pcmu"}}}}})
        error = ws.receive_json()["error"]
        assert error["code"] == "unsupported_feature" and error["event_id"] == "bad-update"
        ws.send_json({"type": "session.update", "session": {}})
        assert ws.receive_json()["session"]["instructions"] == "说中文"
        ws.send_text("invalid json")
        assert ws.receive_json()["error"]["code"] == "invalid_json"
        ws.send_json({"type": "input_audio_buffer.append", "audio": "AAAAA"})
        assert ws.receive_json()["error"]["param"] == "audio"
        ws.send_json({"type": "conversation.item.truncate"})
        assert ws.receive_json()["error"]["code"] == "unsupported_event"


@pytest.mark.parametrize("beta", [False, True])
def test_vad_round_trip_event_lifecycle_and_audio(client, gateway, beta):
    hdrs = headers()
    if beta:
        hdrs["OpenAI-Beta"] = "realtime=v1"
    with client.websocket_connect("/v1/realtime?model=gpt-realtime", headers=hdrs) as ws:
        receive_until(ws, "conversation.created")
        ws.send_json({"type": "input_audio_buffer.append", "audio": audio([8000] * 4800 + [0] * 24000)})
        events = receive_until(ws, "response.done")
        kinds = [e["type"] for e in events]
        assert kinds[:4] == ["input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped",
                             "input_audio_buffer.committed", "conversation.item.created"]
        delta_name = "response.audio.delta" if beta else "response.output_audio.delta"
        delta = next(e for e in events if e["type"] == delta_name)
        assert np.array_equal(np.frombuffer(base64.b64decode(delta["delta"]), dtype="<i2"), [0, 16384, -16384])
        done = events[-1]["response"]
        assert done["status"] == "completed"
        assert done["output"][0]["content"][0]["transcript"] == "你好"
        assert done["id"] == delta["response_id"]
        native = next(e for e in gateway.instances[0].sent if e["type"] == "audio_chunk")
        assert len(base64.b64decode(native["audio_base64"])) == 16000 * 4
        assert len({e["event_id"] for e in events}) == len(events)
    assert gateway.instances[0].closed


def test_manual_commit_create_and_clear(client, gateway):
    with connect(client) as ws:
        receive_until(ws, "conversation.created")
        ws.send_json({"type": "session.update", "session": {"audio": {"input": {"turn_detection": None}}}})
        assert ws.receive_json()["type"] == "session.updated"
        ws.send_json({"type": "input_audio_buffer.commit"})
        assert ws.receive_json()["error"]["code"] == "input_audio_buffer_commit_empty"
        ws.send_json({"type": "input_audio_buffer.append", "audio": audio([1000] * 24000)})
        ws.send_json({"type": "input_audio_buffer.commit"})
        receive_until(ws, "conversation.item.created")
        assert not gateway.instances  # no Worker reserved before response.create
        ws.send_json({"type": "response.create", "response": {"metadata": {"test": "manual"}}})
        events = receive_until(ws, "response.done")
        assert events[-1]["response"]["metadata"] == {"test": "manual"}
        ws.send_json({"type": "session.update", "session": {"instructions": "new"}})
        assert ws.receive_json()["error"]["code"] == "session_configuration_locked"
        ws.send_json({"type": "input_audio_buffer.clear"})
        assert ws.receive_json()["type"] == "input_audio_buffer.cleared"


def test_browser_ephemeral_secret_origin_and_config(settings):
    settings.allowed_origins = ("https://client.example",)
    with TestClient(create_app(settings)) as client:
        assert client.post("/v1/realtime/client_secrets").status_code == 401
        minted = client.post("/v1/realtime/client_secrets", headers=headers(), json={
            "session": {"type": "realtime", "model": "duplex", "instructions": "secret config"}})
        assert minted.status_code == 200 and minted.headers["cache-control"] == "no-store"
        token = minted.json()["value"]
        assert token.startswith("ek_")
        with client.websocket_connect("/v1/realtime", headers={"Origin": "https://client.example"},
                                      subprotocols=["realtime", "openai-insecure-api-key." + token]) as ws:
            assert ws.accepted_subprotocol == "realtime"
            session = ws.receive_json()["session"]
            assert session["instructions"] == "secret config" and session["model"] == "duplex"
        with pytest.raises(Exception) as rejected:
            with client.websocket_connect("/v1/realtime", headers={**headers(), "Origin": "https://evil.example"}):
                pass
        assert rejected.value.status_code == 403
        token = client.post("/v1/realtime/sessions", headers=headers(), json={}).json()["client_secret"]["value"]
        with client.websocket_connect("/v1/realtime", headers={"Authorization": "Bearer " + token}) as ws:
            assert ws.receive_json()["type"] == "session.created"


def test_limits_and_validation(settings, monkeypatch):
    settings.max_connections = 1
    with TestClient(create_app(settings)) as client:
        with connect(client):
            with pytest.raises(Exception) as rejected:
                with connect(client):
                    pass
            assert rejected.value.status_code == 429
        with connect(client) as ws:
            receive_until(ws, "conversation.created")
        secret = client.post("/v1/realtime/client_secrets", headers=headers(), json={}).json()["value"]
        monkeypatch.setattr("realtime.time.time", lambda: 10**12)
        with pytest.raises(Exception) as rejected:
            with client.websocket_connect("/v1/realtime", headers={"Authorization": "Bearer " + secret}):
                pass
        assert rejected.value.status_code == 401


def test_streaming_resampler_boundary_invariance_and_antialias():
    rng = np.random.default_rng(42)
    data = rng.integers(-20000, 20000, 24000, dtype=np.int16).astype("<i2").tobytes()
    whole = AudioResampler().convert(data)
    converter = AudioResampler()
    packets = b"".join(converter.convert(data[start:start + 334]) for start in range(0, len(data), 334))
    assert packets == whole
    assert len(whole) == 16000 * 4
    t = np.arange(24000) / 24000
    high = (10000 * np.sin(2 * np.pi * 11000 * t)).astype("<i2").tobytes()
    result = np.frombuffer(AudioResampler().convert(high), dtype="<f4")[100:]
    assert np.sqrt(np.mean(result ** 2)) < .002
    clipped = output_pcm16(base64.b64encode(np.array([-2, 2, np.nan], dtype="<f4").tobytes()).decode())
    assert list(np.frombuffer(base64.b64decode(clipped), dtype="<i2")) == [-32768, 32767, 0]


@pytest.mark.asyncio
async def test_cancel_discards_late_audio_and_keeps_epochs(settings):
    class Socket:
        def __init__(self):
            self.events = []

        async def send_json(self, event):
            self.events.append(event)

    ws = Socket()
    session = RealtimeSession(ws, settings)
    session.response_allowed = True
    await session.native_event({"type": "result", "text": "old", "audio_epoch": 3})
    await session.cancel()
    assert ws.events[-1]["response"]["status"] == "cancelled"
    before = len(ws.events)
    await session.native_event({"type": "audio_only", "audio_data": "stale-invalid-audio", "audio_epoch": 3})
    assert len(ws.events) == before
    await session.native_event({"type": "interrupted", "audio_epoch": 4})
    session.response_allowed = True
    await session.native_event({"type": "result", "text": "new", "audio_epoch": 4})
    assert ws.events[-1]["delta"] == "new"


def test_generated_key_is_stable_and_local(tmp_path, monkeypatch):
    monkeypatch.delenv("MODELBEST_API_KEY", raising=False)
    monkeypatch.delenv("DUPLEX_REALTIME_API_KEY", raising=False)
    path = tmp_path / "private" / "api-key"
    first = load_key(path)
    assert len(first) >= 32 and first == load_key(path)
    monkeypatch.setenv("DUPLEX_REALTIME_API_KEY", "override" * 8)
    assert load_key(path) == "override" * 8


def test_settings_reject_insecure_values():
    with pytest.raises(ValueError):
        Settings(api_key="short")
    with pytest.raises(ValueError):
        Settings(api_key=KEY, gateway_url="https://localhost:8040")


@pytest.mark.asyncio
async def test_manual_clear_keeps_committed_audio(settings):
    class Socket:
        async def send_json(self, event):
            pass
    session = RealtimeSession(Socket(), settings, initial={"turn_detection": None})
    await session.append(audio([1000] * 2400))
    await session.commit()
    await session.append(audio([2000] * 2400))
    await session.handle({"type": "input_audio_buffer.clear"})
    assert not session.manual
    assert len(session.committed_pcm) == 4800
    assert session.committed
    session.pcm.extend(session.committed_pcm)
    session.committed_pcm.clear()
    await session.handle({"type": "input_audio_buffer.clear"})
    assert len(session.pcm) == 4800  # active response input is already committed


@pytest.mark.asyncio
async def test_native_turn_end_waits_for_audio_and_retains_text(settings):
    class Socket:
        def __init__(self):
            self.events = []
        async def send_json(self, event):
            self.events.append(event)
    ws = Socket()
    session = RealtimeSession(ws, settings)
    session.response_allowed = True
    await session.native_event({"type": "result", "text": "last words", "is_listen": True,
                                "end_of_turn": True, "native_stream": True, "audio_epoch": 0})
    assert session.response
    assert ws.events[-1]["delta"] == "last words"
    await session.native_event({"type": "audio_only", "speech_end": True, "audio_epoch": 0})
    assert ws.events[-1]["type"] == "response.done"
    assert ws.events[-1]["response"]["output"][0]["content"][0]["transcript"] == "last words"


@pytest.mark.asyncio
async def test_manual_cancel_closes_generation_and_preserves_partial_text(settings):
    class Socket:
        def __init__(self):
            self.events = []
        async def send_json(self, event):
            self.events.append(event)
    ws = Socket()
    session = RealtimeSession(ws, settings, initial={"turn_detection": None})
    session.response_allowed = True
    await session.begin()
    session.transcript = "partial"
    session.backend_task = asyncio.create_task(asyncio.Event().wait())
    task = session.backend_task
    await session.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert task.cancelled()
    assert session.history[-1] == {"role": "assistant", "content": "partial"}
    assert ws.events[-1]["response"]["status"] == "cancelled"


def test_manual_turns_include_entire_audio_and_previous_history(client, gateway):
    with connect(client) as ws:
        receive_until(ws, "conversation.created")
        ws.send_json({"type": "session.update", "session": {"turn_detection": None}})
        ws.receive_json()
        for _ in range(2):
            ws.send_json({"type": "input_audio_buffer.append", "audio": audio([1000] * 24000)})
            ws.send_json({"type": "input_audio_buffer.commit"})
            receive_until(ws, "conversation.item.created")
            ws.send_json({"type": "response.create"})
            assert receive_until(ws, "response.done")[-1]["response"]["status"] == "completed"
        request = gateway.instances[-1].sent[0]
        assert [m["role"] for m in request["messages"]] == ["system", "user", "assistant", "user"]
        assert len(base64.b64decode(request["messages"][-1]["content"][0]["data"])) == 64000


def test_bad_upstream_closes_and_releases_connection(client, monkeypatch):
    def failing(*args, **kwargs):
        raise ConnectionError("internal detail must not be reflected")
    monkeypatch.setattr("gateway_modules.realtime.connect", failing)
    with connect(client) as ws:
        receive_until(ws, "conversation.created")
        ws.send_json({"type": "input_audio_buffer.append", "audio": audio([0] * 2400)})
        error = receive_until(ws, "error")[-1]
        assert error["error"]["code"] == "upstream_unavailable"
        assert "internal detail" not in str(error)
        assert ws.receive()["code"] == 1011
    assert client.app.state.active_connections == 0


def test_binary_frames_close_cleanly(client):
    with connect(client) as ws:
        receive_until(ws, "conversation.created")
        ws.send_bytes(b"not a JSON event")
        assert ws.receive()["code"] == 1003
    assert client.app.state.active_connections == 0


def test_buffer_overflow_is_bounded(settings):
    settings.max_buffer_seconds = 1
    with TestClient(create_app(settings)) as client:
        with connect(client) as ws:
            receive_until(ws, "conversation.created")
            ws.send_json({"type": "session.update", "session": {"turn_detection": None}})
            ws.receive_json()
            ws.send_json({"type": "input_audio_buffer.append", "audio": audio([0] * 24001)})
            assert ws.receive_json()["error"]["code"] == "audio_buffer_full"
            assert ws.receive()["code"] == 1009


def test_secret_body_size_and_unsupported_features(client):
    assert client.post("/v1/realtime/client_secrets", headers=headers(), content="x" * 32769).status_code == 413
    for patch in ({"tools": [{"type": "function", "name": "fake"}]},
                  {"input_audio_transcription": {"model": "whisper-1"}},
                  {"audio": {"input": {"turn_detection": {"type": "semantic_vad"}}}},
                  {"turn_detection": {"type": "server_vad", "interrupt_response": False}},
                  {"audio": {"output": {"voice": "marin"}}}):
        reply = client.post("/v1/realtime/client_secrets", headers=headers(), json={"session": patch})
        assert reply.status_code == 400 and reply.json()["error"]["code"] == "unsupported_feature"

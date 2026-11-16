"""A disconnected queue client must never retain an assigned Worker."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi import WebSocketDisconnect
import pytest

import gateway
from gateway_modules.models import GatewayWorkerStatus
from gateway_modules.worker_pool import WorkerPool


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", [
    "immediate", "assigned_before_queue_done", "assigned_during_disconnect",
    "assigned_during_cancel", "waiting_disconnect",
])
async def test_disconnected_client_releases_only_its_reservation(monkeypatch, scenario):
    pool = WorkerPool(["127.0.0.1:22440"])
    worker = next(iter(pool.workers.values()))
    worker.status = GatewayWorkerStatus.IDLE
    if scenario != "immediate":
        worker.mark_busy(GatewayWorkerStatus.DUPLEX_ACTIVE, "audio_duplex", "occupied")
    monkeypatch.setattr(gateway, "worker_pool", pool)
    monkeypatch.setattr(gateway, "app_registry", SimpleNamespace(is_enabled=lambda app: True))
    connect = Mock(side_effect=AssertionError("Disconnected client must not connect to a Worker"))
    monkeypatch.setattr("websockets.connect", connect)

    class Socket:
        async def accept(self):
            pass

        async def send_json(self, event):
            if event["type"] == "queued":
                if scenario != "waiting_disconnect":
                    pool.release_worker(worker)  # dispatch wins before disconnect handling
                if scenario == "assigned_before_queue_done":
                    return
                if scenario == "assigned_during_cancel":
                    raise asyncio.CancelledError()
            raise WebSocketDisconnect(1006)

    try:
        if scenario == "assigned_during_cancel":
            with pytest.raises(asyncio.CancelledError):
                await gateway.duplex_ws(Socket(), "adx_cleanup")
        else:
            await gateway.duplex_ws(Socket(), "adx_cleanup")
        assert pool.queue_length == 0
        connect.assert_not_called()
        if scenario == "waiting_disconnect":
            assert worker.current_session_id == "occupied"
            assert worker.status == GatewayWorkerStatus.DUPLEX_ACTIVE
        else:
            assert worker.current_session_id is None
            assert worker.status in (GatewayWorkerStatus.IDLE, GatewayWorkerStatus.LOADING)
    finally:
        await pool.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["disconnect", "stop", "failed_prepare"])
async def test_worker_resets_only_after_prepare_started(monkeypatch, scenario):
    import worker as worker_module
    fake_worker = SimpleNamespace(
        processor=None, _http_client=object(),
        state=worker_module.WorkerState(status=worker_module.WorkerStatus.IDLE),
        duplex_stop=Mock(), full_reinit=Mock(),
    )
    monkeypatch.setattr(worker_module, "worker", fake_worker)

    class Socket:
        query_params = {"session_id": "adx_cleanup"}

        async def accept(self):
            pass

        async def receive_text(self):
            if scenario == "stop":
                return json.dumps({"type": "stop"})
            if scenario == "failed_prepare":
                return json.dumps({"type": "prepare", "config": {"length_penalty": "invalid"}})
            raise WebSocketDisconnect(1000)

        async def send_json(self, event):
            assert event["type"] == "stopped"

    await worker_module.duplex_ws(Socket())
    assert fake_worker.state.status == worker_module.WorkerStatus.IDLE
    assert fake_worker.state.current_session_id is None
    assert fake_worker.full_reinit.call_count == (1 if scenario == "failed_prepare" else 0)
    assert fake_worker.duplex_stop.call_count == (1 if scenario == "failed_prepare" else 0)

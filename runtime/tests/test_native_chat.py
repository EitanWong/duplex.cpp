"""Regression coverage for structured browser chat and reference audio loading."""
import base64
import json
from types import SimpleNamespace
from unittest.mock import Mock
import threading

import numpy as np
import pytest
import soundfile as sf

from core.audio import load_mono_audio
from core.processors.cpp_backend import CppBackendWorker
from core.processors.native_chat import stream_chat


def test_reference_audio_preserves_mono_16k(tmp_path):
    samples = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
    path = tmp_path / 'ref.wav'
    sf.write(path, samples, 16000, subtype='FLOAT')
    np.testing.assert_array_equal(load_mono_audio(path), samples)


def test_reference_audio_downmixes_and_resamples(tmp_path):
    t = np.arange(22050) / 22050
    wave = np.sin(2*np.pi*440*t).astype(np.float32) * 0.4
    path = tmp_path / 'ref.wav'
    sf.write(path, np.stack([wave, np.zeros_like(wave)], axis=1), 22050, subtype='FLOAT')
    result = load_mono_audio(path)
    assert result.shape == (16000,) and result.dtype == np.float32
    expected = 0.2*np.sin(2*np.pi*440*np.arange(16000)/16000)
    np.testing.assert_allclose(result[100:-100], expected[100:-100], atol=0.0005)


@pytest.mark.parametrize('samples', [[], [0, float('nan')]])
def test_reference_audio_rejects_invalid_samples(tmp_path, samples):
    path = tmp_path / 'ref.wav'
    sf.write(path, np.asarray(samples), 16000, subtype='FLOAT')
    with pytest.raises(ValueError, match='finite samples'):
        load_mono_audio(path)


def native_fixture(monkeypatch, events):
    ws = Mock()
    ws.__enter__ = Mock(return_value=ws)
    ws.__exit__ = Mock(return_value=False)
    ws.recv.side_effect = [json.dumps(e) for e in [
        {'type': 'session.created', 'session_id': 'test'}, *events]]
    monkeypatch.setattr('core.processors.native_chat.connect', lambda *a, **k: ws)
    owner = SimpleNamespace(_cpp_server_url='http://127.0.0.1:19080',
                            _maybe_update_kv_cache_length=Mock(), _http_client=Mock())
    owner._http_client.post.return_value.status_code = 200
    request = {'messages': [{'role': 'user', 'content': 'hello'}], 'ref_audio_path': '',
               'sampling': {}, 'max_slice_nums': 1, 'enable_thinking': False,
               'cancel': threading.Event()}
    return owner, request, ws


def test_browser_chat_owns_full_history_and_honors_generation_options(monkeypatch):
    owner, request, ws = native_fixture(monkeypatch, [
        {'type': 'response.output.delta', 'kind': 'text', 'text': 'ok'},
        {'type': 'response.done', 'text': 'ok'}])
    chunks = list(stream_chat(owner, request, max_new_tokens=8,
                              generate_audio=False, length_penalty=1.0))
    assert ''.join(c.text_delta or '' for c in chunks) == 'ok'
    assert chunks[-1].is_final and not any(c.audio_data for c in chunks)
    sent = json.loads(ws.send.call_args_list[1].args[0])['input']
    assert sent['messages'] == request['messages']
    assert sent['generation']['max_new_tokens'] == 8
    assert not sent['tts']['enabled'] and not sent['use_tts_template']
    owner._http_client.post.assert_called_once()


def test_non_streaming_preserves_done_text_and_pcm(monkeypatch):
    pcm = base64.b64encode(np.zeros(4, dtype='<f4').tobytes()).decode()
    owner, request, _ = native_fixture(monkeypatch, [
        {'type': 'response.done', 'text': 'ok', 'audio': pcm}])
    chunks = list(stream_chat(owner, request, streaming=False, max_new_tokens=8,
                              generate_audio=True, length_penalty=1.0))
    assert len(chunks) == 1 and chunks[0].is_final
    assert chunks[0].text_delta == 'ok' and chunks[0].audio_data == pcm


def test_native_failure_reaches_browser_instead_of_empty_done(monkeypatch):
    owner, request, _ = native_fixture(monkeypatch, [
        {'type': 'session.closed', 'reason': 'prefill_failed'}])
    with pytest.raises(RuntimeError, match='prefill_failed'):
        list(stream_chat(owner, request, max_new_tokens=8,
                         generate_audio=False, length_penalty=1.0))
    owner._http_client.post.assert_called_once()


def test_browser_prefill_does_not_send_user_text_as_legacy_index_zero():
    owner = CppBackendWorker.__new__(CppBackendWorker)
    owner.ref_audio_path = ''
    owner._call_prefill = Mock(side_effect=AssertionError('legacy prefill'))
    messages = [{'role': 'user', 'content': 'hello'}]
    owner.chat_prefill('browser', msgs=[], native_messages=messages)
    assert owner._native_chat_request['messages'] == messages


def test_cancellation_during_initialization_does_not_start_generation(monkeypatch):
    owner, request, ws = native_fixture(monkeypatch, [])
    request['cancel'].set()
    with pytest.raises(RuntimeError, match='cancelled'):
        list(stream_chat(owner, request, max_new_tokens=8,
                         generate_audio=False, length_penalty=1.0))
    assert ws.send.call_count == 1
    owner._http_client.post.assert_called_once()
    assert owner._active_native_chat_session is None


def test_cancel_chat_stops_the_active_session():
    owner = CppBackendWorker.__new__(CppBackendWorker)
    owner._cpp_server_url = 'http://127.0.0.1:19080'
    owner._active_native_chat_request = {'cancel': threading.Event()}
    owner._active_native_chat_session = 'test'
    owner._http_client = Mock()
    owner._http_client.post.return_value.status_code = 200
    owner.cancel_native_chat()
    assert owner._active_native_chat_request['cancel'].is_set()
    owner._http_client.post.assert_called_once_with(
        'http://127.0.0.1:19080/sessions/test/close', timeout=30)


def test_cancel_chat_before_generation_marks_staged_request():
    owner = CppBackendWorker.__new__(CppBackendWorker)
    owner.ref_audio_path = ''
    owner.chat_prefill('browser', msgs=[], native_messages=[{'role': 'user', 'content': 'hello'}])
    owner.cancel_native_chat()
    assert owner._native_chat_request['cancel'].is_set()

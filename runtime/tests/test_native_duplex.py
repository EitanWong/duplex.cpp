"""Native duplex output must survive a switch back to listening."""
import json
import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.processors.m5_duplex import NativeDuplex


@pytest.mark.parametrize('terminal, expected', [
    ({'kind': 'listen'}, (True, '你好。', {})),
    ({'kind': 'listen', 'text': '完整回复。'}, (True, '完整回复。', {})),
    ({'type': 'response.done'}, (False, '你好。', {})),
    ({'type': 'response.done', 'text': '完整回复。'}, (False, '完整回复。', {})),
])
def test_text_is_preserved_at_the_input_boundary(terminal, expected):
    native = NativeDuplex.__new__(NativeDuplex)
    native.owner = SimpleNamespace(_maybe_update_kv_cache_length=Mock())
    native.closed = False
    native.results = queue.Queue()
    native.ws = [json.dumps(event) for event in [
        {'kind': 'text', 'text': '你好'},
        {'kind': 'text', 'text': '。'},
        terminal,
        {'kind': 'listen'},
    ]]

    native._read()

    assert native.results.get_nowait() == expected
    assert native.results.get_nowait() == (True, '', {})
    assert native.results.empty()


def generating_native():
    native = NativeDuplex.__new__(NativeDuplex)
    native.owner = SimpleNamespace(_duplex_chunk_counter=1)
    native.results = queue.Queue()
    native.audio = queue.Queue()
    native.audio_lock = threading.Lock()
    native.audio_ready = threading.Event()
    native.interruptions = queue.Queue()
    native.audio_epoch = 0
    native.audio_muted = False
    native.forced_listen_active = False
    native.interrupt_requested = False
    native.assistant_spoke = True
    native.pending_speech_hint = {'force_listen': True}
    native.ws = Mock()
    return native


def generate_result(native, listen, text='', force=False):
    native.pending = {'audio': 'input'}
    native.results.put((listen, text, {}))
    result = native.generate(force_listen=force)
    return result, json.loads(native.ws.send.call_args.args[0])['input']


def test_automatic_interrupt_releases_after_listen_acknowledgement():
    native = generating_native()
    _, first = generate_result(native, True)
    assert first['force_listen']

    # The user is still speaking, but the previous assistant speech has stopped.
    result, second = generate_result(native, False, '北京是中国的首都。')
    assert not second['force_listen']
    assert result.text == '北京是中国的首都。'
    assert native.audio_epoch == 1
    assert not native.audio_muted


def test_explicit_force_listen_stays_active_until_the_caller_releases():
    native = generating_native()
    for _ in range(2):
        _, sent = generate_result(native, True, force=True)
        assert sent['force_listen']
    _, released = generate_result(native, False, '新回复。')
    assert not released['force_listen']
    assert native.audio_epoch == 1


def test_new_assistant_speech_rearms_automatic_interruption():
    native = generating_native()
    generate_result(native, True)
    generate_result(native, False, '新回复。')
    _, sent = generate_result(native, True)
    assert sent['force_listen']
    assert native.audio_epoch == 2


def test_memory_acknowledgements_do_not_become_listen_or_speak_results():
    native = NativeDuplex.__new__(NativeDuplex)
    native.owner = SimpleNamespace(_maybe_update_kv_cache_length=Mock())
    native.closed = False
    native.results = queue.Queue()
    native.memory_results = queue.Queue()
    native.ws = [json.dumps(event) for event in [
        {'type': 'memory.error', 'error': 'source turn is not completed'},
        {'type': 'memory.updated', 'revision': 1},
        {'kind': 'listen'},
    ]]
    native._read()
    assert native.memory_results.get_nowait()['type'] == 'memory.error'
    assert native.memory_results.get_nowait()['revision'] == 1
    assert native.results.get_nowait() == (True, '', {})
    assert native.results.empty()


def test_rejected_memory_update_retains_the_previous_retrieval_query():
    native = generating_native()
    native.closed = False
    native.memory_request_lock = threading.Lock()
    native.memory_results = queue.Queue()
    native.memory_query = 'previous'
    native.memory_results.put({'type': 'memory.error', 'error': 'assistant is not a user source'})
    with pytest.raises(RuntimeError, match='not a user source'):
        native.update_memory([{'source': 'assistant'}], 'new query')
    assert native.memory_query == 'previous'

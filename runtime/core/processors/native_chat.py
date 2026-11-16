"""Run browser chat through the native structured turn-based protocol."""
import base64
import json
import logging
import time

from websockets.sync.client import connect

from core.audio import load_mono_audio
from core.schemas.streaming import StreamingChunk


logger = logging.getLogger('cpp_backend')


def stream_chat(owner, request, *, max_new_tokens, generate_audio,
                length_penalty, streaming=True):
    # Keep the resident TTS weights available across turns; the input controls
    # whether this particular response actually runs speech synthesis.
    payload = {'mode': 'turn_based', 'use_tts': True,
               'config': request['sampling']}
    if request['ref_audio_path']:
        pcm = load_mono_audio(request['ref_audio_path'])
        encoded = base64.b64encode(pcm.tobytes()).decode('ascii')
        payload['voice'] = {'ref_audio': encoded, 'tts_ref_audio': encoded}
    index = 0
    session_id = None
    started = time.perf_counter()
    owner._active_native_chat_request = request
    url = owner._cpp_server_url.replace('http://', 'ws://') + '/backend'
    try:
        with connect(url, proxy=None, max_size=128*1024*1024, open_timeout=30,
                     ping_interval=None, close_timeout=5) as ws:
            ws.send(json.dumps({'type': 'session.init', 'payload': payload}))
            created = json.loads(ws.recv(timeout=120))
            if created.get('type') != 'session.created':
                raise RuntimeError(f'Native chat init failed: {created}')
            session_id = created['session_id']
            owner._active_native_chat_session = session_id
            if request['cancel'].is_set():
                raise RuntimeError('Chat cancelled during initialization')
            owner._last_duplex_mode = False
            owner._maybe_update_kv_cache_length(created.get('metrics', {}))
            ws.send(json.dumps({'type': 'input.append', 'input': {
                'messages': request['messages'], 'streaming': streaming,
                'generation': {'max_new_tokens': max_new_tokens,
                               'length_penalty': length_penalty},
                'max_slice_nums': request['max_slice_nums'],
                'use_tts_template': generate_audio,
                'enable_thinking': request['enable_thinking'],
                'tts': {'enabled': generate_audio}}}))
            while True:
                event = json.loads(ws.recv(timeout=180))
                owner._maybe_update_kv_cache_length(event.get('metrics', {}))
                event_type = event.get('type')
                if event_type in ('error', 'session.closed'):
                    raise RuntimeError(f'Native chat failed: {event}')
                if event_type == 'response.done':
                    logger.info('[NativeChat] done in %.0fms, metrics=%s',
                                (time.perf_counter()-started)*1000, event.get('metrics', {}))
                    yield StreamingChunk(
                        chunk_index=index,
                        text_delta=(event.get('text') or None) if not streaming else None,
                        audio_data=(event.get('audio') or None) if generate_audio and not streaming else None,
                        is_final=True)
                    return
                text = event.get('text') if event.get('kind') == 'text' else None
                audio = event.get('audio') if generate_audio and event.get('kind') == 'audio' else None
                if text or audio:
                    yield StreamingChunk(chunk_index=index, text_delta=text or None,
                                         audio_data=audio or None, is_final=False)
                    index += 1
    finally:
        # The session registry survives a transport disconnect. Release its
        # reservation as well so the next browser turn can use the worker.
        if session_id:
            try:
                response = owner._http_client.post(
                    owner._cpp_server_url + '/sessions/' + session_id + '/close', timeout=20)
                if response.status_code not in (200, 404):
                    response.raise_for_status()
            except Exception:
                logger.exception('Could not release native chat session %s', session_id)
        owner._active_native_chat_session = None
        owner._active_native_chat_request = None

"""Bridge the demo to the engine's native full_duplex WebSocket protocol."""
import base64
import io
import json
import logging
import math
import os
import queue
import threading
import time

import numpy as np
import soundfile as sf
from websockets.sync.client import connect


logger = logging.getLogger('cpp_backend')


class NativeDuplex:
    def __init__(self, owner, prompt, ref_audio_path, sampling, length_penalty):
        self.owner = owner
        self.results = queue.Queue()
        self.audio = queue.Queue()
        self.audio_lock = threading.Lock()
        self.audio_ready = threading.Event()
        self.interruptions = queue.Queue()
        self.audio_epoch = 0
        self.last_collected_audio_epoch = 0
        self.last_result_audio_epoch = 0
        self.last_audio_metadata = []
        self.audio_muted = False
        self.forced_listen_active = False
        self.interrupt_requested = False
        self.pending = None
        self.closed = False
        self.assistant_spoke = False
        self.speech_gate = _ForegroundSpeech()
        self.pending_speech_hint = None
        self.ws = connect(owner._cpp_server_url.replace('http://', 'ws://')+'/backend',
                          proxy=None, max_size=20*1024*1024, open_timeout=30,
                          ping_interval=None, close_timeout=5)
        payload = {'mode': 'full_duplex', 'use_tts': True,
                   'system_prompt': prompt or 'Streaming Duplex Conversation.',
                   'config': {**(sampling or {}), 'length_penalty': length_penalty}}
        # Bound the native per-second speak budget so long answers keep accepting input.
        budget = max(1, min(26, int(os.environ.get('OMNI_SPEAK_TOKENS', '10'))))
        payload['config']['max_new_speak_tokens_per_chunk'] = min(
            budget, int(payload['config'].get('max_new_speak_tokens_per_chunk', budget)))
        ref = ref_audio_path or owner.ref_audio_path
        if ref:
            wave, sr = sf.read(ref, dtype='float32')
            if wave.ndim > 1:
                wave = wave.mean(axis=1)
            if sr != 16000:
                raise ValueError('Reference audio must be mono 16 kHz')
            encoded = base64.b64encode(wave.astype(np.float32).tobytes()).decode()
            payload['voice'] = {'ref_audio': encoded, 'tts_ref_audio': encoded}
        self.ws.send(json.dumps({'type': 'session.init', 'payload': payload}))
        created = json.loads(self.ws.recv(timeout=120))
        if created.get('type') != 'session.created':
            self.ws.close()
            raise RuntimeError(f'Native duplex init failed: {created}')
        self.session_id = created['session_id']
        owner._maybe_update_kv_cache_length(created.get('metrics', {}))
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for raw in self.ws:
                event = json.loads(raw)
                metrics = event.get('metrics', {})
                self.owner._maybe_update_kv_cache_length(metrics)
                if event.get('kind') == 'audio' and (event.get('audio') or event.get('speech_end')):
                    with self.audio_lock:
                        producer_epoch = event.get('audio_epoch', self.audio_epoch)
                        if not self.audio_muted and producer_epoch == self.audio_epoch:
                            metadata = {**event.get('trace', {}), 'reader_ms': time.perf_counter()*1000, 'reader_wall_ms': time.time()*1000,
                                        'speech_end': event.get('speech_end', False)}
                            self.audio.put((producer_epoch, base64.b64decode(event.get('audio') or ''), metadata))
                            self.audio_ready.set()
                elif event.get('kind') == 'listen':
                    self.results.put((True, '', metrics))
                elif event.get('type') == 'response.done':
                    self.results.put((False, event.get('text', ''), metrics))
                elif event.get('type') == 'session.closed':
                    if not self.closed:
                        self.results.put(RuntimeError(f'Native session closed: {event}'))
                    break
        except Exception as error:
            if not self.closed:
                self.results.put(error)

    def prefill(self, audio, frames, max_slices, speech_hint=None):
        self.pending_speech_hint = speech_hint if speech_hint is not None else self.speech_gate.feed(audio)
        images = []
        for frame in (frames or []):
            buf = io.BytesIO()
            frame.convert('RGB').save(buf, format='JPEG', quality=85)
            images.append(base64.b64encode(buf.getvalue()).decode())
        self.pending = {'audio': base64.b64encode(np.asarray(audio, dtype=np.float32).tobytes()).decode(),
                        'video_frames': images, 'max_slice_nums': max_slices}
        return {'n_vision_images': len(images)}

    def generate(self, force_listen=False):
        from core.schemas.duplex import DuplexGenerateResult
        if self.pending is None:
            raise RuntimeError('No audio/video input is pending')
        hint = self.pending_speech_hint or {}
        automatic_listen = self.assistant_spoke and bool(hint.get('force_listen'))
        with self.audio_lock:
            forced = bool(force_listen) or automatic_listen or self.interrupt_requested
            self.interrupt_requested = False
            if forced and not self.forced_listen_active:
                self.audio_epoch += 1
                while not self.audio.empty():
                    self.audio.get_nowait()
                self.interruptions.put({'type': 'interrupted', 'audio_epoch': self.audio_epoch,
                                        'reason': 'foreground_speech' if automatic_listen else 'force_listen'})
                self.audio_ready.set()
            self.audio_muted = forced
            self.forced_listen_active = forced
            request_epoch = self.audio_epoch
        self.pending['force_listen'] = forced
        logger.info('[AfterSpeak] %s', json.dumps({
            'chunk': self.owner._duplex_chunk_counter,
            'assistant_spoke': self.assistant_spoke,
            'caller_force': bool(force_listen),
            'automatic_force': automatic_listen,
            'foreground_ms': hint.get('foreground_ms', 0),
            'trailing_silence_ms': hint.get('trailing_silence_ms'),
            'reference_rms': hint.get('reference_rms'),
            'cpu_ms': hint.get('cpu_ms', 0),
        }))
        start = time.perf_counter()
        self.ws.send(json.dumps({'type': 'input.append', 'input': self.pending}))
        self.pending = None
        item = self.results.get(timeout=120)
        if isinstance(item, Exception):
            raise item
        listen, text, metrics = item
        self.last_native_metrics = metrics
        self.last_result_audio_epoch = request_epoch
        if not listen and text:
            self.assistant_spoke = True
        elif listen and not automatic_listen and not force_listen:
            # A natural LISTEN ends our intervention. The next user turn starts
            # with the model's own decision until it actually speaks again.
            self.assistant_spoke = False
        return DuplexGenerateResult(is_listen=listen, text=text, audio_data=None,
                                    end_of_turn=listen, current_time=self.owner._duplex_chunk_counter,
                                    cost_all_ms=(time.perf_counter()-start)*1000,
                                    cost_llm_ms=metrics.get('generate_ms', 0))

    def request_interrupt(self):
        """Mute now; commit LISTEN with the next intact native input unit."""
        with self.audio_lock:
            self.interrupt_requested = True
            if not self.forced_listen_active:
                self.audio_epoch += 1
                while not self.audio.empty():
                    self.audio.get_nowait()
                self.interruptions.put({'type': 'interrupted', 'audio_epoch': self.audio_epoch,
                                        'reason': 'force_listen'})
            self.forced_listen_active = True
            self.audio_muted = True
            self.audio_ready.set()

    def collect_audio(self, text=''):
        chunks = []
        metadata = []
        with self.audio_lock:
            self.last_collected_audio_epoch = self.audio_epoch
            while True:
                try:
                    epoch, pcm, trace = self.audio.get_nowait()
                    if epoch == self.audio_epoch and not self.audio_muted:
                        chunks.append(pcm)
                        metadata.append(trace)
                except queue.Empty:
                    break
            if self.interruptions.empty():
                self.audio_ready.clear()
        self.last_audio_metadata = metadata
        return (base64.b64encode(b''.join(chunks)).decode() if chunks else None, text)

    def close(self):
        self.closed = True
        self.audio_ready.set()
        self.ws.close()
        if self.reader is not threading.current_thread():
            self.reader.join(timeout=6)
        response = self.owner._http_client.post(
            self.owner._cpp_server_url+'/sessions/'+self.session_id+'/close', timeout=20)
        # Normal WebSocket cleanup may already have released this session.
        if response.status_code not in (200, 404):
            response.raise_for_status()


class _ForegroundSpeech:
    def __init__(self):
        from vad.vad import get_vad_model
        self.model=get_vad_model()
        self.state=self.model.get_initial_state(1)
        self.pending=np.empty(0,dtype=np.float32)
        self.reference_rms=0.003

    def feed(self,audio):
        started=time.perf_counter()
        self.pending=np.concatenate((self.pending,np.asarray(audio,dtype=np.float32)))
        windows=[];used=0
        while len(self.pending)-used>=1024:
            x=self.pending[used:used+1024];used+=1024
            prob,self.state=self.model(x,self.state,16000)
            prob=float(prob.item());rms=float(np.sqrt(np.mean(x*x)))
            self.reference_rms*=math.exp(-0.064/60)
            if prob>=0.7:self.reference_rms=max(self.reference_rms,rms)
            threshold=max(0.003,0.3*self.reference_rms)
            foreground=prob>=0.7 and rms>=threshold
            windows.append({'prob':prob,'rms':rms,'threshold':threshold,'foreground':foreground})
        self.pending=self.pending[used:].copy()
        # Require at least 128 ms of foreground speech in the received chunk.
        # A quiet chunk releases immediately; no whole-utterance buffering.
        count=sum(w['foreground'] for w in windows)
        return {'force_listen':count>=2,'foreground_ms':count*64,
                'reference_rms':self.reference_rms,'cpu_ms':1000*(time.perf_counter()-started),
                'windows':windows}

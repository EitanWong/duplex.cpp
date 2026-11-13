"""Experimental source-time endpoint submission; native audio units remain 1 s."""
import numpy as np


class EndpointInputBuffer:
    def __init__(self, speech_gate, silence_ms=384, chunk_samples=16000):
        self.gate = speech_gate
        self.silence_ms = silence_ms
        self.chunk_samples = chunk_samples
        self.pending = np.empty(0, dtype=np.float32)
        self.source_samples = 0
        self.unit_start = 0
        self.speech_run_ms = 0
        self.armed = False
        self.unit_windows = []
        self.source_tail_ms = 0
        self.last_speech_sample = 0
        self.last_endpoint_sample = -16000

    def _emit(self, count, endpoint):
        valid = self.pending[:count].copy()
        audio = np.zeros(self.chunk_samples, dtype=np.float32)
        audio[:count] = valid
        self.pending = self.pending[count:].copy()
        foreground_ms = 64 * sum(w['foreground'] for w in self.unit_windows)
        tail_ms = self.source_tail_ms
        result = dict(audio=audio, valid_samples=count,
                      padding_samples=self.chunk_samples-count,
                      source_start=self.unit_start, source_end=self.unit_start+count,
                      endpoint=endpoint, speech_end_sample=self.last_speech_sample,
                      hint=dict(force_listen=foreground_ms >= 128 and tail_ms < self.silence_ms,
                                foreground_ms=foreground_ms, trailing_silence_ms=tail_ms,
                                reference_rms=self.gate.reference_rms, cpu_ms=0))
        self.unit_start += count
        self.unit_windows = []
        return result

    def feed(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or len(audio) > 1024:
            raise ValueError('feed source frames of at most 1024 mono 16-kHz samples')
        processed_before = self.source_samples-len(self.gate.pending)
        hint = self.gate.feed(audio)
        self.source_samples += len(audio)
        self.pending = np.concatenate((self.pending, audio))
        self.unit_windows.extend(hint['windows'])
        for index,w in enumerate(hint['windows']):
            # The interruption gate uses a high relative volume threshold.
            # Endpointing must retain quieter syllables, so continue activity
            # at the lower probability/volume floor until it really goes quiet.
            active = w['prob'] >= 0.35 and w['rms'] >= max(0.0015, 0.1*self.gate.reference_rms)
            self.source_tail_ms = 0 if active else self.source_tail_ms+64
            if active:self.last_speech_sample=processed_before+(index+1)*1024
            self.speech_run_ms = self.speech_run_ms + 64 if w['foreground'] else 0
            if self.speech_run_ms >= 128:
                self.armed = True
        units = []
        while len(self.pending) >= self.chunk_samples:
            units.append(self._emit(self.chunk_samples, False))
            if self.source_tail_ms >= self.silence_ms:
                self.armed = False
        # Preserve the original three complete startup units. Only real source
        # frames reach VAD; padding is supplied solely to the native encoder.
        if (self.source_samples >= 3*self.chunk_samples and self.armed
                and self.source_tail_ms >= self.silence_ms
                and self.source_samples-self.last_endpoint_sample >= 16000
                and len(self.pending) >= 1024):
            units.append(self._emit(len(self.pending), True))
            self.last_endpoint_sample=self.source_samples
            self.armed = False
        return units

    def finish(self):
        return [self._emit(len(self.pending), False)] if len(self.pending) else []

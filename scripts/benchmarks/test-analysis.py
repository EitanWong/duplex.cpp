"""Regression checks for measurement boundaries and interrupted journals."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parent

def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def test_device_clock_is_distinct_from_pcm_arrival_and_schedule():
    rows = [
        {'type':'input', 'index':0, 'native_input':2, 'browser_ms':100},
        {'type':'pcm_arrival', 'input':2, 'epoch':0, 'turn':1, 'browser_ms':600},
        {'type':'schedule', 'turn':1, 'browser_ms':850, 'outputContextTime':1,
         'outputPerformanceTime':800, 'scheduledAudioTime':1.2, 'outputLatencyMs':64},
    ]
    result = module('analyze-playback').analyze(rows)
    assert result['input_to_pcm_arrival_ms']['p50'] == 500
    assert result['pcm_arrival_to_schedule_callback_ms']['p50'] == 250
    assert result['input_to_estimated_device_presentation_ms']['p50'] == pytest.approx(900)
    assert result['schedule_underflow_ms']['n'] == 0


def test_legacy_trace_requires_explicit_verified_offset_and_cancels_pending():
    rows = [
        {'type':'input', 'index':4, 'browser_ms':100},
        {'type':'pcm_arrival', 'input':6, 'epoch':0, 'turn':1, 'browser_ms':500},
        {'type':'interrupt', 'browser_ms':510},
        {'type':'pcm_arrival', 'input':6, 'epoch':1, 'turn':2, 'browser_ms':600},
        {'type':'schedule', 'turn':2, 'browser_ms':850, 'outputContextTime':None,
         'outputPerformanceTime':None, 'scheduledAudioTime':2, 'outputLatencyMs':64},
    ]
    tool = module('analyze-playback')
    assert tool.analyze(rows)['input_to_pcm_arrival_ms']['n'] == 0
    result = tool.analyze(rows, 2)
    assert result['input_to_pcm_arrival_ms']['n'] == 2
    assert result['pcm_arrival_to_schedule_callback_ms']['p50'] == 250
    assert result['canceled_unscheduled_packets'] == 1
    assert result['input_to_estimated_device_presentation_ms']['n'] == 0


def test_durable_journal_ignores_only_incomplete_tail(tmp_path):
    (tmp_path/'backend.log').write_text('', encoding='utf-8')
    journal = tmp_path/'round-0/events.jsonl'; journal.parent.mkdir()
    row = {'kind':'input','index':0,'forced_listen':True,'elapsed_ms':10,
           'received_ms':10,'metrics':{'kv_cache_length':1,'kv':{'accounted_tokens':1}},
           'latency_ms':5,'schedule_lag_ms':0}
    journal.write_text(json.dumps(row)+'\n{"kind":', encoding='utf-8')
    analyze = module('analyze-steady').analyze
    result = analyze(tmp_path)
    assert result['inputs'] == 1
    assert result['windows'][0]['accounting_mismatches'] == []
    journal.write_text(json.dumps(row)+'\ncorrupt\n', encoding='utf-8')
    with pytest.raises(json.JSONDecodeError): analyze(tmp_path)
@pytest.mark.asyncio
async def test_burst_replay_keeps_complete_inputs_fifo_and_drains_partial_batch(tmp_path, monkeypatch):
    import argparse
    import asyncio
    tool = module('native')
    class Socket:
        def __init__(self):
            self.queue = asyncio.Queue(); self.sent = 0; self.batches = []
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def send(self, raw):
            row = json.loads(raw)
            if row['type'] == 'session.init': return
            assert row['type'] == 'input.append'
            assert row['input']['audio'] == 'fixture'
            self.sent += 1
            await self.queue.put(json.dumps({'kind':'text', 'text':str(self.sent), 'response_id':str(self.sent)}))
            await self.queue.put(json.dumps({'type':'response.done', 'text':str(self.sent), 'response_id':str(self.sent), 'metrics':{}}))
        async def recv(self): return json.dumps({'type':'session.created'})
        def __aiter__(self): return self
        async def __anext__(self):
            self.batches.append(self.sent)
            return await self.queue.get()
    socket = Socket()
    monkeypatch.setattr(tool, 'connect', lambda *a, **k: socket)
    monkeypatch.setattr(tool, 'pcm', lambda path: 'fixture')
    args = argparse.Namespace(output=tmp_path, url='ws://fake', count=5, interval_ms=0,
        audio_only=True, interrupt_at=2, burst_size=2, recovery_count=1, tail_seconds=0)
    report = await tool.replay(args, 0)
    assert [f['text'] for f in report['frames']] == ['1','2','3','4','5','6']
    assert [f['index'] for f in report['frames']] == list(range(6))
    assert [f['forced_listen'] for f in report['frames']] == [False,False,True,False,False,False]
    assert report['frames'][-1]['phase'] == 'recovery'
    assert report['frames'][-2]['phase'] == 'primary'
    assert report['burst_size'] == 2
    assert all(f['first_text_ms'] is not None and f['first_text_ms'] <= f['latency_ms'] for f in report['frames'])
    assert sum(json.loads(line)['kind'] == 'input' for line in
               (tmp_path/'round-0/events.jsonl').read_text().splitlines()) == 6


def test_adjacent_forced_listen_inputs_do_not_invent_extra_audio_epochs(tmp_path):
    (tmp_path/'backend.log').write_text('', encoding='utf-8')
    journal = tmp_path/'round-0/events.jsonl'; journal.parent.mkdir()
    frames = []
    for i, forced in enumerate([True, True, False, True]):
        frames.append({'kind':'input','index':i,'forced_listen':forced,'elapsed_ms':(i+1)*10,
            'sent_ms':i*10,'received_ms':(i+1)*10,'metrics':{},'latency_ms':10,'schedule_lag_ms':0})
    audio = [{'kind':'audio','received_ms':25,'samples':1,'epoch':1},
             {'kind':'audio','received_ms':45,'samples':1,'epoch':1}]
    journal.write_text(''.join(json.dumps(r)+'\n' for r in frames+audio),encoding='utf-8')
    result = module('analyze-steady').analyze(tmp_path)
    assert result['interruption_epoch_transitions'] == 2
    assert result['stale_pcm_after_ack'] == 1

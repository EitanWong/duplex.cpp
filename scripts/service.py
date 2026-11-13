#!/usr/bin/env python3
"""Local-only MiniCPM service lifecycle; never stops unrelated processes."""
from pathlib import Path
import argparse
import fcntl
import tempfile
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'service-state.json'
PYTHON = ROOT / '.venv/bin/python'

STOP = ROOT / 'service-stop.request'

def fingerprint(pid):
    q = subprocess.run(['/bin/ps', '-p', str(pid), '-o', 'lstart=', '-o', 'args='], capture_output=True, text=True)
    return q.stdout.strip() if q.returncode == 0 else ''

def alive(pid):
    q = subprocess.run(['/bin/ps', '-p', str(pid), '-o', 'stat='], capture_output=True, text=True)
    return q.returncode == 0 and bool(q.stdout.strip()) and not q.stdout.strip().startswith('Z')

def owned(entry):
    if not alive(entry['pid']):
        return False
    current = fingerprint(entry['pid'])
    if entry.get('identity'):
        return current == entry['identity']
    # Upgrade old service state only when its known command still matches.
    command = entry.get('command', [])
    return bool(command) and Path(command[0]).name in current and all(str(x) in current for x in command[1:])

def save_state(state):
    with tempfile.NamedTemporaryFile('w', dir=ROOT, prefix='.service-state-', delete=False) as f:
        json.dump(state, f, indent=2); f.write('\n'); name = f.name
    os.replace(name, STATE)

def entry_for(process, command, **extra):
    return dict(pid=process.pid, command=command, identity=fingerprint(process.pid), started_at=time.time(), **extra)

def stop_owned(entry):
    if not owned(entry):
        return
    os.killpg(entry['pid'], signal.SIGTERM)
    deadline = time.monotonic() + 20
    while owned(entry) and time.monotonic() < deadline:
        time.sleep(0.1)
    if owned(entry):
        os.killpg(entry['pid'], signal.SIGKILL)

def register_native(state):
    worker = state.get('worker')
    if not worker or not owned(worker):
        return state
    children = subprocess.run(['/usr/bin/pgrep', '-P', str(worker['pid'])], capture_output=True, text=True)
    for raw in children.stdout.split():
        pid = int(raw)
        identity = fingerprint(pid)
        if str(ROOT) not in identity or 'llama-omni-server' not in identity or '--port 19080' not in identity:
            continue
        if os.getpgid(pid) != pid:
            continue
        entry = dict(pid=pid, command=['llama-omni-server'], identity=identity, worker_pid=worker['pid'])
        if state.get('native') != entry:
            state['native'] = entry
            save_state(state)
        break
    return state

def supervise():
    # One monitor per workspace; an intentional stop marker prevents resurrection.
    with (ROOT/'service-supervisor.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        failures = 0
        backoff = 5
        while not STOP.exists():
            try:
                with (ROOT/'service-control.lock').open('a') as control:
                    fcntl.flock(control, fcntl.LOCK_EX)
                    state = json.loads(STATE.read_text()) if STATE.exists() else {}
                    state = register_native(state)
                missing = [n for n in ('worker', 'gateway') if n not in state or not owned(state[n])]
                if missing:
                    print('Recovering stopped processes:', ', '.join(missing), flush=True)
                    q = subprocess.run([str(PYTHON), str(ROOT/'scripts/service.py'), 'repair'], cwd=ROOT, timeout=60)
                    backoff = min(60, backoff * 2) if q.returncode else 5
                    failures = 0
                elif time.time()-state['worker'].get('started_at', time.time()) >= 120:
                    try:
                        with urllib.request.urlopen('http://127.0.0.1:22440/health', timeout=5) as r:
                            health = json.load(r)
                        failures = failures + 1 if health.get('status') == 'error' and not health.get('model_loaded') else 0
                    except (OSError, ValueError):
                        failures = 0  # A slow HTTP response alone is not evidence of a dead model.
                    if failures >= 3:
                        print('Native backend failed three health checks; recovering worker; active context is lost.', flush=True)
                        with (ROOT/'service-control.lock').open('a') as control:
                            fcntl.flock(control, fcntl.LOCK_EX)
                            latest = json.loads(STATE.read_text())
                            if not STOP.exists() and latest.get('worker') == state['worker']:
                                stop_owned(state['worker'])
                        failures = 0
                for _ in range(backoff):
                    if STOP.exists(): return
                    time.sleep(1)
            except Exception as e:
                print('Supervisor:', type(e).__name__, str(e), flush=True)
                time.sleep(5)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['start', 'stop', 'status', 'gateway', 'supervise', 'repair'])
    args = parser.parse_args()
    if args.action == 'supervise':
        supervise()
        return
    control = (ROOT/'service-control.lock').open('a')
    fcntl.flock(control, fcntl.LOCK_EX)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    state = register_native(state)
    if args.action == 'stop':
        STOP.touch()
        for name in ['supervisor', 'gateway', 'worker', 'native', 'caffeinate']:
            entry = state.get(name)
            if entry and owned(entry):
                stop_owned(entry)
                print(f'Stopped {name} ({entry["pid"]})')
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and any(owned(entry) for entry in state.values()):
            time.sleep(0.1)
        remaining = [name for name, entry in state.items() if owned(entry)]
        if remaining:
            raise SystemExit('Still shutting down: ' + ', '.join(remaining))
        save_state({})
        return
    if args.action == 'status':
        for name, entry in state.items():
            print(name, 'running' if owned(entry) else 'stopped', entry)
        for name, port in [('gateway',8040), ('worker',22440), ('engine',19080)]:
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3) as r:
                    print(name, r.read().decode())
            except Exception as e:
                print(name, str(e))
        return
    if args.action == 'repair' and STOP.exists():
        return
    if args.action in ('start', 'gateway'):
        STOP.unlink(missing_ok=True)
    if args.action in ('start', 'repair') and 'native' in state and ('worker' not in state or not owned(state['worker'])):
        # The native backend owns a separate process group. Retire its verified
        # previous identity before opening the same port for a replacement worker.
        stop_owned(state['native'])
        state.pop('native')
        save_state(state)
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT/'runtime')
    env['NO_PROXY'] = 'localhost,127.0.0.1,::1'
    env['no_proxy'] = env['NO_PROXY']
    env.setdefault('OMNI_CPU_THREADS', '4')
    profile = ROOT/'selected-profile.json'
    selected = json.loads(profile.read_text()) if profile.exists() else {}
    release_dir = selected.get('native_release_dir')
    if release_dir:
        release = Path(release_dir)
        if not release.is_absolute():
            release = ROOT / release
        if not (release/'llama-omni-server').is_file():
            raise SystemExit(f'Selected native release missing: {release}')
        env['OMNI_NATIVE_RELEASE_DIR'] = str(release)
        env['DYLD_LIBRARY_PATH'] = str(release) + ':' + str(ROOT/'engine/build/bin')
    env['OMNI_DUPLEX_KEEP_ARTIFACTS'] = '1' if selected.get('duplex_keep_artifacts', False) else '0'
    env['OMNI_VISION_BACKEND'] = selected.get('vision_backend','metal')
    env['OMNI_LLM_KV_TYPE'] = selected.get('llm_kv_type', 'f16')
    env['OMNI_DUPLEX_EXACT_FRAME_CACHE'] = '1' if selected.get('vision_exact_frame_cache', False) else '0'
    env['OMNI_TTS_CTX_SIZE'] = str(selected.get('tts_context_size', 4096))
    env['OMNI_TTS_GPU_LAYERS'] = str(selected.get('tts_gpu_layers', 99))
    for switch in ('OMNI_T2W_TIME_CACHE', 'OMNI_T2W_FLASH', 'OMNI_T2W_PROFILE', 'OMNI_TTS_GPU_HEAD'):
        env[switch] = '0'
    env['OMNI_CONTINUITY_TRACE'] = '1'
    env['OMNI_T2W_PROFILE'] = '2'
    env['OMNI_T2W_FUSED_QKV'] = '1'
    env['OMNI_T2W_THREADS'] = str(selected.get('tts_cpu_threads', 8))
    env['OMNI_TTS_TEXT_LOGITS'] = '0' if selected.get('tts_hidden_only', True) else '1'
    env['OMNI_VOC_BLAS'] = '1' if selected.get('vocoder_accelerate', True) else '0'
    env['OMNI_VOC_WARMUP'] = '1' if selected.get('vocoder_warmup', False) else '0'
    env['OMNI_T2W_WARMUP'] = '1' if selected.get('t2w_full_path_warmup', False) else '0'
    env['OMNI_DUPLEX_MERGE_WAV'] = '1' if selected.get('duplex_merge_wav', False) else '0'
    if selected.get('metal_residency', True):
        env.pop('GGML_METAL_NO_RESIDENCY', None)
    else:
        env['GGML_METAL_NO_RESIDENCY'] = '1'
    env['OMNI_SPEAK_TOKENS'] = str(selected.get('max_new_speak_tokens_per_chunk', 10))
    env['VECLIB_MAXIMUM_THREADS'] = '4'
    jobs = [('gateway', [str(PYTHON),'gateway.py','--host','127.0.0.1','--port','8040','--workers','127.0.0.1:22440','--http'])]
    if args.action in ('start', 'repair'):
        model = ROOT/'models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf'
        if not model.exists():
            raise SystemExit('Model download is not complete. See logs/model-download.log.')
        jobs.insert(0, ('worker', [str(PYTHON),'worker.py','--host','127.0.0.1','--port','22440','--gpu-id','0','--worker-index','0']))
    for name, command in jobs:
        if name in state and owned(state[name]):
            print(f'{name} already running ({state[name]["pid"]})')
            continue
        log = ROOT/'logs'/f'{name}.log'
        with log.open('ab') as output:
            p = subprocess.Popen(command,cwd=ROOT/'runtime',env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        state[name] = entry_for(p, command, log=str(log))
        if name == 'worker':
            c = subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(p.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
            state['caffeinate'] = entry_for(c, ['/usr/bin/caffeinate','-i','-w',str(p.pid)])
        save_state(state)
        print(f'Started {name} ({p.pid})')
    if args.action == 'start' and ('supervisor' not in state or not owned(state['supervisor'])):
        command = [str(PYTHON), str(ROOT/'scripts/service.py'), 'supervise']
        with (ROOT/'logs/supervisor.log').open('ab') as output:
            p = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        state['supervisor'] = entry_for(p, command)
        save_state(state)
        print(f'Started supervisor ({p.pid})')
    print('Open http://localhost:8040/omni')

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""One local entry point for the existing MiniCPM service lifecycle."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import shlex
import signal
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT/'.venv/bin/python'
SERVICE = ROOT/'scripts/service.py'

def service(action):
    result = subprocess.run([str(PYTHON), str(SERVICE), action], cwd=ROOT)
    if result.returncode:
        raise SystemExit(result.returncode)

def snapshot():
    spec = importlib.util.spec_from_file_location('project_service', SERVICE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = ROOT/'service-state.json'
    state = json.loads(path.read_text()) if path.exists() else {}
    return {name:entry for name,entry in state.items() if module.owned(entry)}

def monitors():
    output = subprocess.check_output(['/bin/ps','-axo','pid=,args='], text=True)
    found = []
    for line in output.splitlines():
        parts = line.strip().split(None,1)
        if len(parts) != 2:
            continue
        try:
            args = shlex.split(parts[1])
        except ValueError:
            continue
        if len(args) < 2 or args[1] not in ('scripts/runtime_monitor.py',str(ROOT/'scripts/runtime_monitor.py')):
            continue
        pid = int(parts[0])
        cwd = subprocess.run(['/usr/sbin/lsof','-a','-p',str(pid),'-d','cwd','-Fn'],capture_output=True,text=True).stdout
        if '\nn'+str(ROOT)+'\n' in cwd:
            found.append((pid,args))
    return found

def stop_monitors():
    for pid,args in monitors():
        try:
            os.kill(pid,signal.SIGTERM)
        except ProcessLookupError:
            continue
        deadline = time.monotonic()+5
        while any(p == pid for p,_ in monitors()) and time.monotonic()<deadline:
            time.sleep(.1)
        if any(p == pid for p,_ in monitors()):
            raise SystemExit(f'Monitoring process is still shutting down: {pid}')
        if '--output' in args:
            target = Path(args[args.index('--output')+1])
            if not target.is_absolute():
                target = ROOT/target
            target = target.resolve()
            if target.is_relative_to(ROOT) and (target/'checkpoint.json').exists():
                path = target/'checkpoint.json'
                data = json.loads(path.read_text())
                data.update(phase='stopped_by_user',stopped_epoch=time.time())
                path.write_text(json.dumps(data,indent=2)+'\n')
        print(f'Stopped project monitoring: {pid}')

def ready():
    for port in [8040,22440,19080]:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as response:
                health = json.load(response)
            if health.get('status') not in ('healthy','ok'):
                return False
            if port == 22440 and not health.get('model_loaded'):
                return False
        except (OSError,ValueError):
            return False
    return True

def main():
    parser = argparse.ArgumentParser(description='MiniCPM-o 4.5 project controls')
    parser.add_argument('action',choices=['start','stop','status','restart','help'],nargs='?',default='help')
    args = parser.parse_args()
    if args.action == 'help':
        print('Usage: minicpm.sh {start|stop|status|restart}\nYou can also run start.sh / stop.sh / status.sh in this folder.')
        return
    if args.action in ('stop','restart'):
        service('stop')
        stop_monitors()
        if snapshot() or monitors():
            raise SystemExit('Some project processes are still running. Check the project logs.')
        print('Project stopped (services, model, supervisor and project monitoring have exited).')
        if args.action == 'stop':
            return
    if args.action in ('start','restart'):
        service('start')
        print('Waiting for the model to become ready...',flush=True)
        deadline = time.monotonic()+180
        while time.monotonic()<deadline:
            if ready():
                print('Project started: http://localhost:8040/omni\nClosing this terminal will not stop the service. Run stop.sh to stop it.')
                return
            time.sleep(1)
        service('stop')
        raise SystemExit('Startup timed out; services have been stopped. Check project-control/logs/worker.log.')
    state = snapshot()
    monitor = monitors()
    if not state and not monitor:
        print('Status: stopped')
    else:
        print('Status: running' if ready() else 'Status: starting or some services are unhealthy')
        for name,entry in state.items():
            print(f'  {name}: PID {entry["pid"]}')
        for pid,_ in monitor:
            print(f'  runtime_monitor: PID {pid}')
    print(f'Project: {ROOT}\nWeb UI: http://localhost:8040/omni\nLogs: {ROOT/"logs"}')

if __name__ == '__main__':
    main()

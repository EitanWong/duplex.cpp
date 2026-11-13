#!/usr/bin/env python3
"""Record local service availability and memory without opening media devices."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import ctypes
import json
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['ri_user_time','ri_system_time','ri_pkg_idle_wkups','ri_interrupt_wkups','ri_pageins','ri_wired_size','ri_resident_size','ri_phys_footprint','ri_proc_start_abstime','ri_proc_exit_abstime','ri_child_user_time','ri_child_system_time','ri_child_pkg_idle_wkups','ri_child_interrupt_wkups','ri_child_pageins','ri_child_elapsed_abstime','ri_diskio_bytesread','ri_diskio_byteswritten']
class Usage(ctypes.Structure):
    _fields_ = [('uuid', ctypes.c_uint8*16)]+[(name, ctypes.c_uint64) for name in FIELDS]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hours', type=float, default=24)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.hours <= 0:
        raise SystemExit('hours must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    lib = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
    lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    start = time.time()
    deadline = time.monotonic()+args.hours*3600
    samples = 0
    errors = 0
    with (args.output/'samples.jsonl').open('x') as stream:
        while True:
            row = {'at':datetime.now(timezone.utc).isoformat(), 'epoch':time.time(), 'processes':{}, 'health':{}}
            try:
                state = json.loads((ROOT/'service-state.json').read_text())
                for name, entry in state.items():
                    usage = Usage()
                    status = lib.proc_pid_rusage(entry['pid'], 2, ctypes.byref(usage))
                    row['processes'][name] = {'pid':entry['pid'], 'return_code':status}
                    if status == 0:
                        row['processes'][name].update({field:getattr(usage,field) for field in ['ri_phys_footprint','ri_resident_size','ri_proc_start_abstime','ri_pageins','ri_diskio_byteswritten']})
                for name, port in [('gateway',8040),('worker',22440)]:
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=3) as response:
                            row['health'][name] = json.load(response)
                    except Exception as error:
                        row['health'][name] = {'error':str(error)}
                        errors += 1
            except Exception as error:
                row['error'] = str(error)
                errors += 1
            stream.write(json.dumps(row)+'\n')
            stream.flush()
            samples += 1
            complete = time.monotonic() >= deadline
            progress = {'phase':'completed' if complete else 'running', 'started_epoch':start, 'latest_epoch':row['epoch'], 'samples':samples, 'health_probe_errors':errors, 'planned_hours':args.hours, 'continuous_dialogue_verified':False, 'meaning':'Availability and process-memory observation; does not generate input or verify continuous conversational quality.'}
            (args.output/'checkpoint.json').write_text(json.dumps(progress,indent=2)+'\n')
            if complete:
                break
            time.sleep(min(30,max(0,deadline-time.monotonic())))

if __name__ == '__main__':
    main()

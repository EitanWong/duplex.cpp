"""Detach a service without inheriting a caller's console or pipe handles."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shell', required=True)
    parser.add_argument('--launcher', required=True)
    parser.add_argument('--role', required=True, choices=('worker', 'gateway'))
    parser.add_argument('--stdout', required=True)
    parser.add_argument('--stderr', required=True)
    parser.add_argument('--cuda-root', default='')
    args = parser.parse_args()
    if sys.platform != 'win32':
        parser.error('This launcher requires Windows.')
    command = [args.shell, '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
               '-File', args.launcher, '-Role', args.role]
    if args.cuda_root:
        command.extend(('-CudaRoot', args.cuda_root))
    with open(args.stdout, 'wb') as stdout, open(args.stderr, 'wb') as stderr:
        child = subprocess.Popen(
            command, cwd=Path(args.launcher).resolve().parents[2],
            stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
            close_fds=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW,
        )
    print(child.pid)


if __name__ == '__main__':
    main()

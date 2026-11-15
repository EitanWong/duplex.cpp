#!/usr/bin/env python3
"""Check source syntax, entry points and publication hygiene without loading models."""
import ast
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = subprocess.check_output(
        ['git', '-C', str(ROOT), 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
        text=True, encoding='utf-8'
    ).split('\0')
    # Include pending source additions and renames without staging local data.
    paths = sorted({p for p in paths if p and ((ROOT/p).exists() or (ROOT/p).is_symlink())})
    assert paths, 'No tracked source files'
    assert not (ROOT/'project-control').exists(), 'Retired control directory: use scripts/windows or scripts/macos'
    required = ['README.md', 'README.zh-CN.md', 'LICENSE', 'THIRD_PARTY.md',
                'UPSTREAM.json', 'runtime/config.example.json',
                'profiles/apple-m5.json', 'selected-profile.json',
                'engine/CMakeLists.txt', 'scripts/configure.py',
                'scripts/README.md', 'docs/README.md', 'docs/project-layout.md',
                'scripts/windows/start.ps1', 'scripts/windows/stop.ps1',
                'scripts/windows/status.ps1', 'scripts/windows/build.ps1',
                'scripts/windows/service.ps1', 'scripts/windows/run.ps1',
                'scripts/windows/detach.py', 'scripts/windows/start.bat',
                'scripts/windows/stop.bat', 'scripts/windows/status.bat',
                'scripts/macos/start.sh', 'scripts/macos/stop.sh',
                'scripts/macos/status.sh', 'scripts/macos/build.sh',
                'scripts/macos/Start.command', 'scripts/macos/Stop.command',
                'scripts/macos/Status.command', 'scripts/macos/service.sh',
                'scripts/macos/service.py', 'scripts/macos/manage.py',
                'scripts/macos/monitor.py', 'scripts/benchmarks/native.py',
                'scripts/benchmarks/session-reuse.py',
                'runtime/assets/vad_model/silero_vad.onnx']
    for name in required:
        assert name in paths and (ROOT / name).is_file(), f'Missing: {name}'
    private = re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----')
    token = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})')
    parsed = 0
    for name in paths:
        if Path(name).parent.as_posix() == 'scripts':
            assert Path(name).name in ('README.md', 'configure.py', 'configure-lan.py', 'check-repository.py'), f'Classify this script by platform or purpose: {name}'
        p = ROOT / name
        assert not p.is_symlink(), f'Unexpected symlink: {name}'
        assert p.stat().st_size < 20 * 1024 * 1024, f'Oversized file: {name}'
        assert not name.startswith(('.local/', 'models/', 'logs/', 'releases/', 'optimization/', 'experiments/', 'verification/')), name
        assert name != 'runtime/config.json' and not name.endswith(('.pem', '.key', '.gguf', '.safetensors', '.pyc')), name
        assert not any(x in Path(name).parts for x in ('.venv', 'node_modules', '__pycache__')), name
        if p.suffix in ('.py', '.md', '.sh', '.ps1', '.bat', '.command', '.json', '.yml', '.yaml', '.env'):
            data = p.read_bytes()
            assert not private.search(data) and not token.search(data), f'Credential material: {name}'
        if name.startswith(('runtime/', 'scripts/')) and p.suffix == '.py':
            ast.parse(p.read_text(encoding='utf-8'), filename=name)
            parsed += 1
    example = json.loads((ROOT / 'runtime/config.example.json').read_text(encoding='utf-8'))
    assert example['backend'] == 'cpp'
    for key in ('llamacpp_root', 'model_dir'):
        assert not Path(example['cpp_backend'][key]).is_absolute(), key
    assert json.loads((ROOT / 'profiles/apple-m5.json').read_text(encoding='utf-8')) == json.loads((ROOT / 'selected-profile.json').read_text(encoding='utf-8'))
    for name in paths:
        if '/' not in name:
            assert Path(name).suffix not in ('.sh', '.ps1', '.bat', '.command', '.log', '.lock'), f'Platform script or generated file at root: {name}'
        if name.startswith('scripts/') and Path(name).suffix in ('.ps1', '.bat'):
            assert name.startswith('scripts/windows/'), f'Windows script outside platform directory: {name}'
        if name.startswith('scripts/') and Path(name).suffix in ('.sh', '.command'):
            assert name.startswith('scripts/macos/'), f'macOS script outside platform directory: {name}'
        if name.startswith(('scripts/', 'docs/')) or name in ('README.md', 'README.zh-CN.md'):
            body = (ROOT/name).read_text(encoding='utf-8')
            for target in re.findall(r'scripts/[A-Za-z0-9_./-]+\.(?:ps1|py|bat|sh|command)\b', body):
                assert (ROOT/target).is_file(), f'{name}: stale script reference {target}'
    docs = [name for name in paths if name.endswith('.md') and name.startswith(('docs/', 'scripts/'))]
    for name in ('README.md', 'README.zh-CN.md', *docs):
        body = (ROOT / name).read_text(encoding='utf-8')
        if name in ('README.md', 'README.zh-CN.md'):
            assert body.splitlines()[0] == '# duplex.cpp'
        for target in re.findall(r'\]\(([^)]+)\)', body):
            if '://' not in target and not target.startswith('#'):
                assert (ROOT / name).parent.joinpath(target.split('#')[0]).exists(), f'{name}: broken link {target}'
    print(f'PASS: {len(paths)} source files; {parsed} Python files parsed; platform layout, script references, documentation links and publication hygiene checked')


if __name__ == '__main__':
    main()

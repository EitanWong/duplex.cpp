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
        ['git', '-C', str(ROOT), 'ls-files', '-z'], text=True
    ).split('\0')
    paths = [p for p in paths if p]
    assert paths, 'No tracked source files'
    required = ['README.md', 'README.zh-CN.md', 'LICENSE', 'THIRD_PARTY.md',
                'UPSTREAM.json', 'runtime/config.example.json',
                'profiles/apple-m5.json', 'selected-profile.json',
                'engine/CMakeLists.txt', 'scripts/configure.py',
                'runtime/assets/vad_model/silero_vad.onnx']
    for name in required:
        assert name in paths and (ROOT / name).is_file(), f'Missing: {name}'
    private = re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----')
    token = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})')
    parsed = 0
    for name in paths:
        p = ROOT / name
        assert not p.is_symlink(), f'Unexpected symlink: {name}'
        assert p.stat().st_size < 20 * 1024 * 1024, f'Oversized file: {name}'
        assert not name.startswith(('models/', 'logs/', 'releases/', 'optimization/', 'experiments/')), name
        assert name != 'runtime/config.json' and not name.endswith(('.pem', '.key', '.gguf', '.safetensors', '.pyc')), name
        assert not any(x in Path(name).parts for x in ('.venv', 'node_modules', '__pycache__')), name
        if p.suffix in ('.py', '.md', '.sh', '.json', '.yml', '.yaml', '.env'):
            data = p.read_bytes()
            assert not private.search(data) and not token.search(data), f'Credential material: {name}'
        if name.startswith(('runtime/', 'scripts/', 'project-control/')) and p.suffix == '.py':
            ast.parse(p.read_text(encoding='utf-8'), filename=name)
            parsed += 1
    example = json.loads((ROOT / 'runtime/config.example.json').read_text())
    assert example['backend'] == 'cpp'
    for key in ('llamacpp_root', 'model_dir'):
        assert not Path(example['cpp_backend'][key]).is_absolute(), key
    assert json.loads((ROOT / 'profiles/apple-m5.json').read_text()) == json.loads((ROOT / 'selected-profile.json').read_text())
    for name in ('README.md', 'README.zh-CN.md'):
        body = (ROOT / name).read_text()
        assert body.splitlines()[0] == '# duplex.cpp'
        for target in re.findall(r'\]\(([^)]+)\)', body):
            if '://' not in target and not target.startswith('#'):
                assert (ROOT / target.split('#')[0]).exists(), f'{name}: broken link {target}'
    print(f'PASS: {len(paths)} source files; {parsed} Python files parsed; native config, README links and publication hygiene checked')


if __name__ == '__main__':
    main()

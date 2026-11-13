#!/usr/bin/env python3
"""Generate machine-local paths without modifying the preserved Apple profile."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'models/MiniCPM-o-4_5-gguf')
    parser.add_argument('--force', action='store_true', help='Replace an existing local config')
    args = parser.parse_args()
    target = ROOT / 'runtime/config.json'
    if target.exists() and not args.force:
        parser.exit(1, 'Local config already exists; use --force to replace it.\n')
    config = json.loads((ROOT / 'runtime/config.example.json').read_text(encoding='utf-8'))
    config['cpp_backend']['llamacpp_root'] = str(ROOT / 'engine')
    config['cpp_backend']['model_dir'] = str(args.model_dir.expanduser().resolve())
    target.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    for directory in ('logs', 'releases', 'runtime/tmp', 'runtime/data'):
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    print(f'Configured: {target}')
    print(f'Models: {config["cpp_backend"]["model_dir"]}')


if __name__ == '__main__':
    main()

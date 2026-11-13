#!/bin/bash
# Rebuild the production package entirely from the formal engine sources.
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
DEST="$ROOT/releases/m5-integrated-v3/native"
mkdir -p "$ROOT/releases"
if [[ -f "$ROOT/service-state.json" ]] && "$ROOT/.venv/bin/python" - "$ROOT/service-state.json" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1])) else 1)
PY
then
    echo "Stop the project with project-control/stop.sh before rebuilding its native release." >&2
    exit 1
fi
cmake -S "$ROOT/engine" -B "$ROOT/engine/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
    -DGGML_ACCELERATE=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple \
    -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build "$ROOT/engine/build" --target llama-omni-server -j 4
STAGE="$(mktemp -d "$ROOT/releases/.native-build.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
cp -P "$ROOT/engine/build/bin/llama-omni-server" "$STAGE/"
cp -P "$ROOT/engine/build/bin/"*.dylib "$STAGE/"
"$ROOT/.venv/bin/python" - "$ROOT" "$STAGE" <<'PY'
from pathlib import Path
import hashlib, json, subprocess, sys
root, stage = map(Path, sys.argv[1:])
files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
         for p in stage.iterdir() if p.is_file() and not p.is_symlink()}
source = {}
for p in (root/'engine').rglob('*'):
    if p.suffix in ('.h', '.cpp', '.mm', '.m', '.c', '.metal') and p.is_file() and not any(x in p.parts for x in ('build', '.git')):
        source[str(p.relative_to(root/'engine'))] = hashlib.sha256(p.read_bytes()).hexdigest()
(stage/'build-manifest.json').write_text(json.dumps(dict(
    upstream_commit=json.loads((root/'UPSTREAM.json').read_text())['engine']['commit'],
    project_commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
    ai_assisted_integration=True, binaries=files, sources=source), indent=2)+'\n')
PY
mkdir -p "$(dirname -- "$DEST")"
if [[ -d "$DEST" ]]; then
    mv "$DEST" "$STAGE/previous"
fi
mv "$STAGE" "$DEST"
trap - EXIT
rm -rf "$DEST/previous"
echo "Built production release: $DEST"

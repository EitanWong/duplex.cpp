#!/bin/bash
set -euo pipefail
CONTROL_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_DIR="$(CDPATH= cd -- "$CONTROL_DIR/.." && pwd)"
PYTHON="$PROJECT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    printf 'Project Python was not found: %s\n' "$PYTHON" >&2
    exit 1
fi
exec "$PYTHON" "$CONTROL_DIR/manage.py" "${@:-help}"

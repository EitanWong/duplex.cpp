#!/bin/bash
set -euo pipefail
CONTROL_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec "$CONTROL_DIR/minicpm.sh" stop "$@"

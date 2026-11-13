#!/bin/bash
set -euo pipefail
PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec "$PROJECT_DIR/project-control/stop.sh" "$@"

#!/usr/bin/env bash
# Local PR1 Studio. Default port 8890, bound on all interfaces for Pinggy and LAN.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${1:-${PR1_PORT:-8890}}"
cd "$ROOT"
exec python3 server.py --host 0.0.0.0 --port "$PORT"

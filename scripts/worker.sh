#!/usr/bin/env bash
# Optional second process. The site already runs a worker thread; this exits if that lock is held.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec python3 -c 'from pathlib import Path; from pr1store.jobs import JobStore; from pr1store.worker import Worker; Worker(JobStore(Path("."))).run_forever()'

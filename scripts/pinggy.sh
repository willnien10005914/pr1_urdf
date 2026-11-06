#!/usr/bin/env bash
# Free Pinggy HTTPS tunnel to the local PR1 Studio.
# Usage: bash scripts/pinggy.sh [local-port]
set -euo pipefail
PORT="${1:-${PR1_PORT:-8890}}"
HOST="${PR1_TUNNEL_HOST:-127.0.0.1}"
PINGGY_USER="${PR1_PINGGY_USER:-x:https}"
if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 [local-port]" >&2
  exit 2
fi
echo "PR1 Studio Pinggy: public HTTPS → http://${HOST}:${PORT}"
echo "Free tunnels last about 60 minutes. Press Ctrl-C to stop."
SSH_TTY_FLAG=()
if [[ -t 0 ]]; then
  SSH_TTY_FLAG=(-t)
fi
exec ssh -4 "${SSH_TTY_FLAG[@]+"${SSH_TTY_FLAG[@]}"}" -p 443 \
  -o StrictHostKeyChecking=accept-new \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes \
  -o ConnectTimeout=20 \
  -R0:"${HOST}:${PORT}" \
  "${PINGGY_USER}@a.pinggy.io"

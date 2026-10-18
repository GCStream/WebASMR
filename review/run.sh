#!/usr/bin/env bash
# Start the accuracy review console on loopback; Caddy serves it at /review/.
set -euo pipefail
cd "$(dirname "$0")/.."
export QC_SECRET="$(cat /root/.qc_secret)"
exec python3 review/server.py --port 8141

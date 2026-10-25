#!/bin/bash
set -e
cd "$(dirname "$0")"

WORKERS="${WORKERS:-4}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

echo "Starting Atelier backend with WORKERS=$WORKERS on http://$HOST:$PORT"
export WORKERS
python3 -m uvicorn main:app --host "$HOST" --port "$PORT"
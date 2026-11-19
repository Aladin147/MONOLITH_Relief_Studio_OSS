#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# ---------------------------------------------------------------------------
# Auto-bootstrap Python virtualenv and frontend dependencies on first run.
# ---------------------------------------------------------------------------
if [[ ! -f "$ROOT_DIR/.venv/bin/activate" ]]; then
  echo "Python virtualenv not found at $ROOT_DIR/.venv - bootstrapping..."
  if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 is required to create the virtualenv" >&2
    exit 1
  fi
  python3 -m venv "$ROOT_DIR/.venv"
  # shellcheck source=/dev/null
  source "$ROOT_DIR/.venv/bin/activate"
  pip install --upgrade pip
  # Prefer the pinned lock (reproducible output); fall back to the lower-bound list.
  if [[ -f "$ROOT_DIR/backend/requirements.lock" ]]; then
    pip install -r "$ROOT_DIR/backend/requirements.lock"
  else
    pip install -r "$ROOT_DIR/backend/requirements.txt"
  fi
else
  # shellcheck source=/dev/null
  source "$ROOT_DIR/.venv/bin/activate"
fi

if [[ ! -d "$ROOT_DIR/frontend/node_modules" ]]; then
  echo "frontend/node_modules not found - running npm ci..."
  if ! command -v npm >/dev/null 2>&1; then
    echo "Error: npm is required to install frontend dependencies" >&2
    exit 1
  fi
  (cd "$ROOT_DIR/frontend" && npm ci)
fi

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  trap - SIGINT SIGTERM EXIT
  echo ""
  echo "Stopping MONOLITH Relief Studio..."
  if [[ -n "$FRONTEND_PID" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
    kill "$FRONTEND_PID" 2>/dev/null || true
    wait "$FRONTEND_PID" 2>/dev/null || true
  fi
  if [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" 2>/dev/null || true
  fi
}

trap cleanup SIGINT SIGTERM EXIT

# Private by default: only this machine can reach the studio and the artwork.
# Set MONOLITH_HOST=0.0.0.0 to deliberately expose the UI on your local network.
HOST="${MONOLITH_HOST:-127.0.0.1}"

# The backend is only ever reached through the Vite proxy (127.0.0.1:8000), so it stays
# on loopback even when the UI is shared on the LAN.
echo "Starting MONOLITH FastAPI Backend on http://127.0.0.1:8000 ..."
uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

echo "Starting MONOLITH Vite Frontend on http://${HOST}:5173 ..."
(
  cd "$ROOT_DIR/frontend"
  npm run dev -- --host "$HOST" --port 5173
) &
FRONTEND_PID=$!

echo ""
echo "=================================================="
echo "  MONOLITH Relief Studio is running!"
echo "  Frontend UI : http://localhost:5173"
echo "  Backend API : http://localhost:8000"
echo "  API Docs    : http://localhost:8000/docs"
echo "  Press Ctrl+C to stop both services cleanly."
echo "=================================================="
echo ""

wait -n "$BACKEND_PID" "$FRONTEND_PID"

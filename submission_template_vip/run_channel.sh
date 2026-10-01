#!/usr/bin/env bash
# Start the Partner Channel Health MVP (Problem 05): FastAPI backend + Streamlit dashboard.
#   ./run_channel.sh                        API on 8001, dashboard on http://localhost:8502
#   API_PORT=8101 APP_PORT=8602 ./run_channel.sh  (other ports, e.g. if 8001/8502 are busy)
# Ports differ from Problem 03 (8000/8501) so both MVPs can run at the same time.
# Works from the project root OR from inside the team folder (both copies are identical).
# Stop with Ctrl+C (stops both).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# The team folder is the one that contains 05_channel_analytics/: either this folder itself,
# or submission_template_vip/ below it (when the script sits in the project root).
if [[ -d "$SCRIPT_DIR/05_channel_analytics" ]]; then
  TEAM_DIR="$SCRIPT_DIR"
else
  TEAM_DIR="$SCRIPT_DIR/submission_template_vip"
fi
ROOT_DIR="$(dirname "$TEAM_DIR")"   # contains SG_Hackathon_Pack/ (and usually .venv/)
SRC="$TEAM_DIR/05_channel_analytics/src"
API_PORT="${API_PORT:-8001}"
APP_PORT="${APP_PORT:-8502}"

# Use the Python 3.12 virtual env next to the team folder (or inside it).
PYTHON=""
for candidate in "$ROOT_DIR/.venv/bin/python" "$TEAM_DIR/.venv/bin/python"; do
  if [[ -x "$candidate" ]]; then PYTHON="$candidate"; break; fi
done
if [[ -z "$PYTHON" ]]; then
  echo "ERROR: .venv not found in $ROOT_DIR or $TEAM_DIR. Create it with Python 3.12 first:" >&2
  echo "  cd \"$ROOT_DIR\" && python3.12 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt" >&2
  exit 1
fi
if [[ ! -d "$SRC" ]]; then
  echo "ERROR: $SRC not found - is the script next to (or inside) the team folder?" >&2
  exit 1
fi

API_PID=""
APP_PID=""
cleanup() {
  # Runs on Ctrl+C or when the script exits for any reason: stop both servers.
  echo
  echo "Stopping servers..."
  [[ -n "$APP_PID" ]] && kill "$APP_PID" 2>/dev/null || true
  [[ -n "$API_PID" ]] && kill "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting API on http://127.0.0.1:$API_PORT ..."
"$PYTHON" -m uvicorn api:app --app-dir "$SRC" --port "$API_PORT" &
API_PID=$!

# Loading data and building features takes a few seconds - wait until /health answers (max 60 s).
for _ in $(seq 1 60); do
  if curl -s "http://127.0.0.1:$API_PORT/health" > /dev/null; then
    echo "API is up."
    break
  fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "ERROR: the API stopped while starting - see the messages above (port $API_PORT busy?)." >&2
    exit 1
  fi
  sleep 1
done
if ! curl -s "http://127.0.0.1:$API_PORT/health" > /dev/null; then
  echo "ERROR: API did not answer on port $API_PORT within 60 s." >&2
  exit 1
fi

echo "Starting dashboard on http://localhost:$APP_PORT ..."
API_URL="http://127.0.0.1:$API_PORT" "$PYTHON" -m streamlit run "$SRC/dashboard.py" \
  --server.port "$APP_PORT" --server.headless true &
APP_PID=$!

echo
echo "Open http://localhost:$APP_PORT in your browser (API docs: http://127.0.0.1:$API_PORT/docs)."
echo "Press Ctrl+C to stop both."
wait

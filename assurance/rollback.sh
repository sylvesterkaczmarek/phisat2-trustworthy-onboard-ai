#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXAMPLE="$ROOT/examples/phi2-eo-tile-filter"
python "$ROOT/assurance/model_store.py" rollback \
  --active "$EXAMPLE/models/active.onnx" \
  --previous "$EXAMPLE/models/previous.onnx" \
  --manifest "$EXAMPLE/models/model_state.json"

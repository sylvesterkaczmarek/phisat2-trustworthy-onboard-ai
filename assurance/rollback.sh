#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXAMPLE="$ROOT/examples/phi2-eo-tile-filter"
python "$ROOT/assurance/model_store.py" rollback \
  --store "$EXAMPLE/models/bundles" \
  --state "$EXAMPLE/models/deployment_state.json"

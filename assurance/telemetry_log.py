from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "phi2-eo-tile-filter"
EXAMPLE_SRC = EXAMPLE_ROOT / "src"
if str(EXAMPLE_SRC) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_SRC))

from phi2_tile_filter.bandwidth_filter import load_policy  # noqa: E402
from phi2_tile_filter.runtime import OnnxRunner  # noqa: E402
from phi2_tile_filter.utils import CLASS_NAMES, discover_labeled_tiles  # noqa: E402


def emit_telemetry(
    model_path: str | Path,
    data_root: str | Path,
    policy_path: str | Path,
    output: str | Path,
) -> dict:
    runner = OnnxRunner(model_path)
    policy = load_policy(policy_path, runner)
    data_root = Path(data_root)
    runner.assert_data_schema(data_root)
    items = discover_labeled_tiles(data_root)
    if not items:
        raise ValueError(f"no labeled tiles found under {data_root}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with output.open("w", encoding="utf-8") as handle:
        for path, true_class in items:
            record = runner.evaluate_file(path, policy)
            record["file"] = str(path.relative_to(data_root))
            record["true_class"] = int(true_class)
            record["true_class_name"] = CLASS_NAMES[true_class]
            if not record["inference_ok"]:
                failures += 1
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    result = {
        "schema_version": 3,
        "samples": len(items),
        "inference_failures": failures,
        "model_sha256": runner.model_sha256,
        "input_schema_sha256": runner.input_schema_sha256,
        "input_band_ids": list(runner.band_ids),
        "preprocessing_version": runner.input_schema["preprocessing"]["version"],
        "output": str(output),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--out", default="logs/test.jsonl")
    args = parser.parse_args()
    emit_telemetry(args.onnx, args.data, args.policy, args.out)


if __name__ == "__main__":
    main()

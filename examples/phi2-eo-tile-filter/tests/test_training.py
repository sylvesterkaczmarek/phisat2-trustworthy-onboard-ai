from __future__ import annotations

from pathlib import Path

import torch

from phi2_tile_filter.input_schema import input_schema_sha256
from phi2_tile_filter.synth import write_dataset
from phi2_tile_filter.train import train


def test_training_is_reproducible_and_binds_input_contract(tmp_path: Path) -> None:
    data = tmp_path / "tiles"
    write_dataset(data, n=48, bands=4, size=16, seed=3)
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    train(data, epochs=1, base=4, batch=8, seed=19, output=first, device_name="cpu")
    train(data, epochs=1, base=4, batch=8, seed=19, output=second, device_name="cpu")
    a = torch.load(first, map_location="cpu", weights_only=True)
    b = torch.load(second, map_location="cpu", weights_only=True)
    assert a["format_version"] == b["format_version"] == 3
    assert a["in_ch"] == b["in_ch"] == 4
    assert a["input_schema_sha256"] == b["input_schema_sha256"]
    assert a["input_schema_sha256"] == input_schema_sha256(a["input_schema"])
    assert [band["id"] for band in a["input_schema"]["tensor"]["bands"]] == [
        "band_01",
        "band_02",
        "band_03",
        "band_04",
    ]
    for key in a["state_dict"]:
        assert torch.equal(a["state_dict"][key], b["state_dict"][key])

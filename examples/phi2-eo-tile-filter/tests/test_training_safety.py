from pathlib import Path

import pytest
import torch

import phi2_tile_filter.filesystem as filesystem
import phi2_tile_filter.train as training
from phi2_tile_filter.synth import write_dataset
from phi2_tile_filter.utils import sha256_file


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / 'tiles'
    write_dataset(root, n=32, bands=3, size=16, seed=7)
    return root


@pytest.mark.parametrize('rate', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_learning_rate_is_rejected_before_data_access(tmp_path, rate):
    output = tmp_path / 'model.pt'
    output.write_bytes(b'previous checkpoint')
    with pytest.raises(ValueError, match='finite'):
        training.train(tmp_path / 'absent', lr=rate, output=output)
    assert output.read_bytes() == b'previous checkpoint'


def test_nonfinite_model_does_not_replace_checkpoint(dataset, tmp_path, monkeypatch):
    output = tmp_path / 'model.pt'
    output.write_bytes(b'previous checkpoint')
    summary = output.with_suffix('.pt.json')
    summary.write_bytes(b'previous summary')
    original = torch.optim.Adam.step

    def corrupt_parameters(optimizer, *args, **kwargs):
        result = original(optimizer, *args, **kwargs)
        with torch.no_grad():
            for parameter in optimizer.param_groups[0]['params']:
                parameter.fill_(float('nan'))
        return result

    monkeypatch.setattr(torch.optim.Adam, 'step', corrupt_parameters)
    with pytest.raises(RuntimeError, match='non-finite'):
        training.train(dataset, epochs=1, base=2, batch=64, output=output)
    assert output.read_bytes() == b'previous checkpoint'
    assert summary.read_bytes() == b'previous summary'


def test_interrupted_checkpoint_save_preserves_existing_pair(dataset, tmp_path, monkeypatch):
    output = tmp_path / 'model.pt'
    output.write_bytes(b'previous checkpoint')
    summary = output.with_suffix('.pt.json')
    summary.write_bytes(b'previous summary')

    def interrupted_save(checkpoint, path):
        Path(path).write_bytes(b'partial checkpoint')
        raise OSError('simulated disk failure')

    monkeypatch.setattr(torch, 'save', interrupted_save)
    with pytest.raises(OSError, match='disk failure'):
        training.train(dataset, epochs=1, base=2, batch=64, output=output)
    assert output.read_bytes() == b'previous checkpoint'
    assert summary.read_bytes() == b'previous summary'


def test_summary_publication_failure_restores_checkpoint_pair(dataset, tmp_path, monkeypatch):
    output = tmp_path / 'model.pt'
    output.write_bytes(b'previous checkpoint')
    summary = output.with_suffix('.pt.json')
    summary.write_bytes(b'previous summary')
    real_replace = filesystem.os.replace
    injected = False

    def fail_summary_once(source, target):
        nonlocal injected
        if Path(target) == summary and '.stage-' in Path(source).name and not injected:
            injected = True
            raise OSError('simulated summary publication failure')
        return real_replace(source, target)

    monkeypatch.setattr(filesystem.os, 'replace', fail_summary_once)
    with pytest.raises(OSError, match='summary publication failure'):
        training.train(dataset, epochs=1, base=2, batch=64, output=output)
    assert injected
    assert output.read_bytes() == b'previous checkpoint'
    assert summary.read_bytes() == b'previous summary'


def test_checkpoint_staging_preserves_reproducible_bytes(dataset, tmp_path):
    first = tmp_path / 'first' / 'model.pt'
    second = tmp_path / 'second' / 'model.pt'
    for path in (first, second):
        summary = training.train(dataset, epochs=1, base=2, batch=64, seed=19, output=path)
        assert summary['checkpoint_sha256'] == sha256_file(path)
    assert first.read_bytes() == second.read_bytes()


def test_checkpoint_cannot_overwrite_dataset_schema(dataset):
    output = dataset / 'input_schema.json'
    before = output.read_bytes()
    with pytest.raises(ValueError, match='overlapping'):
        training.train(dataset, epochs=1, output=output)
    assert output.read_bytes() == before

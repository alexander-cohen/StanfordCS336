import io
import random

import numpy as np
import pytest
import torch

from cs336_basics import training
from cs336_basics.run_io import atomic_save


def seed():
    random.seed(12)
    np.random.seed(12)
    torch.manual_seed(12)


def configs():
    return (
        dict(vocab_size=16, context_length=4, num_heads=2, num_layers=1, d_model=8, d_ff=16, rope_theta=10000.0),
        dict(batch_size=2, steps=4, grad_clip=1.0, lr_mode="basic", learning_rate=0.001),
    )


def test_resume_matches_uninterrupted(tmp_path, capsys):
    model_config, train_config = configs()
    data = np.arange(64) % 16
    seed()
    original, continued = [], []
    full = training.training_loop(data, model_config, train_config, tmp_path / "full", verbose=False,
                                  checkpoint_every=2, log_metrics=original.append)
    seed()
    first = training.training_loop(
        data, model_config, train_config, tmp_path / "first", verbose=False, checkpoint_every=2, stop_after=2)
    # Perturb all generators to ensure resume actually restores the checkpoint.
    random.random()
    np.random.random()
    torch.rand(3)
    resumed = training.training_loop(
        data, model_config, train_config, tmp_path / "resumed", verbose=False, checkpoint_every=2,
        resume_from=tmp_path / "first" / "2.pt", log_metrics=continued.append)
    assert resumed["run_id"] == first["run_id"]
    for key, tensor in full["model"].state_dict().items():
        torch.testing.assert_close(tensor, resumed["model"].state_dict()[key], rtol=0, atol=0)
    from tests.test_serialization import are_optimizers_equal
    assert are_optimizers_equal(full["optimizer"].state_dict(), resumed["optimizer"].state_dict(), 0, 0)
    assert [r["step"] for r in continued] == [3, 4]
    assert [r["train_loss"] for r in original[2:]] == [r["train_loss"] for r in continued]
    saved = torch.load(tmp_path / "resumed" / "4.pt", weights_only=True)
    assert saved["format_version"] == training.CHECKPOINT_FORMAT_VERSION
    assert saved["tokens_seen"] == 32
    assert "cross_entropy" not in saved
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("version", [None, training.CHECKPOINT_FORMAT_VERSION + 1])
def test_resume_rejects_unsupported_format(tmp_path, version):
    checkpoint = {} if version is None else {"format_version": version}
    path = tmp_path / "unsupported.pt"
    torch.save(checkpoint, path)
    model_config, train_config = configs()
    with pytest.raises(ValueError, match=f"version: {version!r}; expected {training.CHECKPOINT_FORMAT_VERSION}"):
        training.training_loop(
            np.arange(64) % 16, model_config, train_config, verbose=False,
            resume_from=path,
        )


def test_atomic_failure_preserves_previous_checkpoint(tmp_path, monkeypatch):
    destination = tmp_path / "checkpoint.pt"
    atomic_save({"iteration": 2}, destination)
    original = destination.read_bytes()

    def fail(payload, stream):
        stream.write(b"partial")
        raise OSError("disk failure")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk failure"):
        atomic_save({"iteration": 4}, destination)
    assert destination.read_bytes() == original
    assert list(tmp_path.iterdir()) == [destination]


def test_file_object_checkpoint_interface():
    model = torch.nn.Linear(2, 2)
    opt = training.AdamW(model.parameters())
    stream = io.BytesIO()
    training.save_checkpoint(model, opt, 7, stream)
    stream.seek(0)
    assert training.load_checkpoint(stream, model, opt) == 7


def test_resume_rejects_changed_data(tmp_path):
    model_config, train_config = configs()
    data = np.arange(64) % 16
    training.training_loop(data, model_config, train_config, tmp_path / "first",
                           verbose=False, checkpoint_every=2, stop_after=2)
    data[0] = 7
    with pytest.raises(ValueError, match="differs"):
        training.training_loop(data, model_config, train_config, tmp_path / "resumed",
                               verbose=False, checkpoint_every=2, resume_from=tmp_path / "first" / "2.pt")
    assert not (tmp_path / "resumed").exists()


def test_failed_checkpoint_is_not_reported_as_saved(tmp_path, monkeypatch):
    model_config, train_config = configs()

    def fail(*args, **kwargs):
        raise OSError("disk failure")

    monkeypatch.setattr(training, "save_checkpoint", fail)
    with pytest.raises(OSError, match="disk failure"):
        training.training_loop(np.arange(64) % 16, model_config, train_config,
                               tmp_path / "run", verbose=False, checkpoint_every=2)
    assert not list((tmp_path / "run").glob("*.pt"))


def test_cadence_final_checkpoint_and_no_overwrite(tmp_path):
    model_config, train_config = configs()
    data = np.arange(64) % 16
    records = []
    training.training_loop(data, model_config, train_config, tmp_path / "run", verbose=False,
                           checkpoint_every=2, log_every=3, log_metrics=records.append)
    assert [record["step"] for record in records] == [3, 4]
    assert (tmp_path / "run" / "4.pt").is_file()
    with pytest.raises(FileExistsError):
        training.training_loop(data, model_config, train_config, tmp_path / "run", verbose=False, checkpoint_every=2)

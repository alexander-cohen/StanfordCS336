from types import SimpleNamespace

import numpy as np

import cs336_scaling.schemas  # noqa: F401
from cs336_scaling.tokenized_data import DCLMDataResult
from cs336_scaling.training.data import download_data


def test_validation_and_training_with_same_basename_stay_separate(tmp_path):
    source = tmp_path / "source"
    (source / "validation").mkdir(parents=True)
    (source / "train").mkdir()
    validation = source / "validation" / "0.npy"
    np.save(validation, np.full(513, 17, dtype=np.uint16))
    training = [source / "train" / f"{i}.npy" for i in range(2)]
    for i, path in enumerate(training):
        np.save(path, np.full(1024, i + 23, dtype=np.uint16))
    data = DCLMDataResult(
        validation_filepath=validation,
        train_filepaths=training,
        n_train_tokens=2048,
        n_validation_tokens=513,
        n_source_chunks=1,
    )
    config = SimpleNamespace(total_train_tokens=512, eval_every_tokens=512)
    local = tmp_path / "staged"
    staged = download_data(data, config, local_data_dir=local)
    assert staged.validation_filepath not in staged.train_filepaths
    np.testing.assert_array_equal(
        np.load(staged.validation_filepath), np.load(validation)
    )
    for actual, expected in zip(staged.train_filepaths, training, strict=True):
        np.testing.assert_array_equal(np.load(actual), np.load(expected))
    # Reusing a staged directory must preserve the separate data files too.
    assert download_data(data, config, local_data_dir=local) == staged

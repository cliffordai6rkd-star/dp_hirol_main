"""Overwriting a run resets its artifacts without disturbing other paths."""

from contextlib import contextmanager
import logging

import pytest

from diffusion_policy.common.training_output import reset_training_output


@contextmanager
def _active_file_logger(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"training-output-test.{path}")
    old_level, old_propagate = logger.level, logger.propagate
    handler = logging.FileHandler(path)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        yield logger, handler
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(old_level)
        logger.propagate = old_propagate


def test_reset_clears_only_selected_run_and_preserves_current_hydra_metadata(tmp_path):
    output = tmp_path / "selected"
    metadata = output / ".hydra"
    metadata.mkdir(parents=True)
    (metadata / "config.yaml").write_text("current: configuration\n")
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "best.ckpt").write_bytes(b"old checkpoint")
    (output / "logs.json.txt").write_text('{"train_loss": 1}\n')
    (output / "training_losses.png").write_bytes(b"old plot")
    sibling = tmp_path / "another_run"
    sibling.mkdir()
    (sibling / "best.ckpt").write_bytes(b"unrelated checkpoint")

    reset_training_output(output)

    assert sorted(path.name for path in output.iterdir()) == [".hydra"]
    assert (metadata / "config.yaml").read_text() == "current: configuration\n"
    assert (sibling / "best.ckpt").read_bytes() == b"unrelated checkpoint"


@pytest.mark.parametrize("relative_log", ["train.log", "nested/live.log"])
def test_reset_truncates_active_file_log_and_logging_continues_at_same_path(tmp_path, relative_log):
    output = tmp_path / "run"
    path = output / relative_log
    with _active_file_logger(path) as (logger, handler):
        logger.info("old run")
        handler.flush()
        obsolete = path.parent / "obsolete.txt"
        obsolete.write_text("old artifact")

        reset_training_output(output)

        assert path.is_file()
        assert path.read_text() == ""
        assert not obsolete.exists()
        logger.info("new run")
        handler.flush()
        assert path.read_text() == "new run\n"


def test_reset_does_not_truncate_active_log_outside_selected_run(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "old.txt").write_text("remove")
    unrelated_log = tmp_path / "other" / "train.log"
    with _active_file_logger(unrelated_log) as (logger, handler):
        logger.info("outside run")
        handler.flush()

        reset_training_output(output)

        assert unrelated_log.read_text() == "outside run\n"
        assert list(output.iterdir()) == []


def test_reset_with_no_hydra_subdirectory_preserves_no_metadata(tmp_path):
    output = tmp_path / "run"
    metadata = output / ".hydra"
    metadata.mkdir(parents=True)
    (metadata / "config.yaml").write_text("old: true\n")

    reset_training_output(output, hydra_output_subdir=None)

    assert output.is_dir()
    assert list(output.iterdir()) == []


def test_reset_preserves_configured_hydra_subdirectory(tmp_path):
    output = tmp_path / "run"
    metadata = output / "run_metadata"
    metadata.mkdir(parents=True)
    (metadata / "config.yaml").write_text("current: true\n")
    (output / "old.txt").write_text("remove")

    reset_training_output(output, hydra_output_subdir="run_metadata")

    assert [path.name for path in output.iterdir()] == ["run_metadata"]
    assert (metadata / "config.yaml").read_text() == "current: true\n"


def test_reset_unlinks_external_symlinks_without_touching_their_targets(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    external_file = external / "keep.txt"
    external_file.write_text("keep external data")
    directory_link = output / "linked_directory"
    directory_link.symlink_to(external, target_is_directory=True)
    file_link = output / "linked_file"
    file_link.symlink_to(external_file)

    reset_training_output(output)

    assert not directory_link.is_symlink()
    assert not file_link.is_symlink()
    assert external.is_dir()
    assert external_file.read_text() == "keep external data"
    assert list(output.iterdir()) == []


@pytest.mark.parametrize("protected_relative", [".", "protected", "protected/nested"])
def test_reset_refuses_protected_path_or_its_ancestor_before_removing_anything(tmp_path, protected_relative):
    output = tmp_path / "run"
    protected = output / protected_relative
    protected.mkdir(parents=True)
    (protected / "keep.txt").write_text("protected")
    (output / "existing.txt").write_text("must remain on refusal")

    with pytest.raises(ValueError):
        reset_training_output(output, protected_paths=(protected,))

    assert (protected / "keep.txt").read_text() == "protected"
    assert (output / "existing.txt").read_text() == "must remain on refusal"


def test_reset_allows_unrelated_protected_path(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "old.txt").write_text("remove")
    protected = tmp_path / "source"
    protected.mkdir()
    (protected / "keep.txt").write_text("source")

    reset_training_output(output, protected_paths=(protected,))

    assert list(output.iterdir()) == []
    assert (protected / "keep.txt").read_text() == "source"


@pytest.mark.parametrize("already_exists", [False, True])
def test_reset_accepts_new_or_empty_output_directory(tmp_path, already_exists):
    output = tmp_path / "run"
    if already_exists:
        output.mkdir()

    reset_training_output(output)

    assert output.is_dir()
    assert list(output.iterdir()) == []

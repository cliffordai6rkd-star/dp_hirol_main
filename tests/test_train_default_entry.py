"""The no-argument training entry dispatches before loading the Hydra workspace."""

from pathlib import Path
import runpy
import sys
from types import ModuleType

import pytest


def test_no_argument_train_dispatches_to_teacher_student_pipeline(monkeypatch):
    entry = Path(__file__).resolve().parents[1] / "train.py"
    calls = []
    pipeline = ModuleType("train_pipeline")

    def main():
        calls.append(list(sys.argv))
        # The pipeline owns execution; no legacy Hydra setup should follow.
        raise SystemExit(71)

    pipeline.main = main
    monkeypatch.setitem(sys.modules, "train_pipeline", pipeline)
    monkeypatch.setattr(sys, "argv", [str(entry)])

    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(entry), run_name="__main__")

    assert stopped.value.code == 71
    assert calls == [[str(entry)]]

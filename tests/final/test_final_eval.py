from pathlib import Path

import numpy as np
import pytest

from churnshield.final_eval import _bootstrap_precision, run_final_evaluation


def test_final_evaluation_requires_explicit_authorization(tmp_path):
    with pytest.raises(PermissionError, match="S01 está bloqueado"):
        run_final_evaluation(output_dir=tmp_path)
    assert not (Path("logs") / "reserved_access.log").exists()


def test_final_evaluation_cli_help_mentions_confirmation():
    import sys
    from subprocess import run
    import os

    env = {**os.environ, "PYTHONPATH": "src"}
    result = run(
        [sys.executable, "-m", "churnshield", "evaluate-final", "--help"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0
    assert "--confirm-final" in result.stdout


def test_bootstrap_precision_is_independent_of_input_order():
    forward = _bootstrap_precision(np.array([1, 0, 1, 0, 0]))
    reverse = _bootstrap_precision(np.array([0, 0, 1, 0, 1]))

    assert forward == reverse


@pytest.mark.parametrize(
    "labels",
    [np.array([[0, 1]]), np.array([0, 0.5, 1]), np.array(["no", "yes"])],
)
def test_bootstrap_precision_rejects_invalid_labels(labels):
    with pytest.raises(ValueError, match="unidimensional"):
        _bootstrap_precision(labels)

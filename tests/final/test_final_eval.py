from pathlib import Path

import pytest

from churnshield.final_eval import run_final_evaluation


def test_final_evaluation_requires_explicit_authorization(tmp_path):
    with pytest.raises(PermissionError, match="S01 está bloqueado"):
        run_final_evaluation(output_dir=tmp_path)
    assert not (Path("logs") / "reserved_access.log").exists()


def test_final_evaluation_cli_help_mentions_confirmation():
    from subprocess import run
    import os

    env = {**os.environ, "PYTHONPATH": "src"}
    result = run(
        ["python3", "-m", "churnshield", "evaluate-final", "--help"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0
    assert "--confirm-final" in result.stdout

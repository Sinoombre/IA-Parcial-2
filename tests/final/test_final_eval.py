from pathlib import Path

import pytest

from churnshield.final_eval import run_final_evaluation


def test_final_evaluation_requires_explicit_authorization(tmp_path):
    with pytest.raises(PermissionError, match="S01 está bloqueado"):
        run_final_evaluation(output_dir=tmp_path)
    assert not (Path("logs") / "reserved_access.log").exists()


def test_final_evaluation_requires_hash_pinned_authorization(tmp_path):
    with pytest.raises(PermissionError, match="authorization-file"):
        run_final_evaluation(
            output_dir=tmp_path / "final",
            confirm_final=True,
            authorization_file=None,
        )
    assert not (tmp_path / "final").exists()


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


def test_final_evaluation_cli_reports_missing_authorization_without_traceback(
    tmp_path,
):
    import os
    import sys
    from subprocess import run

    env = {**os.environ, "PYTHONPATH": "src"}
    result = run(
        [
            sys.executable,
            "-m",
            "churnshield",
            "evaluate-final",
            "--confirm-final",
            "--output-dir",
            str(tmp_path / "final"),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode != 0
    assert "authorization-file" in result.stderr
    assert "Traceback" not in result.stderr

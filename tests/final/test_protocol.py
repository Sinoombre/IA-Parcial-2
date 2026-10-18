import json

import pytest

from churnshield import protocol


def test_authorization_requires_two_distinct_approvers_and_current_fingerprints(
    monkeypatch, tmp_path
):
    fingerprints = {
        "commit_hash": "commit",
        "dataset_sha256": "dataset",
        "partition_sha256": "partition",
        "config_sha256": "config",
        "thresholds_sha256": "thresholds",
    }
    monkeypatch.setattr(protocol, "protocol_fingerprints", lambda: fingerprints)
    authorization = {
        "authorized": True,
        "approvers": ["Alice", "alice"],
        "authorization_reference": "approval-17",
        "authorized_at": "2026-10-18T08:00:00Z",
        **fingerprints,
    }
    approval_file = tmp_path / "approval.json"
    approval_file.write_text(json.dumps(authorization), encoding="utf-8")

    with pytest.raises(PermissionError, match="dos aprobadores distintos"):
        protocol.validate_final_authorization(
            approval_file, output_dir=tmp_path / "new-output"
        )


def test_authorization_rejects_changed_partition_hash(monkeypatch, tmp_path):
    fingerprints = {
        "commit_hash": "commit",
        "dataset_sha256": "dataset",
        "partition_sha256": "expected-partition",
        "config_sha256": "config",
        "thresholds_sha256": "thresholds",
    }
    monkeypatch.setattr(protocol, "protocol_fingerprints", lambda: fingerprints)
    authorization = {
        "authorized": True,
        "approvers": ["Alice", "Bob"],
        "authorization_reference": "approval-17",
        "authorized_at": "2026-10-18T08:00:00+00:00",
        **{**fingerprints, "partition_sha256": "old-partition"},
    }
    approval_file = tmp_path / "approval.json"
    approval_file.write_text(json.dumps(authorization), encoding="utf-8")

    with pytest.raises(PermissionError, match="partition_sha256"):
        protocol.validate_final_authorization(
            approval_file, output_dir=tmp_path / "new-output"
        )


def test_authorization_rejects_reused_output_directory(monkeypatch, tmp_path):
    fingerprints = {
        "commit_hash": "commit",
        "dataset_sha256": "dataset",
        "partition_sha256": "partition",
        "config_sha256": "config",
        "thresholds_sha256": "thresholds",
    }
    monkeypatch.setattr(protocol, "protocol_fingerprints", lambda: fingerprints)
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(PermissionError, match="debe ser nueva"):
        protocol.validate_final_authorization(
            tmp_path / "unused.json", output_dir=output
        )

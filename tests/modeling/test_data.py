import pytest
import pandas as pd

from churnshield.data import (
    EXPECTED_PARTITION_SHA256,
    dataset_path,
    load_dev,
    load_reserved,
    partition_path,
    partition_sha256,
)


def test_reserved_is_blocked_without_flag():
    with pytest.raises(PermissionError, match="authorization-file"):
        load_reserved(authorization_file=None, output_dir="reserved-not-created")


def test_dataset_dev_contract(monkeypatch):
    import churnshield.data as data

    original_read = data._read_data_rows
    observed_splits = []

    def development_only_reader(indices, *, split):
        observed_splits.append(split)
        assert split == "development"
        return original_read(indices, split=split)

    monkeypatch.setattr(data, "_read_data_rows", development_only_reader)
    df, meta = load_dev()
    assert len(df) == 5634
    assert meta["reserved_count"] == 1409
    assert meta["total_charges_blank_count"] == 5
    assert meta["partition_sha256"] == EXPECTED_PARTITION_SHA256
    assert observed_splits == ["development"]


def test_paths_are_anchored_to_repository(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    assert dataset_path().is_absolute()
    assert partition_path().is_absolute()
    assert dataset_path().is_file()
    assert partition_path().is_file()


def test_partition_hash_is_canonical_and_pinned(tmp_path):
    original = partition_path().read_bytes()
    lf_path = tmp_path / "partition-lf.csv"
    crlf_path = tmp_path / "partition-crlf.csv"
    lf_path.write_bytes(original.replace(b"\r\n", b"\n"))
    crlf_path.write_bytes(original.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))

    assert partition_sha256(lf_path) == EXPECTED_PARTITION_SHA256
    assert partition_sha256(crlf_path) == EXPECTED_PARTITION_SHA256


def test_partition_contents_are_pinned(monkeypatch, tmp_path):
    import churnshield.data as data

    altered = tmp_path / "partition.csv"
    altered.write_bytes(partition_path().read_bytes() + b"\n")
    monkeypatch.setattr(data, "partition_path", lambda: altered)

    with pytest.raises(ValueError, match="SHA-256"):
        data._reserved_indices()


def test_reserved_reader_requires_external_authorization_before_reading(
    monkeypatch, tmp_path
):
    import churnshield.data as data

    def fail_if_reserved_rows_are_read(*args, **kwargs):
        raise AssertionError("reserved rows read before authorization")

    monkeypatch.setattr(data, "_read_data_rows", fail_if_reserved_rows_are_read)
    with pytest.raises(PermissionError, match="No existe"):
        load_reserved(
            authorization_file=tmp_path / "approval.json",
            output_dir=tmp_path / "new-output",
        )
    assert not (tmp_path / "new-output").exists()


def test_reserved_access_is_registered_before_reserved_rows_are_read(
    monkeypatch, tmp_path
):
    import churnshield.data as data
    from churnshield import protocol

    output = tmp_path / "authorized-output"
    monkeypatch.setattr(
        protocol,
        "validate_final_authorization",
        lambda *args, **kwargs: {"authorization_reference": "test"},
    )
    monkeypatch.setattr(data, "_reserved_indices", lambda: pd.Index([1]))

    def assert_registered(indices, *, split):
        assert split == "reserved"
        assert (output / "reserved_access.json").is_file()
        raise RuntimeError("test stop before reading")

    monkeypatch.setattr(data, "_read_data_rows", assert_registered)
    with pytest.raises(RuntimeError, match="test stop"):
        load_reserved(authorization_file=tmp_path / "approved.json", output_dir=output)

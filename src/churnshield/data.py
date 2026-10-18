"""Validated, repository-anchored access to development and reserved data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SHA256 = "1bcbc0ccc9b352175216979102628e579cfbde2c3ff57b005de168a433122640"
EXPECTED_PARTITION_SHA256 = (
    "d7d8ca6501a7c2ad951f5b6991fe0756ad6df077cbdbcbef62e66e23e49c9514"
)
EXPECTED_DEV_ROWS = 5634
EXPECTED_RESERVED_ROWS = 1409
EXPECTED_COLUMNS = 33


def dataset_path() -> Path:
    return REPO_ROOT / "data" / "Telco_customer_churn.xlsx"


def partition_path() -> Path:
    return REPO_ROOT / "artifacts" / "public" / "data" / "particion_prueba_indices.csv"


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def dataset_sha256(path: Path | None = None) -> str:
    return _sha256(path or dataset_path())


def partition_sha256(path: Path | None = None) -> str:
    contents = (path or partition_path()).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(contents).hexdigest()


def _reserved_indices() -> pd.Index:
    path = partition_path()
    if partition_sha256(path) != EXPECTED_PARTITION_SHA256:
        raise ValueError("SHA-256 de la partición reservada inesperado")

    partition = pd.read_csv(path)
    if list(partition.columns) != ["row_index", "split"]:
        raise ValueError("columnas inesperadas en la partición reservada")
    if len(partition) != EXPECTED_RESERVED_ROWS or not partition["row_index"].is_unique:
        raise ValueError("cantidad o índices inesperados en la partición reservada")
    if (
        not partition["row_index"]
        .between(0, EXPECTED_DEV_ROWS + EXPECTED_RESERVED_ROWS - 1)
        .all()
    ):
        raise ValueError("índice fuera de rango en la partición reservada")
    if set(partition["split"]) != {"test"}:
        raise ValueError("la partición reservada debe contener solo split=test")
    return pd.Index(partition["row_index"].astype(int))


def _read_data_rows(reserved_indices: set[int], *, split: str) -> pd.DataFrame:
    if split not in {"development", "reserved"}:
        raise ValueError("split debe ser 'development' o 'reserved'")

    path = dataset_path()
    dataset_hash = dataset_sha256(path)
    if dataset_hash != EXPECTED_SHA256:
        raise ValueError("SHA-256 del dataset inesperado")

    row_count = EXPECTED_DEV_ROWS if split == "development" else EXPECTED_RESERVED_ROWS

    def skip_row(excel_row: int) -> bool:
        if excel_row == 0:
            return False
        data_index = excel_row - 1
        is_reserved = data_index in reserved_indices
        return is_reserved if split == "development" else not is_reserved

    data = pd.read_excel(path, skiprows=skip_row)
    if data.shape != (row_count, EXPECTED_COLUMNS):
        raise ValueError(f"shape inesperado para {split}: {data.shape}")

    required = {"CustomerID", "Churn Value", "Tenure Months", "Total Charges"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"faltan columnas requeridas: {sorted(missing)}")
    if not data["CustomerID"].is_unique:
        raise ValueError("CustomerID debe ser único")
    if not data["Churn Value"].isin([0, 1]).all():
        raise ValueError("Churn Value debe contener solo valores 0 o 1")

    blanks = data["Total Charges"].astype(str).str.strip().eq("")
    blank_count = int(blanks.sum())
    if not bool((data.loc[blanks, "Tenure Months"] == 0).all()):
        raise ValueError("los cargos totales vacíos deben corresponder a tenure cero")
    data["Total Charges"] = pd.to_numeric(
        data["Total Charges"].replace(r"^\s*$", pd.NA, regex=True),
        errors="coerce",
    )
    data.loc[
        data["Total Charges"].isna() & data["Tenure Months"].eq(0), "Total Charges"
    ] = 0.0
    if data["Total Charges"].isna().any():
        raise ValueError("Total Charges contiene valores inválidos")
    data.attrs["total_charges_blank_count"] = blank_count
    return data


def load_dev() -> tuple[pd.DataFrame, dict]:
    indices = _reserved_indices()
    development = _read_data_rows(set(indices), split="development")
    if len(development) != EXPECTED_DEV_ROWS:
        raise ValueError("cantidad inesperada de filas de desarrollo")
    return development, {
        "dataset_sha256": dataset_sha256(),
        "partition_sha256": partition_sha256(),
        "total_charges_blank_count": development.attrs["total_charges_blank_count"],
        "reserved_count": EXPECTED_RESERVED_ROWS,
        "dev_count": len(development),
        "rows_used": len(development),
        "split": "development",
        "reserved_access": False,
    }


def load_reserved(
    *, authorization_file: str | Path, output_dir: str | Path
) -> pd.DataFrame:
    """Validate and record an authorized S01 request before reading reserved rows."""
    from .protocol import validate_final_authorization

    authorization = validate_final_authorization(
        authorization_file,
        output_dir=output_dir,
    )
    log_path = Path(output_dir).expanduser().resolve() / "reserved_access.json"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise PermissionError(
            "La carpeta de salida S01 fue reservada por otra ejecución"
        ) from exc
    log_path.write_text(
        json.dumps(authorization, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    indices = _reserved_indices()
    return _read_data_rows(set(indices), split="reserved").reset_index(drop=True)

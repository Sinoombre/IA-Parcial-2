"""Carga y validación de datos de desarrollo y de la partición reservada."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

EXPECTED_SHA256 = "1bcbc0ccc9b352175216979102628e579cfbde2c3ff57b005de168a433122640"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(name: str, *fallbacks: str) -> Path:
    candidates = [Path(name), *(Path(x) for x in fallbacks)]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No se encontró ningún archivo: {candidates}")


def dataset_path() -> Path:
    return _resolve(
        "data/Telco_customer_churn.xlsx",
        str(REPO_ROOT / "data/Telco_customer_churn.xlsx"),
    )


def partition_path() -> Path:
    return _resolve(
        "artifacts/data/particion_prueba_indices.csv",
        str(REPO_ROOT / "artifacts/data/particion_prueba_indices.csv"),
        "artifacts/particion_prueba_indices.csv",
        str(REPO_ROOT / "entrega_manus/particion_prueba_indices.csv"),
        "particion_prueba_indices.csv",
    )


def dataset_sha256(path: Path | None = None) -> str:
    with (path or dataset_path()).open("rb") as dataset:
        return hashlib.file_digest(dataset, "sha256").hexdigest()


def _read_and_validate() -> tuple[pd.DataFrame, int, str]:
    path = dataset_path()
    dataset_hash = dataset_sha256(path)
    if dataset_hash != EXPECTED_SHA256:
        raise ValueError("SHA-256 del dataset inesperado")

    raw = pd.read_excel(path)
    if raw.shape != (7043, 33):
        raise ValueError(f"shape inesperado: {raw.shape}")

    blanks = raw["Total Charges"].astype(str).str.strip().eq("")
    if int(raw["Churn Value"].sum()) != 1869:
        raise ValueError("cantidad inesperada de clientes con churn")
    if not raw["CustomerID"].is_unique:
        raise ValueError("CustomerID debe ser único")
    if int(blanks.sum()) != 11:
        raise ValueError("cantidad inesperada de cargos totales vacíos")
    if not bool((raw.loc[blanks, "Tenure Months"] == 0).all()):
        raise ValueError("los cargos totales vacíos deben corresponder a tenure cero")

    raw["Total Charges"] = pd.to_numeric(
        raw["Total Charges"].replace(r"^\s*$", pd.NA, regex=True),
        errors="coerce",
    )
    raw.loc[
        raw["Total Charges"].isna() & raw["Tenure Months"].eq(0), "Total Charges"
    ] = 0.0
    return raw, int(blanks.sum()), dataset_hash


def _reserved_indices() -> pd.Index:
    part = pd.read_csv(partition_path())
    if list(part.columns) != ["row_index", "split"]:
        raise ValueError("columnas inesperadas en la partición reservada")
    if len(part) != 1409 or not part["row_index"].is_unique:
        raise ValueError("cantidad o índices inesperados en la partición reservada")
    if not part["row_index"].between(0, 7042).all():
        raise ValueError("índice fuera de rango en la partición reservada")
    # El archivo existente usa split=test; semánticamente sigue siendo la prueba reservada.
    if set(part["split"]) != {"test"}:
        raise ValueError("la partición reservada debe contener solo split=test")
    return pd.Index(part["row_index"].astype(int))


def load_dev() -> tuple[pd.DataFrame, dict]:
    df, blank_count, dataset_hash = _read_and_validate()
    return _partition_development(df, blank_count, dataset_hash)


def load_dev_and_full() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Carga el dataset una sola vez y devuelve desarrollo y población completa."""
    full, blank_count, dataset_hash = _read_and_validate()
    dev, meta = _partition_development(full, blank_count, dataset_hash)
    return dev, full, meta


def _partition_development(
    df: pd.DataFrame,
    blank_count: int,
    dataset_hash: str,
) -> tuple[pd.DataFrame, dict]:
    reserved = _reserved_indices()
    dev = df.drop(index=reserved).reset_index(drop=True)
    if len(dev) != 5634:
        raise ValueError(f"cantidad inesperada de filas de desarrollo: {len(dev)}")
    return dev, {
        "dataset_sha256": dataset_hash,
        "total_charges_blank_count": blank_count,
        "reserved_count": len(reserved),
        "dev_count": len(dev),
    }


def load_full_for_internal_split() -> tuple[pd.DataFrame, dict]:
    """Carga las 7043 filas para una partición interna de desarrollo 70/30.

    Esta función no lee ni calcula la partición reservada. La separación 70/30
    sirve únicamente para diagnosticar arreglos de supuestos antes de congelar
    el modelo final.
    """

    df, blank_count, dataset_hash = _read_and_validate()
    return df, {
        "dataset_sha256": dataset_hash,
        "total_charges_blank_count": blank_count,
        "rows_used": len(df),
        "split": "internal_70_30",
    }


def load_reserved(*, final_evaluation: bool = False) -> pd.DataFrame:
    if not final_evaluation:
        raise PermissionError("load_reserved requiere final_evaluation=True")

    log = Path("logs/reserved_access.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("reserved_access_requested\n")
    df, _, _ = _read_and_validate()
    return df.loc[_reserved_indices()].copy()

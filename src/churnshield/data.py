"""Carga y validación de datos para desarrollo, la prueba reservada en este caso quedará bloqueada."""
from __future__ import annotations
from pathlib import Path
import hashlib
import pandas as pd

EXPECTED_SHA256 = "1bcbc0ccc9b352175216979102628e579cfbde2c3ff57b005de168a433122640" 
REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(name: str, *fallbacks: str) -> Path:
    
    candidates = [Path(name), *(Path(x) for x in fallbacks)]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No se encontró ningún archivo: {candidates}")


def dataset_path() -> Path:
    # RUTA FUTURA: si cambia el dataset de la regresión real, modificar solo data.dataset_path en config/agente.yaml.
    return _resolve("data/Telco_customer_churn.xlsx", str(REPO_ROOT / "data/Telco_customer_churn.xlsx"))


def partition_path() -> Path:
    return _resolve("artifacts/particion_prueba_indices.csv", str(REPO_ROOT / "entrega_manus/particion_prueba_indices.csv"), "particion_prueba_indices.csv")


def dataset_sha256(path: Path | None = None) -> str:
    return hashlib.sha256((path or dataset_path()).read_bytes()).hexdigest()


def _read_and_validate() -> tuple[pd.DataFrame, int]:

    path = dataset_path()
    assert dataset_sha256(path) == EXPECTED_SHA256, "SHA-256 del dataset inesperado"
    raw = pd.read_excel(path)
    assert raw.shape == (7043, 33), f"shape inesperado: {raw.shape}"
    blanks = raw["Total Charges"].astype(str).str.strip().eq("")
    assert int(raw["Churn Value"].sum()) == 1869
    assert raw["CustomerID"].is_unique
    assert int(blanks.sum()) == 11
    assert bool((raw.loc[blanks, "Tenure Months"] == 0).all())
    raw["Total Charges"] = pd.to_numeric(raw["Total Charges"].replace(r"^\s*$", pd.NA, regex=True), errors="coerce")
    raw.loc[raw["Total Charges"].isna() & raw["Tenure Months"].eq(0), "Total Charges"] = 0.0
    return raw, int(blanks.sum())


def _reserved_indices() -> pd.Index:

    part = pd.read_csv(partition_path())
    assert list(part.columns) == ["row_index", "split"]
    assert len(part) == 1409 and part["row_index"].is_unique
    assert part["row_index"].between(0, 7042).all()
    # El archivo existente usa split=test; semánticamente sigue siendo la prueba reservada.
    assert set(part["split"]) == {"test"}
    return pd.Index(part["row_index"].astype(int))


def load_dev() -> tuple[pd.DataFrame, dict]:
    df, blank_count = _read_and_validate()
    reserved = _reserved_indices()
    dev = df.drop(index=reserved).reset_index(drop=True)
    assert len(dev) == 5634
    return dev, {"dataset_sha256": dataset_sha256(), "total_charges_blank_count": blank_count, "reserved_count": len(reserved), "dev_count": len(dev)}


def load_full_for_internal_split() -> tuple[pd.DataFrame, dict]:
    """Carga las 7043 filas para una partición interna de desarrollo 70/30.

    Esta función no lee ni calcula la partición reservada. La separación 70/30
    sirve únicamente para diagnosticar arreglos de supuestos antes de congelar
    el modelo final.
    """

    df, blank_count = _read_and_validate()
    return df.reset_index(drop=True), {
        "dataset_sha256": dataset_sha256(),
        "total_charges_blank_count": blank_count,
        "rows_used": len(df),
        "split": "internal_70_30",
    }


def load_reserved(*, final_evaluation: bool = False) -> pd.DataFrame:
    if not final_evaluation:
        raise PermissionError("load_reserved requiere final_evaluation=True")
    
    log = Path("logs/reserved_access.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.open("a").write("reserved_access_requested\n")
    df, _ = _read_and_validate()
    return df.loc[_reserved_indices()].copy()

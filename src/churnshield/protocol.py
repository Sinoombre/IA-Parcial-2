"""Hash-pinned preflight and authorization for the reserved S01 evaluation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

from .data import REPO_ROOT, dataset_sha256, partition_sha256
from .settings import AGENT_CONFIG_PATH, THRESHOLDS_CONFIG_PATH


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _clean_commit() -> str:
    status = _git("status", "--porcelain", "--untracked-files=normal")
    if status:
        raise PermissionError("S01 requiere un árbol Git limpio")
    return _git("rev-parse", "HEAD")


def protocol_fingerprints() -> dict[str, str]:
    return {
        "commit_hash": _clean_commit(),
        "dataset_sha256": dataset_sha256(),
        "partition_sha256": partition_sha256(),
        "config_sha256": _sha256(AGENT_CONFIG_PATH),
        "thresholds_sha256": _sha256(THRESHOLDS_CONFIG_PATH),
    }


def final_authorization_template() -> dict:
    return {
        "authorized": False,
        "approvers": ["", ""],
        "authorization_reference": "",
        "authorized_at": "",
        **protocol_fingerprints(),
    }


def validate_final_authorization(
    authorization_file: str | Path | None,
    *,
    output_dir: str | Path,
) -> dict:
    if authorization_file is None:
        raise PermissionError("S01 requiere --authorization-file con aprobación")

    output_path = Path(output_dir).expanduser().resolve()
    if output_path.exists():
        raise PermissionError("La carpeta de salida S01 debe ser nueva e inexistente")

    authorization_path = Path(authorization_file).expanduser().resolve()
    if REPO_ROOT == authorization_path or REPO_ROOT in authorization_path.parents:
        raise PermissionError("El archivo de autorización debe estar fuera del repo")
    if not authorization_path.is_file():
        raise PermissionError("No existe el archivo externo de autorización S01")

    try:
        authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PermissionError(
            "El archivo de autorización S01 no es JSON válido"
        ) from exc
    if (
        not isinstance(authorization, dict)
        or authorization.get("authorized") is not True
    ):
        raise PermissionError("La autorización S01 debe confirmar authorized: true")

    approvers = authorization.get("approvers")
    if (
        not isinstance(approvers, list)
        or len(approvers) != 2
        or not all(isinstance(name, str) and name.strip() for name in approvers)
        or len({name.strip().casefold() for name in approvers}) != 2
    ):
        raise PermissionError("S01 requiere dos aprobadores distintos")
    if not isinstance(authorization.get("authorization_reference"), str) or not (
        authorization["authorization_reference"].strip()
    ):
        raise PermissionError("S01 requiere una referencia de aprobación")

    authorized_at = authorization.get("authorized_at")
    if not isinstance(authorized_at, str):
        raise PermissionError("S01 requiere fecha de autorización con zona horaria")
    try:
        parsed_at = datetime.fromisoformat(authorized_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PermissionError("Fecha de autorización S01 inválida") from exc
    if parsed_at.tzinfo is None or parsed_at.utcoffset() is None:
        raise PermissionError("La fecha de autorización S01 requiere zona horaria")

    expected = protocol_fingerprints()
    mismatches = [
        key for key, value in expected.items() if authorization.get(key) != value
    ]
    if mismatches:
        raise PermissionError(
            f"La autorización S01 no coincide con el protocolo actual: {mismatches}"
        )
    return {
        **expected,
        "approvers": [name.strip() for name in approvers],
        "authorization_reference": authorization["authorization_reference"].strip(),
        "authorized_at": authorized_at,
    }

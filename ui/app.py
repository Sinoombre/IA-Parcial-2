"""Interfaz local de demostración para predicciones de desarrollo.

Este módulo nunca carga la partición reservada. Usa únicamente el modelo de
`artifacts/public/dev_eval` y declara esa limitación en cada respuesta.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import BadRequest, UnsupportedMediaType

from churnshield.data import load_dev
from churnshield.features import ALLOWED_PREDICTORS
from churnshield.settings import AGENT_CONFIG_PATH, THRESHOLDS_CONFIG_PATH
from churnshield.utility import action_utility

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(
    os.environ.get(
        "CHURNSHIELD_MODEL",
        ROOT / "artifacts/public/dev_eval/stacking/model_v1.joblib",
    )
)
CARD_PATH = MODEL_PATH.with_name("model_card.json")
DEFAULTS = {
    "Partner": "No",
    "Dependents": "No",
    "Internet Service": "DSL",
    "Contract": "Month-to-month",
    "Paperless Billing": "Yes",
    "Payment Method": "Electronic check",
    "Tenure Months": 12,
    "Monthly Charges": 70.0,
    "Total Charges": 840.0,
}
NUMERIC_FIELDS = ("Tenure Months", "Monthly Charges", "Total Charges")
MAX_GROUP_SIZE = 500

app = Flask(__name__)


class ArtifactUnavailable(RuntimeError):
    """The development model artifact cannot safely serve predictions."""


def _model_card() -> dict:
    if not CARD_PATH.is_file():
        raise ArtifactUnavailable(
            f"No existe {CARD_PATH}. Ejecuta `PYTHONPATH=src python -m churnshield dev-eval`."
        )
    try:
        card = json.loads(CARD_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactUnavailable("El model card falta o no es JSON válido.") from exc
    if not isinstance(card, dict):
        raise ArtifactUnavailable("El model card debe contener un objeto JSON.")
    return card


def _validated_model_card() -> dict:
    card = _model_card()
    if (
        card.get("features") != list(ALLOWED_PREDICTORS)
        or card.get("reserved_evaluation_executed") is not False
    ):
        raise ArtifactUnavailable(
            "El model card no coincide con el contrato de features."
        )
    for key, path in (
        ("config_sha256", AGENT_CONFIG_PATH),
        ("thresholds_sha256", THRESHOLDS_CONFIG_PATH),
    ):
        if card.get(key) != _sha256(path):
            raise ArtifactUnavailable(
                f"El hash {key} no coincide con la configuración."
            )
    try:
        mean_cltv = float(card["mean_cltv_train"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactUnavailable(
            "El model card no tiene mean_cltv_train válido."
        ) from exc
    if not math.isfinite(mean_cltv) or mean_cltv <= 0:
        raise ArtifactUnavailable("mean_cltv_train debe ser positivo y finito.")
    return card


def _sha256(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()
    except OSError as exc:
        raise ArtifactUnavailable(f"No se pudo validar el artefacto {path}.") from exc


def _load_model():
    if not MODEL_PATH.is_file():
        raise ArtifactUnavailable(
            f"No existe {MODEL_PATH}. Ejecuta `PYTHONPATH=src python -m churnshield dev-eval`."
        )
    card = _validated_model_card()
    if card.get("model_sha256") != _sha256(MODEL_PATH):
        raise ArtifactUnavailable("El modelo no coincide con el model card.")
    try:
        model = joblib.load(MODEL_PATH)
    except (OSError, EOFError, ValueError, pickle.UnpicklingError, ImportError) as exc:
        raise ArtifactUnavailable(
            "El artefacto del modelo no se puede cargar."
        ) from exc
    return model


def _coerce_payload(payload: dict) -> dict:
    unexpected = sorted(set(payload) - set(ALLOWED_PREDICTORS))
    if unexpected:
        raise ValueError(f"Campos no permitidos: {unexpected}")
    values = {**DEFAULTS, **payload}
    missing = [field for field in ALLOWED_PREDICTORS if field not in values]
    if missing:
        raise ValueError(f"Faltan campos permitidos: {missing}")
    for field in NUMERIC_FIELDS:
        values[field] = float(values[field])
        if not math.isfinite(values[field]):
            raise ValueError(f"{field} debe ser un número finito.")
    if not 0 <= values["Tenure Months"] <= 72:
        raise ValueError("Tenure Months debe estar entre 0 y 72.")
    if values["Monthly Charges"] < 0 or values["Total Charges"] < 0:
        raise ValueError("Los cargos no pueden ser negativos.")
    return {field: values[field] for field in ALLOWED_PREDICTORS}


def _seeded_profiles(seed: int, count: int = 3) -> list[dict]:
    development, _ = load_dev()
    if count > len(development):
        raise ValueError(
            "El tamaño del grupo supera los perfiles de desarrollo disponibles."
        )
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(development), size=count, replace=False)
    profiles = []
    for position, row_index in enumerate(selected, start=1):
        row = development.iloc[int(row_index)]
        profile = {}
        for field in ALLOWED_PREDICTORS:
            value = row[field]
            if field == "Tenure Months":
                value = int(value)
            elif field in NUMERIC_FIELDS:
                value = float(value)
            else:
                value = str(value)
            profile[field] = value
        profiles.append({"name": f"Perfil {position:02d}", "values": profile})
    return profiles


@app.get("/")
def index():
    return render_template("index.html", fields=ALLOWED_PREDICTORS, defaults=DEFAULTS)


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "mode": "development_only",
            "model_exists": MODEL_PATH.exists(),
            "reserved_evaluation_executed": False,
            "message": "La reserva permanece bloqueada.",
        }
    )


@app.get("/api/sample")
def sample_profiles():
    raw_seed = request.args.get("seed", "")
    try:
        seed = int(raw_seed)
        if not 0 <= seed <= 2**32 - 1:
            raise ValueError("La semilla debe estar entre 0 y 4294967295.")
        raw_count = request.args.get("count", "3")
        count = int(raw_count)
        if not 1 <= count <= MAX_GROUP_SIZE:
            raise ValueError(
                f"El tamaño del grupo debe estar entre 1 y {MAX_GROUP_SIZE}."
            )
        profiles = _seeded_profiles(seed, count)
        return jsonify(
            {
                "seed": seed,
                "count": count,
                "source": "development_only",
                "reserved_evaluation_executed": False,
                "profiles": profiles,
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/predict")
def predict():
    if not request.is_json:
        return jsonify({"error": "El cuerpo debe enviarse como JSON."}), 400
    try:
        payload = request.get_json(silent=False)
        if not isinstance(payload, dict) or not payload:
            raise ValueError("El cuerpo JSON debe ser un objeto no vacío.")
        if isinstance(payload, dict) and "profiles" in payload:
            profiles = payload["profiles"]
            if (
                not isinstance(profiles, list)
                or not profiles
                or len(profiles) > MAX_GROUP_SIZE
            ):
                raise ValueError(
                    f"El grupo debe contener entre 1 y {MAX_GROUP_SIZE} perfiles."
                )
            named_values = []
            for index, item in enumerate(profiles, start=1):
                if not isinstance(item, dict):
                    raise ValueError("Cada perfil del grupo debe ser un objeto.")
                name = str(item.get("name") or f"Perfil {index:02d}")
                profile_values = item.get(
                    "values",
                    {key: value for key, value in item.items() if key != "name"},
                )
                values = _coerce_payload(profile_values)
                named_values.append((name, values))
            return _predict_group(named_values)

        values = _coerce_payload(payload)
        prediction = _predict_profiles([("Perfil", values)])["predictions"][0]
        return {
            **prediction,
            "mode": "development_only",
            "reserved_evaluation_executed": False,
            "disclaimer": (
                "Riesgo estimado, no causalidad; esta interfaz no usa la reserva."
            ),
        }
    except (BadRequest, UnsupportedMediaType):
        return jsonify({"error": "El cuerpo JSON está malformado."}), 400
    except ArtifactUnavailable as exc:
        return jsonify({"error": str(exc)}), 503
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400


def _predict_profiles(named_values: list[tuple[str, dict]]) -> dict:
    model = _load_model()
    card = _validated_model_card()
    features = pd.DataFrame([values for _, values in named_values])
    probabilities = model.predict_proba(features)[:, 1]
    mean_cltv = float(card["mean_cltv_train"])
    predictions = []
    for (name, values), probability in zip(named_values, probabilities):
        p_churn = float(probability)
        v_cliente = float(values["Total Charges"]) / mean_cltv
        utility = action_utility(p_churn, v_cliente)
        predictions.append(
            {
                "name": name,
                "p_churn": p_churn,
                "v_cliente": v_cliente,
                "u_act": utility,
                "u_noact": 0.0,
                "eligible_by_utility": utility > 0,
                "recommendation": "revisar campaña" if utility > 0 else "no actuar",
            }
        )
    return {"predictions": predictions, "reserved_evaluation_executed": False}


def _predict_group(named_values: list[tuple[str, dict]]) -> dict:
    result = _predict_profiles(named_values)
    predictions = result["predictions"]
    probabilities = [item["p_churn"] for item in predictions]
    result.update(
        {
            "mode": "development_only",
            "count": len(predictions),
            "mean_p_churn": float(np.mean(probabilities)),
            "mean_utility": float(np.mean([item["u_act"] for item in predictions])),
            "high_risk_count": sum(probability >= 0.7 for probability in probabilities),
            "reserved_evaluation_executed": False,
            "disclaimer": (
                "Resumen descriptivo de predicciones individuales; no mide churn real "
                "ni utiliza etiquetas de la partición reservada."
            ),
        }
    )
    return result


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)

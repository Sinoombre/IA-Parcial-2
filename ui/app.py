"""Interfaz local de demostración para predicciones de desarrollo.

Este módulo nunca carga la partición reservada. Usa únicamente el modelo de
`artifacts/dev_eval` y declara esa limitación en cada respuesta.
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import BadRequest

from churnshield.data import load_dev
from churnshield.features import ALLOWED_PREDICTORS
from churnshield.utility import action_utility

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(
    os.environ.get("CHURNSHIELD_MODEL", ROOT / "artifacts/dev_eval/model_v1.joblib")
)
CARD_PATH = MODEL_PATH.with_name("model_card.json")
DEFAULTS = {
    "Partner": "No",
    "Dependents": "No",
    "Phone Service": "Yes",
    "Multiple Lines": "No phone service",
    "Internet Service": "DSL",
    "Online Security": "No",
    "Online Backup": "No",
    "Device Protection": "No",
    "Tech Support": "No",
    "Streaming TV": "No",
    "Streaming Movies": "No",
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


@lru_cache(maxsize=4)
def _read_model_card(path: str, _modified_ns: int, _size: int) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _model_card() -> dict:
    if not CARD_PATH.exists():
        raise FileNotFoundError(
            f"No existe {CARD_PATH}. Ejecuta `PYTHONPATH=src python -m churnshield dev-eval`."
        )
    stat = CARD_PATH.stat()
    try:
        return _read_model_card(str(CARD_PATH), stat.st_mtime_ns, stat.st_size)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"La tarjeta del modelo no contiene JSON válido: {CARD_PATH}"
        ) from exc


@lru_cache(maxsize=4)
def _load_model_from_path(path: str, _modified_ns: int, _size: int):
    return joblib.load(path)


def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No existe {MODEL_PATH}. Ejecuta `PYTHONPATH=src python -m churnshield dev-eval`."
        )
    stat = MODEL_PATH.stat()
    return _load_model_from_path(str(MODEL_PATH), stat.st_mtime_ns, stat.st_size)


def _coerce_payload(payload: dict) -> dict:
    values = {**DEFAULTS, **payload}
    missing = [field for field in ALLOWED_PREDICTORS if field not in values]
    if missing:
        raise ValueError(f"Faltan campos permitidos: {missing}")
    for field in NUMERIC_FIELDS:
        try:
            values[field] = float(values[field])
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"{field} debe ser un número finito.") from exc
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
    try:
        if not request.is_json:
            raise ValueError("El cuerpo de la solicitud debe ser JSON.")
        try:
            payload = request.get_json()
        except BadRequest as exc:
            raise ValueError(
                "El cuerpo de la solicitud no contiene JSON válido."
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError("El cuerpo JSON debe ser un objeto.")
        if "profiles" in payload:
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
                values = _coerce_payload(item.get("values", item))
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
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400


def _predict_profiles(named_values: list[tuple[str, dict]]) -> dict:
    model = _load_model()
    features = pd.DataFrame([values for _, values in named_values])
    probabilities = model.predict_proba(features)[:, 1]
    card = _model_card()
    try:
        mean_cltv = float(card["mean_cltv_train"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "La tarjeta del modelo debe incluir un mean_cltv_train válido."
        ) from exc
    if not math.isfinite(mean_cltv) or mean_cltv <= 0:
        raise ValueError("mean_cltv_train debe ser finito y positivo.")
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

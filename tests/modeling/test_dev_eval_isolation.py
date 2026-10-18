import json

import joblib
import numpy as np
import pandas as pd

from churnshield import data, dev_eval
from churnshield.features import ALLOWED_PREDICTORS


def test_dev_eval_uses_only_development_loader(monkeypatch, tmp_path):
    development = pd.DataFrame(
        [
            {
                **{feature: "value" for feature in ALLOWED_PREDICTORS},
                "Tenure Months": index,
                "Monthly Charges": 30.0 + index,
                "Total Charges": 100.0 + index,
                "CustomerID": f"DEV-{index}",
                "Churn Value": index % 2,
                "CLTV": 100.0 + index,
            }
            for index in range(4)
        ]
    )
    metadata = {
        "dataset_sha256": "dataset",
        "partition_sha256": "partition",
        "reserved_count": 1409,
    }
    monkeypatch.setattr(dev_eval, "load_dev", lambda: (development, metadata))
    monkeypatch.setattr(
        dev_eval,
        "select_C",
        lambda *args, **kwargs: (0.1, pd.DataFrame(), pd.DataFrame()),
    )
    monkeypatch.setattr(
        dev_eval,
        "oof_predictions",
        lambda X, y, ids, C, **kwargs: np.linspace(0.1, 0.8, len(X)),
    )
    monkeypatch.setattr(
        dev_eval,
        "oof_stacking_predictions",
        lambda X, y, **kwargs: np.linspace(0.2, 0.9, len(X)),
    )
    monkeypatch.setattr(dev_eval, "precision_at_k", lambda *args: (0.5, 1))
    monkeypatch.setattr(dev_eval, "recall_at_k", lambda *args: (0.5, 1))
    monkeypatch.setattr(dev_eval, "lift_at_k", lambda *args: (1.0, 1))
    monkeypatch.setattr(
        dev_eval,
        "calibration_metrics",
        lambda *args: {"auc": 0.5, "brier": 0.25, "log_loss": 0.7, "ece": 0.1},
    )
    monkeypatch.setattr(dev_eval, "add_utilities", lambda rows: rows)
    monkeypatch.setattr(
        dev_eval,
        "select_candidates",
        lambda *args, **kwargs: {
            "decisions": [{"customer_id": "DEV-0", "selected": True}],
            "n_selected": 1,
            "k": 1,
        },
    )

    class FakePipeline:
        def fit(self, X, y):
            return self

    def write_fake_model(model, path):
        with open(path, "wb") as handle:
            handle.write(b"model")

    monkeypatch.setattr(
        dev_eval, "make_stacking_pipeline", lambda **kwargs: FakePipeline()
    )
    monkeypatch.setattr(joblib, "dump", write_fake_model)
    monkeypatch.setattr(dev_eval, "_git_commit", lambda: "test-commit")
    monkeypatch.setattr(dev_eval, "_git_clean", lambda: True)
    monkeypatch.setattr(
        data,
        "_read_data_rows",
        lambda indices, *, split: (_ for _ in ()).throw(
            AssertionError(f"Unexpected data split access: {split}")
        ),
    )
    monkeypatch.setattr(
        data,
        "load_reserved",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("dev-eval accessed reserved rows")
        ),
    )

    result = dev_eval.run_dev_eval(output_dir=tmp_path / "dev-eval")

    manifest = json.loads(
        (tmp_path / "dev-eval" / "model_manifest.json").read_text(encoding="utf-8")
    )
    assert result["metadata"]["dataset_sha256"] == "dataset"
    assert manifest["features"] == list(ALLOWED_PREDICTORS)
    assert manifest["partition_sha256"] == "partition"
    assert manifest["reserved_evaluation_executed"] is False
    assert manifest["model_type"] == "stacking_logistic_lightgbm_xgboost_catboost"

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from experiments import logistic_regression


def test_pipeline_uses_logistic_regression():
    pipeline = logistic_regression.make_pipeline(logistic_regression.LR_CONFIGS["lr_base"])

    assert "preprocess" in pipeline.named_steps
    assert isinstance(pipeline.named_steps["model"], LogisticRegression)


def test_compare_selects_best_candidate_and_saves_result(monkeypatch, tmp_path):
    y = pd.Series([0, 1] * 15)
    ids = pd.Series([f"C{i:03}" for i in range(len(y))])
    predictions = np.linspace(0.05, 0.95, len(y))

    def fake_oof_predictions(X, y, ids, config):
        score = {0.1: 0.5, 1.0: 0.8}[config["C"]]
        if config["class_weight"] == "balanced":
            score = 0.6
        folds = pd.DataFrame({"precision_at_15": [score, score + 0.02]})
        return predictions, folds

    monkeypatch.setattr(logistic_regression, "oof_predictions", fake_oof_predictions)
    monkeypatch.chdir(tmp_path)

    result = logistic_regression.compare_logistic_regression(
        pd.DataFrame(index=range(len(y))), y, ids
    )
    logistic_regression.save_comparison(result)

    assert result["chosen_model"] == "lr_base"
    assert result["chosen_config"] == logistic_regression.LR_CONFIGS["lr_base"]
    assert len(result["candidates"]) == len(logistic_regression.LR_CONFIGS)
    saved = json.loads(
        (tmp_path / "artifacts" / "logistic_regression_comparison.json").read_text()
    )
    assert saved["chosen_model"] == "lr_base"

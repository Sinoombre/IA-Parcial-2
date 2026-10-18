import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from experiments import random_forest


def test_pipeline_uses_random_forest():
    pipeline = random_forest.make_pipeline(random_forest.RF_CONFIGS["rf_base"])

    assert "preprocess" in pipeline.named_steps
    assert isinstance(pipeline.named_steps["model"], RandomForestClassifier)
    assert pipeline.named_steps["model"].n_jobs == -1


def test_compare_selects_best_candidate_and_saves_result(monkeypatch, tmp_path):
    y = pd.Series([0, 1] * 15)
    ids = pd.Series([f"C{i:03}" for i in range(len(y))])
    predictions = np.linspace(0.05, 0.95, len(y))

    def fake_oof_predictions(X, y, ids, config):
        score = {
            "rf_base": 0.5,
            "rf_regularized": 0.8,
            "rf_balanced": 0.6,
        }[next(name for name, candidate in random_forest.RF_CONFIGS.items() if candidate == config)]
        folds = pd.DataFrame({"precision_at_15": [score, score + 0.02]})
        return predictions, folds

    monkeypatch.setattr(random_forest, "oof_predictions", fake_oof_predictions)
    monkeypatch.chdir(tmp_path)

    result = random_forest.compare_random_forest(
        pd.DataFrame(index=range(len(y))), y, ids
    )
    random_forest.save_comparison(result)

    assert result["chosen_model"] == "rf_regularized"
    assert result["chosen_config"] == random_forest.RF_CONFIGS["rf_regularized"]
    assert len(result["candidates"]) == len(random_forest.RF_CONFIGS)
    saved = json.loads(
        (tmp_path / "artifacts" / "random_forest_comparison.json").read_text()
    )
    assert saved["chosen_model"] == "rf_regularized"

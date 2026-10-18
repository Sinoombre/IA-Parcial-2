import numpy as np
import pandas as pd

from churnshield import stacking
from churnshield.model import make_pipeline
from churnshield.features import ALLOWED_PREDICTORS
from churnshield.stacking import make_stacking_pipeline


def test_pipeline_contains_preprocessing_and_model():
    pipe = make_pipeline(0.1)
    assert "preprocess" in pipe.named_steps
    assert "model" in pipe.named_steps


def test_stacking_pipeline_uses_configured_features_and_regularized_meta_model():
    pipe = make_stacking_pipeline(include_catboost=True)

    assert [name for name, _ in pipe.estimators] == [
        "logistic",
        "lightgbm",
        "xgboost",
        "catboost",
    ]
    for _, estimator in pipe.estimators:
        used_features = {
            feature
            for _, _, columns in estimator.named_steps["preprocess"].transformers
            for feature in columns
        }
        assert used_features == set(ALLOWED_PREDICTORS)
    assert pipe.final_estimator.named_steps["model"].C == 10.0


def test_stacking_oof_predictions_fit_and_predict_on_disjoint_rows(monkeypatch):
    X = pd.DataFrame({"row": range(40)})
    y = pd.Series([0, 1] * 20)
    fitted_rows = []
    factory_kwargs = []

    class FakeModel:
        def fit(self, X_train, y_train):
            self.train_rows = set(X_train.index)
            fitted_rows.append(self.train_rows)
            return self

        def predict_proba(self, X_valid):
            assert not self.train_rows.intersection(X_valid.index)
            positive = np.full(len(X_valid), 0.4)
            return np.column_stack((1 - positive, positive))

    def fake_factory(**kwargs):
        factory_kwargs.append(kwargs)
        return FakeModel()

    monkeypatch.setattr(stacking, "make_stacking_pipeline", fake_factory)

    predictions = stacking.oof_stacking_predictions(X, y, seeds=(3101,))

    assert len(fitted_rows) == 5
    assert predictions.shape == (len(X),)
    assert np.all(predictions == 0.4)
    assert all(kwargs["include_catboost"] for kwargs in factory_kwargs)

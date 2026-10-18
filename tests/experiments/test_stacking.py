from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churnshield.stacking import make_stacking_pipeline
from churnshield.features import ALLOWED_PREDICTORS


def test_stacking_uses_regularized_meta_model_and_only_configured_features():
    model = make_stacking_pipeline(random_state=3101)

    assert isinstance(model, StackingClassifier)
    assert [name for name, _ in model.estimators] == [
        "logistic",
        "lightgbm",
        "xgboost",
    ]
    assert model.cv.n_splits == 3
    assert model.final_estimator.named_steps["model"].C == 10.0
    for _, estimator in model.estimators:
        assert isinstance(estimator, Pipeline)
        preprocessor = estimator.named_steps["preprocess"]
        used_features = {
            feature
            for _, _, columns in preprocessor.transformers
            for feature in columns
        }
        assert used_features == set(ALLOWED_PREDICTORS)


def test_catboost_variant_is_an_optional_base_estimator():
    model = make_stacking_pipeline(include_catboost=True)

    assert [name for name, _ in model.estimators] == [
        "logistic",
        "lightgbm",
        "xgboost",
        "catboost",
    ]
    assert isinstance(model.final_estimator.named_steps["model"], LogisticRegression)

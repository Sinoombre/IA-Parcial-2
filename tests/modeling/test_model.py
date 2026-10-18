import pandas as pd

from churnshield.model import _choose_C, make_pipeline


def test_pipeline_contains_preprocessing_and_model():
    pipe = make_pipeline(0.1)
    assert "preprocess" in pipe.named_steps
    assert "model" in pipe.named_steps


def test_c_selection_breaks_one_standard_error_ties_by_lowest_log_loss():
    grouped = pd.DataFrame(
        [
            {
                "C": 0.01,
                "precision_mean": 0.80,
                "precision_sd": 0.10,
                "log_loss_mean": 0.60,
            },
            {
                "C": 0.1,
                "precision_mean": 0.82,
                "precision_sd": 0.10,
                "log_loss_mean": 0.50,
            },
            {
                "C": 1.0,
                "precision_mean": 0.85,
                "precision_sd": 0.10,
                "log_loss_mean": 0.40,
            },
            {
                "C": 10.0,
                "precision_mean": 0.80,
                "precision_sd": 0.10,
                "log_loss_mean": 0.20,
            },
        ]
    )

    assert _choose_C(grouped, seed_count=3) == 10.0

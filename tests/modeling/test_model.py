from churnshield.model import make_pipeline


def test_pipeline_contains_preprocessing_and_model():
    pipe = make_pipeline(0.1)
    assert "preprocess" in pipe.named_steps
    assert "model" in pipe.named_steps

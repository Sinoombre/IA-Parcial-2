import pytest


@pytest.fixture(autouse=True)
def isolate_evaluator_outputs(monkeypatch, tmp_path):
    from churnshield.simulation import paths

    monkeypatch.setattr(
        paths,
        "PRIVATE_EVALUATOR_ROOT",
        tmp_path.parent / f"{tmp_path.name}-private-evaluator-truth",
    )

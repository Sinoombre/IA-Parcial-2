import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from churnshield.data import load_reserved, load_dev
from churnshield.features import validate_observation
from churnshield.metrics import precision_at_k
from churnshield.utility import action_utility
from churnshield.model import make_pipeline


def test_reserved_is_blocked_without_flag():
    with pytest.raises(PermissionError): load_reserved()


def test_forbidden_observation_fields_fail():

    with pytest.raises(ValueError): validate_observation(['Partner', 'Churn Value'])
    with pytest.raises(ValueError): validate_observation(['Partner', 'would_churn'])


def test_dataset_dev_contract():

    df, meta = load_dev()
    assert len(df) == 5634 and meta['reserved_count'] == 1409
    assert meta['total_charges_blank_count'] == 11


def test_precision_hand_calculated():

    score, meta = precision_at_k(np.array([1, 0, 1, 0, 0, 1, 0]), np.array([.9, .8, .7, .6, .5, .4, .3]), fraction=0.5)
    assert meta['k'] == 3 and score == pytest.approx(2/3)


def test_k_zero_returns_nan_with_reason():

    score, meta = precision_at_k(np.array([1]), np.array([.9]), fraction=0.15)
    assert np.isnan(score) and meta['reason'] == 'k=0'


def test_utility_formula():

    assert action_utility(.8, 1.2, r=.25, cost=.05) == pytest.approx(.19)


def test_pipeline_contains_preprocessing_and_model():
    
    pipe = make_pipeline(.1)
    assert 'preprocess' in pipe.named_steps and 'model' in pipe.named_steps

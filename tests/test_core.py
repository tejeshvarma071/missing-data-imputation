
import numpy as np
import pandas as pd

from missing_data_imputation.data.schema import infer_feature_types
from missing_data_imputation.missingness.masks import (
    generate_mcar_mask,
    generate_mar_mask,
    generate_mnar_mask,
)
from missing_data_imputation.imputation.methods import (
    mean_impute,
    median_impute,
    most_frequent_impute,
    knn_impute,
    iterative_impute,
)
from missing_data_imputation.evaluation.reconstruction import reconstruction_mae
from missing_data_imputation.statistics.multiple_testing import holm_adjust


def test_infer_feature_types():
    df = pd.DataFrame({
        "age": [20, 25, 30],
        "city": ["A", "B", "A"]
    })

    numeric, categorical = infer_feature_types(df)

    assert "age" in numeric
    assert "city" in categorical


def test_mcar_mask():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [5, 4, 3, 2, 1]
    })

    mask = generate_mcar_mask(df, missing_rate=0.2, random_state=42)

    assert mask.shape == df.shape
    assert mask.dtype == bool


def test_mar_mask():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [5, 4, 3, 2, 1]
    })

    mask = generate_mar_mask(df, missing_rate=0.2, random_state=42)

    assert mask.shape == df.shape


def test_mnar_mask():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [5, 4, 3, 2, 1]
    })

    mask = generate_mnar_mask(df, missing_rate=0.2, random_state=42)

    assert mask.shape == df.shape


def test_imputation_methods():
    df = pd.DataFrame({
        "a": [1.0, np.nan, 3.0],
        "b": [4.0, 5.0, np.nan]
    })

    methods = [
        mean_impute,
        median_impute,
        most_frequent_impute,
        knn_impute,
        iterative_impute,
    ]

    for method in methods:
        result = method(df)

        assert not result.isnull().values.any()
        assert result.shape == df.shape


def test_reconstruction_mae():
    original = np.array([[1.0, 2.0], [3.0, 4.0]])
    imputed = np.array([[1.0, 2.0], [2.0, 4.0]])
    mask = np.array([[False, False], [True, False]])

    error = reconstruction_mae(original, imputed, mask)

    assert error >= 0


def test_holm_adjust():
    p_values = [0.01, 0.04, 0.03]

    adjusted = holm_adjust(p_values)

    assert len(adjusted) == 3
    assert all(0 <= p <= 1 for p in adjusted)

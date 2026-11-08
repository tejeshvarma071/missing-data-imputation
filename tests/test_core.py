
import numpy as np
import pandas as pd
import pytest

from src.data.schema import infer_feature_types
from src.missingness.masks import (
    make_mcar_mask,
    fit_mar_driver,
    generate_mar_mask,
    generate_mnar_mask,
    generate_masks,
    validate_mask,
)
from src.imputation.methods import make_numeric_imputer
from src.evaluation.reconstruction import categorical_mode_accuracy
from src.statistics.multiple_testing import holm_adjust


def sample_frame():
    return pd.DataFrame({
        "driver": [1., 2., 3., 4., 5., 6.],
        "value": [10., 11., 12., 13., 14., 15.],
        "kind": ["a", "a", "a", "b", "b", "c"],
    })


def test_mcar_reproducible_and_valid():
    X = sample_frame()
    a = make_mcar_mask(X, .3, 7)
    b = make_mcar_mask(X, .3, 7)

    pd.testing.assert_frame_equal(a, b)
    assert int(a.to_numpy().sum()) == round(
        .3 * X.notna().to_numpy().sum()
    )
    assert validate_mask(X, a)


def test_mar_protects_training_driver_and_reproducible():
    X = sample_frame()
    info = fit_mar_driver(X)

    a = generate_mar_mask(X, info, .5, 11)
    b = generate_mar_mask(X, info, .5, 11)

    pd.testing.assert_frame_equal(a, b)
    assert info["driver_col"] == "driver"
    assert not a["driver"].any()


def test_mnar_validates_partition_and_reproducible():
    X = sample_frame()

    a = generate_mnar_mask(
        X, .5, 3, ["driver", "value"], ["kind"]
    )
    b = generate_mnar_mask(
        X, .5, 3, ["driver", "value"], ["kind"]
    )

    pd.testing.assert_frame_equal(a, b)
    assert validate_mask(X, a)

    with pytest.raises(ValueError):
        generate_mnar_mask(X, .2, 3, ["driver"], ["kind"])


def test_generate_masks_dispatches():
    X = sample_frame()

    train, test, driver = generate_masks(
        "MAR",
        X.iloc[:4],
        X.iloc[4:],
        .5,
        1,
        2,
        ["driver", "value"],
        ["kind"],
    )

    assert driver == "driver"
    assert train.shape == (4, 3)
    assert test.shape == (2, 3)


def test_imputer_factory_names():
    for name in (
        "mean",
        "median",
        "most_frequent",
        "knn",
        "iterative",
    ):
        assert make_numeric_imputer(name) is not None

    with pytest.raises(ValueError):
        make_numeric_imputer("unknown")


def test_categorical_accuracy_uses_unmasked_ground_truth_and_numeric_equality():
    train = pd.DataFrame({"cat": [0, 0, 1, 1]})
    test = pd.DataFrame({"cat": [0.0, 1.0, 0.0]})

    train_mask = pd.DataFrame({
        "cat": [False, False, False, False]
    })
    test_mask = pd.DataFrame({
        "cat": [True, True, True]
    })

    accuracy, total, correct, details = categorical_mode_accuracy(
        train, test, train_mask, test_mask, ["cat"]
    )

    assert total == 3
    assert correct == 2
    assert accuracy == pytest.approx(2 / 3)
    assert details[0]["Masked_Cells"] == 3


def test_holm_adjustment():
    adjusted = holm_adjust([.01, .04, .03])
    np.testing.assert_allclose(adjusted, [.03, .06, .06])

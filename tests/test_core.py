
import numpy as np
import pandas as pd

from missing_data_imputation.data.schema import (
    infer_feature_types,
    validate_feature_types,
    prepare_types,
)

from missing_data_imputation.missingness.masks import (
    make_mcar_mask,
    fit_mar_driver,
    generate_mar_mask,
    generate_mnar_mask,
    validate_mask,
)

from missing_data_imputation.imputation.methods import (
    make_numeric_imputer,
    IMPUTER_NAMES,
)

from missing_data_imputation.evaluation.reconstruction import (
    categorical_mode_accuracy,
)

from missing_data_imputation.statistics.multiple_testing import (
    holm_adjust,
)


def test_infer_feature_types():
    df = pd.DataFrame({
        "age": [20, 25, 30],
        "city": ["A", "B", "A"],
    })

    numeric, categorical = infer_feature_types(df)

    assert numeric == ["age"]
    assert categorical == ["city"]


def test_prepare_and_validate_feature_types():
    df = pd.DataFrame({
        "age": ["20", "25", "30"],
        "city": ["A", None, "B"],
    })

    prepared = prepare_types(df, categorical_columns=["city"])

    assert pd.api.types.is_numeric_dtype(prepared["age"])
    assert validate_feature_types(
        prepared.columns,
        ["age"],
        ["city"],
    )


def test_mcar_mask():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [5, 4, 3, 2, 1],
    })

    mask = make_mcar_mask(
        df,
        missing_rate=0.2,
        seed=42,
    )

    assert mask.shape == df.shape
    assert mask.to_numpy().sum() == 2
    assert validate_mask(df, mask)


def test_mar_mask():
    train = pd.DataFrame({
        "driver": [1, 2, 3, 4, 5, 6],
        "target": [10, 20, 30, 40, 50, 60],
    })

    driver_info = fit_mar_driver(train)

    mask = generate_mar_mask(
        train,
        driver_info,
        missing_rate=0.5,
        seed=42,
    )

    assert mask.shape == train.shape
    assert not mask["driver"].any()
    assert validate_mask(train, mask)


def test_mnar_mask():
    df = pd.DataFrame({
        "age": [20, 25, 30, 35, 40, 45],
        "city": ["A", "A", "A", "B", "B", "C"],
    })

    mask = generate_mnar_mask(
        df,
        missing_rate=0.3,
        seed=42,
        numeric_cols=["age"],
        categorical_cols=["city"],
    )

    assert mask.shape == df.shape
    assert validate_mask(df, mask)


def test_all_imputation_methods():
    df = pd.DataFrame({
        "a": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
        "b": [2.0, 3.0, np.nan, 5.0, 6.0, 7.0],
    })

    assert len(IMPUTER_NAMES) == 5

    for name in IMPUTER_NAMES:
        imputer = make_numeric_imputer(name)
        result = imputer.fit_transform(df)

        assert result.shape == df.shape
        assert np.isfinite(result).all()


def test_categorical_reconstruction_and_holm_adjustment():
    train = pd.DataFrame({
        "city": ["A", "A", "B"],
    })

    test = pd.DataFrame({
        "city": ["A", "A", "C"],
    })

    train_mask = pd.DataFrame({
        "city": [False, False, True],
    })

    test_mask = pd.DataFrame({
        "city": [False, True, False],
    })

    accuracy, total, correct, details = categorical_mode_accuracy(
        train,
        test,
        train_mask,
        test_mask,
        categorical_cols=["city"],
    )

    assert accuracy == 1.0
    assert total == 1
    assert correct == 1
    assert len(details) == 1

    adjusted = holm_adjust([0.01, 0.04, 0.03])

    np.testing.assert_allclose(
        adjusted,
        [0.03, 0.06, 0.06],
    )

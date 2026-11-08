"""Feature typing helpers used by the imputation experiments."""
from __future__ import annotations
import pandas as pd


def infer_feature_types(frame: pd.DataFrame, categorical_columns=None):
    """Return (numeric_columns, categorical_columns), preserving input order.

    Explicit categorical columns take precedence over pandas numeric dtype;
    this is important for coded categorical variables such as Heart Disease.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    explicit = set(categorical_columns or [])
    unknown = explicit.difference(frame.columns)
    if unknown:
        raise ValueError(f"Unknown categorical columns: {sorted(unknown)}")
    categorical = [
        col for col in frame.columns
        if col in explicit or not (
            pd.api.types.is_numeric_dtype(frame[col])
            and not pd.api.types.is_bool_dtype(frame[col])
        )
    ]
    numeric = [col for col in frame.columns if col not in categorical]
    return numeric, categorical


def validate_feature_types(columns, numeric_columns, categorical_columns):
    """Ensure numeric/categorical feature groups partition the columns."""
    columns = list(columns)
    numeric, categorical = set(numeric_columns), set(categorical_columns)
    if numeric.intersection(categorical):
        raise ValueError("Numeric and categorical feature groups overlap")
    if numeric.union(categorical) != set(columns):
        raise ValueError("Feature groups must cover every column exactly once")
    return True


def prepare_types(frame: pd.DataFrame, categorical_columns=None) -> pd.DataFrame:
    """Copy a frame, normalize missing values, and coerce non-categorical data.

    Categorical columns are kept as object values; all other columns are
    converted with pandas.to_numeric(errors='coerce'), matching the notebook.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    categorical = set(categorical_columns or [])
    unknown = categorical.difference(frame.columns)
    if unknown:
        raise ValueError(f"Unknown categorical columns: {sorted(unknown)}")
    result = frame.copy()
    for col in result.columns:
        if col in categorical:
            result[col] = result[col].astype(object)
            result[col] = result[col].where(pd.notna(result[col]), float("nan"))
        else:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    return result

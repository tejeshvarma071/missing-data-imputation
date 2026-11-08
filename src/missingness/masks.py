"""Missingness mask generators aligned with the reviewed notebook.

A True value in a returned mask means that the corresponding observed cell
should be hidden. These functions do not modify the input DataFrame.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..data.schema import validate_feature_types


def _validate_rate(rate):
    if not 0 <= float(rate) <= 1:
        raise ValueError("missing_rate must be between 0 and 1")


def validate_mask(original: pd.DataFrame, mask: pd.DataFrame) -> bool:
    """Validate shape/columns and ensure only originally observed cells are masked."""
    if not isinstance(original, pd.DataFrame) or not isinstance(mask, pd.DataFrame):
        raise TypeError("original and mask must be pandas DataFrames")
    if original.shape != mask.shape or list(original.columns) != list(mask.columns):
        raise ValueError("mask must have the same shape and columns as original")
    if mask.isna().any().any():
        raise ValueError("mask cannot contain missing values")
    if (mask.astype(bool) & original.isna()).any().any():
        raise ValueError("mask selects cells that were already missing")
    return True


def make_mcar_mask(X: pd.DataFrame, missing_rate: float, seed: int) -> pd.DataFrame:
    """Select a uniform random sample of observed cells (not per-column)."""
    _validate_rate(missing_rate)
    rng = np.random.default_rng(seed)
    positions = np.argwhere(X.notna().to_numpy())
    n_missing = int(round(float(missing_rate) * len(positions)))
    mask = pd.DataFrame(False, index=X.index, columns=X.columns)
    if n_missing:
        chosen = rng.choice(len(positions), size=n_missing, replace=False)
        for row_idx, col_idx in positions[chosen]:
            mask.iat[row_idx, col_idx] = True
    validate_mask(X, mask)
    return mask


def fit_mar_driver(train_X: pd.DataFrame) -> dict:
    """Fit empirical-rank MAR driver using training data only."""
    candidates = []
    for col in train_X.columns:
        values = pd.to_numeric(train_X[col], errors="coerce")
        if values.notna().all() and values.nunique() > 1:
            candidates.append(col)
    if not candidates:
        raise ValueError("No complete numeric MAR driver found.")
    driver_col = candidates[0]
    sorted_values = np.sort(pd.to_numeric(
        train_X[driver_col], errors="raise"
    ).to_numpy(dtype=float))
    return {"driver_col": driver_col, "sorted_train_values": sorted_values}


def transform_mar_driver(X: pd.DataFrame, driver_info: dict) -> np.ndarray:
    """Map rows to clipped empirical percentiles from the training reference."""
    values = pd.to_numeric(
        X[driver_info["driver_col"]], errors="raise"
    ).to_numpy(dtype=float)
    reference = np.asarray(driver_info["sorted_train_values"], dtype=float)
    if reference.size == 0:
        raise ValueError("MAR driver reference is empty")
    ranks = np.searchsorted(reference, values, side="right")
    return np.clip(ranks / len(reference), 0.001, 0.999)


def generate_mar_mask(X: pd.DataFrame, driver_info: dict, missing_rate: float,
                      seed: int, strength: float = 2.0) -> pd.DataFrame:
    """Generate fold-level MAR mask; the fitted driver column is protected."""
    _validate_rate(missing_rate)
    rng = np.random.default_rng(seed)
    X = X.reset_index(drop=True).copy()
    driver_col = driver_info["driver_col"]
    if driver_col not in X.columns:
        raise ValueError(f"MAR driver column not found: {driver_col}")
    scores = transform_mar_driver(X, driver_info)
    mask = pd.DataFrame(False, index=X.index, columns=X.columns)
    for target_col in X.columns:
        if target_col == driver_col:
            continue
        eligible = np.flatnonzero(X[target_col].notna().to_numpy())
        n_missing = int(round(float(missing_rate) * len(eligible)))
        if n_missing == 0:
            continue
        weights = np.exp(float(strength) * (scores[eligible] - 0.5))
        probabilities = weights / weights.sum()
        selected = rng.choice(eligible, size=n_missing, replace=False, p=probabilities)
        mask.loc[selected, target_col] = True
    validate_mask(X, mask)
    if mask[driver_col].any():
        raise AssertionError("MAR driver must never be masked")
    return mask


def allocate_categorical_missingness(category_counts, total_missing: int) -> np.ndarray:
    """Allocate categorical missing cells with higher rates for rarer categories."""
    counts = np.asarray(category_counts, dtype=int)
    allocation = np.zeros(len(counts), dtype=int)
    if total_missing < 0 or total_missing > counts.sum():
        raise ValueError("Invalid total missing count.")
    remaining = int(total_missing)
    while remaining > 0:
        available = counts - allocation
        active = np.flatnonzero(available > 0)
        if len(active) == 0:
            raise RuntimeError("No category capacity remaining.")
        weights = 1.0 / np.sqrt(counts[active].astype(float))
        weights /= weights.sum()
        ideal = remaining * weights
        proposed = np.minimum(np.floor(ideal).astype(int), available[active])
        allocation[active] += proposed
        remaining -= int(proposed.sum())
        if remaining == 0:
            break
        available = counts - allocation
        active = np.flatnonzero(available > 0)
        weights = 1.0 / np.sqrt(counts[active].astype(float))
        ideal = remaining * weights / weights.sum()
        order = active[np.argsort(-(ideal - np.floor(ideal)))]
        for idx in order:
            if remaining == 0:
                break
            if allocation[idx] < counts[idx]:
                allocation[idx] += 1
                remaining -= 1
    if allocation.sum() != total_missing or np.any(allocation > counts):
        raise AssertionError("Categorical missingness allocation invariant failed")
    return allocation


def generate_mnar_mask(X: pd.DataFrame, missing_rate: float, seed: int,
                       numeric_cols, categorical_cols,
                       strength: float = 2.0) -> pd.DataFrame:
    """Generate MNAR mask: high numeric values and rare categories are favored."""
    _validate_rate(missing_rate)
    numeric_set, categorical_set = set(numeric_cols), set(categorical_cols)
    validate_feature_types(X.columns, numeric_set, categorical_set)
    rng = np.random.default_rng(seed)
    mask = pd.DataFrame(False, index=X.index, columns=X.columns)
    for col in X.columns:
        positions = np.flatnonzero(X[col].notna().to_numpy())
        n = len(positions)
        if n < 2:
            continue
        total_missing = int(round(float(missing_rate) * n))
        if total_missing == 0:
            continue
        observed = X.iloc[positions][col]
        if col in numeric_set:
            scores = observed.astype(float).rank(method="average", pct=True).to_numpy()
            weights = np.exp(float(strength) * (scores - 0.5))
            probabilities = weights / weights.sum()
            selected = rng.choice(positions, size=total_missing, replace=False, p=probabilities)
        else:
            categories = observed.astype(str).to_numpy()
            _, inverse, counts = np.unique(categories, return_inverse=True, return_counts=True)
            allocations = allocate_categorical_missingness(counts, total_missing)
            selected_list = []
            for category_index, quota in enumerate(allocations):
                if quota:
                    category_positions = positions[inverse == category_index]
                    selected_list.extend(rng.choice(category_positions, size=int(quota), replace=False).tolist())
            selected = np.asarray(selected_list, dtype=int)
            if len(selected) != total_missing:
                raise AssertionError("MNAR categorical allocation count mismatch")
        mask.iloc[selected, mask.columns.get_loc(col)] = True
    validate_mask(X, mask)
    return mask


def generate_masks(mechanism: str, X_train: pd.DataFrame, X_test: pd.DataFrame,
                   missing_rate: float, seed_train: int, seed_test: int,
                   numeric_cols, categorical_cols, strength: float = 2.0):
    """Generate independent train/test masks for MCAR, MAR, or MNAR."""
    mechanism = mechanism.upper()
    driver_name = None
    if mechanism == "MCAR":
        train_mask = make_mcar_mask(X_train, missing_rate, seed_train)
        test_mask = make_mcar_mask(X_test, missing_rate, seed_test)
    elif mechanism == "MAR":
        driver_info = fit_mar_driver(X_train)
        driver_name = driver_info["driver_col"]
        train_mask = generate_mar_mask(X_train, driver_info, missing_rate, seed_train, strength)
        test_mask = generate_mar_mask(X_test, driver_info, missing_rate, seed_test, strength)
    elif mechanism == "MNAR":
        train_mask = generate_mnar_mask(X_train, missing_rate, seed_train, numeric_cols, categorical_cols, strength)
        test_mask = generate_mnar_mask(X_test, missing_rate, seed_test, numeric_cols, categorical_cols, strength)
    else:
        raise ValueError("mechanism must be one of: MCAR, MAR, MNAR")
    validate_mask(X_train, train_mask)
    validate_mask(X_test, test_mask)
    return train_mask, test_mask, driver_name

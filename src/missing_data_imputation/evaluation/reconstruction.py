"""Type-aware categorical reconstruction evaluation."""
from __future__ import annotations
import numpy as np
import pandas as pd


def categorical_mode_accuracy(X_train: pd.DataFrame, X_test: pd.DataFrame,
                               train_mask: pd.DataFrame, test_mask: pd.DataFrame,
                               categorical_cols):
    """Score mode-based categorical predictions on originally masked test cells.

    The mode is learned from the masked training frame. Ground truth is read
    from the original, unmasked X_test. Numeric-aware pandas equality ensures
    that values such as 0 and 0.0 compare equal, as in the corrected notebook.
    Returns (accuracy, masked_count, correct_count, per_feature_details).
    """
    for frame, mask, label in ((X_train, train_mask, "train"), (X_test, test_mask, "test")):
        if frame.shape != mask.shape or list(frame.columns) != list(mask.columns):
            raise ValueError(f"{label} mask must match its source frame")
        if mask.isna().any().any():
            raise ValueError(f"{label} mask cannot contain missing values")
        if (mask.astype(bool) & frame.isna()).any().any():
            raise ValueError(f"{label} mask selects pre-existing missing cells")
    correct = total = 0
    details = []
    for col in categorical_cols:
        if col not in X_train.columns or col not in X_test.columns:
            raise ValueError(f"Categorical column not found: {col}")
        train_values = X_train[col].mask(train_mask[col].astype(bool))
        modes = train_values.dropna().mode()
        if modes.empty:
            continue
        fill_value = modes.iloc[0]
        selected = test_mask[col].to_numpy(dtype=bool)
        true_values = X_test.loc[selected, col]
        if true_values.empty:
            continue
        matches = true_values.eq(fill_value).to_numpy()
        feature_correct, feature_total = int(matches.sum()), int(len(matches))
        correct += feature_correct
        total += feature_total
        details.append({"Feature": col, "Correct": feature_correct,
                        "Masked_Cells": feature_total,
                        "Accuracy": feature_correct / feature_total})
    accuracy = correct / total if total else float("nan")
    return accuracy, total, correct, details

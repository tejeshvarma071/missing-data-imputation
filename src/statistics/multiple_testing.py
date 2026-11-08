"""Holm family-wise error-rate adjustment."""
import numpy as np


def holm_adjust(p_values):
    """Return Holm-adjusted p-values in original order (NaNs remain NaN)."""
    p = np.asarray(p_values, dtype=float)
    if np.any((p[np.isfinite(p)] < 0) | (p[np.isfinite(p)] > 1)):
        raise ValueError("Finite p-values must lie in [0, 1]")
    result = np.full(p.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return result
    order = valid[np.argsort(p[valid])]
    m = len(order)
    adjusted_sorted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    result[order] = np.minimum(adjusted_sorted, 1.0)
    return result

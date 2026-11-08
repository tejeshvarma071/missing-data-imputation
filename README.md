# Missing Data Imputation Research — Reusable Source Package

This package extracts a **reusable core** from the reviewed research notebook:
`notebooks/Missing_Data_Imputation_Research_FINAL_REVIEWED.ipynb`.

It is not a replacement for the notebook and is not a claim that the entire
research pipeline has been converted or independently reproduced. The
notebook remains the authoritative end-to-end experiment record.

## Included modules

- `src/missing_data_imputation/data/`: feature-type inference, validation, and type preparation.
- `src/missing_data_imputation/missingness/`: MCAR, training-fitted MAR, and MNAR mask generators.
- `src/missing_data_imputation/imputation/`: the five numeric imputer constructors (mean, median,
  most-frequent, KNN, iterative).
- `src/missing_data_imputation/evaluation/`: corrected categorical mode-reconstruction evaluator.
- `src/missing_data_imputation/statistics/`: Holm multiple-testing adjustment.
- `tests/`: focused deterministic and invariant tests.

## Installation

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Run tests

```bash
python -m pytest -q
```

## Minimal example

```python
from missing_data_imputation.data import infer_feature_types
from missing_data_imputation.missingness import generate_masks
from missing_data_imputation.imputation import make_numeric_imputer

numeric_cols, categorical_cols = infer_feature_types(X_train)
train_mask, test_mask, mar_driver = generate_masks(
    "MCAR", X_train, X_test, 0.30,
    seed_train=101, seed_test=102,
    numeric_cols=numeric_cols,
    categorical_cols=categorical_cols,
)
median_imputer = make_numeric_imputer("median")
```

The imputer factory creates **numeric** imputers only. Categorical encoding,
classifier pipelines, dataset download/loading, fold orchestration, full
predictive metrics, plotting, and the complete historical experiment runner
remain in the notebook and are intentionally not represented as completed
package functionality here.

## Research fidelity and safeguards

- MCAR selects a fixed rounded fraction of all observed cells.
- MAR fits its empirical-rank driver on training data and protects that driver
  from masking; default MAR strength is `2.0` as in the notebook.
- MNAR favors higher numeric values and rarer categorical values; default
  strength is `2.0` as in the notebook.
- Mask generators return Boolean DataFrames (`True` means hide this cell),
  preserve the input data, and validate that pre-existing missing cells are
  never selected.
- Categorical reconstruction is evaluated against the original, unmasked
  test values and uses numeric-aware equality, reflecting the corrected
  evaluator in the reviewed notebook.

The package tests use small synthetic data and verify software behavior;
they do not rerun the full four-dataset, repeated-fold research experiment.

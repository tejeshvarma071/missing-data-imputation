# Comparative Evaluation of Missing-Data Imputation Methods

A research project comparing missing-data imputation techniques under MCAR, MAR, and MNAR missingness mechanisms, across multiple datasets, missingness levels, and downstream machine-learning models.

## Research objectives

- Compare numerical imputation methods.
- Evaluate reconstruction and predictive performance under MCAR, MAR, and MNAR.
- Study the effect of missingness severity.
- Evaluate runtime, memory use, calibration, and robustness.
- Analyze relationships between reconstruction quality and predictive performance.

## Datasets

The notebook describes four datasets from the UCI Machine Learning Repository:

| Dataset | Sample size reported in project |
|---|---:|
| Adult Income | 5,000 |
| Breast Cancer Wisconsin (Original) | 699 |
| Heart Disease (Cleveland) | 303 |
| Bank Marketing | 5,000 |

The Bank Marketing `duration` feature is excluded from predictors. Consult the notebook and dataset documentation for preprocessing and provenance details. Dataset provenance has not been independently re-verified as part of the cross-mechanism file audit.

## Imputation methods

1. Mean
2. Median
3. Most frequent
4. K-nearest neighbours (KNN)
5. Iterative imputation

Categorical features are handled using most-frequent imputation.

## Missingness mechanisms and severity

- MCAR — Missing Completely At Random
- MAR — Missing At Random
- MNAR — Missing Not At Random

The primary experiments include 10%, 30%, and 50% missingness. Additional severity experiments include 20% and 40%.

## Classifiers

- Logistic Regression
- Decision Tree
- Random Forest

## Evaluation metrics

Predictive metrics include Accuracy, weighted Precision/Recall/F1, ROC-AUC, Average Precision (AP), Matthews Correlation Coefficient (MCC), and Expected Calibration Error (ECE).

Reconstruction metrics include Normalized MAE, Normalized RMSE, and categorical reconstruction accuracy.

## Audited cross-mechanism reconstruction results

**Status:** The corrected v2 cross-mechanism reconstruction outputs included in this repository package were independently recalculated from their matched fold-level data. This is a scoped audit of the supplied cross-mechanism reconstruction files, not certification of every experiment, figure, or result in the full research project.

The corrected v2 analysis compares MCAR, MAR, and MNAR using 25 matched observations per test (5 repetitions × 5 folds). Categorical accuracy is tested once per dataset/split/rate because it is duplicated across the five numeric imputers in the harmonized data.

| Mechanism comparison | Significant after Holm | Tests |
|---|---:|---:|
| MAR vs MNAR | 61 | 129 |
| MCAR vs MAR | 7 | 129 |
| MCAR vs MNAR | 64 | 129 |
| **Total** | **132** | **387** |

Significant results by metric:

| Comparison | Categorical accuracy | Normalized MAE | Normalized RMSE |
|---|---:|---:|---:|
| MAR vs MNAR | 9/9 | 31/60 | 21/60 |
| MCAR vs MAR | 0/9 | 7/60 | 0/60 |
| MCAR vs MNAR | 9/9 | 30/60 | 25/60 |

Higher categorical accuracy is better. Lower Normalized MAE and Normalized RMSE are better.

The statistical calculation uses a corrected resampled paired t-test with variance correction factor `1/25 + 1/4`, 24 degrees of freedom, two-sided p-values, and Holm adjustment across 387 tests. The numerical calculations and adjusted p-values were independently recomputed and matched the supplied corrected v2 outputs. The research methods section must still justify why this inferential procedure and the single 387-test correction family are appropriate for the study design; arithmetic verification alone does not establish that methodological choice.

## MAR categorical-accuracy correction

The supplied Heart Disease MAR reproduction audit reproduced 75 split records across 10%, 30%, and 50% missingness. The type-aware corrected means were 0.560000, 0.574652, and 0.577453. The earlier harmonized file retained the lower string-comparison values; the corrected v2 harmonized output incorporates the corrected values. Adult Income and Bank Marketing corrected MAR values matched their previous harmonized values. Breast Cancer Wisconsin has no categorical features in this analysis.

## Repository contents in this release package

- `notebooks/` — reviewed notebook copy with cross-mechanism downstream inputs updated to corrected v2 files.
- `results/cross_mechanism/final_outputs/` — authoritative corrected v2 cross-mechanism reconstruction files.
- `results/cross_mechanism/legacy_archive/` — superseded files retained for traceability; do not use for final reporting.
- `results/cross_mechanism/supporting_audit/` — MAR correction source and independent reproduction audit.
- `docs/` — audit scope, limitations, and change log.

This package does not include all raw datasets, MCAR/MNAR source experiment outputs, predictive outputs, Phase 3 result folders, figures, or tables. It is therefore not a complete archival copy of the full research workspace.

## Reproducibility

Use the corrected notebook copy and preserve raw experiment outputs. Run the cross-mechanism workflow in this order:

1. Task 4A — corrected harmonization.
2. Task 4B — descriptive summaries (updated to read the v2 harmonized files).
3. Task 4C — predictive statistical testing (updated to read the v2 predictive file).
4. Task 4E V2 — corrected reconstruction statistical testing.

The legacy Task 4E cell is disabled in the reviewed notebook to prevent accidental analysis from the superseded uncorrected harmonized file. Do not overwrite original raw experiment CSVs or legacy outputs.

## Technology

Python, NumPy, Pandas, scikit-learn, SciPy, statsmodels, Matplotlib, Seaborn, Google Colab, and Google Drive.

## Author

Tejesh Varma

## License

No license is included. Add one only after the authors and supervisor confirm the distribution terms.

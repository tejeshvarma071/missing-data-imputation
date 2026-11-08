# Comparative Evaluation of Missing-Data Imputation Methods

A research project investigating the performance of different missing-data imputation techniques under varying missingness mechanisms and severity levels.

## Overview

Missing data is a common challenge in real-world datasets. The choice of imputation technique can influence both the quality of reconstructed data and the performance of downstream machine learning models.

This research presents a comparative evaluation of five imputation methods across four datasets, three missingness mechanisms, multiple missingness severity levels, and three machine learning classifiers.

## Research Objectives

- Compare different numerical imputation techniques.
- Evaluate imputation performance under MCAR, MAR, and MNAR mechanisms.
- Investigate the impact of increasing missingness severity.
- Measure downstream classification performance after imputation.
- Analyze reconstruction accuracy and computational efficiency.
- Examine dataset-level, classifier-level, and feature-level robustness.

## Datasets

The study uses four datasets from the UCI Machine Learning Repository:


|               Dataset             | Sample Size |
|-----------------------------------|-------------|
|            Adult Income           |    5,000    |
| Breast Cancer Wisconsin(Original) |      699    |
|   Heart Disease (Cleveland)       |      303    |
|            Bank Marketing         |    5,000    |


The Bank Marketing `duration` feature is excluded from predictors. Dataset preprocessing and source details are documented separately.

## Imputation Methods

Five imputation techniques are evaluated:

1. Mean Imputation
2. Median Imputation
3. Most Frequent Imputation
4. K-Nearest Neighbours (KNN) Imputation
5. Iterative Imputation

Categorical features are handled using most-frequent imputation.

## Missingness Mechanisms

- **MCAR:** Missing Completely At Random
- **MAR:** Missing At Random
- **MNAR:** Missing Not At Random

The primary experiments evaluate missingness levels of 10%, 30%, and 50%. Additional experiments introduce 20% and 40% severity levels.

## Machine Learning Classifiers

- Logistic Regression
- Decision Tree
- Random Forest

## Evaluation Metrics

### Predictive Performance

- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1-score
- ROC-AUC
- Average Precision (AP)
- Matthews Correlation Coefficient (MCC)
- Expected Calibration Error (ECE)

### Reconstruction Performance

- Normalized Mean Absolute Error (MAE)
- Normalized Root Mean Squared Error (RMSE)
- Categorical Reconstruction Accuracy

### Additional Analyses

- Statistical significance testing
- Runtime and memory benchmarking
- Reconstruction-prediction association
- Feature-level reconstruction difficulty
- Dataset and classifier robustness

## Repository Structure

```text
missing-data-imputation/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── Missing_Data_Imputation_Research.ipynb
│
├── docs/
│   ├── dataset_documentation.md
│   ├── experiment_configuration_register.md
│   ├── results_documentation.md
│   └── reproducibility_guide.md
│
├── results/
│   ├── figures/
│   ├── tables/
│   └── validation/
│
└── configs/
    └── experiment_configurations/
```

## Technology Stack

- Python
- NumPy
- Pandas
- Scikit-learn
- SciPy
- Matplotlib
- Seaborn
- Google Colab
- Google Drive

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/tejeshvarma071/missing-data-imputation.git
cd missing-data-imputation
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the research notebook

Open the notebook in Google Colab or Jupyter Notebook.

### 4. Prepare the datasets

Follow the dataset restoration and preprocessing instructions in the notebook and dataset documentation.

### 5. Execute the experiments

Follow the phase-specific execution order in the reproducibility guide. Preserve the experiment configurations, random seeds, and output files.

## Documentation

Detailed project documentation is available in the `docs/` directory:

- Dataset Documentation
- Experiment Configuration Register
- Results Documentation
- Reproducibility Guide

## Reproducibility

The experiments use documented random seeds, phase-specific configurations, repeated cross-validation where applicable, and validation procedures.

The original fixed-split MCAR experiment and later repeated-CV experiments represent different experimental designs and should not be treated as interchangeable.

Refer to the reproducibility guide for environment setup, execution order, expected output counts, and validation procedures.

## Research Integrity

- Corrected evaluation outputs are used where documented.
- Experimental configurations are preserved.
- Validation and integrity checks are documented.
- Superseded result files should not be used for final reporting.
- Reported limitations and methodological distinctions should be retained when interpreting results.

## Author

**Tejesh Varma**

## License

A license will be added after the project authors and supervisor finalize the repository's distribution terms.

---

**Note:** This repository is intended for academic research, experimentation, and reproducibility of missing-data imputation methods.

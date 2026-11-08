# Source extraction map

Source of truth: `notebooks/Missing_Data_Imputation_Research_FINAL_REVIEWED.ipynb` (357 cells).

The package intentionally extracts reusable functions rather than moving whole experiment cells, because the notebook contains iterative revisions, repeated function names, Colab paths, and experiment orchestration tied to Drive files and notebook globals.

| Package module | Notebook source | Notes |
|---|---:|---|
| `data/schema.py` | Cells 11 and 220 | Type preparation and numeric/categorical feature identification, generalized to explicit categorical columns. |
| `missingness/masks.py` | Cells 198, 200, 223, 242, 275 | Training-fitted MAR driver, MCAR mask, categorical MNAR allocation and MNAR mask, corrected-v2 mechanism dispatch. |
| `imputation/methods.py` | Cell 200 | Five numeric imputer constructors and notebook iterative-imputer defaults. |
| `evaluation/reconstruction.py` | Cell 275 | Corrected categorical mode evaluator; original unmasked test values are ground truth. |
| `statistics/multiple_testing.py` | Notebook Holm adjustment cells | Standalone Holm adjustment utility; full 387-test analysis remains notebook-driven. |

## Deliberate exclusions

Dataset acquisition and preprocessing, full cross-validation orchestration, model/classifier pipelines, predictive metric suites, figure generation, CSV checkpoint/resume logic, and complete end-to-end experiment execution are not extracted in this first package. They rely on notebook-level state, external data files, or multiple historical variants. Do not describe this package as a complete reproduction engine.

## Validation

`tests/test_core.py` contains seven focused tests over synthetic data. They validate deterministic masks, MAR driver protection, MNAR feature partition checks, mechanism dispatch, imputer factory behavior, corrected categorical comparison, and Holm adjustment. These tests are software unit tests, not a rerun of the research experiment.

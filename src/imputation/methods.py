"""Five numeric imputation methods used in the benchmark."""
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

IMPUTER_NAMES = ("mean", "median", "most_frequent", "knn", "iterative")


def make_numeric_imputer(name: str, *, iterative_max_iter: int = 200,
                         iterative_tol: float = 1e-3, random_state: int = 42):
    """Create a fresh sklearn imputer; categorical preprocessing is separate."""
    key = name.lower().strip()
    if key == "mean":
        return SimpleImputer(strategy="mean")
    if key == "median":
        return SimpleImputer(strategy="median")
    if key == "most_frequent":
        return SimpleImputer(strategy="most_frequent")
    if key == "knn":
        return KNNImputer(n_neighbors=5)
    if key == "iterative":
        return IterativeImputer(max_iter=iterative_max_iter, tol=iterative_tol,
                                random_state=random_state)
    raise ValueError(f"Unknown imputer {name!r}. Choose from {IMPUTER_NAMES}.")

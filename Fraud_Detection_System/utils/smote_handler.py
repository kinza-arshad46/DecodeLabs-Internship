"""
utils/smote_handler.py
------------------------
Wraps imbalanced-learn's SMOTE.

apply_smote() is used ONLY for the Preprocessing page's before/after
visualization - it runs SMOTE once on the training data just to show the
class counts. It is NOT how SMOTE is applied to the actual models: for
training, get_smote() builds a fresh SMOTE step that lives inside the
model pipeline (see model_trainer.py) so it's re-applied inside every
cross-validation fold, on training data only.
"""

from imblearn.over_sampling import SMOTE


def get_smote(random_state: int = 42, k_neighbors: int = 5) -> SMOTE:
    return SMOTE(random_state=random_state, k_neighbors=k_neighbors)


def apply_smote(X_train, y_train, k_neighbors: int = 5, random_state: int = 42):
    """
    Runs SMOTE once for visualization purposes. Automatically lowers
    k_neighbors if the minority class has fewer samples than requested,
    to avoid a sklearn error on very small training folds.
    """
    n_minority = int(y_train.sum())
    safe_k = max(1, min(k_neighbors, n_minority - 1)) if n_minority > 1 else 1
    smote = get_smote(random_state=random_state, k_neighbors=safe_k)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res

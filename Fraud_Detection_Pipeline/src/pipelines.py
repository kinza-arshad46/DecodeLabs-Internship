"""
Leak-free pipelines using imblearn.pipeline.Pipeline.
Follows the Zero-Leakage Protocol from the project guide.
"""

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def build_logistic_pipeline(random_state: int = 42) -> ImbPipeline:
    """
    Linear Pipeline:
    StandardScaler -> SMOTE -> LogisticRegression

    StandardScaler MUST live inside the pipeline so that
    mean/variance statistics are computed only on training folds.
    """
    return ImbPipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=random_state)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=random_state,
                    solver="lbfgs",
                    n_jobs=1,
                ),
            ),
        ]
    )


def build_random_forest_pipeline(random_state: int = 42) -> ImbPipeline:
    """
    Ensemble Pipeline:
    SMOTE -> RandomForestClassifier

    Tree-based models are scale-invariant; no StandardScaler needed.
    """
    return ImbPipeline(
        steps=[
            ("smote", SMOTE(random_state=random_state)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=random_state,
                    n_jobs=1,
                    class_weight=None,  # SMOTE already balances
                ),
            ),
        ]
    )


def get_param_grids():
    """
    Hyperparameter grids for GridSearchCV.
    Kept reasonably small so the project runs in acceptable time
    on standard hardware while still demonstrating proper tuning.
    """
    lr_param_grid = {
        "smote__k_neighbors": [3, 5],
        "classifier__C": [0.01, 0.1, 1.0],
        "classifier__penalty": ["l2"],
    }

    rf_param_grid = {
        "smote__k_neighbors": [3, 5],
        "classifier__n_estimators": [50, 100],
        "classifier__max_depth": [10, 20, None],
        "classifier__min_samples_split": [2, 5],
    }

    return lr_param_grid, rf_param_grid

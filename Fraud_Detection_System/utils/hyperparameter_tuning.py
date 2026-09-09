"""
utils/hyperparameter_tuning.py
--------------------------------
GridSearchCV configuration, kept separate from model_trainer.py so the
tuning strategy (scoring metric, CV strategy) can be changed in one place.
"""

from sklearn.model_selection import GridSearchCV, StratifiedKFold


def get_param_grid(model_name: str, quick: bool = False) -> dict:
    """
    quick=True gives a smaller grid - useful while developing, since a full
    grid search with SMOTE inside every fold can be slow on the full
    284,807-row dataset.
    """
    if model_name == "Logistic Regression":
        if quick:
            return {"classifier__C": [0.01, 1.0]}
        return {"classifier__C": [0.01, 0.1, 1.0, 10.0]}
    elif model_name == "Random Forest":
        if quick:
            return {"classifier__n_estimators": [100], "classifier__max_depth": [10]}
        return {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [10, 20, None],
            "classifier__min_samples_split": [2, 5],
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_grid_search(pipeline, model_name, X_train, y_train, cv_folds: int = 5, quick: bool = False):
    """
    Runs GridSearchCV scored on ROC-AUC with StratifiedKFold, so every fold
    keeps the true class ratio and SMOTE is re-applied fresh inside each
    fold by the pipeline itself (never on the held-out fold).

    Returns (best_estimator, best_params: dict).
    """
    param_grid = get_param_grid(model_name, quick=quick)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_

"""
utils/model_trainer.py
------------------------
Builds the leak-free model pipelines and orchestrates training.

Why a pipeline at all?
    StandardScaler and SMOTE are placed INSIDE an imblearn Pipeline so that
    when the pipeline is fit, both steps run only on the data passed to
    .fit() (the training fold). When GridSearchCV cross-validates this
    pipeline, it refits the whole thing - scaler + SMOTE + model - fresh on
    each fold's training portion, and evaluates on that fold's untouched
    validation portion. This is what makes the pipeline "leak-free".

Persistence:
    Trained pipelines are saved as .pkl files under models/, so predictions
    can be made in a later session without retraining, and so a scaler.pkl
    is available for any downstream use outside this app.
"""

import os
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline

from utils.smote_handler import get_smote
from utils.hyperparameter_tuning import run_grid_search

MODELS_DIR = "models"


def build_pipeline(model_name: str, random_state: int = 42, k_neighbors: int = 5) -> ImbPipeline:
    """
    - Logistic Regression: StandardScaler -> SMOTE -> LogisticRegression
      (LR is scale-sensitive, so the scaler must be inside the pipeline)
    - Random Forest: SMOTE -> RandomForestClassifier
      (tree splits are scale-invariant, so no scaler is needed)
    """
    if model_name == "Logistic Regression":
        steps = [
            ("scaler", StandardScaler()),
            ("smote", get_smote(random_state=random_state, k_neighbors=k_neighbors)),
            ("classifier", LogisticRegression(max_iter=2000, random_state=random_state)),
        ]
    elif model_name == "Random Forest":
        steps = [
            ("smote", get_smote(random_state=random_state, k_neighbors=k_neighbors)),
            ("classifier", RandomForestClassifier(random_state=random_state, n_jobs=-1)),
        ]
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return ImbPipeline(steps=steps)


def train_model(model_name, X_train, y_train, cv_folds: int = 5, use_grid_search: bool = True, quick_grid: bool = False):
    """
    Returns (best_estimator, best_params: dict).
    """
    pipeline = build_pipeline(model_name)

    if not use_grid_search:
        pipeline.fit(X_train, y_train)
        return pipeline, {"note": "default parameters (tuning disabled)"}

    return run_grid_search(pipeline, model_name, X_train, y_train, cv_folds=cv_folds, quick=quick_grid)


def save_model(model, model_name: str) -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_")
    path = os.path.join(MODELS_DIR, f"{safe_name}.pkl")
    joblib.dump(model, path)

    # Logistic Regression's fitted scaler is also exported on its own,
    # so it can be reused outside this app's pipelines if ever needed.
    if model_name == "Logistic Regression" and "scaler" in model.named_steps:
        joblib.dump(model.named_steps["scaler"], os.path.join(MODELS_DIR, "scaler.pkl"))

    return path


def save_best_model(model, model_name: str) -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump({"model": model, "name": model_name}, path)
    return path


def load_model(model_name: str):
    safe_name = model_name.lower().replace(" ", "_")
    path = os.path.join(MODELS_DIR, f"{safe_name}.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None


def load_best_model():
    path = os.path.join(MODELS_DIR, "best_model.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None

"""
utils/preprocessing.py
-----------------------
Data cleaning and train/test splitting.

Scaling and SMOTE are deliberately NOT done here on the full dataset - they
are applied later, inside the model pipeline (see model_trainer.py), and
ONLY on the training fold. Splitting must always happen first.
"""

from sklearn.model_selection import train_test_split


def clean_data(df):
    """
    Basic cleaning: drop exact duplicate rows and report missing values.
    Returns (cleaned_df, report: dict).
    """
    report = {
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    cleaned = df.drop_duplicates().reset_index(drop=True)
    return cleaned, report


def get_feature_target(df):
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def split_data(df, test_size: float = 0.2, random_state: int = 42):
    """
    Stratified train/test split so both sets keep the real, extreme fraud
    ratio. This must run BEFORE any scaling or resampling.
    """
    X, y = get_feature_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Constants
MONTH_ORDER = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "June": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

VISITOR_TYPE_MAP = {
    "Returning_Visitor": 0,
    "New_Visitor": 1,
    "Other": 2,
}

# Functions
def load_data(data_path: Path) -> pd.DataFrame:

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    df = pd.read_csv(data_path)
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # Month — ordinal encoding
    df["Month"] = df["Month"].map(MONTH_ORDER)
    # Fill any unmapped months with 0 (shouldn't happen with clean data)
    df["Month"] = df["Month"].fillna(0).astype(int)

    # VisitorType — label encoding
    df["VisitorType"] = df["VisitorType"].map(VISITOR_TYPE_MAP)
    df["VisitorType"] = df["VisitorType"].fillna(2).astype(int)  # 'Other' fallback

    # Boolean columns → int
    df["Weekend"] = df["Weekend"].astype(int)
    df["Revenue"] = df["Revenue"].astype(int)

    return df


def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:

    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Ensure y are numpy arrays
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return X_train, X_test, y_train, y_test, scaler, feature_names


def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> tuple:

    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

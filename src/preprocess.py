# src/preprocess.py
# ─────────────────────────────────────────────────────────
#  Reusable preprocessing functions
#  Import these in any script or notebook
# ─────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


FEATURE_COLS = [
    'GrLivArea',       # Above-ground living area (sq ft)
    'OverallQual',     # Overall material quality (1-10)
    'YearBuilt',       # Original construction year
    'TotalBsmtSF',     # Total basement area (sq ft)
    'FullBath',        # Full bathrooms above grade
    'GarageCars',      # Garage capacity (cars)
    'Neighborhood',    # Physical location (categorical)
]
TARGET_COL = 'SalePrice'


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values: median for numeric, mode for categorical."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)
    return df


def encode_categoricals(df: pd.DataFrame, le: LabelEncoder = None):
    """
    Label-encode the Neighborhood column.
    Returns (transformed_df, fitted_label_encoder)
    Pass an existing le to transform (not fit) — useful for test/prediction data.
    """
    df = df.copy()
    if le is None:
        le = LabelEncoder()
        df['Neighborhood_enc'] = le.fit_transform(df['Neighborhood'])
    else:
        # Handle unseen categories gracefully
        df['Neighborhood_enc'] = df['Neighborhood'].apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else 0
        )
    df.drop(columns=['Neighborhood'], inplace=True)
    return df, le


def scale_features(X_train, X_test=None):
    """
    Fit StandardScaler on training data, transform both train and test.
    Returns (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test) if X_test is not None else None
    return X_train_scaled, X_test_scaled, scaler


def full_pipeline(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Run the full preprocessing pipeline on a raw dataframe.
    Returns: X_train, X_test, y_train, y_test, scaler, label_encoder
    """
    # Keep only relevant columns
    df = df[FEATURE_COLS + [TARGET_COL]].copy()

    # Fix missing values
    df = handle_missing_values(df)

    # Encode categoricals
    df, le = encode_categoricals(df)

    # Split features and target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

    X_train_sc = pd.DataFrame(X_train_sc, columns=X_train.columns)
    X_test_sc  = pd.DataFrame(X_test_sc,  columns=X_test.columns)

    return X_train_sc, X_test_sc, y_train, y_test, scaler, le

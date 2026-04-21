# src/model.py
# ─────────────────────────────────────────────────────────
#  Model training, evaluation, and prediction functions
# ─────────────────────────────────────────────────────────

import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_linear_regression(X_train, y_train):
    """Train a Linear Regression model and return it."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """Train a Random Forest model and return it."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a trained model and print metrics.
    Returns dict of metrics.
    """
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    print(f"\n{'─'*45}")
    print(f"  {model_name} — Evaluation Metrics")
    print(f"{'─'*45}")
    print(f"  MAE  : ${mae:>12,.0f}  (avg prediction error)")
    print(f"  RMSE : ${rmse:>12,.0f}  (penalizes big errors)")
    print(f"  R²   : {r2:>14.4f}  (1.0 = perfect)")
    print(f"{'─'*45}")

    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'predictions': y_pred}


def save_model(model, path):
    """Save a trained model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"✅ Model saved → {path}")


def load_model(path):
    """Load a saved model from disk."""
    model = joblib.load(path)
    print(f"✅ Model loaded ← {path}")
    return model


def predict_single(model, scaler, label_encoder, input_dict):
    """
    Predict price for a single house.
    
    input_dict example:
    {
        'GrLivArea': 1800, 'OverallQual': 7, 'YearBuilt': 2005,
        'TotalBsmtSF': 900, 'FullBath': 2, 'GarageCars': 2,
        'Neighborhood': 'CollgCr'
    }
    """
    neigh = input_dict.get('Neighborhood', '')
    if neigh in label_encoder.classes_:
        neigh_enc = label_encoder.transform([neigh])[0]
    else:
        neigh_enc = 0

    features = np.array([[
        input_dict['GrLivArea'],
        input_dict['OverallQual'],
        input_dict['YearBuilt'],
        input_dict['TotalBsmtSF'],
        input_dict['FullBath'],
        input_dict['GarageCars'],
        neigh_enc
    ]])

    features_scaled = scaler.transform(features)
    price = model.predict(features_scaled)[0]
    return max(price, 0)

# ============================================================
#  HOUSE PRICE PREDICTION USING LINEAR REGRESSION
#  Author  : Yahwanth R
#  Dataset : Ames Housing Dataset (Kaggle)
#  Goal    : Predict sale price of a house based on features
# ============================================================

# ─────────────────────────────────────────────
# STEP 1 — IMPORT LIBRARIES
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


warnings.filterwarnings('ignore')
import logging
logging.captureWarnings(True)

sns.set_theme(style="whitegrid")
print(" All libraries imported successfully!")


# ─────────────────────────────────────────────
# STEP 2 — LOAD DATASET
# ─────────────────────────────────────────────
def load_or_generate_data(filepath="data/train.csv"):
    """Load real CSV if available, else generate a realistic synthetic dataset."""
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        print(f" Dataset loaded from {filepath}")
        print(f"   Shape: {df.shape}")

        # Normalise column names — handles both AmesHousing.csv (spaces) and train.csv (camelCase)
        rename_map = {
            'Gr Liv Area'   : 'GrLivArea',
            'Overall Qual'  : 'OverallQual',
            'Overall Cond'  : 'OverallCond',
            'Year Built'    : 'YearBuilt',
            'Total Bsmt SF' : 'TotalBsmtSF',
            'Full Bath'     : 'FullBath',
            'Garage Cars'   : 'GarageCars',
            'Sale Price'    : 'SalePrice',
        }
        df.rename(columns=rename_map, inplace=True)
        print("   Column names normalised.")
    else:
        print("  Real dataset not found. Generating synthetic data for demo...")
        np.random.seed(42)
        n = 1000

        neighborhoods = ['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel',
                         'Somerst', 'NWAmes', 'OldTown', 'BrkSide', 'Sawyer']
        neigh_multiplier = {nb: v for nb, v in zip(neighborhoods,
                            [1.3, 1.2, 1.1, 1.4, 0.9, 1.15, 1.0, 0.85, 0.8, 0.95])}

        neigh        = np.random.choice(neighborhoods, n)
        gr_liv_area  = np.random.normal(1500, 400, n).clip(600, 4000).astype(int)
        overall_qual = np.random.randint(1, 11, n)
        year_built   = np.random.randint(1900, 2011, n)
        total_bsmt   = np.random.normal(1000, 300, n).clip(0, 3000).astype(int)
        full_bath    = np.random.randint(1, 4, n)
        garage_cars  = np.random.randint(0, 4, n)

        price = (
            50000
            + gr_liv_area   * 60
            + overall_qual  * 8000
            + (2010 - year_built) * (-200)
            + total_bsmt    * 20
            + full_bath     * 5000
            + garage_cars   * 6000
            + np.array([neigh_multiplier[x] for x in neigh]) * 15000
            + np.random.normal(0, 15000, n)
        ).clip(50000, 750000).astype(int)

        df = pd.DataFrame({
            'Neighborhood': neigh,
            'GrLivArea':    gr_liv_area,
            'OverallQual':  overall_qual,
            'YearBuilt':    year_built,
            'TotalBsmtSF':  total_bsmt,
            'FullBath':     full_bath,
            'GarageCars':   garage_cars,
            'SalePrice':    price
        })
        print(f" Synthetic dataset generated. Shape: {df.shape}")

    return df

df = load_or_generate_data()

print("\n First 5 rows:")
print(df.head())
print("\n Dataset Info:")
print(df.info())
print("\n Statistical Summary:")
print(df.describe())
print("\n Missing Values:")
print(df.isnull().sum())


# ─────────────────────────────────────────────
# STEP 3 — DATA PREPROCESSING
# ─────────────────────────────────────────────
print("\n🔧 Starting Preprocessing...")

FEATURES = ['GrLivArea', 'OverallQual', 'YearBuilt',
            'TotalBsmtSF', 'FullBath', 'GarageCars', 'Neighborhood']
TARGET = 'SalePrice'

df_model = df[FEATURES + [TARGET]].copy()

# Fill ALL missing values robustly.
# Key insight: newer pandas (2.x+) reads text columns as dtype 'str' (StringDtype),
# NOT 'object'. So we cannot rely on is_object_dtype() alone.
# Solution: check if numeric explicitly; treat everything else as categorical.
for col in df_model.columns:
    if pd.api.types.is_numeric_dtype(df_model[col]):
        # Numeric (int64 or float64): fill with median
        df_model[col] = df_model[col].fillna(df_model[col].median())
    else:
        # Text/categorical (object or str): fill with most frequent value
        df_model[col] = df_model[col].fillna(df_model[col].mode()[0])

# Convert float64 columns to int now that NaNs are gone (float was only needed for NaN storage)
for col in ['TotalBsmtSF', 'GarageCars']:
    if col in df_model.columns:
        df_model[col] = df_model[col].astype(int)

remaining = df_model.isnull().sum().sum()
print(f"   Missing values after fix: {remaining}")
if remaining > 0:
    # Ultimate safety net: drop any remaining rows with NaN
    df_model = df_model.dropna()
    print(f"   Rows after dropna: {len(df_model)}")
print("    No NaN values — safe to train!")

le = LabelEncoder()
df_model['Neighborhood_enc'] = le.fit_transform(df_model['Neighborhood'])

os.makedirs("model", exist_ok=True)
joblib.dump(le, "model/label_encoder.pkl")
print(f"   Neighborhoods encoded: {list(le.classes_)}")

feature_cols = ['GrLivArea', 'OverallQual', 'YearBuilt',
                'TotalBsmtSF', 'FullBath', 'GarageCars', 'Neighborhood_enc']

X = df_model[feature_cols]
y = df_model[TARGET]

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

joblib.dump(scaler, "model/scaler.pkl")
print("   Feature scaling done (StandardScaler applied).")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"   Training samples : {X_train.shape[0]}")
print(f"   Testing  samples : {X_test.shape[0]}")
print(" Preprocessing complete!")


# ─────────────────────────────────────────────
# STEP 4 — DATA VISUALIZATION
# ─────────────────────────────────────────────
print("\n📊 Generating visualizations...")
os.makedirs("outputs", exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("House Price Prediction — Exploratory Data Analysis",
             fontsize=16, fontweight='bold')

sns.histplot(df_model['SalePrice'], bins=40, kde=True, ax=axes[0,0], color='steelblue')
axes[0,0].set_title("Distribution of Sale Price")
axes[0,0].set_xlabel("Sale Price ($)")

axes[0,1].scatter(df_model['GrLivArea'], df_model['SalePrice'],
                  alpha=0.4, color='coral', edgecolors='none', s=20)
axes[0,1].set_title("Living Area vs Sale Price")
axes[0,1].set_xlabel("Above-Ground Living Area (sq ft)")
axes[0,1].set_ylabel("Sale Price ($)")

sns.boxplot(data=df_model, x='OverallQual', y='SalePrice', ax=axes[0,2], palette='Blues')
axes[0,2].set_title("Overall Quality vs Sale Price")
axes[0,2].set_xlabel("Overall Quality (1-10)")

numeric_df = df_model[feature_cols + [TARGET]].copy()
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm',
            ax=axes[1,0], linewidths=0.5, square=True)
axes[1,0].set_title("Feature Correlation Heatmap")

axes[1,1].scatter(df_model['YearBuilt'], df_model['SalePrice'],
                  alpha=0.4, color='mediumseagreen', edgecolors='none', s=20)
axes[1,1].set_title("Year Built vs Sale Price")
axes[1,1].set_xlabel("Year Built")
axes[1,1].set_ylabel("Sale Price ($)")

avg_price = df_model.groupby('Neighborhood')['SalePrice'].mean().sort_values(ascending=True)
avg_price.plot(kind='barh', ax=axes[1,2], color='mediumpurple')
axes[1,2].set_title("Avg Sale Price by Neighborhood")
axes[1,2].set_xlabel("Average Sale Price ($)")

plt.tight_layout()
plt.savefig("outputs/eda_plots.png", dpi=120, bbox_inches='tight')
plt.close()   # ← FIX 4: close figure instead of plt.show() — avoids GUI/display errors
print(" EDA plots saved to outputs/eda_plots.png")


# ─────────────────────────────────────────────
# STEP 5 — MODEL BUILDING
# ─────────────────────────────────────────────
print("\n Training Linear Regression model...")

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
joblib.dump(lr_model, "model/linear_regression_model.pkl")
print(" Model trained and saved to model/linear_regression_model.pkl")

coeff_df = pd.DataFrame({
    'Feature':     feature_cols,
    'Coefficient': lr_model.coef_
}).sort_values('Coefficient', ascending=False)
print("\n Feature Coefficients (impact on price):")
print(coeff_df.to_string(index=False))
print(f"\n   Intercept: ${lr_model.intercept_:,.0f}")


# ─────────────────────────────────────────────
# STEP 6 — MODEL EVALUATION
# ─────────────────────────────────────────────
print("\n Evaluating model performance...")

y_pred_lr = lr_model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred_lr)
mse  = mean_squared_error(y_test, y_pred_lr)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred_lr)

print("\n┌──────────────────────────────────────────────────┐")
print("│         LINEAR REGRESSION — RESULTS               │")
print("├──────────────────────────────────────────────────┤")
print(f"│  MAE  (Mean Absolute Error)  : ${mae:>12,.0f}    │")
print(f"│  RMSE (Root Mean Sq. Error)  : ${rmse:>12,.0f}    │")
print(f"│  R²   (R-Squared Score)      : {r2:>14.4f}    │")
print("└──────────────────────────────────────────────────┘")
print(f"""
 What do these mean?
   • MAE  = On average, our predictions are off by ${mae:,.0f}
   • RMSE = Penalizes larger errors more. Lower is better.
   • R²   = {r2:.1%} of price variation is explained by our features.
             R² closer to 1.0 means a better model.
""")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_test, y_pred_lr, alpha=0.4, color='steelblue', s=20)
axes[0].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect prediction')
axes[0].set_xlabel("Actual Price ($)")
axes[0].set_ylabel("Predicted Price ($)")
axes[0].set_title("Linear Regression: Actual vs Predicted")
axes[0].legend()

residuals = y_test - y_pred_lr
axes[1].scatter(y_pred_lr, residuals, alpha=0.4, color='coral', s=20)
axes[1].axhline(y=0, color='black', linestyle='--', lw=2)
axes[1].set_xlabel("Predicted Price ($)")
axes[1].set_ylabel("Residual (Actual - Predicted)")
axes[1].set_title("Residuals Plot (should be random around 0)")

plt.tight_layout()
plt.savefig("outputs/model_evaluation.png", dpi=120, bbox_inches='tight')
plt.close()   # ← FIX 4 applied here too
print(" Evaluation plots saved to outputs/model_evaluation.png")


# ─────────────────────────────────────────────
# BONUS — COMPARE WITH RANDOM FOREST
# ─────────────────────────────────────────────
print("\n BONUS: Training Random Forest for comparison...")

rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

rf_mae  = mean_absolute_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_r2   = r2_score(y_test, y_pred_rf)

print("\n┌──────────────────────────────────────────────────┐")
print("│           MODEL COMPARISON                        │")
print("├────────────────────┬─────────────┬───────────────┤")
print("│ Metric             │ Lin. Reg.   │ Random Forest │")
print("├────────────────────┼─────────────┼───────────────┤")
print(f"│ MAE ($)            │ {mae:>10,.0f}  │ {rf_mae:>12,.0f}  │")
print(f"│ RMSE ($)           │ {rmse:>10,.0f}  │ {rf_rmse:>12,.0f}  │")
print(f"│ R² Score           │ {r2:>11.4f}  │ {rf_r2:>13.4f}  │")
print("└────────────────────┴─────────────┴───────────────┘")

joblib.dump(rf_model, "model/random_forest_model.pkl")

feat_imp = pd.DataFrame({
    'Feature':    feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=True)

plt.figure(figsize=(8, 4))
feat_imp.plot(kind='barh', x='Feature', y='Importance',
              color='steelblue', legend=False)
plt.title("Random Forest — Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=120, bbox_inches='tight')
plt.close()   # ← FIX 4 applied here too
print(" Feature importance plot saved to outputs/feature_importance.png")


# ─────────────────────────────────────────────
# STEP 7 — PREDICT ON NEW INPUT
# ─────────────────────────────────────────────
print("\n Making predictions for new houses...")

loaded_model  = joblib.load("model/linear_regression_model.pkl")
loaded_scaler = joblib.load("model/scaler.pkl")
loaded_le     = joblib.load("model/label_encoder.pkl")

def predict_price(gr_liv_area, overall_qual, year_built,
                  total_bsmt, full_bath, garage_cars, neighborhood):
    """Predict house price for given features."""
    if neighborhood in loaded_le.classes_:
        neigh_enc = loaded_le.transform([neighborhood])[0]
    else:
        neigh_enc = 0

    input_data   = np.array([[gr_liv_area, overall_qual, year_built,
                               total_bsmt, full_bath, garage_cars, neigh_enc]])
    input_scaled = loaded_scaler.transform(input_data)
    return max(loaded_model.predict(input_scaled)[0], 0)


test_houses = [
    (1800,  7, 2005,  900, 2, 2, 'CollgCr'),
    (2500,  9, 2010, 1200, 3, 3, 'NoRidge'),
    (900,   4, 1965,  600, 1, 1, 'OldTown'),
    (3200, 10, 2015, 1500, 4, 3, 'Somerst'),
    (1200,  5, 1985,  800, 2, 1, 'BrkSide'),
]

print("\n┌──────┬────────┬──────┬──────────┬──────────────────┬──────────────────┐")
print("│ Area │  Qual  │ Year │   Bsmt   │  Neighborhood    │  Predicted Price │")
print("├──────┼────────┼──────┼──────────┼──────────────────┼──────────────────┤")
for house in test_houses:
    price = predict_price(*house)
    print(f"│{house[0]:>5} │   {house[1]:>2}   │ {house[2]} │  {house[3]:>5}   │ {house[6]:<16} │  ${price:>13,.0f} │")
print("└──────┴────────┴──────┴──────────┴──────────────────┴──────────────────┘")

print("""
 How to read these results:
   • House with area=1800, quality=7 in CollgCr → ~middle-range price
   • Larger area + higher quality = much higher price  ✓
   • Older house in OldTown (1965) = lower price       ✓
   • Luxury house (quality=10, area=3200) = top price  ✓
   These results match real-world expectations!
""")

print(" Project complete! All outputs saved in the outputs/ folder.")
print("   Models saved in the model/ folder.")

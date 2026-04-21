# 🏠 House Price Prediction using Linear Regression

> **A complete end-to-end Machine Learning project for BCA/B.Tech final year students.**  
> Predict the sale price of a house based on features like area, quality, location, and more.

---

## 📌 Problem Statement

Real estate prices fluctuate based on many factors — size, location, age, quality, etc.  
Manually estimating prices is time-consuming and error-prone.  
**Goal:** Build a Machine Learning model that can predict the sale price of a house  
given its features, with measurable accuracy.

---

## 📁 Project Structure

```
house_price_prediction/
│
├── data/
│   └── train.csv                  ← Download from Kaggle (see below)
│
├── notebook/
│   └── house_price_prediction.py  ← Main step-by-step code (run this!)
│
├── src/
│   ├── preprocess.py              ← Data cleaning & encoding functions
│   └── model.py                   ← Model training & prediction functions
│
├── model/
│   ├── linear_regression_model.pkl ← Saved trained model
│   ├── random_forest_model.pkl     ← Saved bonus model
│   ├── scaler.pkl                  ← Saved StandardScaler
│   └── label_encoder.pkl           ← Saved LabelEncoder
│
├── outputs/
│   ├── eda_plots.png              ← Exploratory data analysis charts
│   ├── model_evaluation.png       ← Actual vs Predicted + Residuals
│   └── feature_importance.png     ← Random Forest feature importance
│
├── requirements.txt
└── README.md                      ← You are here!
```

---

## 📊 Dataset

**Ames Housing Dataset** — one of the most popular datasets for regression tasks.

- 🔗 **Kaggle Link:** https://www.kaggle.com/datasets/prevek18/ames-housing-dataset
- 📦 Download `train.csv` and place it in the `data/` folder

### Key Features Used

| Column | Description | Type |
|---|---|---|
| `GrLivArea` | Above-ground living area (sq ft) | Numeric |
| `OverallQual` | Overall material & finish quality (1–10) | Numeric |
| `YearBuilt` | Original construction year | Numeric |
| `TotalBsmtSF` | Total basement area (sq ft) | Numeric |
| `FullBath` | Full bathrooms above grade | Numeric |
| `GarageCars` | Garage capacity in car count | Numeric |
| `Neighborhood` | Physical location within Ames | Categorical |
| `SalePrice` | **Target — house sale price ($)** | Numeric |

---

## 🛠️ Technologies Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting graphs |
| `seaborn` | Statistical visualizations |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `joblib` | Saving and loading models |

---

## ⚙️ Steps to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/yashwanthr12/house-price-prediction.git
cd house-price-prediction
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the Dataset
- Go to: https://www.kaggle.com/datasets/prevek18/ames-housing-dataset
- Download `train.csv`
- Place it in the `data/` folder

> **No Kaggle account?** No problem! The script auto-generates a realistic  
> synthetic dataset if `data/train.csv` is not found.

### 5. Run the Main Script
```bash
python notebook/house_price_prediction.py
```

### 6. Check Outputs
After running, you'll find:
- **Plots** in `outputs/`
- **Saved models** in `model/`

---

## 📈 Model Results

| Metric | Linear Regression | Random Forest |
|---|---|---|
| MAE | ~$22,000 | ~$17,000 |
| RMSE | ~$30,000 | ~$23,000 |
| R² Score | ~0.82 | ~0.89 |

> Results may vary slightly based on the dataset used (real vs synthetic).

---

## 🔍 Sample Predictions

```
Area  |  Quality  |  Year  |  Neighborhood  |  Predicted Price
------|-----------|--------|----------------|------------------
1800  |     7     |  2005  |    CollgCr     |    $198,450
2500  |     9     |  2010  |    NoRidge     |    $312,800
 900  |     4     |  1965  |    OldTown     |    $112,300
3200  |    10     |  2015  |    Somerst     |    $428,600
```

---

## 🖼️ Output Screenshots to Include

When uploading to GitHub, include these screenshots in your README or a `/screenshots` folder:

1. **`eda_plots.png`** — 6-panel EDA dashboard (price distribution, scatter plots, heatmap)
2. **`model_evaluation.png`** — Actual vs Predicted plot + Residuals plot
3. **`feature_importance.png`** — Which features matter most (Random Forest)
4. **Terminal output** — Model metrics table comparing both models

---

## 💡 Key Learnings

- How to handle **missing values** in real-world data
- How to **encode categorical variables** (text → numbers)
- Why **feature scaling** matters for Linear Regression
- How to evaluate models using **MAE, RMSE, and R²**
- How to **save and reload** trained models using `joblib`
- Difference between **Linear Regression vs Random Forest**

---

## 🎯 Interview Questions You Can Answer After This Project

1. *Why did you use Linear Regression for this problem?*
2. *What is the difference between MAE and RMSE?*
3. *Why do we need feature scaling in Linear Regression?*
4. *How did you handle missing values in your dataset?*
5. *Why is R² a useful metric? What does R²=0.82 mean?*
6. *Why did Random Forest perform better than Linear Regression here?*

---

## 👤 Author

**Yashwanth R**  
🔗 [LinkedIn](https://www.linkedin.com/in/yashwanthr12/) | [GitHub](https://github.com/yashwanthr12)

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

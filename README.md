# Car Price Prediction with Machine Learning

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-important?logo=xgboost&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![R²](https://img.shields.io/badge/Best%20R%C2%B2-0.97-success)

> **CodeAlpha Data Science Internship — Task 3**  
> Production-grade regression pipeline — Shapiro-Wilk normality check, VIF multicollinearity analysis, RandomizedSearchCV hyperparameter optimisation, full regression diagnostics, partial dependence plots, and a deployable joblib pipeline.

---

## Problem Statement

Predict the resale (selling) price of used cars based on vehicle attributes: age, mileage, fuel type, transmission, and original showroom price. This enables dealers and consumers to make data-driven pricing decisions rather than relying on intuition.

---

## Dataset

| Attribute | Detail |
|-----------|--------|
| Source | Kaggle Used Car Dataset |
| Samples | 301 cars |
| Features | 9 (mix of numeric and categorical) |
| Target | Selling Price (₹ Lakhs, continuous) |
| Missing Values | None |

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `Car_Name` | Categorical | Brand/model (dropped — too granular) |
| `Year` | Numeric | Manufacturing year → engineered to `Car_Age` |
| `Selling_Price` | Numeric | **Target variable** |
| `Present_Price` | Numeric | Current ex-showroom price |
| `Driven_kms` | Numeric | Total kilometres driven |
| `Fuel_Type` | Categorical | Petrol / Diesel / CNG |
| `Selling_type` | Categorical | Dealer / Individual |
| `Transmission` | Categorical | Manual / Automatic |
| `Owner` | Numeric | Number of previous owners |

---

## Methodology

```
Raw Data
   │
   ├── EDA
   │     ├── Shapiro-Wilk normality test on target
   │     ├── Q-Q plot (raw vs log-transformed price)
   │     ├── Categorical features vs price (box plots with medians)
   │     └── Correlation heatmap + Present vs Selling scatter
   │
   ├── Feature Engineering
   │     ├── Car_Age = 2024 − Year
   │     ├── Price_Depreciation = (Present − Selling) / Present
   │     └── LabelEncoding: Fuel_Type, Selling_type, Transmission
   │
   ├── Multicollinearity Check (VIF)
   │     └── Variance Inflation Factor for all features
   │
   ├── Preprocessing
   │     ├── Train/test split (80/20, random_state=42)
   │     └── StandardScaler
   │
   ├── Baseline Models (+ 5-Fold CV)
   │     ├── Linear Regression
   │     ├── Ridge Regression
   │     ├── Lasso Regression
   │     ├── Random Forest (200 trees)
   │     ├── Gradient Boosting
   │     └── XGBoost
   │
   ├── Hyperparameter Tuning
   │     └── RandomizedSearchCV on XGBoost (n_iter=40)
   │         Parameters: n_estimators, max_depth, learning_rate,
   │                     subsample, colsample_bytree, min_child_weight
   │
   ├── Advanced Diagnostics
   │     ├── Actual vs Predicted scatter
   │     ├── Residuals vs Fitted
   │     ├── Q-Q plot of residuals
   │     └── Residual distribution histogram
   │
   ├── Interpretability
   │     ├── Permutation importance (model-agnostic)
   │     ├── XGBoost gain-based importance
   │     └── Partial Dependence Plots (top 2 features)
   │
   └── Model Persistence
         └── joblib Pipeline (StandardScaler + XGBoost)
```

---

## Results

| Model | MAE (₹L) | RMSE (₹L) | R² | CV R² |
|-------|:--------:|:---------:|:--:|:-----:|
| Linear Regression | ~1.50 | ~2.80 | ~0.83 | ~0.82 |
| Ridge Regression | ~1.50 | ~2.80 | ~0.83 | ~0.82 |
| Lasso Regression | ~1.60 | ~3.00 | ~0.81 | ~0.80 |
| Random Forest | ~0.80 | ~1.50 | ~0.95 | ~0.94 |
| Gradient Boosting | ~0.70 | ~1.40 | ~0.96 | ~0.95 |
| XGBoost (baseline) | ~0.70 | ~1.40 | ~0.96 | ~0.95 |
| **XGBoost (Tuned)** | **~0.55** | **~1.10** | **~0.97** | **~0.96** |

### Key Findings

1. **Present Price** is the dominant predictor — showroom cost anchors resale value (Permutation Importance rank #1).
2. **Car Age** (engineered feature) is the second strongest predictor — value depreciates ~20% per year.
3. **Diesel vehicles** command a significant resale premium over Petrol (confirmed by box plots and model coefficients).
4. **Kilometres driven** beyond ~80K shows accelerating depreciation — captured by XGBoost's non-linearity.
5. **RandomizedSearchCV** improved XGBoost R² from 0.96 → 0.97 by tuning regularisation parameters.
6. **VIF analysis** confirmed no severe multicollinearity (all VIF < 10).

---

## Project Structure

```
3_Car_Data_Analysis/
├── Car_Analysis.ipynb             # Full analysis notebook
├── data/
│   └── car_data.csv
├── models/
│   ├── car_price_predictor.pkl    # Trained pipeline (scaler + XGBoost)
│   └── car_price_metadata.pkl    # Feature names & label encoders
├── reports/
│   ├── 01_target_dist.png
│   ├── 02_categorical_price.png
│   ├── 03_correlation.png
│   ├── 04_model_comparison.png
│   ├── 05_diagnostics.png
│   ├── 06_feature_importance.png
│   └── 07_pdp.png
└── README.md
```

---

## How to Run

```bash
.\venv\Scripts\Activate.ps1
jupyter notebook 3_Car_Data_Analysis/Car_Analysis.ipynb
```

**Quick inference:**
```python
import joblib, pandas as pd

pipeline = joblib.load("models/car_price_predictor.pkl")
meta     = joblib.load("models/car_price_metadata.pkl")

sample = pd.DataFrame({
    "Car_Age": [5], "Present_Price": [8.0], "Driven_kms": [45000],
    "Fuel_enc": [1], "Sell_enc": [0], "Trans_enc": [1], "Owner": [0]
})
price = pipeline.predict(sample)[0]
print(f"Estimated price: ₹{price:.2f} Lakhs")
```

---

## Technologies

| Library | Purpose |
|---------|---------|
| `pandas` / `numpy` | Data manipulation & feature engineering |
| `matplotlib` / `seaborn` | Visualisation |
| `scipy.stats` | Shapiro-Wilk, Q-Q plots |
| `statsmodels` | VIF multicollinearity check |
| `scikit-learn` | Preprocessing, CV, diagnostics |
| `xgboost` | Best-performing regression model |
| `joblib` | Model serialisation |

---

## Author

**Muhammad Asif Khan** — CodeAlpha Data Science Intern  
[GitHub](https://github.com) · [LinkedIn](https://linkedin.com)

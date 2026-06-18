# --- Imports ---
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor

# --- Load Data ---
df = pd.read_csv("prowess data.csv")

# --- Sort for lag creation ---
df = df.sort_values(by=["PlayerID", "Age"])

# --- Create lag features (previous 2 seasons) ---
lag_features = [
    "PA","BB%","K%","BB/K","AVG","OBP","SLG","OPS","ISO",
    "Spd","BABIP","wSB","wRC","wRAA","wOBA","wRC+"
]

# --- Clean invalid values ---
df = df.replace(["#DIV/0!", "#VALUE!", "#N/A"], np.nan)

# Convert everything numeric except identifiers
for col in df.columns:
    if col not in ["PlayerID", "Position", "Name"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill missing values (important for lag data)
df = df.fillna(0)

for lag in [1, 2]:
    for col in lag_features:
        df[f"{col}_lag{lag}"] = df.groupby("PlayerID")[col].shift(lag)


# --- OPTIONAL: Aggregate lag features (mean of last 2 seasons) ---
for col in lag_features:
    df[f"{col}_lag_avg"] = (
        df[f"{col}_lag1"] + df[f"{col}_lag2"]
    ) / 2

# --- Define Features ---
feature_cols = lag_features + [
    f"{col}_lag1" for col in lag_features
] + [
    f"{col}_lag2" for col in lag_features
]

# --- Create X and y ---
X = df[feature_cols]
y = df["3YR WAR"]

#--- Safety Check ---
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Missing target values:", y.isna().sum())

feature_cols = lag_features + [
    f"{col}_lag1" for col in lag_features
] + [
    f"{col}_lag2" for col in lag_features
]

#--- Train/Test Split Fix ---
print(df.shape)
print(df.head())
print(df.columns)

print(X.shape)
print(y.shape)

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Scale (helps XGBoost slightly, RF doesn't need it but fine) ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Random Forest ---
rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=10,
    random_state=42
)
rf.fit(X_train, y_train)

rf_preds = rf.predict(X_test)

# --- XGBoost ---
xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb.fit(X_train, y_train)

xgb_preds = xgb.predict(X_test)

# --- Evaluation ---
def evaluate(name, y_true, preds):
    print(f"\n{name}")
    print("RMSE:", np.sqrt(mean_squared_error(y_true, preds)))
    print("R2:", r2_score(y_true, preds))

evaluate("Random Forest", y_test, rf_preds)
evaluate("XGBoost", y_test, xgb_preds)

# --- Feature Importance (XGBoost) ---
import matplotlib.pyplot as plt

importance = xgb.feature_importances_
feat_names = feature_cols

imp_df = pd.DataFrame({
    "feature": feat_names,
    "importance": importance
}).sort_values(by="importance", ascending=False).head(15)

plt.figure()
plt.barh(imp_df["feature"], imp_df["importance"])
plt.gca().invert_yaxis()
plt.title("Top 15 Features (XGBoost)")
plt.show()

# =====================================
# CORE SCOUTING METRICS MODEL
# =====================================

print("\n" + "="*50)
print("CORE SCOUTING METRICS MODEL")
print("="*50)

core_features = [
    'Age to Level',
    'wRC+',
    'K%',
    'ISO',
    'Spd',
    'wSB',
    'PA'
]

# Create dataframe
core_df = df[core_features + ['3YR WAR']].copy()
core_df = core_df.dropna()

print(f"Core Model Sample Size: {len(core_df)}")

# Define X and y
X_core = core_df[core_features]
y_core = core_df['3YR WAR']

# Train/Test Split
X_train_core, X_test_core, y_train_core, y_test_core = train_test_split(
    X_core,
    y_core,
    test_size=0.20,
    random_state=42
)

# =====================================
# RANDOM FOREST
# =====================================

rf_core = RandomForestRegressor(
    n_estimators=500,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42
)

rf_core.fit(X_train_core, y_train_core)

rf_preds = rf_core.predict(X_test_core)

rf_rmse = np.sqrt(mean_squared_error(y_test_core, rf_preds))
rf_r2 = r2_score(y_test_core, rf_preds)

print("\nRANDOM FOREST RESULTS")
print(f"RMSE: {rf_rmse:.3f}")
print(f"R²: {rf_r2:.3f}")

# =====================================
# XGBOOST
# =====================================

xgb_core = XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)

xgb_core.fit(X_train_core, y_train_core)

xgb_preds = xgb_core.predict(X_test_core)

xgb_rmse = np.sqrt(mean_squared_error(y_test_core, xgb_preds))
xgb_r2 = r2_score(y_test_core, xgb_preds)

print("\nXGBOOST RESULTS")
print(f"RMSE: {xgb_rmse:.3f}")
print(f"R²: {xgb_r2:.3f}")

# =====================================
# FEATURE IMPORTANCE
# =====================================

importance_df = pd.DataFrame({
    'Feature': core_features,
    'Importance': rf_core.feature_importances_
}).sort_values(
    by='Importance',
    ascending=False
)

print("\nRANDOM FOREST FEATURE IMPORTANCE")
print(importance_df)

# =====================================
# MODEL COMPARISON
# =====================================

print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

comparison = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost'],
    'RMSE': [rf_rmse, xgb_rmse],
    'R2': [rf_r2, xgb_r2]
})

print(comparison)

# =====================================
# FEATURE IMPORTANCE PLOT
# =====================================

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.barh(
    importance_df['Feature'],
    importance_df['Importance']
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Core Scouting Metrics Feature Importance")
plt.tight_layout()
plt.show()

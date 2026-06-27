# Auto-generated from member2_xgboost_pipeline.ipynb
# -*- coding: utf-8 -*-

# === Cell 2 ===
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import os
import gc
from pathlib import Path

warnings.filterwarnings("ignore")
sns.set_theme(style="darkgrid", palette="viridis")
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["font.size"] = 12

# Paths
DATA_DIR = Path(".")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print("[OK] All imports successful!")

# === Cell 4 ===
# ?? 1.1 Load raw data ?????????????????????????????????????????????????
print("Loading sales data...")
sales = pd.read_csv(DATA_DIR / "sales_subset_15000.csv")
print(f"  sales shape: {sales.shape}")

print("Loading calendar data...")
calendar = pd.read_csv(DATA_DIR / "calendar.csv", parse_dates=["date"])
print(f"  calendar shape: {calendar.shape}")

print("Loading sell_prices data...")
prices = pd.read_csv(DATA_DIR / "sell_prices.csv")
print(f"  prices shape: {prices.shape}")

print("\n[OK] Data loaded successfully!")

# === Cell 5 ===
# ?? 1.2 Quick EDA on raw data ?????????????????????????????????????????
print("=" * 60)
print("SALES DATA")
print("=" * 60)
print(sales.head(3))
print(f"\nUnique items : {sales['item_id'].nunique()}")
print(f"Unique stores: {sales['store_id'].nunique()}")
print(f"Unique depts : {sales['dept_id'].nunique()}")
print(f"Unique cats  : {sales['cat_id'].nunique()}")
print(f"States       : {sales['state_id'].unique().tolist()}")

print("\n" + "=" * 60)
print("CALENDAR DATA")
print("=" * 60)
print(calendar.head(3))
print(f"\nDate range: {calendar['date'].min()} -> {calendar['date'].max()}")
print(f"Events: {calendar['event_name_1'].nunique()} unique event names")

print("\n" + "=" * 60)
print("SELL PRICES DATA")
print("=" * 60)
print(prices.head(3))
print(f"\nPrice range: ${prices['sell_price'].min():.2f} - ${prices['sell_price'].max():.2f}")

# === Cell 6 ===
# -- 1.3 Melt sales from wide -> long format -----------------------
# Sample items to fit in memory (15000 items x 1913 days = ~29M rows)
# We sample a representative subset for training
SAMPLE_ITEMS = 1000  # Adjust based on available RAM
unique_ids = sales["id"].unique()
if len(unique_ids) > SAMPLE_ITEMS:
    np.random.seed(42)
    sampled_ids = np.random.choice(unique_ids, size=SAMPLE_ITEMS, replace=False)
    sales = sales[sales["id"].isin(sampled_ids)].copy()
    print(f"  Sampled {SAMPLE_ITEMS} items from {len(unique_ids)} (to fit in RAM)")

# This converts d_1, d_2, ... d_1941 columns into rows
id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
d_cols = [c for c in sales.columns if c.startswith("d_")]

print(f"Melting {len(d_cols)} day columns into long format...")
sales_long = sales.melt(
    id_vars=id_cols,
    value_vars=d_cols,
    var_name="d",
    value_name="sales",
)
print(f"  Long-format shape: {sales_long.shape}")

# Free memory
del sales
gc.collect()

sales_long.head()


# === Cell 7 ===
# ?? 1.4 Merge calendar ????????????????????????????????????????????????
print("Merging calendar info...")
df = sales_long.merge(calendar, on="d", how="left")

del sales_long
gc.collect()

print(f"  Merged shape: {df.shape}")
df.head()

# === Cell 8 ===
# ?? 1.5 Merge sell prices ?????????????????????????????????????????????
print("Merging sell prices...")
df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

del prices
gc.collect()

print(f"  Final merged shape: {df.shape}")
print(f"  Null sell_price rows: {df['sell_price'].isna().sum()} ({df['sell_price'].isna().mean()*100:.1f}%)")

df.head()

# === Cell 9 ===
# -- 1.6b Price Features -------------------------------------------------
print("Creating price-based features...")

# Fill missing prices with forward fill per item
df["sell_price"] = df.groupby("id")["sell_price"].ffill()
df["sell_price"] = df["sell_price"].fillna(0)

# Price momentum: current price vs rolling average price (7-day lag)
df["price_rolling_mean"] = (
    df.groupby("id")["sell_price"]
    .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
)
df["price_change"] = (df["sell_price"] - df["price_rolling_mean"]).fillna(0)
print("  [v] price_change (current - 7d rolling avg)")

# Price relative to category average (is this item cheap or expensive?)
cat_store_price = df.groupby(["cat_id", "store_id", "d"])["sell_price"].transform("mean")
df["price_relative"] = (df["sell_price"] / (cat_store_price + 1e-8)).fillna(1.0).astype(np.float32)
print("  [v] price_relative (item price / category avg)")

# Is the item on discount? (price < 90% of rolling avg)
df["is_discounted"] = (
    (df["sell_price"] < 0.9 * df["price_rolling_mean"]) & (df["sell_price"] > 0)
).astype(np.int8)
print("  [v] is_discounted (price < 90% of rolling avg)")

# Clean up temporary column
df.drop(columns=["price_rolling_mean"], inplace=True)

print(f"\nPrice features created: 3")


# === Cell 10 ===
# ?? 1.6 Data type optimization ????????????????????????????????????????
print("Optimizing data types to save memory...")
mem_before = df.memory_usage(deep=True).sum() / 1e9

# Extract numeric day index for sorting
df["d_num"] = df["d"].str.extract(r"(\d+)").astype(np.int16)

# Downcast numerics
df["sales"] = df["sales"].astype(np.int16)
df["wday"] = df["wday"].astype(np.int8)
df["month"] = df["month"].astype(np.int8)
df["year"] = df["year"].astype(np.int16)
df["snap_CA"] = df["snap_CA"].astype(np.int8)
df["snap_TX"] = df["snap_TX"].astype(np.int8)
df["snap_WI"] = df["snap_WI"].astype(np.int8)

# Categorical encoding for string columns
for col in ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
            "d", "weekday", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
    df[col] = df[col].astype("category")

mem_after = df.memory_usage(deep=True).sum() / 1e9
print(f"  Memory: {mem_before:.2f} GB -> {mem_after:.2f} GB  ({(1-mem_after/mem_before)*100:.0f}% reduction)")

# Sort by item and day for proper time-series feature engineering
df.sort_values(["id", "d_num"], inplace=True)
df.reset_index(drop=True, inplace=True)

print("\n[OK] Preprocessing complete!")
df.info()

# === Cell 12 ===
# ?? 2.1 Lag Features ??????????????????????????????????????????????????
# Historical sales values shifted by N days
LAG_DAYS = [1, 2, 3, 7, 14, 28]

print("Creating lag features...")
for lag in LAG_DAYS:
    col_name = f"lag_{lag}"
    df[col_name] = df.groupby("id")["sales"].shift(lag)
    print(f"  [v] {col_name}")

print(f"\nLag features created: {len(LAG_DAYS)}")

# === Cell 13 ===
# ?? 2.2 Rolling Averages ??????????????????????????????????????????????
# Smoothed demand trends over different time windows
ROLLING_WINDOWS = [7, 14, 28, 60]

print("Creating rolling average features...")
for window in ROLLING_WINDOWS:
    col_name = f"rolling_mean_{window}"
    df[col_name] = (
        df.groupby("id")["sales"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    print(f"  [v] {col_name}")

print(f"\nRolling average features created: {len(ROLLING_WINDOWS)}")

# === Cell 14 ===
# ?? 2.3 Rolling Variance ??????????????????????????????????????????????
# Measures how spread-out sales have been (demand stability)
VARIANCE_WINDOWS = [7, 14, 28]

print("Creating rolling variance features...")
for window in VARIANCE_WINDOWS:
    col_name = f"rolling_var_{window}"
    df[col_name] = (
        df.groupby("id")["sales"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).var())
    )
    print(f"  [v] {col_name}")

print(f"\nRolling variance features created: {len(VARIANCE_WINDOWS)}")

# === Cell 15 ===
# ?? 2.4 Volatility Indicators ????????????????????????????????????????
# Coefficient of variation (CV) = std / mean - normalized volatility
# Higher CV -> more unpredictable demand

print("Creating volatility indicators...")

# Rolling standard deviation (28-day window)
df["rolling_std_28"] = (
    df.groupby("id")["sales"]
    .transform(lambda x: x.shift(1).rolling(28, min_periods=1).std())
)
print("  [v] rolling_std_28")

# Coefficient of Variation (28-day window)
df["cv_28"] = df["rolling_std_28"] / (df["rolling_mean_28"] + 1e-8)
print("  [v] cv_28 (coefficient of variation)")

# Expanding (cumulative) standard deviation - overall item volatility
df["expanding_std"] = (
    df.groupby("id")["sales"]
    .transform(lambda x: x.shift(1).expanding(min_periods=2).std())
)
print("  [v] expanding_std")

# Sales momentum: ratio of short-term avg to long-term avg
df["momentum_7_28"] = df["rolling_mean_7"] / (df["rolling_mean_28"] + 1e-8)
print("  [v] momentum_7_28")

print(f"\nVolatility indicators created: 4")

# === Cell 16 ===
# ?? 2.5 Event Flags ???????????????????????????????????????????????????
# Binary flags for whether an event is happening on that day

print("Creating event flag features...")

# Binary: is there any event today?
df["has_event_1"] = df["event_name_1"].notna().astype(np.int8)
df["has_event_2"] = df["event_name_2"].notna().astype(np.int8)
df["has_any_event"] = ((df["has_event_1"] == 1) | (df["has_event_2"] == 1)).astype(np.int8)
print("  [v] has_event_1, has_event_2, has_any_event")

# Event type encoding (label encode)
df["event_type_1_enc"] = df["event_type_1"].cat.codes.astype(np.int8)
df["event_type_2_enc"] = df["event_type_2"].cat.codes.astype(np.int8)
print("  [v] event_type_1_enc, event_type_2_enc")

print(f"\nEvent flag features created: 5")
print(f"  Event type 1 categories: {df['event_type_1'].cat.categories.tolist()}")
print(f"  Event type 2 categories: {df['event_type_2'].cat.categories.tolist()}")

# === Cell 17 ===
# ?? 2.6 SNAP Features ????????????????????????????????????????????????
# SNAP (Supplemental Nutrition Assistance Program) - state-specific benefit days
# When SNAP benefits are distributed, food sales tend to spike

print("Creating SNAP features...")

# Per-row SNAP flag based on the item's state
def get_snap_flag(row):
    state = row["state_id"]
    if state == "CA":
        return row["snap_CA"]
    elif state == "TX":
        return row["snap_TX"]
    elif state == "WI":
        return row["snap_WI"]
    return 0

# Vectorized version for performance
df["snap_active"] = np.where(
    df["state_id"] == "CA", df["snap_CA"],
    np.where(
        df["state_id"] == "TX", df["snap_TX"],
        df["snap_WI"]
    )
).astype(np.int8)
print("  [v] snap_active (state-specific SNAP flag)")

# SNAP + Food category interaction (SNAP mainly affects FOODS)
df["snap_x_food"] = ((df["snap_active"] == 1) & (df["cat_id"] == "FOODS")).astype(np.int8)
print("  [v] snap_x_food (SNAP x FOODS interaction)")

# Rolling SNAP days in last 7 days (SNAP benefit density)
df["snap_rolling_7"] = (
    df.groupby("id")["snap_active"]
    .transform(lambda x: x.rolling(7, min_periods=1).sum())
).astype(np.int8)
print("  [v] snap_rolling_7 (SNAP days in last week)")

print(f"\nSNAP features created: 3")

# === Cell 18 ===
# -- 2.6b Interaction Features -------------------------------------------
print("Creating interaction features...")

# SNAP x Weekend: SNAP benefits on weekends may have different effect
df["snap_x_weekend"] = (df["snap_active"] * df["is_weekend"]).astype(np.int8)
print("  [v] snap_x_weekend (SNAP active on weekend)")

# Event x Food category: events may spike food sales differently
df["event_x_food"] = (
    (df["has_any_event"] == 1) & (df["cat_id"] == "FOODS")
).astype(np.int8)
print("  [v] event_x_food (event day + FOODS category)")

# Weekend x Food: weekend food shopping patterns
df["weekend_x_food"] = (
    (df["is_weekend"] == 1) & (df["cat_id"] == "FOODS")
).astype(np.int8)
print("  [v] weekend_x_food (weekend + FOODS)")

# Month-end x SNAP: paycheck + benefits timing
df["monthend_x_snap"] = (
    df["is_month_end"] * df["snap_active"]
).astype(np.int8)
print("  [v] monthend_x_snap (month end + SNAP active)")

print(f"\nInteraction features created: 4")


# === Cell 19 ===
# -- 2.7 Additional calendar features ----------------------------------
print("Creating additional calendar features...")

# -- Basic calendar ----------------------------------------------------
df["day_of_month"] = df["date"].dt.day.astype(np.int8)
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(np.int8)
df["is_weekend"] = ((df["wday"] == 1) | (df["wday"] == 2)).astype(np.int8)
df["is_month_start"] = df["date"].dt.is_month_start.astype(np.int8)
df["is_month_end"] = df["date"].dt.is_month_end.astype(np.int8)
df["quarter"] = df["date"].dt.quarter.astype(np.int8)

# -- Fortnight features (from calendar.csv date info) ------------------
df["is_fortnight_2"] = (df["day_of_month"] > 15).astype(np.int8)
print("  [v] is_fortnight_2 (1 if day > 15, 0 otherwise)")

df["fortnight_of_year"] = ((df["month"] - 1) * 2 + df["is_fortnight_2"]).astype(np.int8)
print("  [v] fortnight_of_year (0-23, biweekly period index)")

# -- Weekend density features ------------------------------------------
df["weekend_rolling_7"] = (
    df.groupby("id")["is_weekend"]
    .transform(lambda x: x.rolling(7, min_periods=1).sum())
).astype(np.int8)
print("  [v] weekend_rolling_7 (weekend days in last 7 days)")

df["weekend_rolling_14"] = (
    df.groupby("id")["is_weekend"]
    .transform(lambda x: x.rolling(14, min_periods=1).sum())
).astype(np.int8)
print("  [v] weekend_rolling_14 (weekend days in last 14 days)")

# -- Days since last weekend (vectorized) ------------------------------
# wday cycles Sat=1, Sun=2, Mon=3, ..., Fri=7
# Days since last weekend = wday - 2 for Mon-Fri (wday 3-7), 0 for Sat/Sun
df["days_since_weekend"] = np.where(
    df["is_weekend"] == 1, 0,
    np.clip(df["wday"].astype(int) - 2, 0, 5)
).astype(np.int8)
print("  [v] days_since_weekend (vectorized, 0 on weekends, 1-5 on weekdays)")

print("  [v] day_of_month, week_of_year, is_weekend")
print("  [v] is_month_start, is_month_end, quarter")
print(f"\nAdditional calendar features created: 11")


# === Cell 20 ===
# -- 2.8 Feature summary ------------------------------------------------
print("=" * 60)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 60)

feature_groups = {
    "Lag features": [c for c in df.columns if c.startswith("lag_")],
    "Rolling averages": [c for c in df.columns if c.startswith("rolling_mean_")],
    "Rolling variance": [c for c in df.columns if c.startswith("rolling_var_")],
    "Volatility indicators": ["rolling_std_28", "cv_28", "expanding_std", "momentum_7_28"],
    "Event flags": ["has_event_1", "has_event_2", "has_any_event", "event_type_1_enc", "event_type_2_enc"],
    "SNAP features": ["snap_active", "snap_x_food", "snap_rolling_7"],
    "Interaction features": ["snap_x_weekend", "event_x_food", "weekend_x_food", "monthend_x_snap"],
    "Calendar features": ["wday", "month", "year", "day_of_month", "week_of_year", "is_weekend",
                          "is_month_start", "is_month_end", "quarter",
                          "is_fortnight_2", "fortnight_of_year",
                          "weekend_rolling_7", "weekend_rolling_14", "days_since_weekend"],
    "Price features": ["sell_price", "price_change", "price_relative", "is_discounted"],
}

total = 0
for group, cols in feature_groups.items():
    present = [c for c in cols if c in df.columns]
    total += len(present)
    print(f"  {group:25s} : {len(present):3d} features -> {present}")

print(f"\n  {'TOTAL':25s} : {total:3d} features")
print(f"  Dataset shape: {df.shape}")


# === Cell 22 ===
# -- 3.1 Reduce dataset for training -------------------------------------
# Use more history for better seasonality learning

# Filter to recent data (expanded from d_1800 to d_1500 for more history)
TRAIN_START_DAY = 1500  # ~413 days of training data (captures yearly patterns)
df_train = df[df["d_num"] >= TRAIN_START_DAY].copy()

print(f"Training subset: d_{TRAIN_START_DAY} onwards")
print(f"  Shape before NaN drop: {df_train.shape}")

# Drop rows where lag/rolling features are NaN (early time-series rows)
lag_cols = [c for c in df_train.columns if c.startswith("lag_")]
df_train.dropna(subset=lag_cols, inplace=True)
print(f"  Shape after NaN drop:  {df_train.shape}")


# === Cell 23 ===
# ?? 3.2 Encode categorical features ??????????????????????????????????
cat_encode_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]

print("Label-encoding categorical columns...")
for col in cat_encode_cols:
    df_train[col + "_enc"] = df_train[col].cat.codes.astype(np.int16)
    print(f"  [v] {col} -> {col}_enc  ({df_train[col].nunique()} categories)")

# === Cell 24 ===
# -- 3.3 Define feature columns -------------------------------------------
FEATURE_COLS = (
    # Encoded categorical
    [c + "_enc" for c in cat_encode_cols]
    # Lag features
    + [c for c in df_train.columns if c.startswith("lag_")]
    # Rolling features
    + [c for c in df_train.columns if c.startswith("rolling_mean_")]
    + [c for c in df_train.columns if c.startswith("rolling_var_")]
    # Volatility
    + ["rolling_std_28", "cv_28", "expanding_std", "momentum_7_28"]
    # Event flags
    + ["has_event_1", "has_event_2", "has_any_event", "event_type_1_enc", "event_type_2_enc"]
    # SNAP
    + ["snap_active", "snap_x_food", "snap_rolling_7"]
    # Interaction features
    + ["snap_x_weekend", "event_x_food", "weekend_x_food", "monthend_x_snap"]
    # Calendar (basic)
    + ["wday", "month", "year", "day_of_month", "week_of_year",
       "is_weekend", "is_month_start", "is_month_end", "quarter"]
    # Calendar (fortnight & weekend density)
    + ["is_fortnight_2", "fortnight_of_year",
       "weekend_rolling_7", "weekend_rolling_14", "days_since_weekend"]
    # Price features
    + ["sell_price", "price_change", "price_relative", "is_discounted"]
)

TARGET = "sales"

# Verify all feature columns exist
missing = [c for c in FEATURE_COLS if c not in df_train.columns]
if missing:
    print(f"[!]  Missing columns: {missing}")
else:
    print(f"[OK] All {len(FEATURE_COLS)} feature columns present!")

print(f"\nFeatures ({len(FEATURE_COLS)}):")
for i, col in enumerate(FEATURE_COLS, 1):
    print(f"  {i:2d}. {col}")


# === Cell 25 ===
# ?? 3.4 Fill remaining NaN values ????????????????????????????????????
print("Filling remaining NaN values...")
nan_before = df_train[FEATURE_COLS].isna().sum()
nan_cols = nan_before[nan_before > 0]
if len(nan_cols) > 0:
    print(f"  Columns with NaN: {dict(nan_cols)}")
    df_train[FEATURE_COLS] = df_train[FEATURE_COLS].fillna(0)
    print("  [v] Filled with 0")
else:
    print("  No NaN values found - clean dataset!")

# === Cell 26 ===
# ?? 3.5 Train / Validation split ?????????????????????????????????????
# Time-based split: last 28 days as validation (simulates forecast horizon)
VAL_DAYS = 28
split_day = df_train["d_num"].max() - VAL_DAYS

train_mask = df_train["d_num"] <= split_day
val_mask = df_train["d_num"] > split_day

X_train = df_train.loc[train_mask, FEATURE_COLS].astype(np.float32)
y_train = df_train.loc[train_mask, TARGET].astype(np.float32)

X_val = df_train.loc[val_mask, FEATURE_COLS].astype(np.float32)
y_val = df_train.loc[val_mask, TARGET].astype(np.float32)

print(f"Split at day d_{split_day}")
print(f"  Train : {X_train.shape[0]:>10,} rows  (d_{TRAIN_START_DAY} -> d_{split_day})")
print(f"  Val   : {X_val.shape[0]:>10,} rows  (d_{split_day+1} -> d_{df_train['d_num'].max()})")
print(f"  Features: {X_train.shape[1]}")

# === Cell 27 ===
# -- 3.6 Train XGBoost (improved) ----------------------------------------
print("Training XGBoost model (with early stopping)...")
print("=" * 60)

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 8,
    "learning_rate": 0.03,        # Lower LR + early stopping = better
    "n_estimators": 2000,          # More trees (early stopping will cut)
    "min_child_weight": 50,        # Reduced from 300 for finer splits
    "subsample": 0.8,
    "colsample_bytree": 0.7,      # Slightly more regularization
    "colsample_bynode": 0.8,      # Per-node column sampling
    "reg_alpha": 0.05,
    "reg_lambda": 1.0,
    "gamma": 0.1,                 # Min loss reduction for split
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 1,
}

model = xgb.XGBRegressor(**xgb_params)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=100,
)

best_iteration = model.best_iteration if hasattr(model, "best_iteration") and model.best_iteration else model.n_estimators
print(f"\n[OK] XGBoost training complete!")
print(f"  Best iteration: {best_iteration}")
print(f"  Trees used: {best_iteration} / {xgb_params['n_estimators']}")


# === Cell 29 ===
# ?? 4.1 Generate Predictions ??????????????????????????????????????????
print("Generating predictions...")

y_pred_val = model.predict(X_val)
y_pred_val = np.clip(y_pred_val, 0, None)  # Sales can't be negative

y_pred_train = model.predict(X_train)
y_pred_train = np.clip(y_pred_train, 0, None)

print(f"  Train predictions: {len(y_pred_train):,}")
print(f"  Val predictions  : {len(y_pred_val):,}")

# === Cell 30 ===
# ?? 4.2 Multi-Horizon Evaluation Metrics ?????????????????????????????
def compute_metrics(y_true, y_pred, label=""):
    """Compute regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE - only on non-zero actuals to avoid division by zero
    mask = y_true > 0
    if mask.sum() > 0:
        mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100
    else:
        mape = float("nan")
    
    # WRMSSE proxy: weighted RMSE
    wrmsse_proxy = rmse / (y_true.mean() + 1e-8)
    
    return {
        "Horizon": label,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "MAPE (%)": round(mape, 2),
        "WRMSSE Proxy": round(wrmsse_proxy, 4),
    }

# ?? Train metrics (overall) ????????????????????????????????????????
train_metrics = compute_metrics(y_train.values, y_pred_train, "Train (all)")

# ?? Validation metrics - broken down by 3 forecast horizons ????????
# The validation set covers the last 28 days of training data.
# We slice it into 7-day, 14-day, and full 28-day windows.
val_d_nums = df_train.loc[val_mask, "d_num"].values
val_d_max = val_d_nums.max()

HORIZONS = [7, 14, 28]
horizon_metrics = []

for h in HORIZONS:
    h_mask = val_d_nums > (val_d_max - h)
    y_true_h = y_val.values[h_mask]
    y_pred_h = y_pred_val[h_mask]
    m = compute_metrics(y_true_h, y_pred_h, f"{h}-day")
    m["N_samples"] = int(h_mask.sum())
    horizon_metrics.append(m)
    print(f"  [v] {h}-day horizon: {h_mask.sum():,} samples")

# Combine all metrics into a single DataFrame
all_metrics = [train_metrics] + horizon_metrics
metrics_df = pd.DataFrame(all_metrics)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE METRICS - PER FORECAST HORIZON")
print("=" * 70)
print(metrics_df)

# Also compute overall validation metrics for backward compatibility
val_metrics = compute_metrics(y_val.values, y_pred_val, "Val (28-day)")

# Save metrics
metrics_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
print(f"\n[OK] Per-horizon metrics saved to {OUTPUT_DIR / 'metrics.csv'}")


# === Cell 31 ===
# ?? 4.3 Feature Importance ????????????????????????????????????????????
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Gain-based importance
importance_gain = model.get_booster().get_score(importance_type="gain")
imp_gain_df = (
    pd.DataFrame.from_dict(importance_gain, orient="index", columns=["gain"])
    .sort_values("gain", ascending=True)
    .tail(20)
)
imp_gain_df.plot.barh(ax=axes[0], color="#2ecc71", legend=False)
axes[0].set_title("Top 20 Features by Gain", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Gain")

# Weight-based importance
importance_weight = model.get_booster().get_score(importance_type="weight")
imp_weight_df = (
    pd.DataFrame.from_dict(importance_weight, orient="index", columns=["weight"])
    .sort_values("weight", ascending=True)
    .tail(20)
)
imp_weight_df.plot.barh(ax=axes[1], color="#3498db", legend=False)
axes[1].set_title("Top 20 Features by Frequency", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Number of Splits")

plt.suptitle("XGBoost Feature Importance", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"[OK] Feature importance plot saved!")

# === Cell 32 ===
# ?? 4.4 Actual vs Predicted Plots ?????????????????????????????????????
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Scatter plot
sample_idx = np.random.choice(len(y_val), size=min(10000, len(y_val)), replace=False)
axes[0].scatter(
    y_val.values[sample_idx], y_pred_val[sample_idx],
    alpha=0.15, s=5, color="#e74c3c"
)
max_val = max(y_val.values[sample_idx].max(), y_pred_val[sample_idx].max())
axes[0].plot([0, max_val], [0, max_val], "k--", linewidth=1, label="Perfect prediction")
axes[0].set_xlabel("Actual Sales")
axes[0].set_ylabel("Predicted Sales")
axes[0].set_title("Actual vs Predicted (Validation Set)", fontsize=14, fontweight="bold")
axes[0].legend()

# Residual distribution
residuals = y_val.values - y_pred_val
axes[1].hist(residuals, bins=100, color="#9b59b6", alpha=0.7, edgecolor="black", linewidth=0.5)
axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
axes[1].set_xlabel("Residual (Actual ? Predicted)")
axes[1].set_ylabel("Count")
axes[1].set_title("Residual Distribution", fontsize=14, fontweight="bold")

plt.suptitle("Model Diagnostics", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_diagnostics.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"[OK] Diagnostic plots saved!")

# === Cell 33 ===
# -- 4.5 Save Predictions (ensemble-ready format) -------------------------
print("Saving predictions...")

val_results = df_train.loc[val_mask, ["id", "d", "d_num", "date", "item_id",
                                       "store_id", "cat_id", "dept_id",
                                       "state_id", "sales"]].copy()
val_results["xgb_pred"] = y_pred_val.round(4)
val_results["residual"] = (val_results["sales"] - val_results["xgb_pred"]).round(4)

# Tag each row with its forecast horizon bucket
val_d_max_save = val_results["d_num"].max()
val_results["horizon"] = "28-day"
val_results.loc[val_results["d_num"] > (val_d_max_save - 14), "horizon"] = "14-day"
val_results.loc[val_results["d_num"] > (val_d_max_save - 7), "horizon"] = "7-day"

# Save for ensemble (LSTM will add its predictions to this file)
val_results.to_csv(OUTPUT_DIR / "xgb_predictions.csv", index=False)
print(f"  [v] XGB predictions saved: {OUTPUT_DIR / 'xgb_predictions.csv'}")
print(f"    Shape: {val_results.shape}")
print(f"    Columns: {val_results.columns.tolist()}")
print(f"    Horizon breakdown:")
print(val_results["horizon"].value_counts().to_string())

print(val_results.head(10).to_string())


# === Cell 34 ===
# ?? 4.6 Save Model ???????????????????????????????????????????????????
print("Saving model artifacts...")

# Save as XGBoost native format (best for deployment)
model_path_xgb = OUTPUT_DIR / "xgboost_m5_model.json"
model.save_model(str(model_path_xgb))
print(f"  [v] XGBoost native: {model_path_xgb}")

# Save as joblib (includes sklearn wrapper metadata)
model_path_joblib = OUTPUT_DIR / "xgboost_m5_model.joblib"
joblib.dump(model, str(model_path_joblib))
print(f"  [v] Joblib backup : {model_path_joblib}")

# Save feature list for reproducibility
feature_list_path = OUTPUT_DIR / "feature_columns.txt"
with open(feature_list_path, "w") as f:
    f.write("\n".join(FEATURE_COLS))
print(f"  [v] Feature list  : {feature_list_path}")

# Save model parameters
params_df = pd.DataFrame.from_dict(xgb_params, orient="index", columns=["value"])
params_df.to_csv(OUTPUT_DIR / "model_params.csv")
print(f"  [v] Parameters    : {OUTPUT_DIR / 'model_params.csv'}")

print(f"\n[OK] All model artifacts saved to '{OUTPUT_DIR}/'")

# === Cell 35 ===
# ?? 4.7 Training Summary ?????????????????????????????????????????????
print("\n" + "=" * 70)
print("[CHART]  PIPELINE SUMMARY")
print("=" * 70)
print(f"")
print(f"  Dataset        : M5 Forecasting (Walmart Sales)")
print(f"  Items x Stores : {df_train['item_id'].nunique()} x {df_train['store_id'].nunique()}")
print(f"  Training days  : d_{TRAIN_START_DAY} -> d_{split_day}")
print(f"  Validation days: d_{split_day+1} -> d_{df_train['d_num'].max()}")
print(f"  Features       : {len(FEATURE_COLS)}")
print(f"  Model          : XGBoost ({model.n_estimators} trees, max_depth={model.max_depth})")
print(f"")
print("  ?? Per-Horizon Accuracy ??")
for _, row in metrics_df.iterrows():
    if row["Horizon"] == "Train (all)":
        continue
    print(f"    {row['Horizon']:>8s}  |  MAE={row['MAE']:.4f}  RMSE={row['RMSE']:.4f}  R2={row['R2']:.4f}  MAPE={row['MAPE (%)']:.1f}%")
print(f"")
print(f"  Outputs saved in '{OUTPUT_DIR}/' :")
for f in sorted(OUTPUT_DIR.iterdir()):
    size_kb = f.stat().st_size / 1024
    print(f"    [FILE] {f.name:35s} ({size_kb:>8.1f} KB)")
print(f"")
print("=" * 70)
print("[DONE] Member 2 pipeline complete!")
print("=" * 70)


# === Cell 37 ===
# -- 5.1 Export Data for LSTM Model ----------------------------------------
print("Exporting data for LSTM model...")
print("=" * 60)

# 5.1a: Save split info (both models MUST use same splits)
split_info = {
    "train_start_day": int(TRAIN_START_DAY),
    "split_day": int(split_day),
    "val_start_day": int(split_day + 1),
    "val_end_day": int(df_train["d_num"].max()),
    "val_days": VAL_DAYS,
    "horizons": [7, 14, 28],
    "sampled_ids": df_train["id"].unique().tolist(),
}
import json as json_lib
with open(OUTPUT_DIR / "split_info.json", "w") as f:
    json_lib.dump(split_info, f, indent=2)
print(f"  [v] Split info saved: {OUTPUT_DIR / 'split_info.json'}")

# 5.1b: Export sales sequences (items x days matrix) for LSTM
# LSTM needs shape: (n_items, sequence_length)
print("\n  Exporting sales sequences...")
sales_pivot = df_train.pivot_table(
    index="id", columns="d_num", values="sales", aggfunc="first"
).fillna(0)
sales_pivot.to_csv(OUTPUT_DIR / "lstm_sales_sequences.csv")
print(f"  [v] Sales sequences: {sales_pivot.shape} (items x days)")

# 5.1c: Export calendar features per day (for LSTM temporal context)
print("\n  Exporting calendar features per day...")
calendar_feats = ["d_num", "wday", "month", "year", "day_of_month",
                  "week_of_year", "is_weekend", "is_month_start",
                  "is_month_end", "quarter", "is_fortnight_2",
                  "fortnight_of_year", "has_any_event", "snap_CA",
                  "snap_TX", "snap_WI"]
cal_existing = [c for c in calendar_feats if c in df_train.columns]
cal_per_day = df_train[cal_existing].drop_duplicates(subset=["d_num"]).sort_values("d_num")
cal_per_day.to_csv(OUTPUT_DIR / "lstm_calendar_features.csv", index=False)
print(f"  [v] Calendar features: {cal_per_day.shape} (days x features)")

# 5.1d: Export item metadata (for LSTM embedding layers)
print("\n  Exporting item metadata...")
item_meta = df_train[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]].drop_duplicates()
item_meta.to_csv(OUTPUT_DIR / "lstm_item_metadata.csv", index=False)
print(f"  [v] Item metadata: {item_meta.shape}")

print(f"\n[OK] All LSTM data exports saved to {OUTPUT_DIR}/")


# === Cell 38 ===
# -- 5.2 Ensemble Combination Logic ----------------------------------------
# This cell defines how XGBoost + LSTM predictions will be combined.
# Run this AFTER the LSTM notebook has generated its predictions.

def ensemble_predictions(xgb_path, lstm_path, output_path, alpha=0.5):
    """
    Combine XGBoost and LSTM predictions using weighted average.
    
    Args:
        xgb_path: Path to XGBoost predictions CSV
        lstm_path: Path to LSTM predictions CSV
        output_path: Path to save ensemble predictions
        alpha: Weight for XGBoost (1-alpha for LSTM)
               0.5 = equal weight, >0.5 = more XGBoost
    """
    xgb_df = pd.read_csv(xgb_path)
    lstm_df = pd.read_csv(lstm_path)
    
    # Merge on id + day
    merged = xgb_df.merge(
        lstm_df[["id", "d_num", "lstm_pred"]],
        on=["id", "d_num"],
        how="left"
    )
    
    # Weighted ensemble
    merged["ensemble_pred"] = (
        alpha * merged["xgb_pred"] + (1 - alpha) * merged["lstm_pred"]
    ).clip(lower=0).round(4)
    
    # Compute ensemble metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    print("=" * 70)
    print("ENSEMBLE RESULTS (XGBoost + LSTM)")
    print(f"  Alpha = {alpha:.2f} (XGBoost weight)")
    print("=" * 70)
    
    for label, pred_col in [("XGBoost", "xgb_pred"), ("LSTM", "lstm_pred"), ("Ensemble", "ensemble_pred")]:
        if pred_col not in merged.columns or merged[pred_col].isna().all():
            continue
        valid = merged.dropna(subset=[pred_col])
        mae = mean_absolute_error(valid["sales"], valid[pred_col])
        rmse = np.sqrt(mean_squared_error(valid["sales"], valid[pred_col]))
        r2 = r2_score(valid["sales"], valid[pred_col])
        print(f"  {label:10s} | MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")
    
    # Find optimal alpha via grid search
    print("\n  Searching for optimal alpha...")
    valid = merged.dropna(subset=["xgb_pred", "lstm_pred"])
    best_alpha, best_rmse = 0.5, float("inf")
    for a in np.arange(0.0, 1.01, 0.05):
        combo = (a * valid["xgb_pred"] + (1 - a) * valid["lstm_pred"]).clip(lower=0)
        rmse_a = np.sqrt(mean_squared_error(valid["sales"], combo))
        if rmse_a < best_rmse:
            best_alpha, best_rmse = a, rmse_a
    
    print(f"  Optimal alpha = {best_alpha:.2f} (RMSE = {best_rmse:.4f})")
    
    # Recompute with optimal alpha
    merged["ensemble_pred_optimal"] = (
        best_alpha * merged["xgb_pred"] + (1 - best_alpha) * merged["lstm_pred"]
    ).clip(lower=0).round(4)
    
    merged.to_csv(output_path, index=False)
    print(f"\n  [v] Ensemble predictions saved: {output_path}")
    return merged, best_alpha

# Save the ensemble function for reuse
print("Ensemble function defined: ensemble_predictions()")
print("\nUsage after LSTM predictions are ready:")
print('  merged, best_alpha = ensemble_predictions(')
print('      "outputs/xgb_predictions.csv",')
print('      "outputs/lstm_predictions.csv",')
print('      "outputs/ensemble_predictions.csv"')
print('  )')

# Check if LSTM predictions already exist
lstm_pred_path = OUTPUT_DIR / "lstm_predictions.csv"
if lstm_pred_path.exists():
    print("\n[OK] LSTM predictions found! Running ensemble...")
    merged, best_alpha = ensemble_predictions(
        OUTPUT_DIR / "xgb_predictions.csv",
        lstm_pred_path,
        OUTPUT_DIR / "ensemble_predictions.csv"
    )
else:
    print(f"\n[INFO] LSTM predictions not found at {lstm_pred_path}")
    print("  Run the LSTM notebook first, then re-run this cell.")


# === Cell 39 ===
# -- 5.3 Ensemble Files Summary -------------------------------------------
print("\n" + "=" * 70)
print("ENSEMBLE & LSTM DATA FILES")
print("=" * 70)

ensemble_files = {
    "For LSTM Training": [
        ("lstm_sales_sequences.csv", "Sales matrix (items x days) for LSTM input"),
        ("lstm_calendar_features.csv", "Calendar features per day for temporal context"),
        ("lstm_item_metadata.csv", "Item/store/category metadata for embeddings"),
        ("split_info.json", "Train/val split boundaries (MUST match XGBoost)"),
    ],
    "For Ensemble": [
        ("xgb_predictions.csv", "XGBoost validation predictions (id, d_num, xgb_pred)"),
    ],
}

for section, files in ensemble_files.items():
    print(f"\n  {section}:")
    for fname, desc in files:
        fpath = OUTPUT_DIR / fname
        if fpath.exists():
            size = fpath.stat().st_size / 1024
            print(f"    [v] {fname:35s} ({size:>8.1f} KB) - {desc}")
        else:
            print(f"    [x] {fname:35s} (missing) - {desc}")

print("\n" + "=" * 70)
print("[OK] XGBoost pipeline + ensemble setup complete!")
print("=" * 70)




from pathlib import Path
import warnings
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 1. USER SETTINGS
# ============================================================

# Option A: Enter your full CSV path here, e.g.
# CSV_PATH = r"C:\Users\YourName\Downloads\FPCPITOTL (1).csv"
# Option B: Keep as None to choose the CSV with a file dialog.
CSV_PATH = ("C:/Users/Asus/Downloads/FPCPITOTL.csv")

SELECTED_VINTAGE = "V_SM2026"
HISTORICAL_END_YEAR = 2024

TRAIN_END = 2018
VALIDATION_START = 2019
VALIDATION_END = 2021
TEST_START = 2022
TEST_END = 2024

RANDOM_SEED = 42
OUTPUT_DIR_NAME = "project_outputs"

np.random.seed(RANDOM_SEED)
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (10, 5)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def choose_csv_file():
    """Open a file dialog to select the CSV."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        file_path = filedialog.askopenfilename(
            title="Select FPCPITOTL CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        root.destroy()

        if file_path:
            return Path(file_path)
    except Exception as exc:
        print("File dialog could not be opened:", exc)

    entered = input(
        "Enter the full path to your CSV file (example: C:/data/FPCPITOTL.csv): "
    ).strip().strip('"').strip("'")

    if not entered:
        raise FileNotFoundError(
            "No CSV file selected. Set CSV_PATH near the top of this script."
        )
    return Path(entered)


def save_and_show_plot(filename):
    """Save the current plot and display it."""
    plt.tight_layout()
    output_path = OUTPUT_DIR / "figures" / filename
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close()


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred)
    }


def make_preprocessor(polynomial=False):
    """
    Preprocess country and numeric features.
    If polynomial=True, create degree-2 polynomial terms only from
    numeric features, avoiding huge country one-hot interactions.
    """
    try:
        one_hot = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        one_hot = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )

    categorical_features = ["REF_AREA"]
    numeric_features = [
        "TIME_PERIOD", "lag_1", "lag_2", "rolling_mean_3"
    ]

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if polynomial:
        numeric_steps.append(
            ("polynomial", PolynomialFeatures(
                degree=2, include_bias=False
            ))
        )
    numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("country", one_hot, categorical_features),
            ("numeric", Pipeline(steps=numeric_steps), numeric_features)
        ],
        remainder="drop"
    )


# ============================================================
# 3. LOAD CSV AND AUDIT DATASET
# ============================================================

if CSV_PATH is None:
    csv_path = choose_csv_file()
else:
    csv_path = Path(CSV_PATH)

if not csv_path.exists():
    raise FileNotFoundError(
        f"CSV file not found: {csv_path}\n"
        "Update CSV_PATH or choose the file when prompted."
    )

OUTPUT_DIR = csv_path.parent / OUTPUT_DIR_NAME
(OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "tables").mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 70)
print("INFLATION RATE PREDICTION USING MACHINE LEARNING")
print("=" * 70)
print("CSV file:", csv_path)

df_raw = pd.read_csv(csv_path, low_memory=False)

print("\n--- DATASET SHAPE ---")
print("Rows:", df_raw.shape[0])
print("Columns:", df_raw.shape[1])

print("\n--- COLUMN NAMES ---")
print(df_raw.columns.tolist())

print("\n--- DATA TYPES ---")
print(df_raw.dtypes.to_string())

print("\n--- FIRST FIVE ROWS ---")
print(df_raw.head().to_string())

print("\n--- MISSING VALUES BY COLUMN ---")
missing_audit = df_raw.isna().sum().sort_values(ascending=False)
print(missing_audit.to_string())
missing_audit.to_csv(OUTPUT_DIR / "tables" / "missing_values_audit.csv")

print("\nExact duplicate rows:", int(df_raw.duplicated().sum()))

required_columns = [
    "REF_AREA", "REF_AREA_LABEL", "TIME_PERIOD",
    "OBS_VALUE", "VINTAGE", "INDICATOR"
]
missing_columns = [c for c in required_columns if c not in df_raw.columns]

if missing_columns:
    raise ValueError(
        "This CSV does not have the expected schema. Missing columns: "
        + ", ".join(missing_columns)
    )

print("\nUnique country codes:", df_raw["REF_AREA"].nunique(dropna=True))
print("Unique country/area labels:", df_raw["REF_AREA_LABEL"].nunique(dropna=True))
print("Raw year range:", df_raw["TIME_PERIOD"].min(), "to", df_raw["TIME_PERIOD"].max())

print("\n--- VINTAGE COUNTS ---")
print(df_raw["VINTAGE"].value_counts(dropna=False).to_string())

print("\n--- VINTAGE LABELS ---")
vintage_labels = (
    df_raw[["VINTAGE", "VINTAGE_LABEL"]]
    .drop_duplicates()
    if "VINTAGE_LABEL" in df_raw.columns
    else df_raw[["VINTAGE"]].drop_duplicates()
)
print(vintage_labels.to_string(index=False))

raw_target = pd.to_numeric(df_raw["OBS_VALUE"], errors="coerce")
print("\nMissing/invalid target values after numeric conversion:",
      int(raw_target.isna().sum()))

if "OBS_STATUS" in df_raw.columns:
    print("\n--- OBSERVATION STATUS COUNTS ---")
    print(df_raw["OBS_STATUS"].value_counts(dropna=False).head(20).to_string())


# ============================================================
# 4. SELECT ONE VINTAGE AND CLEAN
# ============================================================

if SELECTED_VINTAGE not in df_raw["VINTAGE"].astype(str).unique():
    available_vintages = df_raw["VINTAGE"].dropna().astype(str).unique().tolist()
    raise ValueError(
        f"Vintage '{SELECTED_VINTAGE}' was not found. "
        f"Available vintages: {available_vintages}. "
        "Update SELECTED_VINTAGE near the top of the script."
    )

df = df_raw.loc[
    df_raw["VINTAGE"].astype(str) == SELECTED_VINTAGE
].copy()

df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
df["TIME_PERIOD"] = pd.to_numeric(df["TIME_PERIOD"], errors="coerce")
df["REF_AREA"] = df["REF_AREA"].astype("string").str.strip()
df["REF_AREA_LABEL"] = df["REF_AREA_LABEL"].astype("string").str.strip()

# Remove rows without valid identifiers/year; do not impute targets.
df = df.dropna(subset=["REF_AREA", "REF_AREA_LABEL", "TIME_PERIOD"]).copy()
df = df.loc[
    np.isfinite(df["TIME_PERIOD"])
    & df["REF_AREA"].ne("")
    & df["REF_AREA_LABEL"].ne("")
].copy()
df["TIME_PERIOD"] = df["TIME_PERIOD"].astype(int)

# Keep only finite target values for supervised EDA/modeling.
df = df.loc[np.isfinite(df["OBS_VALUE"])].copy()

# Conflicting country-year values in the selected vintage must be investigated.
key_cols = ["REF_AREA", "TIME_PERIOD"]
conflict_counts = df.groupby(key_cols)["OBS_VALUE"].nunique()
conflicting_keys = conflict_counts[conflict_counts > 1]

if not conflicting_keys.empty:
    print("\nConflicting country-year records detected:")
    print(conflicting_keys.head(20).to_string())
    raise ValueError(
        "Conflicting OBS_VALUE values exist for a country-year in the selected "
        "vintage. Resolve these records before modeling."
    )

before_dedup = len(df)
df = df.drop_duplicates(subset=key_cols, keep="first").copy()
removed_duplicate_keys = before_dedup - len(df)

df = df.sort_values(["REF_AREA", "TIME_PERIOD"]).reset_index(drop=True)

print("\n--- CLEANING SUMMARY ---")
print("Selected vintage:", SELECTED_VINTAGE)
print("Rows with usable target and identifiers:", len(df))
print("Duplicate country-year rows removed:", removed_duplicate_keys)
print("Unique countries:", df["REF_AREA"].nunique())
print("Year coverage:", df["TIME_PERIOD"].min(), "to", df["TIME_PERIOD"].max())
print("Target missing values after modeling-row filtering:", df["OBS_VALUE"].isna().sum())

print("\n--- TARGET SUMMARY ---")
target_summary = df["OBS_VALUE"].describe(
    percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
)
print(target_summary.to_string())
target_summary.to_csv(OUTPUT_DIR / "tables" / "target_summary.csv")

print("\n--- TOP 15 OBSERVATIONS BY INFLATION ---")
top_extremes = df.nlargest(
    15, "OBS_VALUE"
)[["REF_AREA", "REF_AREA_LABEL", "TIME_PERIOD", "OBS_VALUE"]]
print(top_extremes.to_string(index=False))
top_extremes.to_csv(OUTPUT_DIR / "tables" / "top_inflation_observations.csv", index=False)


# ============================================================
# 5. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

print("\nGenerating EDA plots...")

# 5.2 Inflation distribution.
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x="OBS_VALUE", bins=70, kde=True)
plt.title("Distribution of Annual Inflation Rates")
plt.xlabel("Inflation rate (% per annum)")
plt.ylabel("Number of observations")
save_and_show_plot("01_inflation_distribution.png")

# 5.4 Year-wise median and interquartile range.
year_summary = (
    df.groupby("TIME_PERIOD")["OBS_VALUE"]
    .agg(
        median="median",
        q1=lambda s: s.quantile(0.25),
        q3=lambda s: s.quantile(0.75),
        count="count"
    )
    .reset_index()
)
year_summary.to_csv(OUTPUT_DIR / "tables" / "yearly_summary.csv", index=False)

plt.figure(figsize=(12, 5))
plt.plot(
    year_summary["TIME_PERIOD"],
    year_summary["median"],
    label="Median inflation"
)
plt.fill_between(
    year_summary["TIME_PERIOD"],
    year_summary["q1"],
    year_summary["q3"],
    alpha=0.2,
    label="25th–75th percentile"
)
plt.title("Country-Level Inflation Distribution Over Time")
plt.xlabel("Year")
plt.ylabel("Inflation rate (% per annum)")
plt.legend()
save_and_show_plot("02_yearly_inflation_trend.png")

# 5.5 Country comparisons using median, with a minimum observation count.
country_summary = (
    df.groupby(["REF_AREA", "REF_AREA_LABEL"])
    .agg(
        median_inflation=("OBS_VALUE", "median"),
        mean_inflation=("OBS_VALUE", "mean"),
        observations=("OBS_VALUE", "count")
    )
    .reset_index()
)
country_summary.to_csv(
    OUTPUT_DIR / "tables" / "country_summary.csv", index=False
)

eligible_countries = country_summary.loc[
    country_summary["observations"] >= 10
].copy()

if not eligible_countries.empty:
    top_countries = eligible_countries.nlargest(
        10, "median_inflation"
    ).sort_values("median_inflation")
    plt.figure(figsize=(11, 6))
    sns.barplot(
        data=top_countries,
        x="median_inflation",
        y="REF_AREA_LABEL"
    )
    plt.title("Countries with Highest Median Inflation (10+ observations)")
    plt.xlabel("Median inflation rate (%)")
    plt.ylabel("Country/area")
    save_and_show_plot("03_high_median_inflation_countries.png")

    bottom_countries = eligible_countries.nsmallest(
        10, "median_inflation"
    ).sort_values("median_inflation")
    plt.figure(figsize=(11, 6))
    sns.barplot(
        data=bottom_countries,
        x="median_inflation",
        y="REF_AREA_LABEL"
    )
    plt.title("Countries with Lowest Median Inflation (10+ observations)")
    plt.xlabel("Median inflation rate (%)")
    plt.ylabel("Country/area")
    save_and_show_plot("04_low_median_inflation_countries.png")

# 5.6 Heatmap for countries with highest inflation in latest observed year.
latest_year = int(df["TIME_PERIOD"].max())
latest_year_rows = df.loc[df["TIME_PERIOD"] == latest_year]

if not latest_year_rows.empty:
    selected_country_codes = latest_year_rows.nlargest(
        15, "OBS_VALUE"
    )["REF_AREA"].tolist()

    heatmap_df = df.loc[
        df["REF_AREA"].isin(selected_country_codes)
    ].pivot(
        index="REF_AREA_LABEL",
        columns="TIME_PERIOD",
        values="OBS_VALUE"
    )

    if not heatmap_df.empty:
        plt.figure(figsize=(15, 7))
        sns.heatmap(
            heatmap_df,
            cmap="coolwarm",
            center=0,
            mask=heatmap_df.isna()
        )
        plt.title(
            f"Inflation Heatmap for Selected Countries "
            f"(Selection year: {latest_year})"
        )
        plt.xlabel("Year")
        plt.ylabel("Country/area")
        save_and_show_plot("05_country_year_heatmap.png")

# 5.7 Country-wise inflation trend comparison.
# Prefer India, United States, and China when available; otherwise use
# the countries with the most historical observations.
preferred_codes = ["IND", "USA", "CHN"]
available_codes = set(df["REF_AREA"].astype(str).str.upper())
selected_codes = [c for c in preferred_codes if c in available_codes]

if len(selected_codes) < 3:
    coverage = (
        df.groupby("REF_AREA")["TIME_PERIOD"]
        .nunique()
        .sort_values(ascending=False)
    )
    for code in coverage.index.astype(str):
        if code not in selected_codes:
            selected_codes.append(code)
        if len(selected_codes) == 3:
            break

trend_df = df.loc[
    df["REF_AREA"].astype(str).str.upper().isin(selected_codes)
].copy()

if not trend_df.empty:
    trend_df = trend_df.sort_values("TIME_PERIOD")
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=trend_df,
        x="TIME_PERIOD",
        y="OBS_VALUE",
        hue="REF_AREA_LABEL",
        estimator="median",
        errorbar=None
    )
    plt.title("Country-wise Inflation Trend Comparison")
    plt.xlabel("Year")
    plt.ylabel("Inflation rate (% per annum)")
    plt.legend(title="Country/area")
    save_and_show_plot("07_country_wise_inflation_trends.png")
else:
    print("Country-wise trend graph skipped: no eligible country data.")

# 5.8 IQR outlier screening (review only; do not delete automatically).
q1 = df["OBS_VALUE"].quantile(0.25)
q3 = df["OBS_VALUE"].quantile(0.75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

extreme = df.loc[
    (df["OBS_VALUE"] < lower_fence)
    | (df["OBS_VALUE"] > upper_fence),
    ["REF_AREA", "REF_AREA_LABEL", "TIME_PERIOD", "OBS_VALUE"]
].copy()

print("\n--- OUTLIER SCREENING ---")
print("IQR lower fence:", lower_fence)
print("IQR upper fence:", upper_fence)
print("Observations outside fences:", len(extreme))
print("These are flagged for review, not automatically removed.")
extreme.to_csv(OUTPUT_DIR / "tables" / "iqr_flagged_observations.csv", index=False)

# 5.9 Correlation interpretation.
# Correlation describes association; it does not establish causation.
print(
    "\nCorrelation is descriptive only. Year and inflation can be "
    "associated without one causing the other."
)


# ============================================================
# 6. FEATURE ENGINEERING
# ============================================================

print("\nCreating country-specific lag features...")

model_data = df.loc[
    df["TIME_PERIOD"].between(1981, HISTORICAL_END_YEAR)
].copy()

model_data = model_data.sort_values(
    ["REF_AREA", "TIME_PERIOD"]
).reset_index(drop=True)

if model_data.duplicated(["REF_AREA", "TIME_PERIOD"]).any():
    raise ValueError("Duplicate country-year keys remain after cleaning.")

# Lags within country.
grouped = model_data.groupby("REF_AREA", sort=False)
model_data["lag_1"] = grouped["OBS_VALUE"].shift(1)
model_data["lag_2"] = grouped["OBS_VALUE"].shift(2)

prev_year_1 = grouped["TIME_PERIOD"].shift(1)
prev_year_2 = grouped["TIME_PERIOD"].shift(2)

# Require exact previous calendar years.
model_data.loc[
    (model_data["TIME_PERIOD"] - prev_year_1) != 1,
    "lag_1"
] = np.nan

model_data.loc[
    (model_data["TIME_PERIOD"] - prev_year_2) != 2,
    "lag_2"
] = np.nan

# Rolling average of the previous three calendar years.
# A dictionary lookup explicitly checks that each calendar year exists.
history_lookup = {
    (row.REF_AREA, int(row.TIME_PERIOD)): float(row.OBS_VALUE)
    for row in model_data[
        ["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]
    ].itertuples(index=False)
}

rolling_values = []
for row in model_data[
    ["REF_AREA", "TIME_PERIOD"]
].itertuples(index=False):
    country = row.REF_AREA
    year = int(row.TIME_PERIOD)
    previous_years = [year - 3, year - 2, year - 1]
    keys = [(country, y) for y in previous_years]

    if all(key in history_lookup for key in keys):
        rolling_values.append(
            float(np.mean([history_lookup[key] for key in keys]))
        )
    else:
        rolling_values.append(np.nan)

model_data["rolling_mean_3"] = rolling_values

FEATURES = [
    "REF_AREA",
    "TIME_PERIOD",
    "lag_1",
    "lag_2",
    "rolling_mean_3"
]
TARGET = "OBS_VALUE"

model_data = model_data.dropna(
    subset=FEATURES + [TARGET]
).copy()

numeric_check = model_data[
    ["TIME_PERIOD", "lag_1", "lag_2", "rolling_mean_3", TARGET]
].to_numpy(dtype=float)

if not np.isfinite(numeric_check).all():
    raise ValueError("Non-finite values remain in model data.")

if model_data.empty:
    raise ValueError(
        "No modeling observations remain after feature engineering. "
        "Check country-year coverage and the selected vintage."
    )

print("Final modeling rows:", len(model_data))
print("Countries:", model_data["REF_AREA"].nunique())
print(
    "Target-year range:",
    model_data["TIME_PERIOD"].min(),
    "to",
    model_data["TIME_PERIOD"].max()
)
print("Features:", FEATURES)
model_data.head(20).to_csv(
    OUTPUT_DIR / "tables" / "model_data_sample.csv", index=False
)


# ============================================================
# 7. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

train = model_data.loc[
    model_data["TIME_PERIOD"] <= TRAIN_END
].copy()

validation = model_data.loc[
    model_data["TIME_PERIOD"].between(
        VALIDATION_START, VALIDATION_END
    )
].copy()

test = model_data.loc[
    model_data["TIME_PERIOD"].between(TEST_START, TEST_END)
].copy()

if len(train) == 0 or len(validation) == 0 or len(test) == 0:
    raise ValueError(
        "At least one chronological split is empty.\n"
        f"Train rows={len(train)}, validation rows={len(validation)}, "
        f"test rows={len(test)}.\n"
        "Check whether your selected vintage contains enough valid "
        "observations in the configured periods."
    )

assert train["TIME_PERIOD"].max() < validation["TIME_PERIOD"].min()
assert validation["TIME_PERIOD"].max() < test["TIME_PERIOD"].min()
assert set(train["TIME_PERIOD"]).isdisjoint(set(validation["TIME_PERIOD"]))
assert set(train["TIME_PERIOD"]).isdisjoint(set(test["TIME_PERIOD"]))
assert set(validation["TIME_PERIOD"]).isdisjoint(set(test["TIME_PERIOD"]))

X_train, y_train = train[FEATURES], train[TARGET]
X_val, y_val = validation[FEATURES], validation[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

print("\n--- CHRONOLOGICAL SPLIT ---")
print(
    f"Training: {len(train)} rows, "
    f"target years {train['TIME_PERIOD'].min()}–{train['TIME_PERIOD'].max()}"
)
print(
    f"Validation: {len(validation)} rows, "
    f"target years {validation['TIME_PERIOD'].min()}–{validation['TIME_PERIOD'].max()}"
)
print(
    f"Test: {len(test)} rows, "
    f"target years {test['TIME_PERIOD'].min()}–{test['TIME_PERIOD'].max()}"
)


# ============================================================
# 8. MODEL TRAINING AND VALIDATION-BASED SELECTION
# ============================================================

# Models are restricted to topics in the supplied syllabus:
# - Multiple Linear Regression / OLS
# - Polynomial Regression
# - Decision Trees
# - Support Vector Machine (SVR for regression)
# - Random Forest (ensemble learning / bagging)
model_specs = {
    "Multiple Linear Regression (OLS)": {
        "estimator": LinearRegression(),
        "polynomial": False
    },
    "Polynomial Regression (degree 2)": {
        "estimator": LinearRegression(),
        "polynomial": True
    },
    "Decision Tree Regression": {
        "estimator": DecisionTreeRegressor(
            max_depth=8,
            min_samples_leaf=5,
            random_state=RANDOM_SEED
        ),
        "polynomial": False
    },
    "Support Vector Regression (SVR)": {
        "estimator": SVR(kernel="rbf", C=10.0, epsilon=0.1),
        "polynomial": False
    },
    "Random Forest Regression": {
        "estimator": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=3,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
        "polynomial": False
    }
}

pipelines = {}
for model_name, spec in model_specs.items():
    pipelines[model_name] = Pipeline(steps=[
        ("preprocessor", make_preprocessor(
            polynomial=spec["polynomial"]
        )),
        ("model", spec["estimator"])
    ])

validation_rows = []
validation_models = {}

print("\\nTraining syllabus-aligned candidate models...")
for model_name, pipeline in pipelines.items():
    print("Training:", model_name)
    pipeline.fit(X_train, y_train)
    val_prediction = pipeline.predict(X_val)
    metrics = calculate_metrics(y_val, val_prediction)

    validation_rows.append({
        "Model": model_name,
        **metrics
    })
    validation_models[model_name] = pipeline

validation_results = pd.DataFrame(validation_rows).sort_values(
    "MAE", ascending=True
).reset_index(drop=True)

print("\\n--- VALIDATION RESULTS (MODEL SELECTION ONLY) ---")
print(validation_results.round(4).to_string(index=False))
validation_results.to_csv(
    OUTPUT_DIR / "tables" / "validation_model_comparison.csv",
    index=False
)

best_model_name = validation_results.iloc[0]["Model"]
print("\\nSelected model based on validation MAE:", best_model_name)


# ============================================================
# 9. REFIT ON TRAIN + VALIDATION AND TEST ONCE
# ============================================================

train_val = pd.concat([train, validation], ignore_index=True)
X_train_val = train_val[FEATURES]
y_train_val = train_val[TARGET]

test_rows = []
final_models = {}
test_predictions = {}

print("\nRefitting models on train + validation and evaluating on test...")
for model_name, pipeline in pipelines.items():
    final_pipeline = clone(pipeline)
    final_pipeline.fit(X_train_val, y_train_val)
    prediction = final_pipeline.predict(X_test)
    metrics = calculate_metrics(y_test, prediction)

    test_rows.append({
        "Model": model_name,
        **metrics
    })
    final_models[model_name] = final_pipeline
    test_predictions[model_name] = prediction

test_results = pd.DataFrame(test_rows).sort_values(
    "MAE", ascending=True
).reset_index(drop=True)

print("\n--- FINAL HELD-OUT TEST RESULTS ---")
print(test_results.round(4).to_string(index=False))
test_results.to_csv(
    OUTPUT_DIR / "tables" / "test_model_comparison.csv",
    index=False
)

# Model performance comparison using held-out test MAE and RMSE.
# Lower values indicate smaller prediction errors.
performance_plot = test_results.melt(
    id_vars="Model",
    value_vars=["MAE", "RMSE"],
    var_name="Metric",
    value_name="Error"
)
plt.figure(figsize=(12, 6))
sns.barplot(
    data=performance_plot,
    x="Model",
    y="Error",
    hue="Metric"
)
plt.title("Model Performance Comparison (Lower Error Is Better)")
plt.xlabel("Regression model")
plt.ylabel("Error (percentage points)")
plt.xticks(rotation=20, ha="right")
plt.legend(title="Metric")
save_and_show_plot("06_model_performance_comparison.png")

# Use the validation-selected model, regardless of which model happens
# to have the lowest test score.
best_model = final_models[best_model_name]
best_test_pred = test_predictions[best_model_name]
selected_test_metrics = calculate_metrics(y_test, best_test_pred)

print("\n--- VALIDATION-SELECTED MODEL TEST METRICS ---")
print("Model:", best_model_name)
for metric, value in selected_test_metrics.items():
    print(f"{metric}: {value:.4f}")

# Compare the selected model against the simplest syllabus model:
# Multiple Linear Regression (OLS).
baseline_name = "Multiple Linear Regression (OLS)"
baseline_mae = test_results.loc[
    test_results["Model"] == baseline_name, "MAE"
].iloc[0]
selected_mae = selected_test_metrics["MAE"]

if best_model_name == baseline_name:
    print("The selected model is the Multiple Linear Regression (OLS) reference.")
elif baseline_mae > 0:
    relative_change = (baseline_mae - selected_mae) / baseline_mae * 100
    print(
        f"MAE improvement vs Multiple Linear Regression (OLS): "
        f"{relative_change:.2f}%"
    )
else:
    print("Reference MAE is zero; relative improvement is undefined.")


# ============================================================
# 10. ACTUAL VS PREDICTED AND RESIDUAL ANALYSIS
# ============================================================

evaluation = test[
    ["REF_AREA", "TIME_PERIOD", TARGET]
].copy()
evaluation["Predicted"] = best_test_pred
evaluation["Residual"] = evaluation[TARGET] - evaluation["Predicted"]
evaluation["Absolute_Error"] = evaluation["Residual"].abs()

evaluation.to_csv(
    OUTPUT_DIR / "tables" / "best_model_test_predictions.csv",
    index=False
)

# Year-wise prediction error analysis (test period only).
yearly_error = (
    evaluation.groupby("TIME_PERIOD")
    .agg(
        MAE=("Absolute_Error", "mean"),
        Observations=("Absolute_Error", "count")
    )
    .reset_index()
)
yearly_error.to_csv(
    OUTPUT_DIR / "tables" / "year_wise_prediction_error.csv",
    index=False
)
plt.figure(figsize=(10, 5))
sns.lineplot(
    data=yearly_error,
    x="TIME_PERIOD",
    y="MAE",
    marker="o"
)
plt.title(f"Year-wise Prediction Error — {best_model_name}")
plt.xlabel("Year")
plt.ylabel("Mean Absolute Error (percentage points)")
plt.xticks(yearly_error["TIME_PERIOD"].astype(int))
save_and_show_plot("08_year_wise_prediction_error.png")

# Actual vs predicted.
plt.figure(figsize=(8, 7))
sns.scatterplot(
    data=evaluation,
    x=TARGET,
    y="Predicted",
    alpha=0.65
)
all_values = np.concatenate([
    evaluation[TARGET].to_numpy(),
    evaluation["Predicted"].to_numpy()
])
plot_min = float(np.nanmin(all_values))
plot_max = float(np.nanmax(all_values))
plt.plot(
    [plot_min, plot_max],
    [plot_min, plot_max],
    linestyle="--",
    color="black",
    label="Perfect prediction"
)
plt.title(f"Actual vs Predicted Inflation — {best_model_name}")
plt.xlabel("Actual inflation (%)")
plt.ylabel("Predicted inflation (%)")
plt.legend()
save_and_show_plot("09_actual_vs_predicted.png")

# Residual distribution.
plt.figure(figsize=(10, 5))
sns.histplot(data=evaluation, x="Residual", bins=50, kde=True)
plt.axvline(0, linestyle="--", color="black")
plt.title("Residual Distribution")
plt.xlabel("Residual = actual − predicted (percentage points)")
plt.ylabel("Observation count")
save_and_show_plot("10_residual_distribution.png")

# Residuals against predicted.
plt.figure(figsize=(10, 5))
sns.scatterplot(
    data=evaluation,
    x="Predicted",
    y="Residual",
    alpha=0.65
)
plt.axhline(0, linestyle="--", color="black")
plt.title("Residuals vs Predicted Inflation")
plt.xlabel("Predicted inflation (%)")
plt.ylabel("Residual (percentage points)")
save_and_show_plot("11_residuals_vs_predicted.png")

print("\nResidual mean:", round(evaluation["Residual"].mean(), 4))
print("Residual median:", round(evaluation["Residual"].median(), 4))
print("Residual standard deviation:", round(evaluation["Residual"].std(), 4))

print("\n--- TEN LARGEST ABSOLUTE TEST ERRORS ---")
largest_errors = evaluation.nlargest(
    10, "Absolute_Error"
)
print(largest_errors.to_string(index=False))
largest_errors.to_csv(
    OUTPUT_DIR / "tables" / "largest_test_errors.csv", index=False
)


# ============================================================
# 11. DEPLOYMENT MODEL AND FUTURE PREDICTION FUNCTION
# ============================================================

# Train a deployment version on all eligible data through the historical
# cutoff. This model is for demonstrating future prediction, not for
# replacing the held-out test evaluation above.
deployment_model = clone(pipelines[best_model_name])
deployment_model.fit(model_data[FEATURES], model_data[TARGET])

country_lookup = (
    df[["REF_AREA", "REF_AREA_LABEL"]]
    .drop_duplicates()
    .copy()
)


def predict_inflation(country, year):
    """
    Recursively forecast inflation for a country using the deployment model.

    country: country code (e.g. 'IND') or exact country/area label
    year: future integer year

    Returns a DataFrame of yearly estimates. These are model forecasts,
    not official economic projections. Forecast uncertainty grows with
    the forecast horizon.
    """
    if not isinstance(year, (int, np.integer)):
        raise TypeError("year must be an integer, e.g. 2025.")

    country_text = str(country).strip()

    code_matches = country_lookup.loc[
        country_lookup["REF_AREA"].astype(str).str.upper()
        == country_text.upper()
    ]
    name_matches = country_lookup.loc[
        country_lookup["REF_AREA_LABEL"].astype(str).str.casefold()
        == country_text.casefold()
    ]

    if not code_matches.empty:
        code = str(code_matches.iloc[0]["REF_AREA"])
        country_name = str(code_matches.iloc[0]["REF_AREA_LABEL"])
    elif not name_matches.empty:
        code = str(name_matches.iloc[0]["REF_AREA"])
        country_name = str(name_matches.iloc[0]["REF_AREA_LABEL"])
    else:
        raise ValueError(
            f"Country '{country}' was not found in the selected vintage."
        )

    history = df.loc[
        (df["REF_AREA"].astype(str) == code)
        & (df["TIME_PERIOD"] <= HISTORICAL_END_YEAR),
        ["TIME_PERIOD", "OBS_VALUE"]
    ].dropna().sort_values("TIME_PERIOD")

    history_map = {
        int(row.TIME_PERIOD): float(row.OBS_VALUE)
        for row in history.itertuples(index=False)
    }

    if not history_map:
        raise ValueError("No usable historical inflation is available.")

    latest_year_for_country = max(history_map)

    if year <= latest_year_for_country:
        raise ValueError(
            f"Requested year {year} is not later than the latest "
            f"available year ({latest_year_for_country}) for {country_name}. "
            "Use the held-out test predictions for historical evaluation."
        )

    forecasts = []

    for target_year in range(latest_year_for_country + 1, year + 1):
        prior_years = [
            target_year - 3,
            target_year - 2,
            target_year - 1
        ]

        if not all(y in history_map for y in prior_years):
            raise ValueError(
                f"Cannot forecast {target_year}: the preceding three "
                "calendar years are not all available. This country may "
                "have gaps in its history."
            )

        feature_row = pd.DataFrame([{
            "REF_AREA": code,
            "TIME_PERIOD": target_year,
            "lag_1": history_map[target_year - 1],
            "lag_2": history_map[target_year - 2],
            "rolling_mean_3": float(np.mean([
                history_map[target_year - 3],
                history_map[target_year - 2],
                history_map[target_year - 1]
            ]))
        }])

        prediction = float(
            deployment_model.predict(feature_row[FEATURES])[0]
        )
        history_map[target_year] = prediction

        forecasts.append({
            "Country": country_name,
            "Country_Code": code,
            "Year": target_year,
            "Predicted_Inflation_Rate": prediction,
            "Prediction_Type": (
                "One-year-ahead forecast"
                if target_year == latest_year_for_country + 1
                else "Recursive multi-year forecast"
            )
        })

    return pd.DataFrame(forecasts)


# Demonstration forecast. Change these values if desired.
print("\n--- EXAMPLE FUTURE PREDICTION ---")
try:
    example_forecast = predict_inflation("IND", 2025)
    print(example_forecast.round(3).to_string(index=False))
    example_forecast.to_csv(
        OUTPUT_DIR / "tables" / "example_future_forecast.csv",
        index=False
    )
except (ValueError, TypeError) as exc:
    print("Example forecast could not be generated:", exc)
    print(
        "This may happen if India is absent or the required preceding "
        "calendar years are not available in the selected vintage."
    )


# ============================================================
# 12. FINAL SUMMARY AND OUTPUT LOCATIONS
# ============================================================

print("\n" + "=" * 70)
print("PROJECT RUN COMPLETED")
print("=" * 70)
print("CSV used:", csv_path)
print("Selected vintage:", SELECTED_VINTAGE)
print("Model selected on validation:", best_model_name)
print("\nHeld-out test metrics:")
for metric, value in selected_test_metrics.items():
    print(f"  {metric}: {value:.4f}")

print("\nOutput folder:", OUTPUT_DIR)
print("Figures folder:", OUTPUT_DIR / "figures")
print("Tables folder:", OUTPUT_DIR / "tables")
print(
    "\nInterpretation reminder: use the measured results above in your "
    "report. Do not claim a model is accurate unless it beats the baseline "
    "on held-out data and the residuals support that conclusion."
)

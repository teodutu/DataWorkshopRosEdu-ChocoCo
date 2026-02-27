# 🍫 Workshop: From Raw Sales Data to a Working Prediction Model (Python)

## 0) The story (why we’re here)

You’ve joined the data team at **ChocoCo**, a company selling chocolate in multiple countries via a sales team.

Leadership asks two practical questions:

1. **What drives sales revenue?**

   * Which products and countries contribute most?
   * Are some salespeople consistently outperforming?

2. **Can we predict revenue per transaction?**

   * For a given (Country, Product, Sales Person, Date, Boxes Shipped), can we estimate **Amount**?

We’ll answer both using a dataset from Kaggle: **Chocolate Sales**. ([Kaggle][1])

---

## 1) The dataset

From the dataset documentation and community analyses, the core fields you’ll use are:

* `Sales Person`
* `Country`
* `Product`
* `Date`
* `Amount` (target / label)
* `Boxes Shipped`

---

## 2) Setup: project structure (scripts *or* notebooks)

You can complete the workshop in **either** of these modes:

### Option A — Script-based

Create a folder like:

```
choco-workshop/
  data/
    raw/
    processed/
  reports/
    figures/
  src/
    clean_data.py
    eda.py
    features.py
    train.py
    evaluate.py
  README.md
  requirements.txt
```

You’ll write small scripts that you run from the terminal.

### Option B — Notebook-based

Create a folder like:

```
choco-workshop/
  data/
    raw/
    processed/
  reports/
    figures/
  notebooks/
    01_cleaning.ipynb
    02_eda.ipynb
    03_features.ipynb
    04_modelling.ipynb
    05_evaluation.ipynb
  requirements.txt
```

You’ll complete each step in a separate notebook.

**Notebook guidance (important):**

* Keep notebooks **linear and reproducible** (Run All should work top-to-bottom).
* Save outputs (cleaned CSVs, plots) to the same `data/processed/` and `reports/figures/` folders as the script version.
* Treat notebooks like “experiment logs”, not like final apps.

### Python environment (same for both)

1. Create a virtual environment (`python -m venv .venv`)
2. Activate it (`source .venv/bin/activate` / Windows activation)
3. Install dependencies from `requirements.txt`

Suggested packages to include:

* `pandas`, `numpy` (data handling)
* `matplotlib` (plots)
* `scikit-learn` (models + evaluation)
* `python-dateutil` (date parsing)
* **Notebook mode only:** `jupyter` or `jupyterlab`

---

## 3) Get the data (Kaggle)

### Manual download

1. Download the CSV from Kaggle. ([Kaggle][1])
2. Put it in `data/raw/`
3. Confirm you can load it using `pandas.read_csv(...)`

**Hint:** If you see weird column names (extra spaces, inconsistent capitalisation), that’s normal — you’ll fix that in cleaning.

---

## 4) Data cleaning & preprocessing

### Why cleaning matters (in real life)

Cleaning is where you prevent:

* **Broken charts** (dates stored as text, numeric values stored as strings)
* **Bad models** (data leakage, incorrect types, missing values)
* **Misleading conclusions** (duplicate rows, inconsistent categories)

### What you should do

1. **Load the raw CSV**

   * Hint: `pandas.read_csv(...)`

2. **Standardise column names**

   * Convert to lowercase, replace spaces with underscores, trim whitespace
   * Hint: `df.columns = ...` with `str.lower()`, `str.strip()`, `str.replace(...)`

3. **Parse the date column**

   * Convert `Date` into a real datetime type
   * Hint: `pandas.to_datetime(..., errors="coerce")`
   * Then check how many dates failed conversion (`isna().sum()`)

4. **Convert numeric columns**

   * Ensure `Amount` and `Boxes Shipped` are numeric types
   * Watch for commas or currency symbols
   * Hint: `str.replace(",", "")`, then `pandas.to_numeric(..., errors="coerce")`

5. **Handle missing values**

   * Decide a strategy:

     * If `Amount` is missing (your prediction target), you usually **drop those rows**
     * If a feature is missing, you may keep rows and later **impute**
   * Hint: `dropna(subset=[...])`, `fillna(...)`, or defer to sklearn imputers later

6. **Remove duplicates carefully**

   * Hint: `drop_duplicates()`

7. **Save a cleaned dataset**

   * Write to `data/processed/` so later steps don’t depend on raw quirks
   * Hint: `to_csv(..., index=False)`

### Exercise checkpoints

* How many rows were removed due to missing `Amount`?
* How many `Date` values failed parsing?
* Are `Amount` and `Boxes Shipped` definitely numeric now (`df.dtypes`)?

---

## 5) Exploratory Data Analysis (EDA): “Can we see patterns?”

EDA is **sanity-checking + curiosity**:

* What ranges do values take?
* Are there obvious outliers?
* Is the dataset balanced across countries/products?
* How does revenue change over time?

### What you should do

1. **Basic inspection**

   * Look at the first few rows, column names, data types
   * Hint: `head()`, `info()`, `describe()`

2. **Missing values overview**

   * Count missing values per column
   * Hint: `isna().sum().sort_values(...)`

3. **Simple grouping questions**

   * Total revenue by country
   * Total revenue by product
   * Total revenue by salesperson
   * Hint: `groupby(...).sum().sort_values(...)`

4. **Time-based summaries**

   * Aggregate revenue by month (or week)
   * Hint: use `dt.to_period("M")` or `dt.month` and `groupby`

### EDA hints (what to look for)

* If one salesperson dominates, ask: “Is that real performance or data coverage bias?”
* If `Boxes Shipped` has zeros/negatives, that’s a data quality red flag.
* If there are only a few dates, your “trend analysis” will be weak.

---

## 6) Quick visualisations (graphs that answer business questions)

We’ll make **3 small plots** that are easy to explain to non-DS stakeholders.

### What you should build

1. **Revenue over time**

   * Plot total `Amount` per month
   * Hint: aggregate first (`groupby month`), then `matplotlib.pyplot.plot(...)`

2. **Top 10 products by revenue**

   * Horizontal bar chart is often easiest to read
   * Hint: `sort_values().head(10)` then `plot(kind="barh")`

3. **Boxes shipped vs revenue**

   * Scatter plot to see whether “more boxes generally means more revenue”
   * Hint: `plt.scatter(x, y)`

**Output habit:** save plots to `reports/figures/` so they can go into slides later.

* Hint: `plt.savefig(...)`

---

## 7) Feature engineering (turn raw columns into model-friendly signals)

### What is feature engineering?

A model can’t directly use text like `"Australia"` or `"Dark Truffles"` unless we encode it.

You’ll build features:

* **Date features**: month, day-of-week (and optionally “is weekend”)
* **Categoricals**: encode `country`, `product`, `sales_person`
* **Numeric**: `boxes_shipped`

### What you should do

1. **Create date-derived features**

   * `month` from the date
   * `day_of_week` from the date
   * Hint: `df["date"].dt.month`, `df["date"].dt.dayofweek`

2. **Choose your target**

   * We predict `Amount` (regression)

3. **Decide which columns are inputs**

   * Likely inputs: `Boxes Shipped`, `Country`, `Product`, `Sales Person`, `month`, `day_of_week`

4. **Encode categorical variables**

   * Use one-hot encoding
   * Hint (sklearn): `OneHotEncoder(handle_unknown="ignore")`
   * Hint (pandas quick version): `get_dummies(...)` (fine for prototyping; less ideal for pipelines)

### A key rule: avoid “cheating” (data leakage)

Don’t include anything that wouldn’t be known at prediction time.

Example leakage:

* If `Amount` is the target, don’t compute “average amount per product” using the whole dataset *before splitting*.
* If you do target encoding / averages, compute them **inside training folds only**.

---

## 8) Modelling (prediction)

1) Pick your train/test split
   - Default (recommended): time-based split
     - Sort by Date
     - Train = first 80% of rows, Test = last 20%
     - Hint: sort_values("date"), then slice with .iloc[...]

   - Only use random split if time doesn’t matter
     - Hint: train_test_split(test_size=0.2, random_state=42)

2) Cross-validation
   - If using time: TimeSeriesSplit(n_splits=5)
   - If random split: KFold / cross_val_score

3) Baselines (always do these first)
- Predict mean Amount from training set
- Predict median Amount from training set (Hint: mean(), median())

4) Models to try
- Ridge regression (great first real model)
- Random forest regressor (handles non-linear patterns)
Hint: RandomForestRegressor(n_estimators=200-500, random_state=42)

---

## 9) Evaluation: “How do we know if the model is any good?”

Evaluation isn’t just a metric—it’s a **decision**.

### Metrics to use

* **MAE** (Mean Absolute Error): “On average, how many currency units off are we?”

  * Hint: `mean_absolute_error`
* **RMSE**: penalises large mistakes more than MAE

  * Hint: compute from MSE (`mean_squared_error`) then square root
* **R²**: fraction of variance explained

  * Hint: `r2_score`

### The most important evaluation choices

1. **Split strategy**

   * If time-ordered: use time-based splits
   * Random split can accidentally train on “future” patterns

2. **Baseline comparison**

   * If baseline MAE ≈ model MAE, your model isn’t adding value

3. **Error analysis**

   * Find where the model performs worst:

     * Which countries/products have the highest errors?
   * Hint: calculate per-group MAE using `groupby` after you generate predictions

### Mini exercise: error slicing

After predicting, compute MAE per country/product:

* “Where is the model weakest and why might that be?”

---

## 10) Feature importance: “What matters most?”

You’ll explain *why* the model makes good predictions.

Two accessible approaches:

1. **Permutation importance (model-agnostic)**

   * Shuffle one feature at a time and see how much the metric worsens
   * Hint: `sklearn.inspection.permutation_importance`

2. **Model-specific importance**

   * Random forests provide built-in importance values
   * Hint: `model.feature_importances_`
   * Caveat: can be misleading with correlated features — use with care

---

## 11) Stretch goals

* Add a new feature: `is_weekend`

  * Hint: `day_of_week >= 5`
* Predict **monthly total sales** instead of transaction `Amount`

  * Hint: aggregate first, then treat it as a time series regression/forecasting problem
* Try a different model: Gradient boosting

  * Hint: `GradientBoostingRegressor` (or `HistGradientBoostingRegressor`)
* Create a Markdown “report generator”

  * Save figures + key numbers, then write a short summary file automatically

---

## 12) What “good” looks like at the end

You’ll have:

* A reproducible pipeline that turns raw CSV → cleaned CSV
* Clear EDA outputs and a few stakeholder-friendly charts
* At least one model that beats the baseline (hopefully!)
* A way to explain feature importance in plain language

---

[1]: https://www.kaggle.com/datasets/saidaminsaidaxmadov/chocolate-sales?utm_source=chatgpt.com "Chocolate Sales - Kaggle"
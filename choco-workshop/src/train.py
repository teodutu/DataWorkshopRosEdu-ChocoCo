

# Stretch: Predict monthly total sales
if 'PROCESSED_DATA_DIR' in globals():
	monthly_sales_path = os.path.join(PROCESSED_DATA_DIR, 'monthly_sales.csv')
	if os.path.exists(monthly_sales_path):
		print('\nStretch: Predicting monthly total sales')
		ms = pd.read_csv(monthly_sales_path)
		ms['month'] = ms['month'].astype(str)
		ms = ms.sort_values('month')
		ms['month_num'] = np.arange(len(ms))
		X_ms = ms[['month_num']]
		y_ms = ms['amount']
		# Simple time series regression
		from sklearn.ensemble import HistGradientBoostingRegressor
		ms_model = HistGradientBoostingRegressor(random_state=42)
		ms_model.fit(X_ms, y_ms)
		y_ms_pred = ms_model.predict(X_ms)
		print(f'Monthly sales RMSE: {np.sqrt(mean_squared_error(y_ms, y_ms_pred)):.2f}')

# Script for model training
# Usage: python src/train.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/processed')
features_csv_path = os.path.join(PROCESSED_DATA_DIR, 'chocolate_sales_features.csv')
model_path = os.path.join(PROCESSED_DATA_DIR, 'rf_model.joblib')

print('Loading feature data...')
df = pd.read_csv(features_csv_path)

# Define features and target
target = 'amount'
exclude_cols = ['amount', 'date']
X = df.drop(columns=[col for col in exclude_cols if col in df.columns], errors='ignore')
y = df[target]


# Remove any non-numeric columns (should be only features after one-hot encoding)
X = X.select_dtypes(include='number')

# Impute missing values in features with median (for Ridge regression compatibility)
if X.isna().any().any():
	print('Imputing missing values in features with median...')
	X = X.fillna(X.median())


# Step 8: Modelling (prediction)
print('\nStep 8: Modelling (prediction)')

# Sort by date for time-based split if date exists
if 'date' in df.columns:
	df = df.sort_values('date')
	X = X.loc[df.index]
	y = y.loc[df.index]

# Time-based split: first 80% train, last 20% test
n = len(df)
split_idx = int(n * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Baseline: mean and median
mean_pred = np.full_like(y_test, y_train.mean(), dtype=np.float64)
median_pred = np.full_like(y_test, y_train.median(), dtype=np.float64)
print(f'Baseline (mean) MAE: {mean_absolute_error(y_test, mean_pred):.2f}')
print(f'Baseline (median) MAE: {mean_absolute_error(y_test, median_pred):.2f}')

# Cross-validation setup
print('\nCross-validation:')
if 'date' in df.columns:
	tscv = TimeSeriesSplit(n_splits=5)
	cv = tscv
	print('Using TimeSeriesSplit (5 folds)')
else:
	kf = KFold(n_splits=5, shuffle=True, random_state=42)
	cv = kf
	print('Using KFold (5 folds)')

# Ridge regression
print('\nTraining Ridge regression...')
ridge = Ridge()
ridge_cv_scores = cross_val_score(ridge, X_train, y_train, cv=cv, scoring='neg_mean_absolute_error')
print(f'Ridge CV MAE: {(-ridge_cv_scores).mean():.2f} (+/- {(-ridge_cv_scores).std():.2f})')
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
print(f'Ridge Test MAE: {mean_absolute_error(y_test, y_pred_ridge):.2f}')
print(f'Ridge Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_ridge)):.2f}')
print(f'Ridge Test R^2: {r2_score(y_test, y_pred_ridge):.3f}')

# Random Forest
print('\nTraining RandomForestRegressor...')
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf_cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='neg_mean_absolute_error')
print(f'RF CV MAE: {(-rf_cv_scores).mean():.2f} (+/- {(-rf_cv_scores).std():.2f})')
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(f'RF Test MAE: {mean_absolute_error(y_test, y_pred_rf):.2f}')
print(f'RF Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f}')
print(f'RF Test R^2: {r2_score(y_test, y_pred_rf):.3f}')

# Save best model (choose RF here)
joblib.dump(rf, model_path)
print(f'Random Forest model saved to: {model_path}')

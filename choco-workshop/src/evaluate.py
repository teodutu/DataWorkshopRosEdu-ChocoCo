import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

# Script for model evaluation
# Usage: python src/evaluate.py

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/processed')
features_csv_path = os.path.join(PROCESSED_DATA_DIR, 'chocolate_sales_features.csv')
model_path = os.path.join(PROCESSED_DATA_DIR, 'rf_model.joblib')

print('Loading feature data and model...')
df = pd.read_csv(features_csv_path)
model = joblib.load(model_path)

# Sort by date for time-based split if date exists
if 'date' in df.columns:
	df = df.sort_values('date')

# Time-based split: first 80% train, last 20% test
n = len(df)
split_idx = int(n * 0.8)
df_test = df.iloc[split_idx:]

target = 'amount'
exclude_cols = ['amount', 'date']
X_test = df_test.drop(columns=[col for col in exclude_cols if col in df_test.columns], errors='ignore')
X_test = X_test.select_dtypes(include='number')
if X_test.isna().any().any():
	X_test = X_test.fillna(X_test.median())
y_test = df_test[target]

# Predict
y_pred = model.predict(X_test)

# Actual vs. Predicted Scatter Plot
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, alpha=0.5, color='navy')
plt.xlabel('Actual Amount')
plt.ylabel('Predicted Amount')
plt.title('Actual vs. Predicted (Test Set)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), '../reports/figures/actual_vs_predicted.png'))
plt.close()

# Error Distribution (Residuals)
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=30, color='crimson', edgecolor='black')
plt.title('Error Distribution (Residuals)')
plt.xlabel('Prediction Error (Actual - Predicted)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), '../reports/figures/error_distribution.png'))
plt.close()

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f'\nTest MAE: {mae:.2f}')
print(f'Test RMSE: {rmse:.2f}')
print(f'Test R^2: {r2:.3f}')

# Error analysis: per country and product
# Error analysis: per country and product
for col in ['country', 'product']:
	if col in df_test.columns:
		print(f'\nMAE by {col}:')
		group_mae = df_test.groupby(col).apply(lambda g: mean_absolute_error(g[target], y_pred[g.index - split_idx]))
		print(group_mae.sort_values(ascending=False))

# Step 10: Feature importance
print('\nStep 10: Feature importance')

# Model-based importance (Random Forest)
if hasattr(model, 'feature_importances_'):
	feature_names = X_test.columns
	importances = model.feature_importances_
	fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
	fi_df = fi_df.sort_values('importance', ascending=False)
	print('\nTop 10 features by model importance:')
	print(fi_df.head(10))
	fi_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'feature_importances_model.csv'), index=False)

# Permutation importance (model-agnostic)
print('\nComputing permutation importance (may take a while)...')
perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
perm_df = pd.DataFrame({'feature': X_test.columns, 'importance': perm.importances_mean})
perm_df = perm_df.sort_values('importance', ascending=False)
print('\nTop 10 features by permutation importance:')
print(perm_df.head(10))
perm_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'feature_importances_permutation.csv'), index=False)

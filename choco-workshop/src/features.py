
# Script for feature engineering
# Usage: python src/features.py

import os
import pandas as pd

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/processed')
cleaned_csv_path = os.path.join(PROCESSED_DATA_DIR, 'chocolate_sales_cleaned.csv')
features_csv_path = os.path.join(PROCESSED_DATA_DIR, 'chocolate_sales_features.csv')

print('Loading cleaned data...')
df = pd.read_csv(cleaned_csv_path)


# Example feature engineering steps:
# 1. Extract year, month, day, day_of_week, is_weekend from date
if 'date' in df.columns:
	df['date'] = pd.to_datetime(df['date'], errors='coerce')
	df['year'] = df['date'].dt.year
	df['month'] = df['date'].dt.month
	df['day'] = df['date'].dt.day
	df['day_of_week'] = df['date'].dt.dayofweek
	df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# 2. Encode categorical variables (one-hot encoding for country, product, sales_person)
for col in ['country', 'product', 'sales_person']:
	if col in df.columns:
		dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
		df = pd.concat([df, dummies], axis=1)

# 3. Example: Create interaction feature (boxes_shipped * amount per box)
if 'boxes_shipped' in df.columns and 'amount' in df.columns:
	df['amount_per_box'] = df['amount'] / df['boxes_shipped'].replace(0, pd.NA)

# 4. Save features
df.to_csv(features_csv_path, index=False)
print(f'Feature dataset saved to: {features_csv_path}')

# Stretch: Predict monthly total sales (aggregate and save)
monthly = df.copy()
if 'date' in monthly.columns and 'amount' in monthly.columns:
	monthly['month'] = monthly['date'].dt.to_period('M')
	monthly_sales = monthly.groupby('month')['amount'].sum().reset_index()
	monthly_sales.to_csv(os.path.join(PROCESSED_DATA_DIR, 'monthly_sales.csv'), index=False)
	print('Monthly sales dataset saved to:', os.path.join(PROCESSED_DATA_DIR, 'monthly_sales.csv'))

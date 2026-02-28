
# Script for cleaning data
# Usage: python src/clean_data.py

import kagglehub
import shutil
import os
import glob
import pandas as pd

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/raw')
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/processed')

# Download latest version of the dataset
print('Downloading dataset from Kaggle...')
dataset_path = kagglehub.dataset_download('saidaminsaidaxmadov/chocolate-sales')
print('Downloaded to:', dataset_path)

# Find the CSV file in the downloaded dataset
downloaded_csvs = glob.glob(os.path.join(dataset_path, '*.csv'))
if not downloaded_csvs:
	raise FileNotFoundError('No CSV file found in the downloaded dataset.')

csv_file = downloaded_csvs[0]

# Copy the CSV to data/raw/
os.makedirs(RAW_DATA_DIR, exist_ok=True)
raw_csv_path = os.path.join(RAW_DATA_DIR, os.path.basename(csv_file))
shutil.copy(csv_file, raw_csv_path)
print('Copied raw CSV to:', raw_csv_path)

# Step 3: Data cleaning
print('Cleaning data...')
df = pd.read_csv(raw_csv_path)

# 1. Standardise column names
df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')

# 2. Parse the date column
df['date'] = pd.to_datetime(df['date'], errors='coerce')
date_parse_failures = df['date'].isna().sum()
print(f'Date parse failures: {date_parse_failures}')

# 3. Convert numeric columns
for col in ['amount', 'boxes_shipped']:
	if col in df.columns:
		df[col] = (
			df[col].astype(str)
			.str.replace(',', '', regex=False)
			.str.replace('$', '', regex=False)
		)
		df[col] = pd.to_numeric(df[col], errors='coerce')

# 4. Handle missing values
rows_before = len(df)
df = df.dropna(subset=['amount'])
rows_after = len(df)
rows_dropped = rows_before - rows_after
print(f'Rows dropped due to missing Amount: {rows_dropped}')

# 5. Remove duplicates
df = df.drop_duplicates()

# 6. Save cleaned dataset
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
cleaned_csv_path = os.path.join(PROCESSED_DATA_DIR, 'chocolate_sales_cleaned.csv')
df.to_csv(cleaned_csv_path, index=False)
print('Cleaned data saved to:', cleaned_csv_path)

# 7. Check dtypes
print('Column types after cleaning:')
print(df.dtypes)

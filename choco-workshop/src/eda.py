
# Script for exploratory data analysis
# Usage: python src/eda.py

import os
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/processed')
cleaned_csv_path = os.path.join(PROCESSED_DATA_DIR, 'chocolate_sales_cleaned.csv')

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '../reports/figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

print('Loading cleaned data...')
df = pd.read_csv(cleaned_csv_path)

# 1. Basic inspection
print('\nFirst 5 rows:')
print(df.head())
print('\nColumn names:')
print(df.columns.tolist())
print('\nData types:')
print(df.dtypes)
print('\nDescribe:')
print(df.describe(include='all'))

# 2. Missing values overview
print('\nMissing values per column:')
print(df.isna().sum().sort_values(ascending=False))

# 3. Simple grouping questions
if 'country' in df.columns:
	print('\nTotal revenue by country:')
	print(df.groupby('country')['amount'].sum().sort_values(ascending=False))
if 'product' in df.columns:
	print('\nTotal revenue by product:')
	print(df.groupby('product')['amount'].sum().sort_values(ascending=False))
if 'sales_person' in df.columns:
	print('\nTotal revenue by salesperson:')
	print(df.groupby('sales_person')['amount'].sum().sort_values(ascending=False))

# 4. Time-based summaries
if 'date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['date']):
	df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
	print('\nTotal revenue by month:')
	print(df.groupby('month')['amount'].sum().sort_index())

# Extra Graphs
print('\nGenerating extra graphs...')

# Revenue by country
if 'country' in df.columns:
	rev_by_country = df.groupby('country')['amount'].sum().sort_values(ascending=False)
	plt.figure(figsize=(8, 5))
	rev_by_country.plot(kind='bar', color='skyblue')
	plt.title('Revenue by Country')
	plt.ylabel('Revenue (Amount)')
	plt.xlabel('Country')
	plt.tight_layout()
	plt.savefig(os.path.join(FIGURES_DIR, 'revenue_by_country.png'))
	plt.close()

# Revenue by salesperson
if 'sales_person' in df.columns:
	rev_by_sales = df.groupby('sales_person')['amount'].sum().sort_values(ascending=False)
	plt.figure(figsize=(10, 6))
	rev_by_sales.plot(kind='bar', color='orchid')
	plt.title('Revenue by Salesperson')
	plt.ylabel('Revenue (Amount)')
	plt.xlabel('Salesperson')
	plt.tight_layout()
	plt.savefig(os.path.join(FIGURES_DIR, 'revenue_by_salesperson.png'))
	plt.close()

# Distribution of Amount
if 'amount' in df.columns:
	plt.figure(figsize=(8, 5))
	plt.hist(df['amount'].dropna(), bins=30, color='goldenrod', edgecolor='black')
	plt.title('Distribution of Amount')
	plt.xlabel('Amount')
	plt.ylabel('Frequency')
	plt.tight_layout()
	plt.savefig(os.path.join(FIGURES_DIR, 'amount_distribution.png'))
	plt.close()

# Distribution of Boxes Shipped
if 'boxes_shipped' in df.columns:
	plt.figure(figsize=(8, 5))
	plt.hist(df['boxes_shipped'].dropna(), bins=30, color='teal', edgecolor='black')
	plt.title('Distribution of Boxes Shipped')
	plt.xlabel('Boxes Shipped')
	plt.ylabel('Frequency')
	plt.tight_layout()
	plt.savefig(os.path.join(FIGURES_DIR, 'boxes_shipped_distribution.png'))
	plt.close()

# Correlation heatmap
plt.figure(figsize=(10, 8))
numeric_cols = df.select_dtypes(include='number')
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'correlation_heatmap.png'))
plt.close()

# Step 5: Quick visualisations
print('\nGenerating and saving business-focused plots...')

# 1. Revenue over time (monthly)
if 'month' in df.columns:
	revenue_by_month = df.groupby('month')['amount'].sum().sort_index()
	plt.figure(figsize=(10, 5))
	revenue_by_month.plot(marker='o')
	plt.title('Total Revenue per Month')
	plt.xlabel('Month')
	plt.ylabel('Revenue (Amount)')
	plt.tight_layout()
	fig1_path = os.path.join(FIGURES_DIR, 'revenue_over_time.png')
	plt.savefig(fig1_path)
	plt.close()
	print(f'Revenue over time plot saved to: {fig1_path}')

# 2. Top 10 products by revenue
if 'product' in df.columns:
	top_products = df.groupby('product')['amount'].sum().sort_values(ascending=False).head(10)
	plt.figure(figsize=(10, 6))
	top_products.sort_values().plot(kind='barh', color='chocolate')
	plt.title('Top 10 Products by Revenue')
	plt.xlabel('Revenue (Amount)')
	plt.ylabel('Product')
	plt.tight_layout()
	fig2_path = os.path.join(FIGURES_DIR, 'top10_products_by_revenue.png')
	plt.savefig(fig2_path)
	plt.close()
	print(f'Top 10 products plot saved to: {fig2_path}')

# 3. Boxes shipped vs revenue (scatter)
if 'boxes_shipped' in df.columns and 'amount' in df.columns:
	plt.figure(figsize=(8, 6))
	plt.scatter(df['boxes_shipped'], df['amount'], alpha=0.5)
	plt.title('Boxes Shipped vs Revenue')
	plt.xlabel('Boxes Shipped')
	plt.ylabel('Revenue (Amount)')
	plt.tight_layout()
	fig3_path = os.path.join(FIGURES_DIR, 'boxes_vs_revenue.png')
	plt.savefig(fig3_path)
	plt.close()
	print(f'Boxes shipped vs revenue plot saved to: {fig3_path}')

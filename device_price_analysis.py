#Device Price Analysis Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

#Load Dataset
file_path = r"C:\Users\dipti\OneDrive\Documents\GitHub\Device-Price-Analysis\Data\used_device_data.csv"
df = pd.read_csv(file_path)

#Drop rows with missing imp data
df = df.dropna(subset=['device_brand', 'normalized_used_price', 'normalized_new_price'])

#Initial Data Check
print("First 5 Rows:\n", df.head())#only 0 to 5records
print("\nInfo:\n", df.info())
print("\nDescription:\n", df.describe())

#Convert Yes/No to 1/0
df['4g'] = df['4g'].replace({'Yes': 1, 'No': 0})
df['5g'] = df['5g'].replace({'Yes': 1, 'No': 0})

#Convert selected columns to int
cols_to_int = ['release_year', 'days_used', 'ram', 'battery', 'weight']
for col in cols_to_int:
    if col in df.columns:
        df[col] = df[col].fillna(0).astype(int)
    else:
        print(f"Column '{col}' not found!!")

#Price Drop Percentage column
df['price_drop_percent'] = ((df['normalized_new_price'] - df['normalized_used_price']) / df['normalized_new_price']) * 100

#Average Price per Brand
brand_price_summary = df.groupby('device_brand')['normalized_used_price'].mean().reset_index()
brand_price_summary = brand_price_summary.sort_values(by='normalized_used_price', ascending=False)
print("\nBrand Price Summary:\n", brand_price_summary)

#5G vs 4G Price Comparison
network_price_summary = df.groupby('5g')['normalized_used_price'].mean().reset_index()
print("\nNetwork Price Summary:\n", network_price_summary)

#Correlation Matrix
print("\n Correlation Matrix:\n")
corr_matrix = df[['normalized_used_price', 'ram', 'battery', 'rear_camera_mp', 'front_camera_mp', 'days_used']].corr()
print(corr_matrix)

#Make outputs folder if not exists
os.makedirs('outputs', exist_ok=True)

#Plot: Average Price per Brand
plt.figure(figsize=(10,6))
sns.barplot(x='device_brand', y='normalized_used_price', data=brand_price_summary)
plt.title('Average Used Price per Brand')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('outputs/brand_avg_price.png')
plt.close()

#Plot: Price Drop Distribution
plt.figure(figsize=(8,6))
sns.histplot(df['price_drop_percent'], bins=30, kde=True)
plt.title('Distribution of Price Drop Percentage')
plt.tight_layout()
plt.savefig('outputs/price_drop_distribution.png')
plt.close()

#Plot:Days Used vs Used Price
plt.figure(figsize=(8,6))
sns.scatterplot(x='days_used', y='normalized_used_price', hue='5g', data=df)
plt.title('Days Used vs Used Price')
plt.tight_layout()
plt.savefig('outputs/days_used_vs_price.png')
plt.close()

#Plot: Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Features')
plt.tight_layout()
plt.savefig('outputs/feature_correlation_heatmap.png')
plt.close()

#Save CSV Output
brand_price_summary.to_csv('outputs/brand_price_summary.csv', index=False)
df.to_csv('outputs/cleaned_devices.csv', index=False)

print("\nAnalysis Completed. Files saved in 'outputs' folder.")

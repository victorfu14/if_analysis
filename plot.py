import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set a nice theme for the plots
sns.set_theme(style="whitegrid")

# --- Configuration ---
file_path_1 = 'results/if_hessian_digits_09.csv'
file_path_2 = 'results/if_hessian_digits_08.csv'
target_column = 'influence_score'

# --- Load Data ---
try:
    df1 = pd.read_csv(file_path_1)
    df2 = pd.read_csv(file_path_2)
    
    print(f"Successfully loaded {file_path_1} ({len(df1)} rows)")
    print(f"Successfully loaded {file_path_2} ({len(df2)} rows)")

except FileNotFoundError as e:
    print(f"Error: {e}")
    
plt.figure(figsize=(12, 6))

# Configuration
target_column = 'influence_score'

# 1. Define a helper to identify non-outliers (returns True if valid)
def get_valid_mask(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return (df[col] >= lower) & (df[col] <= upper)

# 2. Get masks for both datasets
mask1 = get_valid_mask(df1, target_column)
mask2 = get_valid_mask(df2, target_column)

# 3. Combine masks (Intersection)
# We only keep rows that are valid in BOTH datasets to maintain alignment
combined_mask = mask1 & mask2

# 4. Apply filter
df1_clean = df1[combined_mask].copy()
df2_clean = df2[combined_mask].copy()

# Stats
print(f"Original Rows: {len(df1)}")
print(f"Clean Rows:    {len(df1_clean)}")
print(f"Dropped:       {len(df1) - len(df1_clean)} rows containing outliers.")
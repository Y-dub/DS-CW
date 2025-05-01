# STEP 3: Data Cleaning - MPS Camden Crime Dataset

# Technical Notes for Step 3: Data Cleaning
"""
Objective:
- Remove unnecessary columns (IDs, timestamps, highly null fields)
- Handle missing values appropriately
- Format temporal data into analyzable formats
- Prepare clean data for feature engineering and modeling

Approach:
- Drop administrative columns like 'Refreshed Date'
- Drop columns with >40% missing values
- Fill missing numerical fields with median; categorical fields with mode
- Create 'month_number', 'year', and 'week_number' features from column names
"""

# 1. Import Libraries
import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt

# 2. Load Raw Dataset
file_path = "MPS Ward Level Crime (Historical).csv"
crime_df = pd.read_csv(file_path)

# 3. Drop Administrative Columns
columns_to_drop = ['Refreshed Date']
crime_df_clean = crime_df.drop(columns=columns_to_drop)

# 4. Visualize Missing Data
msno.matrix(crime_df_clean)
plt.title('Missing Data Matrix Before Cleaning')
plt.show()

# 5. Drop Columns with >40% Missing Values
threshold = 0.4 * len(crime_df_clean)
crime_df_clean = crime_df_clean.dropna(axis=1, thresh=threshold)

# 6. Handle Missing Values
# Separate categorical and numerical columns
categorical_cols = crime_df_clean.select_dtypes(include=['object']).columns
numerical_cols = crime_df_clean.select_dtypes(include=['int64', 'float64']).columns

# Fill missing numerical values with median
crime_df_clean[numerical_cols] = crime_df_clean[numerical_cols].fillna(crime_df_clean[numerical_cols].median())

# Fill missing categorical values with mode
for col in categorical_cols:
    crime_df_clean[col] = crime_df_clean[col].fillna(crime_df_clean[col].mode()[0])

# 7. Remove Duplicate Rows
crime_df_clean = crime_df_clean.drop_duplicates()

# 8. Format Temporal Data
# Extract columns representing YearMonth (e.g., 201004, 201005)
temporal_cols = [col for col in crime_df_clean.columns if col.isdigit()]

# Melt temporal columns into a long-form structure
crime_temporal_df = crime_df_clean.melt(
    id_vars=['WardName', 'WardCode', 'MajorText', 'MinorText', 'LookUp_BoroughName'],
    value_vars=temporal_cols,
    var_name='YearMonth',
    value_name='CrimeCount'
)

# Parse YearMonth into Year and Month
crime_temporal_df['YearMonth'] = crime_temporal_df['YearMonth'].astype(str)
crime_temporal_df['Year'] = crime_temporal_df['YearMonth'].str[:4].astype(int)
crime_temporal_df['Month'] = crime_temporal_df['YearMonth'].str[4:6].astype(int)

# Approximate Week Number
crime_temporal_df['WeekNumber'] = (crime_temporal_df['Month'] - 1) * 4 + 2

# 9. Save Cleaned Dataset
crime_temporal_df.to_csv("C:/Users/LLR User/Desktop/masters/Data Science Project/MPS Ward Level Crime(cleaned).csv", index=False)

# Final Output
print("\n✅ Data Cleaning Complete. Cleaned file saved as 'MPS Ward Level Crime(cleaned)'.")
print("\nPreview of Cleaned Data:")
print(crime_temporal_df.head())

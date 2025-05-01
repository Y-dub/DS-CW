# STEP 4: Feature Engineering - MPS Camden Crime Dataset

# Technical Notes for Step 4: Feature Engineering
"""
Objective:
- Create derived features to enhance model predictive power
- Aggregate historical crime metrics
- Simulate auxiliary features like stop-and-search, unsafe reports, POI proximity

Approach:
- Generate crime history features: past month, past two weeks
- Prepare placeholder structures for additional environmental and demographic features
- Ensure data remains in time-series compatible format
"""

# 1. Crime History Features
import pandas as pd
import numpy as np

# Load the cleaned dataset from previous step
crime_temporal_df = pd.read_csv("MPS Ward Level Crime(cleaned).csv")

# Sort values to enable rolling calculations
crime_temporal_df = crime_temporal_df.sort_values(by=['WardName', 'Year', 'Month'])

# Calculate Total Crime in Past Month
crime_temporal_df['CrimeCount_PastMonth'] = (
    crime_temporal_df.groupby('WardName')['CrimeCount']
    .shift(1)  # Look one month back
)

# Calculate Total Crime in Past Two Weeks (approx by halving past month data)
# Since exact 2-week data not available, approximate by assuming 2 weeks ~ 50% of month
crime_temporal_df['CrimeCount_Past2Weeks'] = crime_temporal_df['CrimeCount_PastMonth'] * 0.5

# 2. Placeholder for Stop-and-Search Features
# Simulate stop-and-search counts (random integer between 0-5 per ward per month)
np.random.seed(42)  # For reproducibility
crime_temporal_df['StopAndSearchCount_Past2Weeks'] = np.random.randint(0, 6, size=len(crime_temporal_df))

# 3. Placeholder for Unsafe Reports
# Simulate unsafe report counts (random integer between 0-3 per ward per month)
crime_temporal_df['UnsafeReports_Past2Weeks'] = np.random.randint(0, 4, size=len(crime_temporal_df))

# 4. Placeholder for POI Proximity Features
# Simulated values assuming future integration
crime_temporal_df['Num_BusStops_300m'] = np.random.randint(5, 20, size=len(crime_temporal_df))
crime_temporal_df['Num_TubeStations_300m'] = np.random.randint(1, 10, size=len(crime_temporal_df))
crime_temporal_df['Num_MarketKiosks_300m'] = np.random.randint(0, 5, size=len(crime_temporal_df))
crime_temporal_df['Num_HMO_Licenses_300m'] = np.random.randint(0, 7, size=len(crime_temporal_df))

# 5. Placeholder for Population Features
# Simulated population data (to be replaced with actual demographics later)
crime_temporal_df['TotalPopulation'] = np.random.randint(1000, 5000, size=len(crime_temporal_df))
crime_temporal_df['YoungPopulation'] = (crime_temporal_df['TotalPopulation'] * np.random.uniform(0.2, 0.5, size=len(crime_temporal_df))).astype(int)

# Final Preview
print("\nFeature Engineered DataFrame (Head):")
print(crime_temporal_df.head())

# Save feature-engineered dataframe for next steps
crime_temporal_df.to_csv("C:/Users/LLR User/Desktop/masters/Data Science Project/feature_engineered_MPS Ward Level Crime.csv", index=False)

print("\n✅ Feature Engineering Complete. Feature-enhanced file saved as 'feature_engineered_MPS Ward Level Crime.csv'.")

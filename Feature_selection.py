# Here we are performing Feature Selection - 


"""
Objective:
- Identify the most informative features for predicting crime outcomes
- Improve model performance and generalizability
- Reduce noise and prevent overfitting

Approach:
1. Univariate Statistical Filter (SelectKBest with f_regression)
2. Model-Based Importance (RandomForestRegressor)
3. Multicollinearity Check (Variance Inflation Factor - VIF)
4. Final selection: intersection of high-importance, low-collinearity features
"""

# 1. Load Feature-Engineered Dataset
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder

# Load data
crime_df = pd.read_csv("feature_engineered_MPS Ward Level Crime.csv")

# 2. Encode Categorical Columns for Selection Process
label_encoders = {}
categorical_cols = ['WardName', 'MajorText', 'MinorText']
for col in categorical_cols:
    le = LabelEncoder()
    crime_df[col] = le.fit_transform(crime_df[col])
    label_encoders[col] = le

# 3. Prepare Features and Target for Selection
X = crime_df.drop(columns=['CrimeCount'])  # Features
y = crime_df['CrimeCount']               # Target

# 4. Univariate Feature Selection
selector = SelectKBest(score_func=f_regression, k='all')
X_selected = selector.fit_transform(X, y)
feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'F_score': selector.scores_,
    'p_value': selector.pvalues_
}).sort_values(by='F_score', ascending=False)

print("\nTop Features by Univariate F-Score:")
print(feature_scores.head(10))

# 5. Model-Based Feature Importance (Random Forest)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)
rf_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop Features by Random Forest Importance:")
print(rf_importances.head(10))

# 6. Multicollinearity Check (VIF)
# Exclude constant column and target
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\nVariance Inflation Factor (VIF) Analysis:")
print(vif_data.sort_values(by='VIF', ascending=True).head(10))

# 7. Save Top Features
# Optional: Keep top N features from intersection of top univariate & RF importance
top_univariate = feature_scores.nlargest(15, 'F_score')['Feature']
top_modelbased = rf_importances.nlargest(15, 'Importance')['Feature']
selected_features = list(set(top_univariate) & set(top_modelbased))
print("\nFinal Selected Features (intersection of both methods):")
print(selected_features)

# Subset and save for modeling
crime_selected_df = crime_df[selected_features + ['CrimeCount']]
crime_selected_df.to_csv("C:/Users/LLR User/Desktop/masters/Data Science Project/feature_selected_MPS Ward Level Crime.csv", index=False)

print("\n✅ Feature Selection Complete. Saved as 'selected_features_crime_camden.csv'.")



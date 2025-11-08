import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import joblib 
import json

# --- 1. Load the Dataset ---
file_path = '/mnt/khome/requiem/Documents/programs/AI&ML/assignments/Fifa2026WinnerPredicter/Code&Resources v6/backenddata/fifaworldcup_with_win_rates.csv'
df = pd.read_csv(file_path)

# --- 2. Feature Engineering ---
df['Win_Rate_Difference'] = df['Home Team Win Rate'] - df['Away Team Win Rate']
df['Is_Home_Advantage'] = (df['Home Team Name'] == df['Country Name']).astype(int)

# --- 3. *** NEW: Add Sample Weights Based on Year *** ---
# Convert Match Date to datetime to extract the year
df['Match Date'] = pd.to_datetime(df['Match Date'])
df['Year'] = df['Match Date'].dt.year

# Create a weight mapping. Recent matches are more important.
year_weights = {
    2022: 1.0,
    2018: 0.8,
    2014: 0.6,
    2010: 0.4,
    2006: 0.2
}
# Map the weights, using a default for any other years (e.g., 0.1)
df['sample_weight'] = df['Year'].map(year_weights).fillna(0.1)

print("Sample weights generated based on match year.")

# --- 4. Define Features, Target, and Weights ---
numeric_features = ['Win_Rate_Difference', 'Knockout Stage', 'Is_Home_Advantage']
categorical_features = ['Home Team Name', 'Away Team Name']
target = 'Result'

X = df[numeric_features + categorical_features]
y = df[target]
weights = df['sample_weight'] # Get the weights Series

# --- 5. Encode the Target Variable (y) ---
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# --- 6. Split the Data (Including Weights) ---
# We split X, y, and weights together to keep them aligned
X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
    X, y_encoded, weights, test_size=0.2, random_state=42
)

# --- 7. Define Preprocessor (same as before) ---
numeric_transformer = 'passthrough'
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' 
)

# --- 8. Fit Preprocessor (same as before) ---
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# --- 9. Train the Random Forest Model (*** NOW WITH WEIGHTS ***) ---
# We will call this v4
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Pass the weights_train to the .fit() method
rf_classifier.fit(X_train_transformed, y_train, sample_weight=weights_train)
print("Model trained successfully using sample weights.")

# --- 10. Save the new v4 models ---
# We save as v4 to not overwrite your old model
new_model_path = '/mnt/khome/requiem/Documents/programs/AI&ML/assignments/Fifa2026WinnerPredicter/Code&Resources v6/backenddata/rf_classifier_v4.joblib'
new_preprocessor_path = '/mnt/khome/requiem/Documents/programs/AI&ML/assignments/Fifa2026WinnerPredicter/Code&Resources v6/backenddata/preprocessor_v4.joblib'
new_label_encoder_path = '/mnt/khome/requiem/Documents/programs/AI&ML/assignments/Fifa2026WinnerPredicter/Code&Resources v6/backenddata/label_encoder_v4.joblib'

joblib.dump(rf_classifier, new_model_path)
joblib.dump(preprocessor, new_preprocessor_path)
joblib.dump(le, new_label_encoder_path)

print(f"New weighted model saved as: {new_model_path}")
print(f"New preprocessor saved as: {new_preprocessor_path}")
print(f"New label encoder saved as: {new_label_encoder_path}")

# --- 11. Generate team_win_rate_lookup.json (same as before) ---
# This logic doesn't need to change
home_teams = df[['Home Team Name', 'Home Team Win Rate', 'Match Date']].copy()
home_teams.columns = ['Team', 'WinRate', 'Date']
away_teams = df[['Away Team Name', 'Away Team Win Rate', 'Match Date']].copy()
away_teams.columns = ['Team', 'WinRate', 'Date']

all_team_stats = pd.concat([home_teams, away_teams])
all_team_stats = all_team_stats.sort_values(by='Date', ascending=False)
master_win_rates_df = all_team_stats.drop_duplicates(subset='Team', keep='first')
master_win_rates = pd.Series(master_win_rates_df.WinRate.values, index=master_win_rates_df.Team).to_dict()

lookup_path = '/mnt/khome/requiem/Documents/programs/AI&ML/assignments/Fifa2026WinnerPredicter/Code&Resources v6/backenddata/team_win_rate_lookup.json'
with open(lookup_path, 'w') as f:
    json.dump(master_win_rates, f, indent=4)

print(f"Team win rate lookup saved to: {lookup_path}")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import joblib # Import joblib for saving/loading models

# --- 1. Load the Dataset ---
file_path = '/mnt/khome/requiem/Documents/programs/ai&ml precommit/assignment2/FINAL/fifaworldcup_with_win_rates.csv'
df = pd.read_csv(file_path)

# --- 2. Feature Engineering ---
df['Win_Rate_Difference'] = df['Home Team Win Rate'] - df['Away Team Win Rate']
df['Is_Home_Advantage'] = (df['Home Team Name'] == df['Country Name']).astype(int)

# --- 3. Define Features and Target ---
numeric_features = ['Win_Rate_Difference', 'Knockout Stage', 'Is_Home_Advantage']
categorical_features = ['Home Team Name', 'Away Team Name']
target = 'Result'

X = df[numeric_features + categorical_features]
y = df[target]

# --- 4. Encode the Target Variable (y) ---
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# --- 5. Split the Data (even though we're training on all of it for final save) ---
# For robust training, you might train on the full dataset here if appropriate.
# For consistency with previous steps, we'll keep the split logic for now.
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# --- 6. Create Preprocessing Pipeline ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='drop'
)

# --- 7. Apply Transformations (on training data for fitting) ---
X_train_transformed = preprocessor.fit_transform(X_train)

# --- 8. Train the Random Forest Model (v3) ---
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_transformed, y_train)

# --- 9. Save the trained model, preprocessor, and LabelEncoder ---
joblib.dump(rf_classifier, 'rf_classifier_v3.joblib')
joblib.dump(preprocessor, 'preprocessor_v3.joblib')
joblib.dump(le, 'label_encoder_v3.joblib')

print("Model (rf_classifier_v3.joblib), preprocessor (preprocessor_v3.joblib), "
      "and label encoder (label_encoder_v3.joblib) saved successfully.")

# Also, let's ensure the team_win_rate_lookup.json is generated
import json
df['Match Date'] = pd.to_datetime(df['Match Date']) # Ensure date is datetime type

home_teams = df[['Home Team Name', 'Home Team Win Rate', 'Match Date']].copy()
home_teams.columns = ['Team', 'WinRate', 'Date']
away_teams = df[['Away Team Name', 'Away Team Win Rate', 'Match Date']].copy()
away_teams.columns = ['Team', 'WinRate', 'Date']
all_team_stats = pd.concat([home_teams, away_teams])
all_team_stats = all_team_stats.sort_values(by='Date', ascending=False)
master_win_rates_df = all_team_stats.drop_duplicates(subset='Team', keep='first')
team_win_rate_lookup = master_win_rates_df.set_index('Team')['WinRate'].to_dict()

output_file = 'team_win_rate_lookup.json'
with open(output_file, 'w') as f:
    json.dump(team_win_rate_lookup, f, indent=4)
print(f"Master stats lookup saved to '{output_file}'")

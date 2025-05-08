import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

TRAIN_DATA_PATH = "smai/data/merged_normalized_data.csv"
OUTPUT_DIR = "smai/models"
IS_SET_DUMP_WEIGHTS = True

# Load and preprocess data
df = pd.read_csv(TRAIN_DATA_PATH)
drop_cols = ['Race', 'Overtaker', 'Overtaken', 'Turn', 'Session', 'Lap', 'Position',
             'Year', 'OvertakerNumber', 'OvertakenNumber', 'X', 'Y']
df.drop(columns=drop_cols, inplace=True)
df.dropna(inplace=True)

X = df.drop("OvertakeHappened", axis=1)
y = df["OvertakeHappened"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train XGBoost
xgb = XGBClassifier(eval_metric='logloss')
xgb.fit(X_train, y_train)

# Save model and scaler
if IS_SET_DUMP_WEIGHTS:
    os.makedirs("smai/models/", exist_ok=True)
    joblib.dump(xgb, f'{OUTPUT_DIR}/xgboost.pkl')
    joblib.dump(scaler, f'{OUTPUT_DIR}/scaler.pkl')

print(f"XGBoost model and scaler saved to {OUTPUT_DIR}")

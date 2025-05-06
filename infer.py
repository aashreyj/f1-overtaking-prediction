import joblib
import pandas as pd
import os

from google.colab import drive
drive.mount('/content/drive')

os.makedirs("/content/drive/MyDrive/F1/models/", exist_ok=True)

# Load model and scaler

xgb = joblib.load('/content/drive/MyDrive/F1/models/xgboost.pkl')
scaler = joblib.load('/content/drive/MyDrive/F1/models/scaler.pkl')

# Full list of features used for prediction (including NormalizedPosition)
feature_columns = [
    "NormalizedPosition", "CornerAngle", "AverageCornerSpeed", "SpeedDelta",
    "DistanceDelta", "isOvertakerSoft", "isOvertakenSoft",
    "isOvertakerMedium", "isOvertakenMedium", "isOvertakerHard",
    "isOvertakenHard", "isOvertakerWet", "isOvertakenWet",
    "isOvertakerIntermediate", "isOvertakenIntermediate",
    "OvertakerTyreLife", "OvertakenTyreLife", "IsOvertakerFreshTyre",
    "IsOvertakenFreshTyre"
]

# Load CSV
input_file = "/content/overtake_test_data.csv"
df = pd.read_csv(input_file)

# Select only the necessary feature columns
X = df[feature_columns]

# Normalize inputs
X_scaled = scaler.transform(X)

# Make predictions
predictions = xgb.predict(X_scaled)

# Print human-readable predictions
for idx, pred in enumerate(predictions):
    message = "Yes, overtake happened" if pred == 1 else "No, overtake did not happen"
    print(f"Row {idx + 1}: {message}")

import joblib
import pandas as pd

INPUT_FILE_PATH = "smai/data/overtake_test_data.csv"
MODEL_WEIGHTS_PATH = "smai/models/xgboost.pkl"
SCALER_WEIGHTS_PATH = "smai/models/scaler.pkl"

# Load model and scaler

xgb = joblib.load(MODEL_WEIGHTS_PATH)
scaler = joblib.load(SCALER_WEIGHTS_PATH)

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
df = pd.read_csv(INPUT_FILE_PATH)

# Select only the necessary feature columns
X = df[feature_columns]

# Normalize inputs
X_scaled = scaler.transform(X)

# Make predictions
predictions = xgb.predict(X_scaled)

# Print human-readable predictions
for idx, pred in enumerate(predictions):
    message = "Overtake happened!" if pred == 1 else "Overtake did not happen..."
    print(f"Row {idx + 1}: {message}")

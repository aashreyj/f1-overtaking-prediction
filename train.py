import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import os

from google.colab import drive
drive.mount('/content/drive')

os.makedirs("/content/drive/MyDrive/F1/models/", exist_ok=True)

# Load and preprocess data
df = pd.read_csv("/content/merged_normalized_data.csv")
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
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)

# Save model and scaler
joblib.dump(xgb, '/content/drive/MyDrive/F1/models/xgboost.pkl')
joblib.dump(scaler, '/content/drive/MyDrive/F1/models/scaler.pkl')

print("XGBoost model and scaler saved to 'models/'")

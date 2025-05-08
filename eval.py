import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_FILE_PATH = "smai/data/merged_normalized_data.csv"
MODEL_WEIGHTS_PATH = "smai/models/xgboost.pkl"
SCALER_WEIGHTS_PATH = "smai/models/scaler.pkl"

# Load data
df = pd.read_csv(DATA_FILE_PATH)
drop_cols = ['Race', 'Overtaker', 'Overtaken', 'Turn', 'Session', 'Lap', 'Position',
             'Year', 'OvertakerNumber', 'OvertakenNumber', 'X', 'Y']
df.drop(columns=drop_cols, inplace=True)
df.dropna(inplace=True)

X = df.drop("OvertakeHappened", axis=1)
y = df["OvertakeHappened"]

scaler = joblib.load(SCALER_WEIGHTS_PATH)

X_scaled = scaler.transform(X)

_, X_test, _, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Load model
xgb = joblib.load(MODEL_WEIGHTS_PATH)

# Evaluate
y_pred = xgb.predict(X_test)

print("--- XGBoost Evaluation ---")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix: XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

import pandas as pd
import joblib
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 📌 Define File Paths
BASE_DIR = "C:/Users/adhiy/OneDrive/Docs/Fraud/src/backend/data"
PROCESSED_DATA_FILE = os.path.join(BASE_DIR, "processed_data.csv")

MODEL_DIR = "C:/Users/adhiy/OneDrive/Docs/Fraud/src/backend/models"
MODEL_FILE = os.path.join(MODEL_DIR, "xgboost_fraud_model.pkl")

# 📌 Ensure Model Directory Exists
os.makedirs(MODEL_DIR, exist_ok=True)

# 📌 Load Preprocessed Data
df = pd.read_csv(PROCESSED_DATA_FILE)

# ✅ Define Features & Target
X = df.drop(columns=["fraud"])  # Features (excluding fraud label)
y = df["fraud"]  # Target variable

# ✅ Split Data into Training & Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Initialize & Train XGBoost Model
model = xgb.XGBClassifier(
    n_estimators=300,       # 🔥 Increased iterations for better learning
    learning_rate=0.05,     # 🔥 Lowered to improve generalization
    max_depth=6,
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    verbosity=0,            # ✅ Suppresses warnings/logging
    random_state=42
)

model.fit(X_train, y_train)

# ✅ Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n✅ XGBoost Model Training Completed!")
print(f"🎯 Accuracy: {accuracy:.4f}\n")
print(classification_report(y_test, y_pred))

# ✅ Save Model
joblib.dump(model, MODEL_FILE)
print(f"✅ XGBoost Model Saved Successfully at: {MODEL_FILE}")

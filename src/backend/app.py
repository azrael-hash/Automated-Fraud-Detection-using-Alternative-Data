import os
import joblib
import pandas as pd
import xgboost as xgb
from flask import Flask, request, jsonify
from flask_cors import CORS

# 📌 Dynamically Set the Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "models", "xgboost_fraud_model.pkl")

# ✅ Load the Trained Model
try:
    model = joblib.load(MODEL_FILE)
    print("✅ Model Loaded Successfully!")
except FileNotFoundError:
    print("❌ Error: Model file not found! Ensure 'xgboost_fraud_model.pkl' exists.")
    exit(1)

# ✅ Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

@app.route("/")
def home():
    return jsonify({"message": "🚀 Fraud Detection API is running!"})

# ✅ API Endpoint for Fraud Prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 📌 Get JSON data from the request
        data = request.get_json()
        
        if not isinstance(data, list):  # Ensure the input is a list (for batch support)
            data = [data]
        
        df_input = pd.DataFrame(data)  # Convert input to DataFrame
        
        # ✅ Ensure all expected features exist (handle missing or extra columns)
        expected_features = model.get_booster().feature_names
        df_input = df_input.reindex(columns=expected_features, fill_value=0)

        # ✅ Make Predictions
        predictions = model.predict(df_input).tolist()
        fraud_probs = model.predict_proba(df_input)[:, 1].tolist()  # Probabilities

        return jsonify([
            {"fraud_prediction": int(pred), "fraud_probability": round(prob, 4)}
            for pred, prob in zip(predictions, fraud_probs)
        ])

    except Exception as e:
        print(f"❌ Error: {str(e)}")  # Log the error
        return jsonify({"error": str(e)}), 400

# ✅ Run Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

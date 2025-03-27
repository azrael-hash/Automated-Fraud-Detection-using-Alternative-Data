import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import os

# 📌 Define File Paths
BASE_DIR = "C:/Users/adhiy/OneDrive/Docs/Fraud/src/backend/data"
RAW_TRANSACTIONS_FILE = os.path.join(BASE_DIR, "raw_transactions.csv")
REDDIT_POSTS_FILE = os.path.join(BASE_DIR, "reddit_posts.csv")
PROCESSED_DATA_FILE = os.path.join(BASE_DIR, "processed_data.csv")
SCALER_FILE = os.path.join(BASE_DIR, "scaler.pkl")

# 📌 Load Datasets
df_trans = pd.read_csv(RAW_TRANSACTIONS_FILE)
df_reddit = pd.read_csv(REDDIT_POSTS_FILE)

# ✅ Feature Engineering for transactions
df_trans["amount_change_rate"] = df_trans["ratio_to_median_purchase_price"].pct_change().fillna(0)

# ✅ Normalize Sentiment Score
df_reddit["Sentiment_Score"] = (df_reddit["Sentiment_Score"] - df_reddit["Sentiment_Score"].min()) / \
                               (df_reddit["Sentiment_Score"].max() - df_reddit["Sentiment_Score"].min())

# ✅ Merge Reddit Sentiment with Transactions
df_trans["Fraud_Risk_Score"] = df_reddit["Sentiment_Score"].mean()

# ✅ Scale Numeric Features
features = ["distance_from_home", "distance_from_last_transaction", "ratio_to_median_purchase_price",
            "repeat_retailer", "used_chip", "used_pin_number", "online_order",
            "amount_change_rate", "Fraud_Risk_Score"]

scaler = StandardScaler()
df_trans[features] = scaler.fit_transform(df_trans[features])

# ✅ Save Processed Data
df_trans.to_csv(PROCESSED_DATA_FILE, index=False)
joblib.dump(scaler, SCALER_FILE)

print("✅ Data Preprocessing Completed Successfully!")

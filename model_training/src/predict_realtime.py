import os
import requests
import pandas as pd
import joblib

GO_API_URL = "http://127.0.0.1:8080/features"
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "lgbm_model_fold0.pkl"))

def get_realtime_features(user_id):
    try:
		# Call Go API to fetch behavioral features from Redis
        response = requests.get(GO_API_URL, params={"user_id": user_id})
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch features for User {user_id}: {response.text}")
            return None
    except Exception as e:
        print(f"Error connecting to Go API: {e}")
        return None

def main():
    user_id = 2
    print(f"Extracting real-time features for User ID: {user_id} via Go API...")
    
    behavioral_features = get_realtime_features(user_id)
    if not behavioral_features:
        return
        
    # Convert JSON data from Go into LightGBM compatible format
    feat_data = {k: [float(v)] for k, v in behavioral_features.items()}
    
    # Add simulated real-time profile/magnitude data
    # In production, this would come from the current user application input payload
    feat_data['limit_bal'] = [120000.0]
    feat_data['sex'] = [2]
    feat_data['education'] = [2]
    feat_data['marriage'] = [2]
    feat_data['age'] = [26]
    feat_data['mean_bill_amt'] = [2350.0]
    feat_data['max_bill_amt'] = [3272.0]
    feat_data['mean_pay_amt'] = [1000.0]
    feat_data['sum_pay_amt'] = [5000.0]
    
    df_inference = pd.DataFrame(feat_data)
    
    clf = joblib.load(MODEL_PATH)
    
    # Ensure column order matches exactly with training data
    model_features = clf.feature_name_
    df_inference = df_inference[model_features]
    
    # Calculate default probability
    probability = clf.predict_proba(df_inference)[0][1]
    
    print("-" * 50)
    print(f"Credit Decision Result for User ID {user_id}:")
    print(f"Default Probability: {probability:.4f} ({probability * 100:.2f}%)")
    
    if probability > 0.4:
        print("Status: LOAN REJECTED (High Risk)")
    else:
        print("Status: LOAN APPROVED (Safe Risk)")
    print("-" * 50)

if __name__ == "__main__":
    main()
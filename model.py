import pandas as pd
import joblib
import numpy as np
import sys

# List of features from your training set
FEATURE_NAMES = [
    "VIDEO", "BLUETOOTH_INFORMATION", "CALENDAR_INFORMATION", "SMS_MMS", "ACCOUNT_INFORMATION",
    "EMAIL_INFORMATION", "FILE_INFORMATION", "SYNCHRONIZATION_DATA", "PHONE_CONNECTION", "NETWORK",
    "AUDIO", "IMAGE", "ACCOUNT_SETTINGS", "VOIP", "FILE", "DATABASE_INFORMATION",
    "NETWORK_INFORMATION", "HARDWARE_INFO", "NFC", "SYSTEM_SETTINGS", "CONTACT_INFORMATION",
    "VOIP_INFORMATION", "PHONE_INFORMATION", "EMAIL_SETTINGS", "PHONE_STATE", "BROWSER_INFORMATION",
    "NO_CATEGORY", "LOG", "BLUETOOTH", "UNIQUE_IDENTIFIER", "INTER_APP_COMMUNICATION",
    "EMAIL", "ALL", "LOCATION_INFORMATION"
]

def parse_inconsistent_src_txt(src_txt_path, feature_names):
    """
    Parses a `src.txt` where each line is: `FEATURE_NAME val1 val2 val3 ...`
    Extracts a single numeric summary (sum) per feature.

    Args:
        src_txt_path (str): Path to your src.txt
        feature_names (list): List of features to consider.

    Returns:
        pd.DataFrame: DataFrame with one row containing feature values aligned to feature_names, or None if error.
    """
    feature_dict = {feat: 0 for feat in feature_names}

    try:
        with open(src_txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                feat_name = parts[0]
                if feat_name not in feature_dict:
                    print(f"Warning: Unknown feature '{feat_name}' ignored.")
                    continue
                # Convert the rest to numbers safely
                try:
                    values = [float(x) for x in parts[1:] if x.strip() != '']
                    feature_dict[feat_name] = float(np.sum(values)) if values else 0.0
                except ValueError as e:
                    print(f"Warning: Could not parse values for '{feat_name}': {e}. Setting to 0.")
                    feature_dict[feat_name] = 0.0
    except Exception as e:
        print(f"Error parsing src.txt at '{src_txt_path}': {e}")
        return None

    # Create DataFrame for model input
    feature_df = pd.DataFrame([feature_dict])
    # Ensure no NaN values
    feature_df = feature_df.fillna(0)
    return feature_df


MODEL_PATH = 'path to your saved model' 
SCALER_PATH = 'path to your scalar'         

SRC_TXT_PATH = input("Enter the full path to src.txt: ").strip()

# --- PREDICTION CODE ---
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    sys.exit(1)

# Parse features
fv = parse_inconsistent_src_txt(SRC_TXT_PATH, FEATURE_NAMES)


if fv is None or fv.empty:
    print("Error: Feature vector is None or empty. Check src.txt file and parsing logic.")
    sys.exit(1)


print("Parsed Features:", fv)

if fv.shape[1] != len(FEATURE_NAMES):
    print(f"Error: Expected {len(FEATURE_NAMES)} features, got {fv.shape[1]}.")
    sys.exit(1)

# Scale and predict
try:
    fv_scaled = scaler.transform(fv)
    pred = model.predict(fv_scaled)[0]
    proba = model.predict_proba(fv_scaled)[0][1]  # Probability of malware
    print(f"Prediction: {'Malicious' if pred == 1 else 'Benign'}")
    print(f"Malicious probability: {proba:.4f}")
except Exception as e:
    print(f"Error during scaling or prediction: {e}")

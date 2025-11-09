import joblib
import numpy as np
from pathlib import Path

# Load model and encoders
BASE_DIR = Path(__file__).parent / "backend"
model = joblib.load(BASE_DIR / "model.joblib")
encoders = joblib.load(BASE_DIR / "encoders.joblib")

# Test your extreme case
test_case = {
    "Age": 78,
    "Gender": "Male",  # assuming
    "Blood Pressure": 325,
    "Cholesterol Level": 425,
    "BMI": 30,  # assuming
    "Smoking": "Yes"  # assuming worst case
}

print(f"Testing extreme case: {test_case}")

# Process features
features = []
for feature_name in ['Age', 'Gender', 'Blood Pressure', 'Cholesterol Level', 'BMI', 'Smoking']:
    value = test_case[feature_name]
    if feature_name in encoders:
        encoded_value = encoders[feature_name].transform([value])[0]
        features.append(encoded_value)
    else:
        features.append(float(value))

print(f"Processed features: {features}")

# Predict
prediction = model.predict([features])[0]
probability = model.predict_proba([features])[0]

print(f"Prediction: {prediction} ({'High Risk' if prediction == 1 else 'Low Risk'})")
print(f"Disease probability: {probability[1]:.4f} ({probability[1]*100:.1f}%)")

# Check model's training range
print(f"\nModel was trained on data with ranges that might not handle extreme values properly.")
print("Need to retrain with better extreme value handling.")
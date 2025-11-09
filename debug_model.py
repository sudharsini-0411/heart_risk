import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# Load the trained model and encoders
BASE_DIR = Path(__file__).parent / "backend"
MODEL_PATH = BASE_DIR / "model.joblib"
ENCODER_PATH = BASE_DIR / "encoders.joblib"
DATA_PATH = BASE_DIR / "heart.csv"

print("Loading model and encoders...")
model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODER_PATH)

print("Model loaded successfully!")
print(f"Model type: {type(model)}")
print(f"Model features expected: {model.n_features_in_}")

print("\nEncoders available:")
for key, encoder in encoders.items():
    print(f"  {key}: {encoder.classes_}")

# Load original data to check distribution
df = pd.read_csv(DATA_PATH)
print(f"\nOriginal dataset shape: {df.shape}")
print(f"Target distribution:\n{df['target'].value_counts()}")

# Test with some sample data
test_cases = [
    # Low risk case
    {"Age": 30, "Gender": "Female", "Blood Pressure": 110, "Cholesterol Level": 180, "BMI": 22, "Smoking": "No"},
    # Medium risk case  
    {"Age": 45, "Gender": "Male", "Blood Pressure": 130, "Cholesterol Level": 220, "BMI": 26, "Smoking": "No"},
    # High risk case
    {"Age": 65, "Gender": "Male", "Blood Pressure": 160, "Cholesterol Level": 280, "BMI": 32, "Smoking": "Yes"},
]

print("\n" + "="*50)
print("TESTING PREDICTIONS")
print("="*50)

for i, test_case in enumerate(test_cases):
    print(f"\nTest Case {i+1}: {test_case}")
    
    # Prepare features exactly like in app.py
    expected_feature_names = ['Age', 'Gender', 'Blood Pressure', 'Cholesterol Level', 'BMI', 'Smoking']
    features = []
    
    for feature_name in expected_feature_names:
        value = test_case.get(feature_name)
        if value is None:
            features.append(0.0)
        elif feature_name in encoders:
            le = encoders[feature_name]
            if value not in le.classes_:
                le.classes_ = np.append(le.classes_, value)
            encoded_value = le.transform([value])[0]
            features.append(encoded_value)
        else:
            features.append(float(value))
    
    # Ensure correct feature length
    while len(features) < model.n_features_in_:
        features.append(0.0)
    
    print(f"Processed features: {features}")
    
    # Make prediction
    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0]
    
    print(f"Prediction: {prediction} ({'High Risk' if prediction == 1 else 'Low Risk'})")
    print(f"Probabilities: [No Disease: {probability[0]:.4f}, Disease: {probability[1]:.4f}]")
    print(f"Disease probability: {probability[1]:.4f} ({probability[1]*100:.2f}%)")
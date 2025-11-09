import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import joblib

# Paths
BASE_DIR = Path(__file__).parent / "backend"
MODEL_PATH = BASE_DIR / "model.joblib"
ENCODER_PATH = BASE_DIR / "encoders.joblib"

print("[INFO] Creating realistic synthetic heart disease dataset...")

# Create a more realistic synthetic dataset
np.random.seed(42)
n_samples = 2000

# Generate age (20-80)
age = np.random.randint(20, 81, n_samples)

# Generate gender (roughly 50-50 split)
gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.55, 0.45])

# Generate blood pressure (correlated with age and gender)
bp_base = 90 + age * 0.8 + (gender == 'Male') * 10
bp_noise = np.random.normal(0, 15, n_samples)
blood_pressure = (bp_base + bp_noise).clip(80, 200)

# Generate cholesterol (correlated with age and BP)
chol_base = 150 + age * 1.2 + blood_pressure * 0.3
chol_noise = np.random.normal(0, 30, n_samples)
cholesterol = (chol_base + chol_noise).clip(120, 400)

# Generate BMI (slightly correlated with age)
bmi_base = 22 + (age - 40) * 0.05
bmi_noise = np.random.normal(0, 4, n_samples)
bmi = (bmi_base + bmi_noise).clip(16, 45)

# Generate smoking (higher for males and middle-aged)
smoking_prob = 0.15 + (gender == 'Male') * 0.15 + ((age > 30) & (age < 60)) * 0.1
smoking = np.random.binomial(1, smoking_prob, n_samples)
smoking_str = ['No' if x == 0 else 'Yes' for x in smoking]

# Create realistic heart disease risk based on medical knowledge
risk_score = (
    (age - 20) * 0.02 +  # Age factor
    (gender == 'Male') * 0.3 +  # Male factor
    np.maximum(0, (blood_pressure - 120) * 0.01) +  # High BP factor
    np.maximum(0, (cholesterol - 200) * 0.002) +  # High cholesterol factor
    np.maximum(0, (bmi - 25) * 0.05) +  # Overweight factor
    smoking * 0.4 +  # Smoking factor
    np.random.normal(0, 0.2, n_samples)  # Random noise
)

# Convert risk score to binary outcome (sigmoid-like function)
heart_disease_prob = 1 / (1 + np.exp(-2 * (risk_score - 1.2)))
heart_disease = np.random.binomial(1, heart_disease_prob, n_samples)

# Create DataFrame
df = pd.DataFrame({
    'Age': age,
    'Gender': gender,
    'Blood Pressure': blood_pressure.round(0).astype(int),
    'Cholesterol Level': cholesterol.round(0).astype(int),
    'BMI': bmi.round(1),
    'Smoking': smoking_str,
    'Heart Disease Status': heart_disease
})

print(f"Dataset created with {n_samples} samples")
print(f"Heart disease prevalence: {df['Heart Disease Status'].mean()*100:.1f}%")

# Show some statistics
print(f"\nDataset statistics:")
print(df.describe())

print(f"\nHeart disease by risk factors:")
print(f"By Gender: \n{df.groupby('Gender')['Heart Disease Status'].mean()}")
print(f"By Smoking: \n{df.groupby('Smoking')['Heart Disease Status'].mean()}")
print(f"By Age groups: \n{df.groupby(pd.cut(df['Age'], bins=[0, 40, 55, 70, 100]))['Heart Disease Status'].mean()}")

# Encode categorical variables
encoders = {}
for col in ['Gender', 'Smoking']:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    encoders[col] = le
    print(f"[ENCODED] {col}: {le.classes_}")

# Prepare features and target
feature_cols = ['Age', 'Gender_encoded', 'Blood Pressure', 'Cholesterol Level', 'BMI', 'Smoking_encoded']
X = df[feature_cols]
y = df['Heart Disease Status']

# Rename columns to match expected names
X.columns = ['Age', 'Gender', 'Blood Pressure', 'Cholesterol Level', 'BMI', 'Smoking']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
print("[INFO] Training model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# Evaluate model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"[INFO] Training accuracy: {train_score:.3f}")
print(f"[INFO] Test accuracy: {test_score:.3f}")

# Detailed evaluation
y_pred = model.predict(X_test)
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nFeature Importance:")
print(feature_importance)

# Save model and encoders
joblib.dump(model, MODEL_PATH)
joblib.dump(encoders, ENCODER_PATH)
print(f"[INFO] Model saved to {MODEL_PATH}")
print(f"[INFO] Encoders saved to {ENCODER_PATH}")

# Test with realistic cases
print("\n" + "="*60)
print("TESTING REALISTIC MODEL WITH VARIOUS RISK PROFILES")
print("="*60)

test_cases = [
    # Very low risk: young, female, excellent health
    {"Age": 25, "Gender": "Female", "Blood Pressure": 110, "Cholesterol Level": 170, "BMI": 21, "Smoking": "No", "Expected": "Low Risk"},
    
    # Low risk: young male, good health
    {"Age": 30, "Gender": "Male", "Blood Pressure": 120, "Cholesterol Level": 180, "BMI": 23, "Smoking": "No", "Expected": "Low Risk"},
    
    # Medium-low risk: middle-aged female, some risk factors
    {"Age": 45, "Gender": "Female", "Blood Pressure": 135, "Cholesterol Level": 210, "BMI": 26, "Smoking": "No", "Expected": "Low-Medium Risk"},
    
    # Medium risk: middle-aged male, multiple risk factors
    {"Age": 50, "Gender": "Male", "Blood Pressure": 145, "Cholesterol Level": 230, "BMI": 28, "Smoking": "No", "Expected": "Medium Risk"},
    
    # High risk: older male, multiple risk factors
    {"Age": 60, "Gender": "Male", "Blood Pressure": 160, "Cholesterol Level": 260, "BMI": 30, "Smoking": "Yes", "Expected": "High Risk"},
    
    # Very high risk: elderly male, severe risk factors
    {"Age": 70, "Gender": "Male", "Blood Pressure": 170, "Cholesterol Level": 300, "BMI": 32, "Smoking": "Yes", "Expected": "Very High Risk"},
]

for i, test_case in enumerate(test_cases):
    expected = test_case.pop("Expected")
    print(f"\nTest Case {i+1} ({expected}): {test_case}")
    
    # Prepare features
    features = []
    for feature_name in ['Age', 'Gender', 'Blood Pressure', 'Cholesterol Level', 'BMI', 'Smoking']:
        value = test_case[feature_name]
        if feature_name in encoders:
            le = encoders[feature_name]
            encoded_value = le.transform([value])[0]
            features.append(encoded_value)
        else:
            features.append(float(value))
    
    # Make prediction
    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0]
    
    risk_level = "High Risk" if prediction == 1 else "Low Risk"
    disease_prob = probability[1] * 100
    
    print(f"  -> Prediction: {risk_level}")
    print(f"  -> Disease probability: {disease_prob:.1f}%")
    
    # Color coding for probability
    if disease_prob < 20:
        risk_category = "Very Low"
    elif disease_prob < 40:
        risk_category = "Low"
    elif disease_prob < 60:
        risk_category = "Medium"
    elif disease_prob < 80:
        risk_category = "High"
    else:
        risk_category = "Very High"
    
    print(f"  -> Risk Category: {risk_category}")

print(f"\n[SUCCESS] Realistic heart disease prediction model created and saved!")
print(f"The model now provides more intuitive predictions based on medical risk factors.")
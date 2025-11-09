import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import joblib

# Create model that handles extreme values better
np.random.seed(42)
n_samples = 3000

# Generate more diverse data including extreme cases
age = np.concatenate([
    np.random.randint(20, 81, int(n_samples * 0.8)),  # Normal range
    np.random.randint(75, 90, int(n_samples * 0.2))   # Elderly
])

gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.55, 0.45])

# Blood pressure with extreme cases
bp_normal = np.random.normal(130, 20, int(n_samples * 0.7)).clip(90, 180)
bp_extreme = np.random.normal(200, 50, int(n_samples * 0.3)).clip(180, 350)
blood_pressure = np.concatenate([bp_normal, bp_extreme])

# Cholesterol with extreme cases  
chol_normal = np.random.normal(200, 40, int(n_samples * 0.7)).clip(150, 280)
chol_extreme = np.random.normal(350, 80, int(n_samples * 0.3)).clip(280, 500)
cholesterol = np.concatenate([chol_normal, chol_extreme])

bmi = np.random.normal(26, 5, n_samples).clip(18, 45)
smoking = np.random.choice(['No', 'Yes'], n_samples, p=[0.7, 0.3])

# Shuffle all arrays together
indices = np.random.permutation(n_samples)
age = age[indices]
gender = gender[indices]  
blood_pressure = blood_pressure[indices]
cholesterol = cholesterol[indices]
bmi = bmi[indices]
smoking = smoking[indices]

# Create risk score with stronger penalties for extreme values
risk_score = (
    np.maximum(0, (age - 30) * 0.03) +  # Age penalty
    (gender == 'Male') * 0.4 +  # Male penalty
    np.maximum(0, (blood_pressure - 120) * 0.008) +  # BP penalty
    np.maximum(0, (cholesterol - 200) * 0.003) +  # Cholesterol penalty
    np.maximum(0, (bmi - 25) * 0.06) +  # BMI penalty
    (smoking == 'Yes') * 0.5 +  # Smoking penalty
    # EXTREME VALUE PENALTIES
    np.where(blood_pressure > 180, (blood_pressure - 180) * 0.02, 0) +  # Severe hypertension
    np.where(cholesterol > 300, (cholesterol - 300) * 0.005, 0) +  # Severe hypercholesterolemia
    np.where(age > 70, (age - 70) * 0.05, 0) +  # Advanced age
    np.random.normal(0, 0.15, n_samples)  # Noise
)

# Convert to probability and binary outcome
heart_disease_prob = 1 / (1 + np.exp(-1.5 * (risk_score - 1.5)))
heart_disease = np.random.binomial(1, heart_disease_prob, n_samples)

# Create DataFrame
df = pd.DataFrame({
    'Age': age.astype(int),
    'Gender': gender,
    'Blood Pressure': blood_pressure.round(0).astype(int),
    'Cholesterol Level': cholesterol.round(0).astype(int),
    'BMI': bmi.round(1),
    'Smoking': smoking,
    'Heart Disease Status': heart_disease
})

# Encode categorical variables
encoders = {}
for col in ['Gender', 'Smoking']:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    encoders[col] = le

# Train model
X = df[['Age', 'Gender_encoded', 'Blood Pressure', 'Cholesterol Level', 'BMI', 'Smoking_encoded']]
X.columns = ['Age', 'Gender', 'Blood Pressure', 'Cholesterol Level', 'BMI', 'Smoking']
y = df['Heart Disease Status']

model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, class_weight='balanced')
model.fit(X, y)

# Save
BASE_DIR = Path(__file__).parent / "backend"
joblib.dump(model, BASE_DIR / "model.joblib")
joblib.dump(encoders, BASE_DIR / "encoders.joblib")

# Test extreme case
test_features = [78, 1, 325, 425, 30, 1]  # Your case
pred = model.predict([test_features])[0]
prob = model.predict_proba([test_features])[0][1]

print(f"Extreme case (Age:78, BP:325, Chol:425): {pred} ({prob*100:.1f}%)")
print("Model updated to handle extreme values better!")
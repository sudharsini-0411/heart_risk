import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import joblib
import numpy as np

# Paths
BASE_DIR = Path(__file__).parent / "backend"
DATA_PATH = BASE_DIR / "heart.csv"
MODEL_PATH = BASE_DIR / "model.joblib"
ENCODER_PATH = BASE_DIR / "encoders.joblib"

print("[INFO] Loading and analyzing dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Check target distribution
print(f"\nTarget distribution:")
print(df['target'].value_counts())
print(f"Percentage with heart disease: {df['target'].mean()*100:.1f}%")

# Rename columns to match expected feature names
df = df.rename(columns={
    "age": "Age",
    "sex": "Gender",           # 1 = Male, 0 = Female
    "trestbps": "Blood Pressure",
    "chol": "Cholesterol Level",
    "target": "Heart Disease Status"
})

# Convert Gender numeric to text
df["Gender"] = df["Gender"].apply(lambda x: "Male" if x == 1 else "Female")

# Add realistic BMI and Smoking columns
np.random.seed(42)
n_samples = len(df)

# Generate BMI based on age and other factors (more realistic)
base_bmi = np.random.normal(25, 4, n_samples)
# Older people tend to have slightly higher BMI
age_factor = (df["Age"] - 40) * 0.1
bmi = (base_bmi + age_factor).clip(18, 40)
df["BMI"] = bmi

# Generate smoking status (higher probability for older males)
smoking_prob = 0.15  # base probability
male_bonus = (df["Gender"] == "Male") * 0.1  # males more likely to smoke
age_bonus = (df["Age"] > 50) * 0.05  # older people more likely to smoke
final_prob = smoking_prob + male_bonus + age_bonus

smoking_status = np.random.binomial(1, final_prob, n_samples)
df["Smoking"] = ["Yes" if x == 1 else "No" for x in smoking_status]

# Select final columns
feature_cols = ["Age", "Gender", "Blood Pressure", "Cholesterol Level", "BMI", "Smoking"]
df = df[feature_cols + ["Heart Disease Status"]]

print(f"\nFinal dataset shape: {df.shape}")
print(f"Final columns: {list(df.columns)}")

# Check for missing values
print(f"\nMissing values:")
print(df.isnull().sum())

# Show some statistics
print(f"\nDataset statistics:")
print(df.describe())

# Encode categorical columns
encoders = {}
for col in ["Gender", "Smoking"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    print(f"[ENCODED] {col}: {le.classes_}")

# Features and target
X = df[feature_cols]
y = df["Heart Disease Status"]

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target distribution after processing:")
print(y.value_counts())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set: {X_train.shape}, Test set: {X_test.shape}")

# Train model with better parameters
print("[INFO] Training model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'  # Handle class imbalance
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

print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nFeature Importance:")
print(feature_importance)

# Save model and encoders
joblib.dump(model, MODEL_PATH)
joblib.dump(encoders, ENCODER_PATH)
print(f"[INFO] Model saved to {MODEL_PATH}")
print(f"[INFO] Encoders saved to {ENCODER_PATH}")

# Test with sample cases
print("\n" + "="*50)
print("TESTING NEW MODEL")
print("="*50)

test_cases = [
    # Low risk: young, female, good stats
    {"Age": 25, "Gender": "Female", "Blood Pressure": 110, "Cholesterol Level": 180, "BMI": 22, "Smoking": "No"},
    # Medium risk: middle-aged, some risk factors
    {"Age": 45, "Gender": "Male", "Blood Pressure": 140, "Cholesterol Level": 220, "BMI": 27, "Smoking": "No"},
    # High risk: older, male, multiple risk factors
    {"Age": 65, "Gender": "Male", "Blood Pressure": 160, "Cholesterol Level": 280, "BMI": 32, "Smoking": "Yes"},
]

for i, test_case in enumerate(test_cases):
    print(f"\nTest Case {i+1}: {test_case}")
    
    # Prepare features
    features = []
    for feature_name in feature_cols:
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
    
    print(f"Processed features: {features}")
    print(f"Prediction: {prediction} ({'High Risk' if prediction == 1 else 'Low Risk'})")
    print(f"Disease probability: {probability[1]:.4f} ({probability[1]*100:.2f}%)")
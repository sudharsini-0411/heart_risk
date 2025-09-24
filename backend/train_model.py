import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import joblib

# Paths
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "heart.csv"  # Updated to your dataset
MODEL_PATH = BASE_DIR / "model.joblib"
ENCODER_PATH = BASE_DIR / "encoders.joblib"

print("[INFO] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print("[INFO] Original Columns:", list(df.columns))

# Rename Kaggle columns to match your expected feature names
df = df.rename(columns={
    "age": "Age",
    "sex": "Gender",           # 1 = Male, 0 = Female
    "trestbps": "Blood Pressure",
    "chol": "Cholesterol Level",
    "target": "Heart Disease Status"
})

# Select only relevant columns
selected_cols = ["Age", "Gender", "Blood Pressure", "Cholesterol Level", "Heart Disease Status"]
df = df[selected_cols]

print("[INFO] Using columns:", list(df.columns))

# Convert Gender numeric to text for encoding
df["Gender"] = df["Gender"].apply(lambda x: "Male" if x == 1 else "Female")

# Add dummy Smoking column (since Kaggle dataset doesn't have it)
df["Smoking"] = "No"  # Default value for all rows
df = df[["Age", "Gender", "Blood Pressure", "Cholesterol Level", "Smoking", "Heart Disease Status"]]

# Encode categorical columns (Gender, Smoking)
encoders = {}
for col in df.select_dtypes(include="object").columns:
    if col != "Heart Disease Status":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"[ENCODED] {col}")

# Save encoders for future use
joblib.dump(encoders, ENCODER_PATH)
print(f"[INFO] Encoders saved at {ENCODER_PATH}")

# Handle missing values if any
df = df.fillna(df.median(numeric_only=True))
print(f"[INFO] Any NaN left? -> {df.isna().sum().sum()}")

# Features and target split
X = df.drop("Heart Disease Status", axis=1)
y = df["Heart Disease Status"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("[INFO] Training model...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, MODEL_PATH)
print(f"[INFO] Model saved at {MODEL_PATH}")
print(f"[INFO] Model accuracy: {model.score(X_test, y_test):.2f}")

from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import joblib
from pathlib import Path
import numpy as np

app = Flask(__name__, template_folder="frontend")
CORS(app)

# Paths
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model.joblib"
ENCODER_PATH = BASE_DIR / "encoders.joblib"

# Load model and encoders
print("[INFO] Loading model and encoders...")
model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODER_PATH)
print("[INFO] Model and encoders loaded successfully.")

@app.route("/")
def home():
    """Serve the main frontend page."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict heart disease risk based on user-provided features.
    Expects JSON with 'features' key containing the input data.
    Returns prediction result and probability.
    """
    try:
        data = request.get_json()
        if not data or "features" not in data:
            return jsonify({"error": "Missing features in request"}), 400

        features_data = data["features"]

        # Prepare features for prediction
        expected_features = model.n_features_in_
        features = []

        # Define the exact feature names expected by the model
        # Adjust this list to match your model's training features exactly
        expected_feature_names = ['Age', 'Gender', 'Blood Pressure', 'Cholesterol Level', 'BMI']

        for feature_name in expected_feature_names:
            value = features_data.get(feature_name)
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

        # Ensure correct feature length (pad with zeros if needed)
        while len(features) < expected_features:
            features.append(0.0)

        # Make prediction
        try:
            prediction = model.predict([features])[0]
            probability = model.predict_proba([features])[0][1]
            risk_result = "High Risk of Heart Disease 💔" if prediction == 1 else "Low Risk ❤️"
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        # Also get nearby hospitals if location provided in features_data
        location = features_data.get("Location", "")
        nearby_hospitals = get_hospitals_by_location(location)

        return jsonify({
            "prediction": risk_result,
            "probability": round(float(probability), 4),
            "nearby_hospitals": nearby_hospitals
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/hospitals", methods=["POST"])
def get_hospitals():
    """
    Get nearby hospitals based on user location.
    Expects JSON with 'location' key.
    Returns list of nearby hospitals.
    """
    try:
        data = request.get_json()
        location = data.get("location", "").lower() if data else ""

        if not location:
            return jsonify({"nearby_hospitals": ["City Hospital", "Community Health Center", "Global Care Clinic"]})

        # Simple location-based hospital suggestions (can be replaced with real API)
        if "new york" in location:
            hospitals = ["NY General Hospital", "Manhattan Health Center", "Brooklyn Medical"]
        elif "san francisco" in location:
            hospitals = ["SF Medical Center", "Bay Area Hospital", "Golden Gate Clinic"]
        elif "chennai" in location:
            hospitals = ["Apollo Hospital Chennai", "Fortis Malar Hospital", "MIOT International"]
        elif "kolkata" in location:
            hospitals = ["AMRI Hospitals", "Fortis Hospital Kolkata", "Apollo Gleneagles Hospital"]
        elif "coimbatore" in location:
            hospitals = ["PSG Hospitals", "KMCH", "G. Kuppuswamy Naidu Memorial Hospital"]
        else:
            hospitals = ["City Hospital", "Wellness Cardio Center", "Global Health Clinic"]

        return jsonify({"nearby_hospitals": hospitals})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def get_hospitals_by_location(location):
    """Return a list of hospitals based on location."""
    if not location:
        return ["City Hospital", "Community Health Center", "Global Care Clinic"]

    location = location.lower()
    if "new york" in location:
        return ["NY General Hospital", "Manhattan Health Center", "Brooklyn Medical"]
    elif "san francisco" in location:
        return ["SF Medical Center", "Bay Area Hospital", "Golden Gate Clinic"]
    elif "chennai" in location:
        return ["Apollo Hospital Chennai", "Fortis Malar Hospital", "MIOT International"]
    elif "kolkata" in location:
        return ["AMRI Hospitals", "Fortis Hospital Kolkata", "Apollo Gleneagles Hospital"]
    elif "coimbatore" in location:
        return ["PSG Hospitals", "KMCH", "G. Kuppuswamy Naidu Memorial Hospital"]
    else:
        return ["City Hospital", "Wellness Cardio Center", "Global Health Clinic"]

if __name__ == "__main__":
    app.run(debug=True)

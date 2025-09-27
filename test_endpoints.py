import requests
import json

# Test /predict endpoint
predict_data = {
    "features": {
        "Age": 50,
        "Gender": "Male",
        "Blood Pressure": 120,
        "Cholesterol Level": 200,
        "BMI": 25,
        "Smoking": "No"
    },
    "user_id": "test_user_123"
}

response = requests.post("http://localhost:5000/predict", json=predict_data)
print("Predict Response Status:", response.status_code)
print("Predict Response:", response.text)

# Test /history endpoint
history_data = {"user_id": "test_user_123"}
response = requests.post("http://localhost:5000/history", json=history_data)
print("History Response Status:", response.status_code)
print("History Response:", response.text)

import requests
import json

# Test the updated model with various risk profiles
test_cases = [
    {
        "name": "Low Risk - Young Female",
        "features": {
            "Age": 25,
            "Gender": "Female", 
            "Blood Pressure": 110,
            "Cholesterol Level": 170,
            "BMI": 21,
            "Smoking": "No"
        }
    },
    {
        "name": "Medium Risk - Middle-aged Male",
        "features": {
            "Age": 50,
            "Gender": "Male",
            "Blood Pressure": 145,
            "Cholesterol Level": 230,
            "BMI": 28,
            "Smoking": "No"
        }
    },
    {
        "name": "High Risk - Older Male with Risk Factors",
        "features": {
            "Age": 65,
            "Gender": "Male",
            "Blood Pressure": 160,
            "Cholesterol Level": 280,
            "BMI": 32,
            "Smoking": "Yes"
        }
    }
]

print("Testing Heart Disease Prediction API")
print("=" * 50)

for test_case in test_cases:
    print(f"\n{test_case['name']}:")
    print(f"Input: {test_case['features']}")
    
    try:
        response = requests.post(
            "http://localhost:5000/predict",
            json={"features": test_case['features']},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"Result: {data['prediction']}")
            print(f"Probability: {data['probability']:.4f} ({data['probability']*100:.1f}%)")
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server. Make sure the Flask app is running.")
    except Exception as e:
        print(f"Error: {e}")

print("\n" + "=" * 50)
print("To run the server, use: python backend/app.py")
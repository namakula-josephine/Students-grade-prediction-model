from flask import Flask, request, jsonify
import mlflow.pytorch
import torch
import os
import pandas as pd
from src.preprocessing import DataProcessor

# Set MLflow tracking URI
# os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
app = Flask(__name__)
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# Load the trained model

model_uri = 'mlruns/0/f282eb0eddda4a39b2ad75e1d158d40f/artifacts/models/trained_models/grade_predictor.h5'

loaded_model = mlflow.pyfunc.load_model(model_uri)

def preprocess_data(df):
    processor = DataProcessor(df)
    features, target = processor.separate_target_and_features()
    standardized_features = processor.standardize_features(features)
    return standardized_features

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get the input data
            data =  {
            "Student_ID": 123,
            "Sex": "Male",
            "Social Status": "Middle class",
            "Age": 18,
            "Governorate": "Cairo",
            "Living": "Urban",
            "Mother education": "Bachelor's degree",
            "Father education": "Master's degree",
            "Family member Education": "Bachelor's degree",
            "Father Alive": "Yes",
            "Mother Alive": "Yes",
            "Family Size": 4,
            "Parent Apart": "No",
            "The Guardian": "Both parents",
            "Family Relationship": "Good",
            "Father Job": "Engineer",
            "Mother Job": "Doctor",
            "Education Fee": "Paid",
            "Secondary Job": "None",
            "Home Ownership": "Owned",
            "Study Room": "Yes",
            "Family Economic Level": "High",
            "You  chronic disease": "No",
            "Family Chronic Disease": "No",
            "Specialization": "Science",
            "Study willing": "Yes",
            "Reason of study": "Interest",
            "Attendance": "Regular",
            "Failure Year": 0,
            "Higher Education Willing": "Yes",
            "References Usage": "Yes",
            "Internet Usage": "High",
            "TV Usage": "Low",
            "Sleep Hour": 8,
            "Study Hour": 4,
            "Arrival Time": "On time",
            "Transport": "Public",
            "Holiday Effect": "No",
            "Worry Effect": "Low",
            "Parent Meeting": "Regular",
            "Islamea": 90,
            "arabic": 85,
            "english": 88,
            "math": 92,
            "physics": 85,
            "chemistry": 82,
            "economy/bio": 90,
            "Avg1": 87,
            "Islamea.1": 80,
            "arabic.1": 75,
            "english.1": 78,
            "math.1": 82,
            "physics.1": 75,
            "chemistry.1": 72,
            "economy/bio.1": 80,
            "Avg1.1": 78
        }
            
            input_data = pd.DataFrame(data)
            preprocessed_data = preprocess_data(input_data)
            
            input_tensor = torch.tensor(preprocessed_data.values, dtype=torch.float32)
            
            # Make predictions
            with torch.no_grad():
                predictions = loaded_model(input_tensor)
                predictions = (predictions > 0.5).float().numpy().flatten().tolist()

          

            return jsonify({'predictions': predictions})

        except Exception as e:
            return jsonify({'error': str(e)}), 400

@app.route('/')
def index():
    return 'Thank you for visiting students grade prediction'

if __name__ == '__main__':
    app.run(port=5001, debug=True)



from flask import Flask, request, jsonify
import mlflow.pytorch
import torch

app = Flask(__name__)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# Load the trained model
model_uri = 'runs:/867e92ff96814a8382bc29563dd0dba4/models'
loaded_model = mlflow.pyfunc.load_model(model_uri)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get the input data
            data = request.json
            inputs = torch.tensor(data['inputs'])

            # Make predictions
            outputs = loaded_model(inputs)
            predictions = outputs.squeeze().detach().numpy().tolist()

            return jsonify({'predictions': predictions})

        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)



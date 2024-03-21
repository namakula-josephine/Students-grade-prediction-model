from flask import Flask, request, jsonify
import mlflow.pytorch
import torch

app = Flask(__name__)

# Load the trained model
model_uri = "runs:/handsome-doe-904/models"  # Replace <RUN_ID> with the actual run ID
loaded_model = mlflow.pytorch.load_model(model_uri)

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

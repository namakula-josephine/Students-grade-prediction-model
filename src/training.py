import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
import mlflow

def train_model(model, criterion, optimizer, train_loader, epochs):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Set your mlflow URI here
    # mlflow.set_experiment("Students grade prediction")  # Set your experiment name
    
    with mlflow.start_run():
        train_losses = []
        train_accuracies = []

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            y_true = []
            y_pred = []

            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # Store predictions for computing accuracy
                y_true.extend(labels.numpy())
                y_pred.extend(outputs.squeeze().detach().numpy())

            train_losses.append(running_loss / len(train_loader))

            # Compute R-squared
            r2 = r2_score(y_true, y_pred)
            train_accuracies.append(r2)

            print(f'Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]}, Accuracy: {train_accuracies[-1]}')

            # Log metrics with mlflow
            mlflow.log_metric("train_loss", train_losses[-1], step=epoch)
            mlflow.log_metric("train_r2_score", train_accuracies[-1], step=epoch)

        # Save the trained model with mlflow
        mlflow.pytorch.log_model(model, 'models/trained_models/grade_predictor.h5')
        
def save_model(model, path):
    torch.save(model.state_dict(), path)

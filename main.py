import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset
from models.model import GNN
from src.preprocessing import DataProcessor, DataSplitter, TensorConverter
from src.training import train_model
from src.evaluating import evaluate_model
import pandas as pd


def main():
    # Load your data
    data = pd.read_csv('data\Iraqi Student Performance Prediction.csv')  # Change 'your_data.csv' to your actual data file path

    # Preprocess the data
    processor = DataProcessor(data)
    features, target = processor.separate_target_and_features()
    standardized_features = processor.standardize_features(features)

    splitter = DataSplitter()
    X_train, X_test, y_train, y_test = splitter.split_data(standardized_features, target)

    tensor_converter = TensorConverter()
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = tensor_converter.convert_to_tensors(X_train, y_train, X_test, y_test)

    # Instantiate the model
    input_dim = X_train.shape[1]  # Number of features
    hidden_dim = 64  # Hidden layer size
    output_dim = 1  # Output size
    model = GNN(input_dim, hidden_dim, output_dim)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Convert data to DataLoader for batching
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    # Training the model
    epochs = 500
    train_model(model, criterion, optimizer, train_loader, epochs)

    # Evaluate the model
    evaluate_model(model, train_loader)  # Evaluation on training data

if __name__ == "__main__":
    main()

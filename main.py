import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from models.model import GNN
from src.training import train_model
from src.evaluating import evaluate_model

# Load data
data = pd.read_csv('your_data.csv')  # Replace 'your_data.csv' with your actual data file path

# Data preprocessing
class DataProcessor:
    def __init__(self, data):
        self.data = data

    def separate_target_and_features(self):
        target = self.data['Avg1']
        features = self.data.drop(columns=['Avg1'])
        return features, target

    def standardize_features(self, features):
        standardized_features = (features - features.mean()) / features.std()
        return standardized_features

processor = DataProcessor(data)
features, target = processor.separate_target_and_features()
standardized_features = processor.standardize_features(features)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(standardized_features, target, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Convert data to DataLoader for batching
train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Model parameters
input_dim = X_train.shape[1]  # Number of features
hidden_dim = 64  # Hidden layer size
output_dim = 1  # Output size

# Instantiate the model
model = GNN(input_dim, hidden_dim, output_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
epochs = 500
train_model(model, criterion, optimizer, train_loader, epochs)

# Evaluation
test_data = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

print("Evaluation on Test Data:")
evaluate_model(model, test_loader)

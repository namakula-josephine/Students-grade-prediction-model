import torch
from sklearn.model_selection import train_test_split

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

class DataSplitter:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        return X_train, X_test, y_train, y_test

class TensorConverter:
    def convert_to_tensors(self, X_train, y_train, X_test, y_test):
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
        return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score

def train_model(model, criterion, optimizer, train_loader, epochs):
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

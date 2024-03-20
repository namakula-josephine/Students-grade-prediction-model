from sklearn.metrics import r2_score

def evaluate_model(model, data_loader):
    model.eval()
    y_true = []
    y_pred = []

    for inputs, labels in data_loader:
        outputs = model(inputs)
        y_true.extend(labels.numpy())
        y_pred.extend(outputs.squeeze().detach().numpy())

    r2 = r2_score(y_true, y_pred)
    print(f'Evaluation R-squared: {r2}')

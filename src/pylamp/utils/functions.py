import numpy as np

def print_accuracy(model,X_test,y_test):
    y_pred = np.argmax(model.forward(X_test),axis=1)
    accuracy = np.sum(y_test == y_pred)/len(y_test)
    print(f'Accuracy = {accuracy}')

def to_one_hot(y, num_classes):
    return np.eye(num_classes)[y]

def normalize_data(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train_normalized = (X_train - mean) / std
    X_test_normalized = (X_test - mean) / std
    return X_train_normalized, X_test_normalized


import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

def print_accuracy(model,X_test,y_test, print_result=True):
    y_pred = np.argmax(model.forward(X_test),axis=1)
    accuracy = np.sum(y_test == y_pred)/len(y_test)
    if print_result:
        print(f'Accuracy = {accuracy}')
    return accuracy

def to_one_hot(y, num_classes):
    return np.eye(num_classes)[y]

def normalize_data(X_train, X_test, minmax=False):
    if minmax:
        X_train_normalized = (X_train - np.min(X_train, axis=0)) / (np.max(X_train, axis=0) - np.min(X_train, axis=0))
        X_test_normalized = (X_test - np.min(X_train, axis=0)) / (np.max(X_train, axis=0) - np.min(X_train, axis=0))
        return X_train_normalized, X_test_normalized
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train_normalized = (X_train - mean) / std
    X_test_normalized = (X_test - mean) / std
    return X_train_normalized, X_test_normalized


def cluster_kmeans(latent_representations, y_test):
    kmeans = KMeans(n_clusters=10, random_state=0, n_init=10)
    clusters = kmeans.fit_predict(latent_representations)
    ari = adjusted_rand_score(y_test, clusters)
    print(f"Adjusted Rand Index: {ari:.4f}")


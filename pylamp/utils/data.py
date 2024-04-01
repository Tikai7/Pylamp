import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from data.MlTools.mltools import *

class DataGenerator():

    @staticmethod
    def generate_2D_data(centerx=1,centery=1,sigma=0.1,nbex=1000,data_type=0,epsilon=0.02):
        X, y = gen_arti(centerx,centery,sigma,nbex,data_type,epsilon)
        X_train,X_test,y_train,y_test  = train_test_split(X,y,train_size=0.8)
        return X_train,X_test,y_train,y_test 
    
    @staticmethod
    def generate_linear_data(a:float = 6, bias:float = -1, sigma:float = 0.4, size:int = 100, size_test:int = 1000):
        # Génération aléatoire de donnée
        X_train = np.sort(np.random.rand(size))
        X_test = np.sort(np.random.rand(size_test))
        # Ajouts du bruit gaussien
        gaussian_noise_train = np.random.randn(size)*sigma
        gaussian_noise_test = np.random.randn(size_test)*sigma
        # Calcul de y
        y_train = a*X_train+bias+gaussian_noise_train 
        y_test = a*X_test+bias+gaussian_noise_test
        
        X_train = X_train.reshape(-1,1)
        X_test = X_test.reshape(-1,1)
        y_train = y_train.reshape(-1,1)
        y_test = y_test.reshape(-1,1)
        
        return X_train,X_test,y_train,y_test 
    
    @staticmethod
    def batch_generator(data, labels, batch_size):
        num_batches = len(data) // batch_size
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_x = data[start_idx:end_idx]
            batch_y = labels[start_idx:end_idx]
            batch_x = batch_x.reshape(batch_size, -1)
            batch_y = batch_y.reshape(batch_size, -1)

            yield batch_x, batch_y


    @staticmethod
    def plot_linear_data(X_train, y_train, X_test, y_test, 
            alpha_train:float=0.2, alpha_test:float=0.2,
            title_train:str="Train data", title_test:str="Test data", title:str=""
    ):
        plt.style.use('ggplot')
        plt.figure(figsize=(12, 7))
        plt.scatter(X_train, y_train, alpha=alpha_train, label=title_train)
        plt.scatter(X_test, y_test, alpha=alpha_test, label=title_test)
        plt.title(title)
        plt.legend()
        plt.show()

    @staticmethod
    def plot_2D_data(X, y):
        plot_data(X,y)

    def plot_decision_boundary(X, y, model, title):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                            np.arange(y_min, y_max, 0.01))

        Z = model.forward(np.array(np.c_[xx.ravel(), yy.ravel()], dtype=np.float32))
        Z = np.sign(Z)
        Z = Z.reshape(xx.shape)
        plt.figure(figsize=(12, 7))
        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
        plt.title(title)
        plt.show() 
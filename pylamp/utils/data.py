import numpy as np
import matplotlib.pyplot as plt


class DataGenerator():
    @staticmethod
    def generate_linear_data(a:float, bias:float, sigma:float, size:int, size_test:int):
        # Génération aléatoire de donnée
        X_train = np.sort(np.random.rand(size))
        X_test = np.sort(np.random.rand(size_test))
        # Ajouts du bruit gaussien
        gaussian_noise_train = np.random.randn(size)*sigma
        gaussian_noise_test = np.random.randn(size_test)*sigma
        # Calcul de y
        y_train = a*X_train+bias+gaussian_noise_train 
        y_test = a*X_test+bias+gaussian_noise_test
        
        return X_train,y_train,X_test,y_test
    
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
    def plot_data(X_train, y_train, X_test, y_test, 
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
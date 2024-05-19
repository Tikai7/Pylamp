import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

class Display():
    @staticmethod
    def plot_loss(losses):
        plt.style.use('ggplot')
        epochs = range(1, len(losses) + 1)
        plt.plot(epochs, losses, label='Training Loss')
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    @staticmethod
    def compare_images(original, generated, title1='Original Image', title2='Generated Image', shape=None, fig_size=(12, 7)):
        if shape is not None:
            original = original.reshape(shape)
            generated = generated.reshape(shape)
        plt.style.use('ggplot')
        plt.figure(figsize=fig_size)
        plt.subplot(1, 2, 1)
        plt.imshow(original, cmap='gray')
        plt.title(title1)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(generated, cmap='gray')
        plt.title(title2)
        plt.axis('off')
        plt.show()

    @staticmethod
    def plot_latent_space(X, y, encoder, expand_dims=False):
        plt.style.use('ggplot')
        if expand_dims:
            X = np.expand_dims(X, axis=-1)
        latent_space = encoder.forward(X)
        latent_space = latent_space.reshape(latent_space.shape[0], -1)
        # Reduce the dimensionality to 2D using t-SNE
        tsne = TSNE(n_components=2, random_state=0)
        latent_2d = tsne.fit_transform(latent_space)
        # Assuming y_test contains the labels
        y_test_np = np.array(y)
        # Plot the 2D t-SNE projection
        plt.figure(figsize=(10, 8))
        for i in range(10):
            indices = np.where(y_test_np == i)
            plt.scatter(latent_2d[indices, 0], latent_2d[indices, 1], label=f'Class {i}', alpha=0.6)
        plt.legend()
        plt.title("Projection 2D de l'espace latent 2D avec t-SNE")
        plt.xlabel("t-SNE component 1")
        plt.ylabel("t-SNE component 2")
        plt.show()

        return latent_2d

    
    @staticmethod
    def plot_reconstruction(model, X, y, n=1, expand_dims=False, calculation=False):
        plt.style.use('ggplot')
        plt.figure(figsize=(10,5)) 
        for label in np.random.choice(np.unique(y),n):
            target = X[y == label][0]
            target = np.expand_dims(target, axis=-1) if expand_dims else target
            pred = model.forward(np.array([target]))
            Display.compare_images(target,pred,shape=(16,16),fig_size=(7,3))
        plt.show() if not calculation else None

import matplotlib.pyplot as plt

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
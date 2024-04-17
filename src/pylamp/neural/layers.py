import numpy as np 
import scipy
import scipy.fft
import scipy.signal
from pylamp.neural.module import Module

class Linear(Module):
    """Class representing a linear layer in a neural network."""

    def __init__(self, input_size, output_size, use_bias = True):
        super().__init__()
        self.use_bias = use_bias
        self.input_size = input_size
        self.output_size = output_size
        self._parameters = {
            "weights" : np.random.randn(input_size, output_size),
            "bias" : np.random.randn(output_size) if use_bias else np.zeros(output_size)
        }
        self._gradient = {
            "weights" : np.zeros((self.input_size, self.output_size)),
            "bias" : np.zeros(output_size)
        }
        
    def zero_grad(self):
        self._gradient["weights"] = np.zeros((self.input_size, self.output_size))
        self._gradient["bias"] = np.zeros(self.output_size)

    def forward(self, X):
        output = X @ self._parameters["weights"] + self._parameters["bias"]
        return output

    def update_parameters(self, gradient_step=1e-3):
        self._parameters["weights"] -= gradient_step*self._gradient["weights"]
        if self.use_bias : 
            self._parameters["bias"] -= gradient_step*self._gradient["bias"]

    def backward_update_gradient(self, input, delta):
        self._gradient["weights"] += input.T @ delta
        if self.use_bias : 
            self._gradient["bias"] += delta.sum(axis=0)

    def backward_delta(self, input, delta):
        delta_grad = delta @ self._parameters["weights"].T
        return delta_grad
    
class Conv1D(Module):
    """Class representing a 1-dimensional convolution layer in a neural network.
        Padding is always set to 'valid'
    """

    def __init__(self,   kernel_size=9, in_channels=3, out_channels=32, strides=1, use_bias = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strides = strides
        self.use_bias = use_bias
        self._parameters = {
            "weights" : np.random.randn(kernel_size, in_channels, out_channels),
            "bias" : np.random.randn(out_channels)
        }
        self._gradient = {
            "weights" : np.zeros((kernel_size, in_channels, out_channels)),
            "bias" : np.zeros(out_channels)
        }
        
    def zero_grad(self):
        self._gradient["weights"] = np.zeros((self.kernel_size*self.in_channels, self.out_channels))
        self._gradient["bias"] = np.zeros(self.out_channels)

    def forward(self, X):
        print(X.shape)
        batch, size, in_channel = X.shape
        output_size = (size - self.kernel_size) // self.strides + 1
        # Redéfinition de l'entrée pour appliquer la convolution
        windows_view = np.lib.stride_tricks.as_strided(
            X,
            shape=(batch, output_size, self.kernel_size, in_channel),
            strides=(X.strides[0], self.strides * X.strides[1], X.strides[1], X.strides[2]),
            writeable=False
        )
        # Calcul de la convolution en une seule opération de multiplication matricielle
        output = np.tensordot(windows_view, self._parameters["weights"], axes=[[2, 3], [1, 2]]) + self.bias
        return output


    def update_parameters(self, gradient_step=1e-3):
        self._parameters["weights"] -= gradient_step*self._gradient["weights"]
        if self.use_bias : 
            self._parameters["bias"] -= gradient_step*self._gradient["bias"]

    def backward_update_gradient(self, input, delta):
        X = input
        batch, size, in_channel = X.shape
        output_size = (size - self.kernel_size) // self.strides + 1
        # Redéfinition de l'entrée pour appliquer la convolution
        windows_view = np.lib.stride_tricks.as_strided(
            X,
            shape=(batch, output_size, self.kernel_size, in_channel),
            strides=(X.strides[0], self.strides * X.strides[1], X.strides[1], X.strides[2]),
            writeable=False
        )
        self._gradient["weights"] += np.tensordot(delta, windows_view, axes=[[0, 1], [0, 1]])
        self._gradient["bias"] += delta.sum(axis=(0, 1))


    """ CHATGPT A FAIT CETTE FONCTION JAI RIEN COMPRIS
    """
    def backward_delta(self, input, delta):
        # Initialisation du gradient d'entrée
        delta_grad = np.zeros_like(input)
        # Pour chaque canal de sortie
        for i in range(self.out_channels):
            # Pour chaque delta
            for j in range(delta.shape[1]):  # delta.shape[1] est la longueur de sortie
                # Appliquer chaque poids retourné à la région appropriée de l'entrée
                # L'intervalle d'application dépend de la position dans le delta et du stride
                start = j * self.strides
                end = start + self.kernel_size
                # Nous ajoutons le produit du delta pour ce point et le poids correspondant à toutes les entrées qui ont été convoluées pour produire cette sortie
                delta_grad[:, start:end, :] += delta[:, j, np.newaxis, np.newaxis] *self._parameters["weights"][i, ::-1, :]
        return delta_grad
    
class MaxPool1D(Module):
    """Class representing a pooling layer in a neural network.
    """

    def __init__(self, kernel_size=2, strides=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.indices = None

    def forward(self, X):
        batch, size, in_channel = X.shape
        output_size = (size - self.kernel_size) // self.strides + 1
        
        windows_view = np.lib.stride_tricks.as_strided(
            X,
            shape=(batch, output_size, self.kernel_size, in_channel),
            strides=(X.strides[0], self.strides * X.strides[1], X.strides[1], X.strides[2]),
            writeable=False
        )
        
        # Storing the indices of the max for use in backpropagation
        self.indices = np.argmax(windows_view, axis=2)
        output = np.max(windows_view, axis=2)
        return output


    def zero_grad(self):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        # Initialize gradient array with zeros
        delta_grad = np.zeros(input.shape)
        batch, output_size, in_channel = input.shape
        for b in range(batch):
            for i in range(output_size):
                for c in range(in_channel):
                    index = self.indices[b, i, c]
                    # Gradient of the max positions
                    delta_grad[b, i * self.strides + index, c] += delta[b, i, c]

        return delta_grad

    def update_parameters(self, learning_rate):
        pass

class Flatten(Module):
    """Class representing a flattening layer in a neural network.
    """
    def __init__(self):
        super().__init__()

    def forward(self, X):
        batch_size = X.shape[0]
        return np.reshape(X, (batch_size, -1))

    def zero_grad(self):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        return np.reshape(delta, input.shape)

    def update_parameters(self, learning_rate):
        pass
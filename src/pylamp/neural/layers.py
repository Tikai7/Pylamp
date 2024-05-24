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
        limit = np.sqrt(6 / (input_size + output_size))
        self.parameters = {
            "weights": np.random.uniform(-limit, limit, (input_size, output_size)),
            "bias": np.random.uniform(-limit, limit, output_size) if use_bias else np.zeros(output_size)
        }
        self.gradient = {
            "weights" : np.zeros((self.input_size, self.output_size)),
            "bias" : np.zeros(output_size)
        }
        
    def zero_grad(self):
        self.gradient["weights"].fill(0)
        self.gradient["bias"].fill(0)

    def forward(self, X):
        output = X @ self.parameters["weights"] + self.parameters["bias"]
        return output

    def update_parameters(self, gradient_step=1e-3, clip_value=1.0):
        np.clip(self.gradient["weights"], -clip_value, clip_value, out=self.gradient["weights"])
        np.clip(self.gradient["bias"], -clip_value, clip_value, out=self.gradient["bias"])
        self.parameters["weights"] -= gradient_step*self.gradient["weights"]
        if self.use_bias : 
            self.parameters["bias"] -= gradient_step*self.gradient["bias"]

    def backward_update_gradient(self, input, delta):
        self.gradient["weights"] += input.T @ delta
        if self.use_bias : 
            self.gradient["bias"] += delta.sum(axis=0)

    def backward_delta(self, input, delta):
        delta_grad = delta @ self.parameters["weights"].T
        return delta_grad
    
class Conv1D(Module):
    """Class representing a 1-dimensional convolution layer in a neural network.
        Padding is always set to 'valid'
    """

    def __init__(self,   kernel_size=9, in_channels=3, out_channels=32, strides=1, use_bias = True, padding='valid'):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strides = strides
        self.use_bias = use_bias
        self.padding = padding
        limit = np.sqrt(6 / (kernel_size * in_channels + out_channels))
        self.parameters = {
            "weights": np.random.uniform(-limit, limit, (kernel_size, in_channels, out_channels)),
            "bias": np.random.uniform(-limit, limit, out_channels) if use_bias else np.zeros(out_channels)
        }
        self.gradient = {
            "weights" : np.zeros((kernel_size, in_channels, out_channels)),
            "bias" : np.zeros(out_channels)
        }
        
    def zero_grad(self):
        self.gradient["weights"].fill(0)
        self.gradient["bias"].fill(0)

    def pad_input(self, X):
        if self.padding == 'same':
            pad_total = max((self.strides - 1) * X.shape[1] + self.kernel_size - self.strides, 0)
            pad_start = pad_total // 2
            pad_end = pad_total - pad_start
            X = np.pad(X, ((0, 0), (pad_start, pad_end), (0, 0)), mode='constant')
        return X

    def forward(self, X):
        X = self.pad_input(X)
        batch, size, in_channel = X.shape
        output_size = (size - self.kernel_size) // self.strides + 1
        # Redéfinition de l'entrée pour appliquer la convolution
        windows_view = np.lib.stride_tricks.as_strided(
            X,
            shape=(batch, output_size, self.kernel_size, in_channel),
            strides=(X.strides[0], self.strides * X.strides[1], X.strides[1], X.strides[2]),
            writeable=False
        )
        windows_view = windows_view.transpose(0, 1, 3, 2)  # (batch, output_size, in_channels, kernel_size)
        # Calcul de la convolution en une seule opération de multiplication matricielle
        output = np.tensordot(windows_view, self.parameters["weights"], axes=[[3, 2], [0, 1]]) + self.parameters["bias"]

        return output


    def update_parameters(self, gradient_step=1e-3, clip_value=1.0):
        # np.clip(self.gradient["weights"], -clip_value, clip_value, out=self.gradient["weights"])
        # np.clip(self.gradient["bias"], -clip_value, clip_value, out=self.gradient["bias"])
        self.parameters["weights"] -= gradient_step*self.gradient["weights"]
        if self.use_bias : 
            self.parameters["bias"] -= gradient_step*self.gradient["bias"]

    def backward_update_gradient(self, input, delta):
        X = self.pad_input(input)
        batch, size, in_channel = X.shape
        output_size = (size - self.kernel_size) // self.strides + 1
        windows_view = np.lib.stride_tricks.as_strided(
            X,
            shape=(batch, output_size, self.kernel_size, in_channel),
            strides=(X.strides[0], self.strides * X.strides[1], X.strides[1], X.strides[2]),
            writeable=False
        )
        grad_weights = np.tensordot(delta, windows_view, axes=([0, 1], [0, 1]))
        grad_weights = grad_weights.transpose(1, 2, 0)

        self.gradient["weights"] += grad_weights/batch
        if self.use_bias :
            self.gradient["bias"] += np.sum(delta, axis=(0, 1))/batch


    def backward_delta(self, input, delta):
        X = self.pad_input(input)
        _, size, _ = X.shape
        output_size = (size - self.kernel_size) // self.strides + 1
        d_out = np.zeros_like(X)
        d_in = np.einsum("bod, kcd -> kboc", delta, self.parameters["weights"])
        for i in range(self.kernel_size):
            d_out[:, i : i + output_size * self.strides : self.strides, :] += d_in[i]

        return d_out[:, :input.shape[1], :]

    
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
        windows_view = windows_view.transpose(0, 1, 3, 2)  # (batch, output_size, in_channels, kernel_size)
        # Storing the indices of the max for use in backpropagation
        self.indices = np.argmax(windows_view, axis=3)
        output = np.max(windows_view, axis=3)
        return output


    def zero_grad(self):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        delta_grad = np.zeros_like(input)
        batch, size, in_channel = input.shape
        output_size = (size - self.kernel_size) // self.strides + 1  

        for b in range(batch):
            for i in range(output_size):
                for c in range(in_channel):
                    index = self.indices[b, i, c]  # Retrieve the max index
                    # Gradient of the max positions
                    delta_grad[b, i * self.strides + index, c] += delta[b, i, c]

        return delta_grad

    def update_parameters(self, learning_rate):
        pass


class Upsampling1D(Module):
    """Class representing a basic upsampling layer in a neural network using nearest-neighbor upsampling."""

    def __init__(self, size=2):
        super().__init__()
        self.size = size

    def forward(self, X):
        upsampled = np.repeat(X, self.size, axis=1)
        return upsampled
    
    def zero_grad(self):
        pass

    def update_parameters(self, learning_rate):
        pass

    def backward_delta(self, input, delta):
        _, input_size, _ = input.shape
        delta_grad = np.zeros_like(input)
        for i in range(input_size):
            delta_grad[:, i, :] = np.sum(delta[:, i * self.size:(i + 1) * self.size, :], axis=1)
        return delta_grad

    def backward_update_gradient(self, input, delta):
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
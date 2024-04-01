import numpy as np 
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
import numpy as np
from pylamp.neural.module import Module


class TanH(Module):
    """Class representing a TanH activation function."""

    def __init__(self):
        super().__init__()

    def forward(self, X):
        return np.tanh(X)

    def zero_grad(self):
        pass

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        return delta * (1 - self.forward(input) ** 2)


class Sigmoid(Module):
    """Class representing a Sigmoid activation function."""

    def __init__(self):
        super().__init__()

    def forward(self, X):
        return 1/(1+np.exp(-X))

    def zero_grad(self):
        pass

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        sigmoid_x = self.forward(input)
        return delta*sigmoid_x*(1-sigmoid_x)


class ReLU(Module):
    """Class representing a ReLU activation function."""

    def __init__(self):
        super().__init__()

    def forward(self, X):
        return np.maximum(0, X)

    def zero_grad(self):
        pass

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        return  delta * (input > 0)


class Softmax(Module):
    """Class representing a Softmax activation function."""

    def __init__(self):
        super().__init__()

    def forward(self, X):
        exps = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def zero_grad(self):
        pass

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        softmax_x = self.forward(input)
        return delta * softmax_x * (1 - softmax_x)

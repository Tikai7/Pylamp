import numpy as np
from pylamp.neural.module import Module

class TanH(Module):
    """Class representing a TanH activation function."""
    def __init__(self):
        super().__init__()

    def forward(self,X):
        return np.tanh(X)
    
    def backward(self):
        pass

    def zero_grad(self):
        pass

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        pass


class Sigmoid(Module):
    """Class representing a Sigmoid activation function."""
    def __init__(self):
        super().__init__()

    def forward(self,X):
        return 1/(1+np.exp(-X))
    
    def backward(self):
        pass

    def zero_grad(self):
        pass

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        pass

class ReLU(Module):
    """Class representing a ReLU activation function."""
    def __init__(self):
        super().__init__()

    def forward(self,X):
        return np.where(X>0, X, 0)
    
    def backward(self):
        pass
    
    def zero_grad(self):
        pass

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        pass


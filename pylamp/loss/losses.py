import numpy as np 

class Loss(object):
    """Class representing a loss function.
    @methods:
    - forward: compute the forward pass
    - backward: compute the backward pass
    """

    def forward(self, y, yhat):
        """Compute the forward pass :
        @param y: the target
        @param yhat: the prediction
        """
        pass

    def backward(self, y, yhat):
        """Compute the backward pass :
        @param y: the target
        @param yhat: the prediction
        """
        pass

class MSE(Loss):
    """Class representing the Mean Squared Error loss."""
    def forward(self, y, yhat):
        assert y.shape == yhat.shape, "y and yhat must have the same shape"
        return np.mean((y-yhat)**2)

    def backward(self, y, yhat):
        assert y.shape == yhat.shape, "y and yhat must have the same shape"
        return -2*(y-yhat)
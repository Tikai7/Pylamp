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

class MSELoss(Loss):
    """Class representing the Mean Squared Error loss."""
    def forward(self, y, yhat):
        assert y.shape == yhat.shape, "y and yhat must have the same shape, got {} and {}".format(y.shape, yhat.shape)
        return np.mean((y-yhat)**2)

    def backward(self, y, yhat):
        assert y.shape == yhat.shape, "y and yhat must have the same shape, got {} and {}".format(y.shape, yhat.shape)
        return -2*(y-yhat)/y.shape[0]
    
class BCELoss(Loss):
    """Class representing the Binary Cross Entropy loss."""
    def forward(self, y, yhat):
        assert y.shape == yhat.shape, "y and yhat must have the same shape, got {} and {}".format(y.shape, yhat.shape)
        yhat_1 = np.clip(yhat,1e-10,1)
        yhat_2 = np.clip(1-yhat,1e-10,1)
        return -np.mean(y*np.log(yhat_1)+(1-y)*np.log(yhat_2))

    def backward(self, y, yhat):
        assert y.shape == yhat.shape, "y and yhat must have the same shape, got {} and {}".format(y.shape, yhat.shape)
        yhat_1 = np.clip(yhat,1e-10,1)
        yhat_2 = np.clip(1-yhat,1e-10,1)
        return -((y/yhat_1) - (1-y)/(yhat_2))/y.shape[0]
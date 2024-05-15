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
        assert y.shape == yhat.shape, "y and yhat must have the same shape, got {} and {}".format(
            y.shape, yhat.shape)
        return np.mean((y-yhat)**2)

    def backward(self, y, yhat):
        assert y.shape == yhat.shape, "y and yhat must have the same shape, got {} and {}".format(
            y.shape, yhat.shape)
        return -2*(y-yhat)/y.shape[0]

class BCELoss(Loss):
    """Class representing the Binary Cross Entropy loss."""
    def forward(self, y, yhat):
        assert y.shape == yhat.shape, "y and yhat must have the same shape, got {} and {}".format(
            y.shape, yhat.shape)
        yhat_1 = np.clip(yhat, 1e-10, 1)
        yhat_2 = np.clip(1-yhat, 1e-10, 1)
        return -np.mean(y*np.log(yhat_1)+(1-y)*np.log(yhat_2))

    def backward(self, y, yhat):
        assert y.shape == yhat.shape, "y and yhat must have the same shape, got {} and {}".format(
            y.shape, yhat.shape)
        yhat_1 = np.clip(yhat, 1e-10, 1)
        yhat_2 = np.clip(1-yhat, 1e-10, 1)
        return -((y/yhat_1) - (1-y)/(yhat_2))/y.shape[0]

# class CrossEntropyLoss(Loss):
#     """Class representing the Cross Entropy loss."""

#     def forward(self, y, y_hat):
#         """Compute the cross-entropy loss given ground truth labels y and predicted probabilities y_hat.
#         @param y: numpy array of shape (batch_size, num_classes) - ground truth labels (one-hot encoded)
#         @param y_hat: numpy array of shape (batch_size, num_classes) - predicted probabilities
#         @return loss: float - mean cross-entropy loss for the batch
#         """
#         log_probs = self._log_softmax(y_hat)
#         loss = -np.sum(y * log_probs) / y.shape[0]  # Compute mean loss
#         return loss

#     def backward(self, y, y_hat):
#         """Compute the gradient of the cross-entropy loss with respect to y_hat.
#         @param y: numpy array of shape (batch_size, num_classes) - ground truth labels (one-hot encoded)
#         @param y_hat: numpy array of shape (batch_size, num_classes) - predicted probabilities
#         @return grad: numpy array of shape (batch_size, num_classes) - gradient of the loss with respect to y_hat
#         """
#         grad = y_hat - y
#         return grad

#     def _log_softmax(self, x):
#         """Compute the log softmax of input x.
#         @param x: numpy array of shape (batch_size, num_classes) - input logits
#         @return log_probs: numpy array of shape (batch_size, num_classes) - log softmax probabilities
#         """
#         max_val = np.max(x, axis=1, keepdims=True)
#         exp_x = np.exp(x - max_val)
#         log_probs = np.log(exp_x / np.sum(exp_x, axis=1, keepdims=True))
#         return log_probs

class CrossEntropyLoss(Loss):
    """Class representing the Cross Entropy loss."""

    def forward(self, y, y_hat):
        """Compute the cross-entropy loss given ground truth labels y and predicted probabilities y_hat.
        @param y: numpy array of shape (batch_size, num_classes) - ground truth labels (one-hot encoded)
        @param y_hat: numpy array of shape (batch_size, num_classes) - predicted probabilities
        @return loss: float - mean cross-entropy loss for the batch
        """
        log_probs = self._log_softmax(y_hat)
        loss = -np.sum(y * log_probs) / y.shape[0]  # Compute mean loss
        return loss

    def backward(self, y, y_hat):
        """Compute the gradient of the cross-entropy loss with respect to y_hat.
        @param y: numpy array of shape (batch_size, num_classes) - ground truth labels (one-hot encoded)
        @param y_hat: numpy array of shape (batch_size, num_classes) - predicted probabilities
        @return grad: numpy array of shape (batch_size, num_classes) - gradient of the loss with respect to y_hat
        """
        grad = y_hat - y
        return grad

    def _log_softmax(self, x):
        """Compute the log softmax of input x.
        @param x: numpy array of shape (batch_size, num_classes) - input logits
        @return log_probs: numpy array of shape (batch_size, num_classes) - log softmax probabilities
        """
        max_val = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - max_val)
        sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
        log_probs = np.log(exp_x / sum_exp_x + 1e-10)  # Adding epsilon to avoid log(0)
        return log_probs

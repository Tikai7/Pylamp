import numpy as np
from pylamp.utils.data import DataGenerator as dg

class Optim:
    def __init__(self, net, loss, eps):
        """Initialize the Optimizer.
        @param net: The neural network model.
        @param loss: The loss function.
        @param eps: The learning rate (step size) for gradient descent.
        """
        self.net = net
        self.loss = loss
        self.eps = eps

    def step(self, batch_x, batch_y):
        """Perform one step of gradient descent.
        @param batch_x: Input data (batch) for training.
        @param batch_y: Target labels (batch) for training.
        @return loss_value: The loss value for the current batch.
        """
        # Reset gradients to zero
        self.net.zero_grad()
        # Forward pass
        output = self.net.forward(batch_x)
        # Compute loss
        loss_value = self.loss.forward(batch_y, output)
        # Backward pass
        loss_grad = self.loss.backward(batch_y, output)

        self.net.backward_delta(batch_x, loss_grad)
        self.net.backward_update_gradient(batch_x, loss_grad)
        self.net.update_parameters(self.eps)

        return loss_value


def SGD(network, X_train, y_train, batch_size, epochs, verbose=False, add_channel_x=False, add_channel_y=False):
    """Perform Stochastic Gradient Descent (SGD) on the given network using the provided dataset.
    @param network: The neural network model to train.
    @param X_train (numpy array): The input features of the training dataset.
    @param y_train (numpy array): The target labels of the training dataset.
    @param batch_size (int): The size of each mini-batch used for training.
    @param epochs (int): The number of complete passes through the training dataset.
    @param verbose (bool, optional): If True, prints the average loss for each epoch during training. Default is False.
    @return train_loss_tracker: A list containing the average loss for each epoch.
    """

    num_batches = len(X_train) // batch_size
    train_loss_tracker = []

    for epoch in range(epochs):
        total_loss = 0.0

        # Generate a list of indices
        indices = list(range(len(X_train)))
        # Shuffle the indices list
        np.random.shuffle(indices)
        # Sort X_train and y_train according to the shuffled indices
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        train_data = dg.batch_generator(
            X_train_shuffled, y_train_shuffled, batch_size, add_channel_x, add_channel_y)

        for batch_x, batch_y in train_data:
            total_loss += network.step(batch_x, batch_y)

        avg_loss = total_loss / num_batches
        train_loss_tracker.append(avg_loss)
        if (epochs < 10 or epoch % (epochs//10) == 0) and verbose:
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss}")

    print("Training finished.")
    return train_loss_tracker

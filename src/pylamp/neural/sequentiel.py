class Sequentiel():
    def __init__(self):
        """
        The Sequential model represents a linear stack of layers/modules.
        """
        super().__init__()
        self.modules = []
        self.layer_inputs = None

    def add_module(self, module):
        """Add a module (layer) to the Sequential model.
        @param module: The module (layer) to add.
        """
        self.modules.append(module)

    def forward(self, X):
        """Perform a forward pass through the Sequential model.
        @param X: Input data for the first layer.
        @return: The output of the last layer.
        """
        inputs = [X]

        for module in self.modules:
            output = module.forward(inputs[-1])
            inputs.append(output)

        self.layer_inputs = inputs

        return inputs[-1]

    def backward(self, loss_grad, lr):
        """Update gradients and parameters through a backward pass (backpropagation) in the Sequential model.
        @param loss_grad: The gradient of the loss function with respect to the output of the model.
        @param lr: Learning rate for updating parameters.
        """
        delta_grad = loss_grad

        for i, module in reversed(list(enumerate(self.modules))):
            input_data = self.layer_inputs[i]
            module.backward_update_gradient(input_data, delta_grad)
            delta_grad = module.backward_delta(input_data, delta_grad)
            module.update_parameters(lr)

    def zero_grad(self):
        """Reset gradients of all parameters in the model to zero.
        """
        for module in self.modules:
            module.zero_grad()

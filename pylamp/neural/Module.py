class Module(object):
    """Class representing a module in a neural network. 

    @attributes:
    - _parameters: the parameters of the module
    - _gradient: the gradient of the module

    @methods:
    - zero_grad: set the gradient to zero
    - forward: compute the forward pass
    - update_parameters: update the parameters
    - backward_update_gradient: update the gradient
    - backward_delta: compute gradient of the loss with respect to the input.
    """
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        """Set the gradient to zero."""
        ## Annule gradient
        pass

    def forward(self, X):
        """Compute the forward pass : 
        @param X: the input of the module
        """
        ## Calcule la passe forward
        pass

    def update_parameters(self, gradient_step=1e-3):
        """Update the parameters :
        @param gradient_step: the step of the gradient descent.
        """
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient

    def backward_update_gradient(self, input, delta):
        """Update the gradient :
        @param input: the input of the module
        @param delta: the delta of the module
        """
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        """Compute gradient of the loss with respect to the input : 
        @param input: the input of the modules
        @param delta: the delta of the module
        """
        ## Calcul la derivee de l'erreur
        pass



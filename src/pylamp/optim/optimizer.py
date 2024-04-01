import numpy as np
from pylamp.loss.losses import Loss
from pylamp.neural.module import Module
from pylamp.utils.data import DataGenerator as dg
from IPython.display import clear_output

class SGD():
    @staticmethod
    def step_multiple(fc1 : Module,fc2 : Module,activation1 : Module,activation2 : Module,loss : Loss, X_train : np.ndarray, y_train : np.ndarray, 
        epochs : int = 1000, lr : float = 1e-3, batch_size : int = 32,
        verbose : bool = True, plot_boundary=False, model_to_plot=None):
        train_loss_tracker = []
        nb_time_updated = 0

        for i in range(epochs):
            train_data = dg.batch_generator(X_train, y_train, batch_size)
            train_loss = 0
            for batch_x, batch_y in train_data:
                fc1.zero_grad()
                fc2.zero_grad()

                output_fc1 = fc1.forward(batch_x)
                output_ac1 = activation1.forward(output_fc1)
                output_fc2 = fc2.forward(output_ac1)
                output_ac2 = activation2.forward(output_fc2)
                train_loss += loss.forward(batch_y, output_ac2)

                loss_grad = loss.backward(batch_y, output_ac2)
                delta_grad = activation2.backward_delta(output_fc2, loss_grad)
                fc2.backward_update_gradient(output_ac1, delta_grad)
                delta_grad_fc2 = fc2.backward_delta(output_ac1, delta_grad)
                fc2.update_parameters(lr)
                delta_grad_ac1 = activation1.backward_delta(output_fc1, delta_grad_fc2)
                fc1.backward_update_gradient(batch_x, delta_grad_ac1)
                fc1.backward_delta(batch_x, delta_grad_ac1)
                fc1.update_parameters(lr)
                nb_time_updated += 1

            loss_item = train_loss/batch_size
            train_loss_tracker.append(loss_item)
           
            if (epochs < 10 or i%(epochs//10) == 0) and verbose:
                if(plot_boundary):
                    dg.plot_decision_boundary(X_train,y_train,model_to_plot,title="Boundary evolution")
                    if i < epochs-1:
                        clear_output(wait=True)

                print(f"Epoch {i} : Train loss : {loss_item}")

        print(f"Model updated {nb_time_updated} times.")
        return train_loss_tracker

    @staticmethod
    def step(
        model : Module, loss : Loss, X_train : np.ndarray, y_train : np.ndarray, 
        X_val : np.ndarray = None, y_val : np.ndarray = None, epochs : int = 1000, lr : float = 1e-3, batch_size : int = 32,
        verbose : bool = True
    ):
        train_loss_tracker = []
        val_loss_tracker = []
        val_loss_item = None
        nb_time_updated = 0
        for i in range(epochs):
            # Training
            train_data = dg.batch_generator(X_train, y_train, batch_size)
            val_data = dg.batch_generator(X_val, y_val, batch_size) if X_val is not None else None
            train_loss = 0
            for batch_x, batch_y in train_data:
                model.zero_grad()
                output = model.forward(batch_x)
                train_loss += loss.forward(batch_y, output)
                loss_grad = loss.backward(batch_y, output)
                # on s'en fou de la delta_grad, car pour l'instant on a un seul module
                model.backward_delta(batch_x, loss_grad)
                model.backward_update_gradient(batch_x, loss_grad)
                model.update_parameters(lr)
                nb_time_updated += 1

            loss_item = train_loss/batch_size
            train_loss_tracker.append(loss_item)
            # Validation
            if val_data is not None:
                val_loss = 0
                for val_x, val_y in val_data:
                    val_output = model.forward(val_x)
                    val_loss += loss.forward(val_output, val_y)
                val_loss_item = val_loss/batch_size
                val_loss_tracker.append(val_loss_item)
            
            if (epochs < 10 or i%(epochs//10) == 0) and verbose:
                print(f"Epoch {i} : Train loss : {loss_item} - Val loss : {val_loss_item}")

        print(f"Model updated {nb_time_updated} times.")
        return train_loss_tracker, val_loss_tracker


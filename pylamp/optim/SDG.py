import numpy as np
from pylamp.loss.losses import Loss
from pylamp.layers.Module import Module
from pylamp.utils.data import DataGenerator as dg

class GradientDescent():

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
                delta_grad = model.backward_delta(batch_x, loss_grad)
                model.backward_update_gradient(batch_x, delta_grad)
                nb_time_updated += 1
                model.update_parameters(lr)
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
            
            if i%(epochs//10) == 0 and verbose:
                print(f"Epoch {i} : Train loss : {loss_item} - Val loss : {val_loss_item}")
        print(f"Model updated {nb_time_updated} times.")
        return train_loss_tracker, val_loss_tracker


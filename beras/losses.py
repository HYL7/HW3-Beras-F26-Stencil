import numpy as np

from beras.core import Diffable, Tensor

import tensorflow as tf


class Loss(Diffable):
    @property
    def weights(self) -> list[Tensor]:
        return []

    def get_weight_gradients(self) -> list[Tensor]:
        return []


class MeanSquaredError(Loss):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return Tensor(np.mean((y_pred - y_true)**2))

    def get_input_gradients(self) -> list[Tensor]:
        y_pred, y_true = self.inputs

        pred_grad = Tensor(2 * (y_pred - y_true) / y_pred.size)
        true_grad = Tensor(np.zeros_like(y_true))

        return [pred_grad, true_grad]

class CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""
        return Tensor(np.mean(-np.sum(y_true * np.log(np.clip(y_pred, 1e-7, 1 - 1e-7)),axis=-1)))

    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        y_pred, y_true = self.inputs
        pred_grad = Tensor((-y_true * np.clip(y_pred, 1e-7, 1 - 1e-7)**(-1))/y_pred.shape[0])
        true_grad = Tensor(np.zeros_like(y_true))
        
        return [pred_grad, true_grad]

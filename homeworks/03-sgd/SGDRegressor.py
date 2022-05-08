import typing as ty

from _losses import LossFunction, SquaredLoss
from _sgd import sgd


class SGDRegressor():
    def __init__(
        self,
        loss: str = 'squared_loss',
        # <YOUR CODE HERE>
    ):
        if loss == 'squared_loss':
            self.__loss: ty.Type[LossFunction] = SquaredLoss()
        else:
            raise ValueError(f'The loss {loss} is not supported.')

        # Save constructor params
        # <YOUR CODE HERE>

        # Validate saved params in method below
        self.__validate_params()

    def __validate_params(self):
        # <YOUR CODE HERE>
        raise NotImplementedError()

    def fit(self, X, y, sample_weight=None):
        # <YOUR CODE HERE>
        return self

    def partial_fit(self, X, y, sample_weight=None):
        # <YOUR CODE HERE>
        return self

    def predict(self, X):
        # <YOUR CODE HERE>
        raise NotImplementedError()

    def score(self, X, y):
        # <YOUR CODE HERE>
        raise NotImplementedError()

import typing as ty

import numpy as np

from _losses import LossFunction


def sgd(
    weights: np.ndarray,
    intercept: np.ndarray,
    loss: ty.Type[LossFunction],
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int,
    fit_intercept: bool,
    verbose: bool,
    shuffle: bool,
    seed: ty.Optional[int],
    eta0: float,
    sample_weight: ty.Optional[np.ndarray],
):
    epoch = 0
    eta = eta0
    n_samples = len(X)

    for epoch in range(max_iter):
        if verbose:
            print(f'-- Epoch {epoch + 1}')

        indices = list(range(n_samples))

        for i in indices:
            # Calculate prediction for current sample.
            y_hat = 0.0  # <YOUR CODE HERE>
            # Calculate squared error gradient by prediction. Use loss.dloss
            dloss = 0.0  # <YOUR CODE HERE>
            print_dloss(dloss, verbose)

            # Calculate prediction gradient by weights.
            dp_dw = 0.0  # <YOUR CODE HERE>
            # Update weights, using gradients. Don't forget about learning rate.
            weights += 0.0  # <YOUR CODE HERE>

            if fit_intercept:
                # Calculate prediction gradient by intercept.
                dp_dw = 0.0  # <YOUR CODE HERE>
                # Update intercept, using gradients. Don't forget about learning rate.
                intercept += 0.0  # <YOUR CODE HERE>

    return weights, intercept, epoch + 1


def print_dloss(dloss, verbose=True):
    # <YOUR CODE HERE>
    pass

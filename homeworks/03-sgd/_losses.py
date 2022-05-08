class LossFunction():
    """Base class for convex loss functions"""

    def loss(self, p: float, y: float) -> float:
        """Evaluate the loss function.
        Parameters
        ----------
        p : double
            The prediction, p = w^T x + intercept
        y : double
            The true value (aka target)
        Returns
        -------
        double
            The loss evaluated at `p` and `y`.
        """
        return 0

    def dloss(self, p: float, y: float) -> float:
        """Evaluate the derivative of the loss function with respect to
        the prediction `p`.
        Parameters
        ----------
        p : double
            The prediction, p = w^T x
        y : double
            The true value (aka target)
        Returns
        -------
        double
            The derivative of the loss function with regards to `p`.
        """
        return 0


class ClassificationLoss(LossFunction):
    """Base class for loss functions for classification"""

    def loss(self, p: float, y: float) -> float:
        return 0

    def dloss(self, p: float, y: float) -> float:
        return 0


class RegressionLoss(LossFunction):
    """Base class for loss functions for regression"""

    def loss(self, p: float, y: float) -> float:
        return 0

    def dloss(self, p: float, y: float) -> float:
        return 0


class SquaredLoss(RegressionLoss):
    def loss(self, p: float, y: float) -> float:
        # <YOUR CODE HERE>
        raise NotImplementedError()

    def dloss(self, p: float, y: float) -> float:
        # <YOUR CODE HERE>
        raise NotImplementedError()

import numpy as np


class MaxAbsScaler:
    def fit(self, X):
        raise NotImplementedError()

    def transform(self, X):
        raise NotImplementedError()

    def fit_transform(self, X):
        raise NotImplementedError()

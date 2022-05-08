import numpy as np


class MinMaxScaler:
    def __init__(self, *, feature_range=(0, 1)):
        raise NotImplementedError()

    def fit(self, X):
        raise NotImplementedError()

    def transform(self, X):
        raise NotImplementedError()

    def fit_transform(self, X):
        raise NotImplementedError()

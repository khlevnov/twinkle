import numpy as np


class StandardScaler:
    def __init__(self, *, with_mean=False, with_std=False):
        raise NotImplementedError()
        
    def fit(self, X):
        raise NotImplementedError()

    def transform(self, X):
        raise NotImplementedError()

    def fit_transform(self, X):
        raise NotImplementedError()
